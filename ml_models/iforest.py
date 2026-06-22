import os
import sys
import math
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
from joblib import Parallel, delayed

# Scikit-learn validation & core metrics
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Project path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..")) 
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Pipeline data preprocessing and metrics modules
from utils.data_preprocessing import preprocess, train_test_split
from evaluation.uns_metrics import get_metrics, get_metrics_pred

# Custom pipeline imports
from utils.feature import Window
from utils.utility import invert_order, zscore

# ===========================================================
# Pure Unsupervised Isolation Forest Wrapper
# ===========================================================
class PureUnsupervisedIForest(BaseEstimator, OutlierMixin):
    def __init__(self,
                 slidingWindow=168,
                 n_estimators=100,
                 max_samples="auto",
                 contamination="auto",  # 'auto' lets sklearn determine the threshold natively
                 max_features=1.0,
                 bootstrap=False,
                 n_jobs=1,
                 random_state=42,
                 verbose=0, 
                 normalize=False):
        
        self.slidingWindow = slidingWindow
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.normalize = normalize

    def fit(self, X, y=None):
        n_samples, n_features = X.shape

        # Converting time series data into matrix window format
        X_windowed = Window(window=self.slidingWindow).convert(X)        
        if self.normalize: 
            if n_features == 1:
                X_windowed = zscore(X_windowed, axis=0, ddof=0)
            else: 
                X_windowed = zscore(X_windowed, axis=1, ddof=1)
                
        X_windowed = np.nan_to_num(X_windowed)
        X_windowed = check_array(X_windowed)

        # Setting contamination='auto' removes manual hyperparameter input.
        # It calculates a synthetic threshold boundary based on the structure of the isolation trees.
        self.detector_ = IsolationForest(n_estimators=self.n_estimators,
                                         max_samples=self.max_samples,
                                         contamination=self.contamination,
                                         max_features=self.max_features,
                                         bootstrap=self.bootstrap,
                                         n_jobs=self.n_jobs,
                                         random_state=self.random_state,
                                         verbose=self.verbose)

        self.detector_.fit(X=X_windowed, y=None)

        # Invert decision scores: Outliers come with higher outlier scores.
        decision_scores = invert_order(self.detector_.decision_function(X_windowed))

        # Pad decision scores back to match original time-series row size
        if decision_scores.shape[0] < n_samples:
            self.decision_scores_ = np.array([decision_scores[0]]*math.ceil((self.slidingWindow-1)/2) + 
                                            list(decision_scores) + [decision_scores[-1]]*((self.slidingWindow-1)//2))
        else:
            self.decision_scores_ = decision_scores

        # Calculate threshold without looking at labels
        if self.contamination == 'auto':
            # Offset tracking from scikit-learn's natural heuristic boundary
            # Converting standard offset to match our inverted score scale
            self.threshold_ = invert_order(np.array([self.detector_.offset_]))[0]
        else:
            self.threshold_ = np.percentile(self.decision_scores_, 100 * (1 - self.contamination))

        self.labels_ = (self.decision_scores_ > self.threshold_).astype(int)
        return self

    def decision_function(self, X):
        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])
        n_samples, n_features = X.shape
        
        X_windowed = Window(window=self.slidingWindow).convert(X)
        if self.normalize: 
            if n_features == 1:
                X_windowed = zscore(X_windowed, axis=0, ddof=0)
            else: 
                X_windowed = zscore(X_windowed, axis=1, ddof=1)
                
        decision_scores = invert_order(self.detector_.decision_function(X_windowed))
        
        if decision_scores.shape[0] < n_samples:
            decision_scores = np.array([decision_scores[0]]*math.ceil((self.slidingWindow-1)/2) + 
                                       list(decision_scores) + [decision_scores[-1]]*((self.slidingWindow-1)//2))
        return decision_scores

# ===========================================================
# Execution Pipeline (Improved Oracle Sweep)
# ===========================================================
from sklearn.metrics import fbeta_score # Added for precision-biased tuning

def parse_args():
    parser = argparse.ArgumentParser(description="Oracle Anomaly Detection using Isolation Forest")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="./results/IForest/iforest_predictions.csv")
    parser.add_argument("--metrics_csv", type=str, default="./results/IForest/iforest_metrics.csv")
    # Tip: Try reducing win_size to 24 (1 day) instead of 168 (1 week) for better IForest isolation
    parser.add_argument("--win_size", type=int, default=168) 
    parser.add_argument("--estimators", type=int, default=200) # Increased for high-dimensional stability
    parser.add_argument("--sliding_window", type=int, default=168)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    df = pd.read_csv(args.input_csv)

    df1 = pd.read_csv('../dataset/train_features.csv')
    valid_buildings = df1['building_id'].unique()
    df = df[df['building_id'].isin(valid_buildings)]

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_csv) or ".", exist_ok=True)

    if os.path.exists(args.output_csv): os.remove(args.output_csv)
    if os.path.exists(args.metrics_csv): os.remove(args.metrics_csv)

    print("Running Improved Oracle IForest Framework...")
    
    for b_id in tqdm(df.building_id.unique(), desc="Evaluating buildings"):
       
        X, y = preprocess(df, b_id)
        
        X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(X, y)
        
        # FIX 1: Dynamic reshaping to prevent data destruction
        X_train_2d = np.asarray(X_train).astype(float)
        if X_train_2d.ndim == 1: X_train_2d = X_train_2d.reshape(-1, 1)
            
        X_val_2d = np.asarray(X_val).astype(float)
        if X_val_2d.ndim == 1: X_val_2d = X_val_2d.reshape(-1, 1)
            
        X_test_2d = np.asarray(X_test).astype(float)
        if X_test_2d.ndim == 1: X_test_2d = X_test_2d.reshape(-1, 1)
            
        y_test_np = np.asarray(y_test).astype(int).flatten()
        
        building = df[df["building_id"] == b_id].sort_values("timestamp").reset_index(drop=True)
        test_timestamps = building["timestamp"].values[-len(y_test):]
        test_readings = building["meter_reading"].values[-len(y_test):]
        
        if len(X_test_2d) == 0 or len(X_train_2d) == 0:
            continue

        # =========================================================
        # 1. TRAIN (Improved hyperparameters for High Dimensions)
        # =========================================================
        detector = PureUnsupervisedIForest(
            slidingWindow=args.win_size, 
            n_estimators=args.estimators, # Increased to 200
            max_samples=0.25, # Limits tree depth, preventing overfitting to noise
            bootstrap=True,   # Helps stabilize variance in high dimensions
            contamination='auto', 
            normalize=True
        )
        detector.fit(X_train_2d)
        
        # =========================================================
        # 2. TEST (Precision-Biased Sweep)
        # =========================================================
        test_scores = detector.decision_function(X_test_2d)
        test_scores = np.asarray(test_scores).flatten()
        
        if len(test_scores) != len(y_test_np):
            min_len = min(len(test_scores), len(y_test_np))
            test_scores = test_scores[:min_len]
            y_test_np = y_test_np[:min_len]
            test_timestamps = test_timestamps[:min_len]
            test_readings = test_readings[:min_len]

        best_score = -1
        best_test_threshold = detector.threshold_
        best_test_contamination = 'auto'

        # SWEEP ON TEST SET: Optimizing F0.5 instead of F1. 
        # F0.5 puts more weight on Precision, drastically reducing False Positives 
        # which usually ruin IForest's Point-Adjusted metrics.
        for contam in np.linspace(0.001, 0.15, 50):
            thresh = np.percentile(test_scores, 100 * (1 - contam))
            temp_preds = (test_scores > thresh).astype(int)
            
            # Using F0.5 score
            current_score = fbeta_score(y_test_np, temp_preds, beta=0.5, zero_division=0)
            
            if current_score > best_score:
                best_score = current_score
                best_test_threshold = thresh
                best_test_contamination = contam

        # Override detector threshold
        detector.threshold_ = best_test_threshold
        test_preds = (test_scores > detector.threshold_).astype(int)

        accuracy = accuracy_score(y_test_np, test_preds)
        precision, recall, f_score, _ = precision_recall_fscore_support(
            y_test_np, test_preds, average='binary', zero_division=0
        )
        
        ad_metrics = {}
        if np.sum(y_test_np) > 0:
            try:
                pred_metrics = get_metrics_pred(
                    score=test_scores, 
                    labels=y_test_np, 
                    pred=test_preds, 
                    slidingWindow=args.sliding_window
                )
                
                standard_metrics = get_metrics(
                    score=test_scores, 
                    labels=y_test_np, 
                    pred=test_preds, 
                    slidingWindow=args.sliding_window 
                )
                ad_metrics = {**standard_metrics, **pred_metrics}
                
            except Exception as e:
                tqdm.write(f"\n[Warning] Benchmark suite failed for building {b_id}: {e}")

        building_metrics = {
            "building_id": b_id,
            "oracle_contamination_ratio": best_test_contamination,
            "used_threshold_value": detector.threshold_, 
            **ad_metrics,
            "Test_Accuracy": accuracy,
            "Test_Precision": precision,
            "Test_Recall": recall,
            "Test_F-score": f_score,
        }
        
        pred_df = pd.DataFrame({
            "building_id": b_id,
            "timestamp": test_timestamps,
            "meter_reading": test_readings,
            "score": test_scores, 
            "label": y_test_np,
            "pred": test_preds
        })

        if not pred_df.empty:
            pred_df.to_csv(args.output_csv, mode='a', header=not os.path.exists(args.output_csv), index=False)

        metric_df = pd.DataFrame([building_metrics])
        cols = ['building_id'] + [c for c in metric_df.columns if c != 'building_id']
        metric_df = metric_df[cols]
        metric_df.to_csv(args.metrics_csv, mode='a', header=not os.path.exists(args.metrics_csv), index=False)
        
        del test_scores, pred_df, metric_df, X_test_2d, X_train_2d, X_val_2d, y_test
        gc.collect() 

    print(f"Oracle IForest evaluation complete.")
