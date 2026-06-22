import os
import sys
import math
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc

# Scikit-learn metrics & LOF
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.utils.validation import check_array
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..")) 
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Keep your local custom pipeline imports
from utils.data_preprocessing import preprocess, train_test_split
from evaluation.metrics import get_metrics, get_metrics_pred
from utils.feature import Window
from utils.utility import invert_order, zscore

# ===========================================================
# Standalone LOF Detector (Refined based on PyOD logic)
# ===========================================================
class LOF:
    """Wrapper of scikit-learn LOF Class with more functionalities.
    Unsupervised Outlier Detection using Local Outlier Factor (LOF).
    """

    def __init__(self, slidingWindow=100, sub=True, n_neighbors=20, algorithm='auto', leaf_size=30,
                 metric='minkowski', p=2, metric_params=None,
                 contamination=0.1, n_jobs=1, novelty=True, normalize=True):
        
        self.contamination = contamination
        self.slidingWindow = slidingWindow
        self.sub = sub
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.novelty = novelty
        self.normalize = normalize

    def fit(self, X, y=None):
        n_samples, n_features = X.shape

        # Converting time series data into matrix format
        X = Window(window=self.slidingWindow).convert(X)
        if self.normalize: 
            if n_features == 1:
                X = zscore(X, axis=0, ddof=0)
            else: 
                X = zscore(X, axis=1, ddof=1)
                
        # validate inputs X
        X = check_array(X)

        self.detector_ = LocalOutlierFactor(n_neighbors=self.n_neighbors,
                                            algorithm=self.algorithm,
                                            leaf_size=self.leaf_size,
                                            metric=self.metric,
                                            p=self.p,
                                            metric_params=self.metric_params,
                                            contamination=self.contamination,
                                            n_jobs=self.n_jobs,
                                            novelty=self.novelty)
        self.detector_.fit(X=X, y=y)

        # Invert decision_scores_. Outliers comes with higher outlier scores
        self.decision_scores_ = invert_order(self.detector_.negative_outlier_factor_)

        # padded decision_scores_
        if self.decision_scores_.shape[0] < n_samples:
            self.decision_scores_ = np.array([self.decision_scores_[0]]*math.ceil((self.slidingWindow-1)/2) + 
                        list(self.decision_scores_) + [self.decision_scores_[-1]]*((self.slidingWindow-1)//2))

        # --- REFINEMENT: Calculate PyOD native threshold based on contamination ---
        self.threshold_ = np.percentile(self.decision_scores_, 100 * (1 - self.contamination))
        self.labels_ = (self.decision_scores_ > self.threshold_).astype(int)

        return self

    def decision_function(self, X):
        if not hasattr(self, 'decision_scores_'):
            raise ValueError("This LOF instance is not fitted yet.")

        n_samples, n_features = X.shape
        # Converting time series data into matrix format
        X = Window(window=self.slidingWindow).convert(X)
        if self.normalize: 
            if n_features == 1:
                X = zscore(X, axis=0, ddof=0)
            else: 
                X = zscore(X, axis=1, ddof=1)
                
        # Invert outlier scores. Outliers comes with higher outlier scores
        try:
            decision_scores_ = invert_order(self.detector_._score_samples(X))
        except AttributeError:
            try:
                decision_scores_ = invert_order(self.detector_._decision_function(X))
            except AttributeError:
                decision_scores_ = invert_order(self.detector_.score_samples(X))

        # padded decision_scores_
        if decision_scores_.shape[0] < n_samples:
            decision_scores_ = np.array([decision_scores_[0]]*math.ceil((self.slidingWindow-1)/2) + 
                        list(decision_scores_) + [decision_scores_[-1]]*((self.slidingWindow-1)//2))
        return decision_scores_

    @property
    def n_neighbors_(self):
        return self.detector_.n_neighbors_


# ===========================================================
# Argument Parsing & Main Execution
# ===========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Anomaly detection using Custom LOF Class")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_csv", type=str, default="./results/LOF/lof_predictions.csv", help="Output CSV filename for predictions")
    parser.add_argument("--metrics_csv", type=str, default="./results/LOF/lof_metrics.csv", help="Output CSV filename for evaluation metrics")
    parser.add_argument("--win_size", type=int, default=168, help="Sliding window size for LOF matrix generation")
    parser.add_argument("--neighbors", type=int, default=20, help="Number of neighbors for LOF")
    parser.add_argument("--sliding_window", type=int, default=168, help="Tolerance window for metric evaluation")
    
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load and filter data
    df = pd.read_csv(args.input_csv)
    remove_bid = [32, 534, 558, 653, 693, 723, 739, 855, 910, 970, 1147, 1183, 1264, 1282]
    df = df[~df['building_id'].isin(remove_bid)]

    df1 = pd.read_csv('dataset/train_features.csv')
    valid_buildings = df1['building_id'].unique()
    df = df[df['building_id'].isin(valid_buildings)]

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_csv) or ".", exist_ok=True)

    if os.path.exists(args.output_csv): os.remove(args.output_csv)
    if os.path.exists(args.metrics_csv): os.remove(args.metrics_csv)

    print(f"Initializing Custom LOF Detector (Window: {args.win_size}, Neighbors: {args.neighbors})...")
    
    # Initialize with default contamination, though we will sweep dynamically later
    detector = LOF(slidingWindow=args.win_size, n_neighbors=args.neighbors, novelty=True, contamination=0.1)

    for b_id in tqdm(df.building_id.unique(), desc="Evaluating buildings"):
       
        X, y = preprocess(df, b_id)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(X, y)
        
        X_train_2d = np.asarray(X_train).astype(float).reshape(-1, 1)
        X_test_2d = np.asarray(X_test).astype(float).reshape(-1, 1)
        y_test_np = np.asarray(y_test).astype(int).flatten()
        
        building = df[df["building_id"] == b_id].sort_values("timestamp").reset_index(drop=True)
        timestamps = building["timestamp"].values
        
        test_timestamps = timestamps[-len(y_test):]
        reading_col = "meter_reading" 
        test_readings = building[reading_col].values[-len(y_test):]
        
        if len(X_test_2d) == 0 or len(X_train_2d) == 0:
            continue

        # ---------------------------------------------------------
        # LOF Training & Testing
        # ---------------------------------------------------------
        detector.fit(X_train_2d)
        
        scores = detector.decision_function(X_test_2d)
        scores = np.asarray(scores).flatten()

        best_f1 = -1.0
        best_threshold = np.min(scores)
        best_pred_adj = None
        best_gt_adj = None
    
        # --- REFINEMENT: Sweep the Contamination Percentile! ---
        # Instead of raw score values, we sweep assuming anywhere from 0.1% to 50% of the data is an anomaly
        candidate_contaminations = np.linspace(0.001, 0.5, 100) 
            
        for contam in candidate_contaminations:
            # Calculate threshold using PyOD's exact percentile definition
            th = np.percentile(scores, 100 * (1 - contam))
            
            binary_preds = (scores >= th).astype(int)
            
            # Strict point-wise matching (no point-adjustment padding)
            gt_adj_temp = y_test_np.copy()
            pred_adj_temp = binary_preds.copy()
            
            _, _, f_score_temp, _ = precision_recall_fscore_support(
                gt_adj_temp, pred_adj_temp, average='binary', zero_division=0
            )
            
            if f_score_temp > best_f1:
                best_f1 = f_score_temp 
                best_threshold = th
                best_pred_adj = pred_adj_temp
                best_gt_adj = gt_adj_temp

        if best_pred_adj is None:
            best_pred_adj = np.zeros(len(y_test_np))
            best_gt_adj = y_test_np.copy()

        gt_adj = np.asarray(best_gt_adj).flatten()
        pred_adj = np.asarray(best_pred_adj).flatten()
        
        accuracy = accuracy_score(gt_adj, pred_adj)
        precision, recall, f_score, _ = precision_recall_fscore_support(
            y_test_np, pred_adj, average='binary', zero_division=0
        )
        
        ad_metrics = {}
        if np.sum(gt_adj) == 0:
            tqdm.write(f"\n[Notice] Building {b_id} has 0 true anomalies. AUC/VUS skipped.")
        else:
            try:
                ad_metrics = get_metrics_pred(score=scores, labels=gt_adj, pred=pred_adj, slidingWindow=args.sliding_window)
                ad_metrics.update(get_metrics(
                    score=scores, 
                    labels=gt_adj, 
                    pred=None, 
                    slidingWindow=args.sliding_window 
                ))  
            except Exception as e:
                tqdm.write(f"\n[Warning] AUC/VUS metrics failed for building {b_id}. Reason: {e}")

        building_metrics = {
            "building_id": b_id,
            "best_threshold": best_threshold, 
            **ad_metrics,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F-score": f_score,
        }
        
        pred_df = pd.DataFrame({
            "building_id": b_id,
            "timestamp": test_timestamps,
            "meter_reading": test_readings,
            "score": scores, 
            "label": gt_adj,
            "pred": pred_adj
        })

        if not pred_df.empty:
            pred_df.to_csv(args.output_csv, mode='a', header=not os.path.exists(args.output_csv), index=False)

        metric_df = pd.DataFrame([building_metrics])
        cols = ['building_id'] + [c for c in metric_df.columns if c != 'building_id']
        metric_df = metric_df[cols]
        metric_df.to_csv(args.metrics_csv, mode='a', header=not os.path.exists(args.metrics_csv), index=False)
        
        del scores, pred_df, metric_df, X_test_2d, X_train_2d, y_test
        gc.collect() 

    print(f"LOF evaluation complete.")
    print(f"Predictions incrementally saved to: {args.output_csv}")
    print(f"Metrics incrementally saved to: {args.metrics_csv}")