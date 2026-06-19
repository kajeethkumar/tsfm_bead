import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc

from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_recall_fscore_support # Added imports
from numpy.lib.stride_tricks import sliding_window_view
import warnings

warnings.filterwarnings('ignore')

# Project path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..")) 
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.data_preprocessing import preprocess, train_test_split
from evaluation.metrics import get_metrics, get_metrics_pred
from utils.utility import zscore

# ===========================================================
# Source Match: TimeEval-adapted KMeansAD
# ===========================================================
class KMeansAD(BaseEstimator, OutlierMixin):
    """
    This function is adapted from [TimeEval-algorithms] by [CodeLionX&wenig]
    Original source: [https://github.com/TimeEval/TimeEval-algorithms]
    """
    def __init__(self, k, window_size, stride, n_jobs=1, normalize=True):
        self.k = k
        self.window_size = window_size
        self.stride = stride
        self.model = KMeans(n_clusters=k, random_state=42)
        self.padding_length = 0
        self.normalize = normalize

    def _preprocess_data(self, X: np.ndarray) -> np.ndarray:
        flat_shape = (X.shape[0] - (self.window_size - 1), -1)  
        slides = sliding_window_view(X, window_shape=self.window_size, axis=0).reshape(flat_shape)[::self.stride, :]
        self.padding_length = X.shape[0] - (slides.shape[0] * self.stride + self.window_size - self.stride)
        if self.normalize: 
            slides = zscore(slides, axis=1, ddof=1)
        return slides

    def _custom_reverse_windowing(self, scores: np.ndarray) -> np.ndarray:
        begins = np.array([i * self.stride for i in range(scores.shape[0])])
        ends = begins + self.window_size

        unwindowed_length = self.stride * (scores.shape[0] - 1) + self.window_size + self.padding_length
        mapped = np.full(unwindowed_length, fill_value=np.nan)

        indices = np.unique(np.r_[begins, ends])
        for i, j in zip(indices[:-1], indices[1:]):
            window_indices = np.flatnonzero((begins <= i) & (j-1 < ends))
            mapped[i:j] = np.nanmean(scores[window_indices])

        np.nan_to_num(mapped, copy=False)
        return mapped

    def fit(self, X: np.ndarray, y=None, preprocess=True) -> 'KMeansAD':
        if preprocess:
            X = self._preprocess_data(X)
        self.model.fit(X)
        return self

    def predict(self, X: np.ndarray, preprocess=True) -> np.ndarray:
        if preprocess:
            X = self._preprocess_data(X)
        clusters = self.model.predict(X)
        diffs = np.linalg.norm(X - self.model.cluster_centers_[clusters], axis=1)
        return self._custom_reverse_windowing(diffs)

    def fit_predict(self, X, y=None) -> np.ndarray:
        X = self._preprocess_data(X)
        self.fit(X, y, preprocess=False)
        return self.predict(X, preprocess=False)


# ===========================================================
# Execution Pipeline
# ===========================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--window_size", type=int, default=168)
    parser.add_argument("--output_csv", type=str, default="./results/KMeans/kmeans_predictions.csv")
    parser.add_argument("--metrics_csv", type=str, default="./results/KMeans/kmeans_metrics.csv")
    parser.add_argument("--sliding_window", type=int, default=168)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Load and filter datasets
    df = pd.read_csv(args.input_csv)
    remove_bid = [32, 534, 558, 653, 693, 723, 739, 855, 910, 970, 1147, 1183, 1264, 1282]
    df = df[~df['building_id'].isin(remove_bid)]

    df1 = pd.read_csv('dataset/train_features.csv')
    valid_buildings = df1['building_id'].unique()
    df = df[df['building_id'].isin(valid_buildings)]
    
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_csv) or ".", exist_ok=True)

    for b_id in tqdm(df.building_id.unique(), desc="Evaluating buildings"):
        X, y = preprocess(df, b_id)

        X_train, X_val, X_test, \
        y_train, y_val, y_test = train_test_split(X, y)

        X_train_np = np.asarray(X_train).reshape(-1, 1)
        X_val_np   = np.asarray(X_val).reshape(-1, 1)
        X_test_np  = np.asarray(X_test).reshape(-1, 1)

        if len(X_train_np) < args.window_size:
            tqdm.write(f"Skipping building {b_id}: train length < window size")
            continue

        building = (
            df[df["building_id"] == b_id]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        timestamps = building["timestamp"].values
        test_timestamps = timestamps[-len(y_test):]
        test_readings = building["meter_reading"].values[-len(y_test):]

        # =====================================================
        # STEP 1 & 2: Train and Validate Hyperparameters
        # =====================================================
        candidate_k = [5, 10, 20, 30]
        best_k = args.k
        best_metric = -np.inf
        y_val_np = np.asarray(y_val).flatten()

        trained_models_pool = {}

        # Use validation set purely to find the best 'k' structure
        for k in candidate_k:
            try:
                candidate = KMeansAD(
                    k=k,
                    window_size=args.window_size,
                    stride=1,
                    normalize=True
                )
                candidate.fit(X_train_np)
                trained_models_pool[k] = candidate
                
                val_scores = candidate.predict(X_val_np)

                metrics = get_metrics(
                    score=val_scores,
                    labels=y_val_np,
                    pred=None,
                    slidingWindow=args.sliding_window
                )

                score = metrics.get("AUC_PR", metrics.get("AUC-PR", 0))

                if score > best_metric:
                    best_metric = score
                    best_k = k
            except Exception:
                continue

        if best_k in trained_models_pool:
            final_model = trained_models_pool[best_k]
        else:
            final_model = KMeansAD(k=best_k, window_size=args.window_size, stride=1, normalize=True)
            final_model.fit(X_train_np)

        # =====================================================
        # STEP 3: Test Set Inference Generation
        # =====================================================
        test_scores = final_model.predict(X_test_np)
        y_test_np = np.asarray(y_test).flatten()

        if len(test_scores) != len(y_test_np):
            min_len = min(len(test_scores), len(y_test_np))
            test_scores = test_scores[:min_len]
            y_test_np = y_test_np[:min_len]

        # =====================================================
        # STEP 4: Oracle Threshold Sweep (Test Set Only)
        # =====================================================
        best_test_f1 = -1.0
        best_test_threshold = 0.0
        best_test_contamination = 0.0
        best_test_preds = np.zeros_like(y_test_np)

        for contam in np.linspace(0.001, 0.15, 50):
            thresh = np.percentile(test_scores, 100 * (1 - contam))
            temp_preds = (test_scores >= thresh).astype(int)
            
            _, _, f1, _ = precision_recall_fscore_support(
                y_test_np, temp_preds, average='binary', zero_division=0
            )
            
            if f1 > best_test_f1:
                best_test_f1 = f1
                best_test_threshold = thresh
                best_test_contamination = contam
                best_test_preds = temp_preds

        # =====================================================
        # Metric Computations
        # =====================================================
        accuracy = accuracy_score(y_test_np, best_test_preds)
        precision, recall, f_score, _ = precision_recall_fscore_support(
            y_test_np, best_test_preds, average='binary', zero_division=0
        )

        ad_metrics = {}
        try:
            standard_metrics = get_metrics(
                score=test_scores,
                labels=y_test_np,
                pred=best_test_preds,
                slidingWindow=args.sliding_window
            )
            
            pred_metrics = get_metrics_pred(
                score=test_scores,
                labels=y_test_np,
                pred=best_test_preds,
                slidingWindow=args.sliding_window
            )
            
            ad_metrics = {**standard_metrics, **pred_metrics}
            
        except Exception as e:
            tqdm.write(f"[Warning] Building {b_id} test pipeline failed: {e}")

        # =====================================================
        # Write Outputs To File System
        # =====================================================
        building_metrics = {
            "building_id": b_id,
            "selected_k": best_k,
            "oracle_contamination": best_test_contamination,
            "used_threshold_value": best_test_threshold,
            **ad_metrics,
            "Test_Accuracy": accuracy,
            "Test_Precision": precision,
            "Test_Recall": recall,
            "Test_F-score": f_score,
        }

        pred_df = pd.DataFrame({
            "building_id": b_id,
            "timestamp": test_timestamps[:len(test_scores)],
            "meter_reading": test_readings[:len(test_scores)],
            "score": test_scores, 
            "label": y_test_np,
            "pred": best_test_preds 
        })

        if not pred_df.empty:
            pred_df.to_csv(args.output_csv, mode='a', header=not os.path.exists(args.output_csv), index=False)

        metric_df = pd.DataFrame([building_metrics])
        cols = ['building_id'] + [c for c in metric_df.columns if c != 'building_id']
        metric_df = metric_df[cols]
        metric_df.to_csv(args.metrics_csv, mode='a', header=not os.path.exists(args.metrics_csv), index=False)
        
        gc.collect() 

    print(f"KMeansAD Oracle evaluation complete.")
    print(f"Predictions incrementally saved to: {args.output_csv}")
    print(f"Metrics incrementally saved to: {args.metrics_csv}")