import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc

# Scikit-learn metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.exceptions import UndefinedMetricWarning
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Keep your local custom imports
from utils.data_preprocessing import preprocess, train_test_split
from evaluation.metrics import get_metrics, get_metrics_pred

# ===========================================================
# Stateless Modified Z-Score Detector
# ===========================================================
class ModifiedZScoreDetector:
    def __init__(self):
        pass

    def decision_function(self, X_test):
        """
        Calculates the Modified Z-Score purely on the provided sequence.
        Returns the SIGNED MZ Score so the K loop can check both positive and negative bounds.
        """
        X_test = np.asarray(X_test).astype(float).flatten()
        epsilon = 1e-9
        
        if len(X_test) == 0:
            return np.array([])
            
        # Calculate statistics directly on the test set
        median = np.median(X_test)
        diff = X_test - median
        mad = np.median(np.abs(diff))
        
        # Prevent division by zero if the test sequence is perfectly flat
        if mad == 0:
            mad = epsilon
            
        # M_i = 1.486 * (x_i - median) / MAD
        mz_scores = (0.6745 * diff) / (mad + epsilon)
        
        return mz_scores # Returned signed scores (NO np.abs)


# ===========================================================
# Argument Parsing & Main Execution
# ===========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Anomaly detection using Modified Z-Score (Test-Only)")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_csv", type=str, default="./results/MZ_Score/mz_predictions.csv")
    parser.add_argument("--metrics_csv", type=str, default="./results/MZ_Score/mz_metrics.csv")
    
    # Leftover arguments to ensure your CLI command doesn't crash
    parser.add_argument("--device", type=str, default="cpu", help="Ignored for MZ Score")
    parser.add_argument("--win_size", type=int, default=168, help="Ignored for MZ Score")
    parser.add_argument("--batch_size", type=int, default=64, help="Ignored for MZ Score")
    parser.add_argument("--sliding_window", type=int, default=168, help="Tolerance window for metric evaluation")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load and filter data
    df = pd.read_csv(args.input_csv)
    df1 = pd.read_csv('dataset/train_features.csv')
    valid_buildings = df1['building_id'].unique()
    df = df[df['building_id'].isin(valid_buildings)]

    print("Initializing Stateless Modified Z-Score Detector...")
    detector = ModifiedZScoreDetector()

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_csv) or ".", exist_ok=True)

    for b_id in tqdm(df.building_id.unique(), desc="Evaluating buildings"):
       
        X, y = preprocess(df, b_id)
        
        # Split purely to isolate the test set
        X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(X, y)
        y_test_np = np.asarray(y_test).astype(int).flatten()
        
        building = df[df["building_id"] == b_id].sort_values("timestamp").reset_index(drop=True)
        timestamps = building["timestamp"].values
        
        # Extract corresponding timestamps and readings for the test split
        test_timestamps = timestamps[-len(y_test):]
        reading_col = "meter_reading" 
        test_readings = building[reading_col].values[-len(y_test):]
        
        if len(X_test) == 0:
            continue

        # ---------------------------------------------------------
        # Direct Testing (No Training Phase)
        # ---------------------------------------------------------
        z_scores = detector.decision_function(X_test)
        
        # Continuous proxy score strictly to prevent AUC/VUS metrics from crashing
        scores_for_metrics = np.abs(z_scores)

        # ---------------------------------------------------------
        # Oracle Threshold Sweep (Test Set Only)
        # ---------------------------------------------------------
        best_f1 = -1.0
        best_k = 0.0
        best_pred_adj = np.zeros_like(y_test_np)
    
        # A finer K sweep (0.5 to 10.0 in 0.1 increments) to ensure we hit the true Oracle peak
        for K in np.arange(0.5, 10.1, 0.1):
            low = -K
            high = K
            
            # Identify anomalies exceeding the K thresholds
            temp_preds = ((z_scores > high) | (z_scores < low)).astype(int)
            
            _, _, f1, _ = precision_recall_fscore_support(
                y_test_np, temp_preds, average='binary', zero_division=0
            )
            
            if f1 > best_f1:
                best_f1 = f1
                best_k = K
                best_pred_adj = temp_preds

        # ---------------------------------------------------------
        # Metric Computations
        # ---------------------------------------------------------
        accuracy = accuracy_score(y_test_np, best_pred_adj)
        precision, recall, f_score, _ = precision_recall_fscore_support(
            y_test_np, best_pred_adj, average='binary', zero_division=0
        )
        
        ad_metrics = {}
        if np.sum(y_test_np) == 0:
            tqdm.write(f"\n[Notice] Building {b_id} has 0 true anomalies. AUC/VUS skipped.")
        else:
            try:
                # 1. Standard Metrics (AUC-ROC, AUC-PR, VUS)
                standard_metrics = get_metrics(
                    score=scores_for_metrics, 
                    labels=y_test_np, 
                    pred=best_pred_adj, # Successfully feeding best test predictions
                    slidingWindow=args.sliding_window 
                )  
                
                # 2. Prediction-based adjustments (Point-Adjusted Metrics)
                pred_metrics = get_metrics_pred(
                    score=scores_for_metrics, 
                    labels=y_test_np, 
                    pred=best_pred_adj, # Successfully feeding best test predictions
                    slidingWindow=args.sliding_window
                )
                
                ad_metrics = {**standard_metrics, **pred_metrics}
                
            except Exception as e:
                tqdm.write(f"\n[Warning] Evaluation suite failed for building {b_id}. Reason: {e}")

        # ---------------------------------------------------------
        # Write Outputs To File System
        # ---------------------------------------------------------
        building_metrics = {
            "building_id": b_id,
            "best_k_multiplier": round(best_k, 2), 
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
            "score": scores_for_metrics,
            "label": y_test_np,
            "pred": best_pred_adj
        })

        if not pred_df.empty:
            pred_df.to_csv(args.output_csv, mode='a', header=not os.path.exists(args.output_csv), index=False)

        metric_df = pd.DataFrame([building_metrics])
        cols = ['building_id'] + [c for c in metric_df.columns if c != 'building_id']
        metric_df = metric_df[cols]
        metric_df.to_csv(args.metrics_csv, mode='a', header=not os.path.exists(args.metrics_csv), index=False)
        
        del z_scores, pred_df, metric_df, X_test, y_test
        gc.collect() 

    print(f"Test evaluation complete.")
    print(f"Predictions incrementally saved to: {args.output_csv}")
    print(f"Metrics incrementally saved to: {args.metrics_csv}")
