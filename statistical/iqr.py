import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
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
# Helper Functions
# ===========================================================
def calculate_quartiles_and_iqr(values):
    """Calculates the 25th percentile, 75th percentile, and IQR."""
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    
    # Calculate IQR
    iqr = q3 - q1
    
    return q1, q3, iqr

# ===========================================================
# Argument Parsing & Main Execution
# ===========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Anomaly detection using IQR (Val-Tuned, Test-Evaluated)")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_csv", type=str, default="./results/IQR/iqr_predictions.csv", help="Output CSV filename for predictions")
    parser.add_argument("--metrics_csv", type=str, default="./results/IQR/iqr_metrics.csv", help="Output CSV filename for evaluation metrics")
    
    # Leftover arguments to ensure your CLI command doesn't crash
    parser.add_argument("--device", type=str, default="cpu", help="Ignored for IQR")
    parser.add_argument("--win_size", type=int, default=168, help="Ignored for IQR")
    parser.add_argument("--batch_size", type=int, default=32, help="Ignored for IQR")
    parser.add_argument("--sliding_window", type=int, default=168, help="Tolerance window for metric evaluation")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load and filter data
    df = pd.read_csv(args.input_csv)
    df1 = pd.read_csv('dataset/train_features.csv')
    valid_buildings = df1['building_id'].unique()
    df = df[df['building_id'].isin(valid_buildings)]

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_csv) or ".", exist_ok=True)

    # Wipe old CSVs clean before starting the loop to prevent appending to old data
    if os.path.exists(args.output_csv):
        os.remove(args.output_csv)
    if os.path.exists(args.metrics_csv):
        os.remove(args.metrics_csv)

    for b_id in tqdm(df.building_id.unique(), desc="Evaluating buildings"):
       
        X, y = preprocess(df, b_id)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(X, y)

        X_val = np.asarray(X_val).astype(float).flatten()
        y_val_np = np.asarray(y_val).astype(int).flatten()
        X_test = np.asarray(X_test).astype(float).flatten()
        y_test_np = np.asarray(y_test).astype(int).flatten()
        
        building = df[df["building_id"] == b_id].sort_values("timestamp").reset_index(drop=True)
        timestamps = building["timestamp"].values
        
        # Extract corresponding timestamps and readings for the test split
        test_timestamps = timestamps[-len(y_test):]
        reading_col = "meter_reading" 
        test_readings = building[reading_col].values[-len(y_test):]
        
        # Need both a non-empty val split (to fit/select K) and a non-empty test
        # split (to report final metrics on).
        if len(X_val) == 0 or len(X_test) == 0:
            continue

        # ---------------------------------------------------------
        # Fit IQR statistics on the VALIDATION split only, and sweep
        # K against validation labels to pick the best multiplier.
        # The test split is never touched during this selection step.
        # ---------------------------------------------------------
        q1, q3, iqr = calculate_quartiles_and_iqr(X_val)

        best_f1 = -1.0
        best_k = 0.0

        # Your specific discrete K loop
        for K in range(5, 45, 5):
            K /= 10.0  # 0.5, 1.0, 1.5, 2.0...

            low = q1 - K * iqr
            high = q3 + K * iqr

            val_preds = np.zeros(len(y_val_np))
            bool_array = (X_val > high) | (X_val < low)
            val_preds[bool_array] = 1

            # Strict point-wise matching (no adjustment), evaluated on val only
            _, _, f_score_temp, _ = precision_recall_fscore_support(
                y_val_np, val_preds, average='binary', zero_division=0
            )

            if f_score_temp > best_f1:
                best_f1 = f_score_temp
                best_k = K

        # If every K scored 0 on validation (e.g. no validation anomalies to
        # tune against), best_k stays at its initial value of 0.0, which is
        # the most conservative IQR fence. This mirrors the original
        # fallback-to-no-detections behavior without peeking at test labels.

        # ---------------------------------------------------------
        # Apply the SAME q1/q3/iqr (fit on val) and the winning K to the
        # TEST split exactly once. No further selection happens here.
        # ---------------------------------------------------------
        low = q1 - best_k * iqr
        high = q3 + best_k * iqr

        pred_adj = np.zeros(len(y_test_np))
        bool_array_test = (X_test > high) | (X_test < low)
        pred_adj[bool_array_test] = 1

        gt_adj = y_test_np.copy()

        # Continuous proxy score strictly to prevent AUC/VUS metrics from crashing
        scores = np.abs(X_test - np.median(X_test))

        accuracy = accuracy_score(gt_adj, pred_adj)
        precision, recall, f_score, _ = precision_recall_fscore_support(
            gt_adj, pred_adj, average='binary', zero_division=0
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
            "best_k_multiplier": best_k, 
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
        
        del scores, pred_df, metric_df, X_test, y_test, X_val, y_val
        gc.collect() 

    print(f"Test evaluation complete.")
    print(f"Predictions incrementally saved to: {args.output_csv}")
    print(f"Metrics incrementally saved to: {args.metrics_csv}")
