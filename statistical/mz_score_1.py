import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc

# PyTorch (for Dataset/DataLoader)
import torch
from torch.utils.data import Dataset, DataLoader

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
# Sliding Window Dataset
# ===========================================================
class SlidingWindowDataset(Dataset):
    """
    Creates sequences of length `win_size` for time series evaluation.
    """
    def __init__(self, X, y, win_size=168):
        # Flatten to ensure univariate time series is 1D
        self.X = np.asarray(X).flatten()
        self.y = np.asarray(y).flatten()
        self.win_size = win_size

    def __len__(self):
        return len(self.X) - self.win_size + 1

    def __getitem__(self, idx):
        X_win = self.X[idx : idx + self.win_size]
        y_win = self.y[idx : idx + self.win_size]
        return torch.tensor(X_win, dtype=torch.float32), torch.tensor(y_win, dtype=torch.float32)


# ===========================================================
# Local/Rolling Modified Z-Score Detector
# ===========================================================
class ModifiedZScoreDetector:
    def __init__(self):
        pass

    def decision_function(self, X_batch):
        """
        Calculates the Modified Z-Score over batches of sequences.
        X_batch shape: (batch_size, win_size)
        Returns the SIGNED MZ Score for the LAST point in each sequence.
        """
        epsilon = 1e-6
        
        if len(X_batch) == 0:
            return np.array([])
            
        # Calculate statistics across the sequence length (axis=1)
        median = np.median(X_batch, axis=1, keepdims=True)
        diff = X_batch - median
        mad = np.median(np.abs(diff), axis=1, keepdims=True)
        
        # Prevent division by zero
        mad[mad == 0] = epsilon
            
        # CORRECTED: M_i = 1.486 * (x_i - median) / MAD
        mz_scores = (1.486 * diff) / mad
        
        # Return the score of the LAST element in each sequence window
        return mz_scores[:, -1].flatten()


# ===========================================================
# Argument Parsing & Main Execution
# ===========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Rolling Anomaly detection using Modified Z-Score")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_csv", type=str, default="./results/MZ_Score/mz_predictions.csv")
    parser.add_argument("--metrics_csv", type=str, default="./results/MZ_Score/mz_metrics.csv")
    
    # Statistical Threshold & DataLoader parameters
    parser.add_argument("--threshold", type=float, default=3.5, help="Static threshold for Modified Z-Score")
    parser.add_argument("--win_size", type=int, default=168, help="Sequence length for data loader")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for DataLoader evaluation")
    
    parser.add_argument("--device", type=str, default="cpu", help="Ignored for MZ Score")
    parser.add_argument("--sliding_window", type=int, default=168, help="Tolerance window for metric evaluation")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load and filter data
    df = pd.read_csv(args.input_csv)
    df1 = pd.read_csv('dataset/train_features.csv')
    valid_buildings = df1['building_id'].unique()
    df = df[df['building_id'].isin(valid_buildings)]

    print("Initializing Rolling Modified Z-Score Detector...")
    detector = ModifiedZScoreDetector()

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_csv) or ".", exist_ok=True)

    for b_id in tqdm(df.building_id.unique(), desc="Evaluating buildings"):
       
        X, y = preprocess(df, b_id)
        
        # Split purely to isolate the test set
        X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(X, y)
        
        building = df[df["building_id"] == b_id].sort_values("timestamp").reset_index(drop=True)
        timestamps = building["timestamp"].values
        
        # Extract corresponding timestamps and readings for the test split
        # We slice off the first `win_size - 1` elements to align with sequence targets
        test_timestamps = timestamps[-len(y_test):][args.win_size - 1 :]
        reading_col = "meter_reading" 
        test_readings = building[reading_col].values[-len(y_test):][args.win_size - 1 :]

        # ---------------------------------------------------------
        # DataLoader Testing
        # ---------------------------------------------------------
        test_dataset = SlidingWindowDataset(X_test, y_test, win_size=args.win_size)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        z_scores_list = []
        labels_list = []
        
        for X_batch, y_batch in test_loader:
            # Calculate MZ Scores for the batch of sequences
            batch_scores = detector.decision_function(X_batch.numpy())
            z_scores_list.extend(batch_scores)
            
            # Extract the ground truth label of the LAST point in the sequence
            labels_list.extend(y_batch[:, -1].numpy())
            
        z_scores = np.array(z_scores_list)
        y_test_np = np.array(labels_list).astype(int)
        
        # Continuous proxy score strictly to prevent AUC/VUS metrics from crashing
        scores_for_metrics = np.abs(z_scores)

        # ---------------------------------------------------------
        # Static Threshold Application
        # ---------------------------------------------------------
        K = args.threshold
        best_pred_adj = ((z_scores > K) | (z_scores < -K)).astype(int)

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
                # Standard Metrics (AUC-ROC, AUC-PR, VUS)
                standard_metrics = get_metrics(
                    score=scores_for_metrics, 
                    labels=y_test_np, 
                    pred=best_pred_adj, 
                    slidingWindow=args.sliding_window 
                )  
                
                # Prediction-based adjustments (Point-Adjusted Metrics)
                pred_metrics = get_metrics_pred(
                    score=scores_for_metrics, 
                    labels=y_test_np, 
                    pred=best_pred_adj, 
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
            "threshold_used": K, 
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
        
        del z_scores, pred_df, metric_df, X_test, y_test, test_dataset, test_loader
        gc.collect() 

    print(f"Test evaluation complete.")
    print(f"Predictions incrementally saved to: {args.output_csv}")
    print(f"Metrics incrementally saved to: {args.metrics_csv}")
