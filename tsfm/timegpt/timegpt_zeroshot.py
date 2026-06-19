import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
from nixtla import NixtlaClient

# PyTorch imports for Dataloader
import torch
from torch.utils.data import Dataset, DataLoader

# Scikit-learn metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.exceptions import UndefinedMetricWarning
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.data_preprocessing import preprocess, train_test_split
from evaluation.metrics import get_metrics, get_metrics_pred
from utils.tools import adjustment  

# ===========================================================
# PyTorch Sequence Dataset
# ===========================================================
class TimeSeriesDataset(Dataset):
    """
    Slices the time-series into overlapping windows of size `win_size`.
    Returns the sequence and its starting index so we can map timestamps later.
    """
    def __init__(self, X, win_size=168):
        self.X = np.asarray(X).astype(float)
        self.win_size = win_size

    def __len__(self):
        return max(0, len(self.X) - self.win_size + 1)

    def __getitem__(self, idx):
        seq = self.X[idx : idx + self.win_size]
        return torch.tensor(seq, dtype=torch.float32), idx

# ===========================================================
# TimeGPT Windowed Evaluator
# ===========================================================
class TimeGPTWindowedDetector:
    def __init__(self, api_key, win_size=168):
        self.nixtla_client = NixtlaClient(api_key=api_key)
        self.win_size = win_size

    def decision_function(self, dataloader, full_timestamps):
        """
        Processes batches of 168-length sequences. 
        Constructs a multi-series dataframe to minimize API calls to Nixtla.
        Scores the LAST point of each sequence.
        """
        all_scores = []
        
        for batch_x, batch_idx in dataloader:
            batch_x_np = batch_x.numpy()
            if batch_x_np.ndim == 3:
                batch_x_np = batch_x_np.squeeze(-1)

            df_list = []
            # 1. Build a multi-series DataFrame for the batch
            for b in range(len(batch_idx)):
                start_idx = batch_idx[b].item()
                seq = batch_x_np[b]
                ts_seq = full_timestamps[start_idx : start_idx + self.win_size]
                
                temp_df = pd.DataFrame({
                    "unique_id": f"win_{start_idx}",
                    "timestamp": ts_seq,
                    "meter_reading": seq
                })
                df_list.append(temp_df)
                
            batch_df = pd.concat(df_list, ignore_index=True)

            # 2. Call TimeGPT Anomaly Detection on the entire batch at once
            anomalies = self.nixtla_client.detect_anomalies(
                df=batch_df,
                time_col="timestamp",
                target_col="meter_reading",
                id_col="unique_id",
                freq="h",
                level=99 
            )

            # 3. Calculate continuous anomaly score = |Actual - Expected|
            merged = batch_df.merge(anomalies, on=["unique_id", "timestamp"], how="left")
            merged["score"] = np.abs(merged["meter_reading"] - merged["TimeGPT"]).fillna(0.0)

            # 4. Extract the score ONLY for the last point of each sequence
            last_rows = merged.groupby("unique_id").tail(1)
            score_dict = dict(zip(last_rows["unique_id"], last_rows["score"]))

            # Append in exact batch order
            for b in range(len(batch_idx)):
                start_idx = batch_idx[b].item()
                uid = f"win_{start_idx}"
                all_scores.append(score_dict.get(uid, 0.0))
                
        return np.array(all_scores)

# ===========================================================
# Argument Parsing / Setup
# ===========================================================
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot windowed anomaly detection using TimeGPT")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="./results/TimeGPT/timegpt_zeroshot_predictions.csv")
    parser.add_argument("--metrics_csv", type=str, default="./results/TimeGPT/timegpt_zeroshot_metrics.csv")
    parser.add_argument("--win_size", type=int, default=168)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--sliding_window", type=int, default=168)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # INITIALIZATION
    api_key = "nixak-899469406fc7fa38ac133c374ef0fde8484a06565f7ad9dc44f518c35b0b6c6b616aab0c6c8c6926" # Replace or use os.environ.get("NIXTLA_API_KEY")
    detector = TimeGPTWindowedDetector(api_key=api_key, win_size=args.win_size)

    df = pd.read_csv(args.input_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    remove_bid = [32, 534, 558, 653, 693, 723, 739, 855, 910, 970, 1147, 1183, 1264, 1282]
    df = df[~df['building_id'].isin(remove_bid)]

    if os.path.exists('dataset/train_features.csv'):
        df1 = pd.read_csv('dataset/train_features.csv')
        valid_buildings = df1['building_id'].unique()
        df = df[df['building_id'].isin(valid_buildings)]

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_csv) or ".", exist_ok=True)

    print(f"Evaluating TimeGPT via DataLoader (Window Size: {args.win_size})...")

    # -----------------------
    # PER-BUILDING LOOP
    # -----------------------
    for b_id in tqdm(df["building_id"].unique(), desc="Evaluating buildings"):
        try:
            X, y = preprocess(df, b_id)
            
            _, _, X_test, _, _, y_test = train_test_split(X, y)
            y_test_np = np.asarray(y_test).astype(int).flatten()

            if len(X_test) < args.win_size:
                tqdm.write(f"Skipping building {b_id}: test length < win_size")
                continue

            building = df[df["building_id"] == b_id].sort_values("timestamp").reset_index(drop=True)
            test_timestamps = building["timestamp"].values[-len(y_test):]
            test_readings = building["meter_reading"].values[-len(y_test):]

            # ---------------------------------------------------------
            # Dataloader Execution
            # ---------------------------------------------------------
            test_dataset = TimeSeriesDataset(X_test, win_size=args.win_size)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

            raw_scores = detector.decision_function(test_loader, test_timestamps)
            
            # Pad the beginning of the scores to match the original test length 
            # (since sliding window consumes the first win_size-1 points)
            pad_len = len(y_test_np) - len(raw_scores)
            if pad_len > 0:
                scores = np.pad(raw_scores, (pad_len, 0), mode='constant', constant_values=0)
            else:
                scores = raw_scores

            # ---------------------------------------------------------
            # Threshold Calculation: Maximize Point-Adjusted F1 Score
            # ---------------------------------------------------------
            best_f1 = -1.0
            best_threshold = np.min(scores)
            best_pred_adj = np.zeros_like(y_test_np)
            best_gt_adj = y_test_np.copy()

            candidate_thresholds = np.linspace(np.min(scores), np.max(scores), 100)

            for th in candidate_thresholds:
                binary_preds = (scores >= th).astype(int)
                
                gt_adj_temp, pred_adj_temp = adjustment(y_test_np.copy(), binary_preds.copy())
                
                _, _, f_score_temp, _ = precision_recall_fscore_support(
                    gt_adj_temp, pred_adj_temp, average='binary', zero_division=0
                )
                
                if f_score_temp > best_f1:
                    best_f1 = f_score_temp
                    best_threshold = th
                    best_pred_adj = pred_adj_temp
                    best_gt_adj = gt_adj_temp

            gt_adj = np.asarray(best_gt_adj).flatten()
            pred_adj = np.asarray(best_pred_adj).flatten()

            # Calculate Basic Metrics 
            accuracy = accuracy_score(gt_adj, pred_adj)
            precision, recall, f_score, _ = precision_recall_fscore_support(
                y_test_np, pred_adj, average='binary', zero_division=0
            )

            # Initialize advanced metrics
            ad_metrics = {}
            if np.sum(gt_adj) == 0:
                tqdm.write(f"\n[Notice] Building {b_id} has 0 true anomalies. AUC/VUS skipped.")
            else:
                try:
                    pred_metrics = get_metrics_pred(
                        score=scores, 
                        labels=gt_adj, 
                        pred=pred_adj, 
                        slidingWindow=args.sliding_window
                    )
                    
                    standard_metrics = get_metrics(
                        score=scores, 
                        labels=gt_adj, 
                        pred=pred_adj, 
                        slidingWindow=args.sliding_window 
                    )  
                    
                    ad_metrics = {**standard_metrics, **pred_metrics}
                except Exception as e:
                    tqdm.write(f"\n[Warning] Evaluation suite failed for building {b_id}: {e}")

            # ---------------------------------------------------------
            # Write Outputs To File System
            # ---------------------------------------------------------
            building_metrics = {
                "building_id": b_id,
                "best_threshold": best_threshold,
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

            del raw_scores, scores, pred_df, metric_df, test_loader, test_dataset, X_test, y_test
            gc.collect()

        except Exception as e:
            tqdm.write(f"\n[Error] Building {b_id} skipped due to internal failure: {e}")

    print(f"TimeGPT Windowed Oracle evaluation complete.")
    print(f"Predictions saved to: {args.output_csv}")
    print(f"Metrics saved to: {args.metrics_csv}")