import os
import gc
# Limit CPU threading to prevent CPU memory spikes
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import math
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.set_num_threads(4)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.exceptions import UndefinedMetricWarning
import warnings

from momentfm import MOMENTPipeline

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.dataset import ReconstructDataset_Moment
from utils.torch_utility import get_gpu
from utils.tools import adjustment  
from utils.data_preprocessing import preprocess, train_test_split
from evaluation.metrics import get_metrics, get_metrics_pred

# ===========================================================
# MOMENT Zero-Shot Model
# ===========================================================
class MOMENTZeroShot:
    def __init__(
        self,
        win_size=None,
        batch_size=None,
        model_name="AutonLab/MOMENT-1-base",
        cuda=True,
    ):
        self.win_size = win_size
        self.batch_size = batch_size
        self.device = get_gpu(cuda)
        
        self.anomaly_criterion = nn.MSELoss(reduction='none')

        self.model = MOMENTPipeline.from_pretrained(
            model_name,
            model_kwargs={"task_name": "reconstruction"},
        )

        self.model.init()
        self.model = self.model.to(self.device).float()
        self.model.eval()

    @torch.no_grad()
    def decision_function(self, data):
        data = np.asarray(data)

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        test_loader = DataLoader(
            dataset=ReconstructDataset_Moment(data, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=False,
        )

        score_list = []
        for batch_x, batch_masks in test_loader:
            batch_x = batch_x.float().to(self.device)
            batch_masks = batch_masks.to(self.device)                    
            
            # Permute to [Batch, Channels, Window]
            batch_x = batch_x.permute(0, 2, 1)

            output = self.model(x_enc=batch_x, input_mask=batch_masks)
            score = torch.mean(self.anomaly_criterion(batch_x, output.reconstruction), dim=-1).detach().cpu().numpy()[:, -1]
            score_list.append(score)

        anomaly_score = np.concatenate(score_list, axis=0).reshape(-1)

        if anomaly_score.shape[0] < len(data):
            anomaly_score = np.array([anomaly_score[0]]*math.ceil((self.win_size-1)/2) + 
                        list(anomaly_score) + [anomaly_score[-1]]*((self.win_size-1)//2))
            
        return anomaly_score

# ===========================================================
# Argument Parsing & Main Execution
# ===========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot anomaly detection using MOMENT")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_csv", type=str, default="./results/MOMENT/zeroshot/moment_zeroshot_predictions.csv")
    parser.add_argument("--metrics_csv", type=str, default="./results/MOMENT/zeroshot/moment_zeroshot.csv")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    
    parser.add_argument("--win_size", type=int, default=168, help="Window size for the model context")
    parser.add_argument("--sliding_window", type=int, default=168, help="Tolerance window for metric evaluation")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for data loader processing")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    use_cuda = (args.device == "cuda") and torch.cuda.is_available()
    if args.device == "cuda" and not use_cuda:
        print("CUDA requested but not available. Falling back to CPU.")

    df = pd.read_csv(args.input_csv)

    df1 = pd.read_csv('../../dataset/train_features.csv')
    valid_buildings = df1['building_id'].unique()
    df = df[df['building_id'].isin(valid_buildings)]
    # df = df[df['building_id'] > 1318]

    print("Loading MOMENT Model...")
    detector = MOMENTZeroShot(win_size=args.win_size, batch_size=args.batch_size, cuda=use_cuda)

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_csv) or ".", exist_ok=True)

    for b_id in tqdm(df.building_id.unique(), desc="Evaluating buildings"):
       
        X, y = preprocess(df, b_id)
        
        # Isolate Test Set 
        _, _, X_test, _, _, y_test = train_test_split(X, y)
        y_test_np = np.asarray(y_test).astype(int).flatten()
        
        if len(X_test) == 0:
            continue

        # 1. Get anomaly scores purely on Test Set
        scores = detector.decision_function(X_test)
        scores = np.asarray(scores).astype(float).flatten()
        
        building = df[df["building_id"] == b_id].sort_values("timestamp").reset_index(drop=True)
        timestamps = building["timestamp"].values
        test_timestamps = timestamps[-len(y_test):]
        test_readings = building["meter_reading"].values[-len(y_test):]

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
            
            # 1. Apply point adjustment
            gt_adj_temp, pred_adj_temp = adjustment(y_test_np.copy(), binary_preds.copy())
            
            # 2. Calculate F1-score on the ADJUSTED arrays
            _, _, f_score_temp, _ = precision_recall_fscore_support(
                gt_adj_temp, pred_adj_temp, average='binary', zero_division=0
            )
            
            # 3. Keep the best threshold and corresponding adjusted arrays
            if f_score_temp > best_f1:
                best_f1 = f_score_temp
                best_threshold = th
                best_pred_adj = pred_adj_temp
                best_gt_adj = gt_adj_temp

        # Force strict 1D arrays
        gt_adj = np.asarray(best_gt_adj).flatten()
        pred_adj = np.asarray(best_pred_adj).flatten()
        scores = np.asarray(scores).flatten()

        # Calculate Basic Metrics based on the best adjusted arrays
        accuracy = accuracy_score(gt_adj, pred_adj)
        precision, recall, f_score, _ = precision_recall_fscore_support(
            y_test_np, pred_adj, average='binary', zero_division=0
        )
        
        # Initialize advanced metrics safely
        ad_metrics = {}
        if np.sum(gt_adj) == 0:
            tqdm.write(f"\n[Notice] Building {b_id} has 0 true anomalies. AUC/VUS skipped.")
        else:
            try:
                # Prediction metrics
                pred_metrics = get_metrics_pred(
                    score=scores, 
                    labels=gt_adj, 
                    pred=pred_adj, 
                    slidingWindow=args.sliding_window
                )
                
                # Standard threshold-independent metrics
                standard_metrics = get_metrics(
                    score=scores, 
                    labels=gt_adj, 
                    pred=pred_adj, 
                    slidingWindow=args.sliding_window 
                )  
                
                ad_metrics = {**standard_metrics, **pred_metrics}
                
            except Exception as e:
                tqdm.write(f"\n[Warning] Evaluation suite failed for building {b_id}. Reason: {e}")

        # LOG SUMMARY RESULTS
        building_metrics = {
            "building_id": b_id,
            "best_threshold": best_threshold, 
            **ad_metrics,
            "Test_Accuracy": accuracy,
            "Test_Precision": precision,
            "Test_Recall": recall,
            "Test_F-score": f_score,
        }
        
        # Store predictions
        pred_df = pd.DataFrame({
            "building_id": b_id,
            "timestamp": test_timestamps,
            "meter_reading": test_readings,
            "score": scores,
            "label": gt_adj,
            "pred": pred_adj
        })

        # Incremental Saving
        if not pred_df.empty:
            pred_df.to_csv(args.output_csv, mode='a', header=not os.path.exists(args.output_csv), index=False)

        metric_df = pd.DataFrame([building_metrics])
        cols = ['building_id'] + [c for c in metric_df.columns if c != 'building_id']
        metric_df = metric_df[cols]
        metric_df.to_csv(args.metrics_csv, mode='a', header=not os.path.exists(args.metrics_csv), index=False)
        
        # Explicit VRAM Cleanup
        del scores, pred_df, metric_df, X_test, y_test
        if use_cuda:
            torch.cuda.empty_cache()
        gc.collect()

    print(f"Zero-shot Oracle evaluation complete.")
    print(f"Predictions incrementally saved to: {args.output_csv}")
    print(f"Metrics incrementally saved to: {args.metrics_csv}")
