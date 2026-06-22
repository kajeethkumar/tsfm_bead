import faulthandler
faulthandler.enable()
import gc
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'      # or '4'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
torch.set_num_threads(4)
import os
import sys
import math
import argparse
import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import subprocess as sp
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.exceptions import UndefinedMetricWarning
torch.backends.cudnn.enabled = False
from momentfm import MOMENTPipeline
from momentfm.utils.masking import Masking
import warnings

# Suppress the specific torch checkpoint warning
warnings.filterwarnings("ignore", message=".*use_reentrant.*")
from sklearn.exceptions import UndefinedMetricWarning

# Silences the precision zero-division warning to keep the console clean
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.dataset import ReconstructDataset_Moment
from utils.torch_utility import get_gpu # Removed EarlyStoppingTorch
from utils.tools import adjustment
from evaluation.metrics import get_metrics, get_metrics_pred
from utils.data_preprocessing import preprocess, train_test_split

# ===========================================================
# MOMENT Fine-Tuning Model
# ===========================================================
class MOMENTFineTuner:
    def __init__(
        self,
        win_size=168,
        batch_size=64,
        model_name="AutonLab/MOMENT-1-base", 
        cuda=True,
        epochs=20,
        lr=1e-4
    ):
        self.win_size = win_size
        self.batch_size = batch_size
        self.device = get_gpu(cuda)
        self.epochs = epochs
        self.lr = lr
        self.anomaly_criterion = nn.MSELoss(reduction='none')
        self.train_criterion = nn.MSELoss()

        self.model = MOMENTPipeline.from_pretrained(
            model_name,
            model_kwargs={"task_name": "reconstruction"},
        )

        self.model.init()
        self.model = self.model.to(self.device).float()
        self.criterion = torch.nn.MSELoss() 
        
        # Cache the pristine, pre-trained weights in memory
        # This allows us to instantly reset the model for each new building without re-downloading
        # self.base_state_dict = copy.deepcopy(self.model.state_dict())
        self.base_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def reset_weights(self):
        self.model.load_state_dict(self.base_state_dict)

    def fit(self, X_train, X_val, epochs=None, lr=1e-4, mask_ratio=0.3):
        if epochs is None:
            epochs = self.epochs
        
        X_train = np.asarray(X_train)
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
            
        X_val = np.asarray(X_val)
        if X_val.ndim == 1:
            X_val = X_val.reshape(-1, 1)

        # --- Setup Trainable Parameters ---
        for param in self.model.parameters(): 
            param.requires_grad = False
            
        for name, param in self.model.named_parameters():
            if "head" in name.lower(): 
                param.requires_grad = True

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.75)

        mask_generator = Masking(mask_ratio=mask_ratio)
        train_loader = DataLoader(ReconstructDataset_Moment(X_train, window_size=self.win_size), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(ReconstructDataset_Moment(X_val, window_size=self.win_size), batch_size=self.batch_size, shuffle=False)

        # In-Memory Early Stopping Variables
        best_loss = float('inf')
        patience = 3
        patience_counter = 0
        best_model_weights = None

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_train_loss = 0.0
            
            for batch_x, batch_masks in tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch}", leave=False):
                batch_x = batch_x.to(self.device).float().permute(0, 2, 1)
                original = batch_x
                n_channels = batch_x.shape[1]
                batch_x = batch_x.reshape((-1, 1, self.win_size)) 
                batch_masks = batch_masks.to(self.device).long().repeat_interleave(n_channels, axis=0)
                
                mask = mask_generator.generate_mask(x=batch_x, input_mask=batch_masks).to(self.device).long()
                mask = torch.nn.functional.pad(mask, (0, batch_masks.size(1) - mask.size(1)), mode='constant', value=1)

                optimizer.zero_grad()
                output = self.model(x_enc=batch_x, input_mask=batch_masks, mask=mask).reconstruction
                output = torch.nn.functional.pad(output, (0, original.size(2)-output.size(2)), mode='replicate')
                output = output.reshape(original.size(0), n_channels, self.win_size)

                loss = self.train_criterion(output, original)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                
            self.model.eval()
            avg_loss = 0
            with torch.no_grad():
                for batch_x, batch_masks in val_loader:
                    batch_x = batch_x.to(self.device).float()
                    batch_masks = batch_masks.to(self.device)
                    batch_x = batch_x.permute(0,2,1)

                    output = self.model(x_enc=batch_x, input_mask=batch_masks) 
                    loss = self.criterion(output.reconstruction.reshape(-1, n_channels, self.win_size), batch_x)
                    avg_loss += loss.cpu().item()

            valid_loss = avg_loss/max(len(val_loader), 1)
            scheduler.step() # Fixed scheduler warning
            
            # --- In-Memory Early Stopping Logic ---
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                tqdm.write(f"   Early stopping at epoch {epoch}<<<")
                break

        # --- CRITICAL: Load Best Weights from RAM ---
        if best_model_weights is not None:
            self.model.load_state_dict(best_model_weights)
            # tqdm.write("Restored best model weights from memory.")

    @torch.no_grad()
    def decision_function(self, data):
        self.model.eval()
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
    parser = argparse.ArgumentParser(description="Fine-tuned anomaly detection using MOMENT")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_csv", type=str, default="./results/MOMENT/finetuned/moment_finetuned_predictions.csv", help="Output CSV filename for predictions")
    parser.add_argument("--metrics_csv", type=str, default="./results/MOMENT/finetuned/moment_finetuned.csv", help="Output CSV filename for evaluation metrics")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    
    parser.add_argument("--win_size", type=int, default=168, help="Window size for the model context and metric evaluation")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for data loader processing")
    parser.add_argument("--epochs", type=int, default=20, help="Number of fine-tuning epochs per building")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for fine-tuning")
    parser.add_argument("--sliding_window", type=int, default=168, help="Tolerance window for metric evaluation")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    use_cuda = (args.device == "cuda") and torch.cuda.is_available()
    if args.device == "cuda" and not use_cuda:
        print("CUDA requested but not available. Falling back to CPU.")

    df = pd.read_csv(args.input_csv)

    df1 = pd.read_csv('dataset/train_features.csv')

    # Keep only building_ids present in df1
    valid_buildings = df1['building_id'].unique()
    df = df[df['building_id'].isin(valid_buildings)]
    df = df[df['building_id'] > 1319]


    print("Loading MOMENT Model...")
    detector = MOMENTFineTuner(win_size=args.win_size, batch_size=args.batch_size, cuda=use_cuda)

    all_predictions = []
    all_metrics = []

    # Make sure we iterate through all valid buildings, not just the first one ([:1])
    for b_id in tqdm(df.building_id.unique(), desc="Evaluating buildings"):
        X, y = preprocess(df, b_id)
        
        X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(X, y)
        y_test_np = np.asarray(y_test).astype(int).flatten()
        
        building = df[df["building_id"] == b_id].sort_values("timestamp").reset_index(drop=True)
        timestamps = building["timestamp"].values
        test_timestamps = timestamps[-len(y_test):]
        
        reading_col = "meter_reading" 
        test_readings = building[reading_col].values[-len(y_test):]

        # ---------------------------------------------------------
        # Per-Building Fine-Tuning 
        # ---------------------------------------------------------
        # 1. Purge knowledge of previous buildings and restore to pre-trained base
        detector.reset_weights() 
        
        # 2. Train uniquely on this building's data
        tqdm.write(f"\n--- Fine-tuning Building {b_id} ---")
        detector.fit(X_train, X_val, epochs=args.epochs, lr=args.lr)
        
        # ---------------------------------------------------------
        # Testing Execution (Test Set Only)
        # ---------------------------------------------------------
        # 3. Evaluate the newly adapted model
        scores = detector.decision_function(X_test)
        scores = np.asarray(scores).astype(float).flatten()
        
        # ---------------------------------------------------------
        # Threshold Calculation: Maximize Point-Adjusted F1 Score
        # ---------------------------------------------------------
        best_f1 = 0
        best_threshold = np.min(scores)  # Default to min score if all thresholds fail
        best_pred_adj = None
        best_gt_adj = None
    
        candidate_thresholds = np.linspace(np.min(scores), np.max(scores), 100)
            
        for th in candidate_thresholds:
            binary_preds = (scores >= th).astype(int)
            
            # Apply point adjustment
            gt_adj_temp, pred_adj_temp = adjustment(y_test_np.copy(), binary_preds.copy())
            
            # Calculate F1-score on the ADJUSTED arrays
            _, _, f_score_temp, _ = precision_recall_fscore_support(
                gt_adj_temp, pred_adj_temp, average='binary', zero_division=0
            )
            
            # Keep the best threshold and corresponding adjusted arrays
            if f_score_temp > best_f1:
                best_f1 = f_score_temp # Fixed np.max() issue here as well
                best_threshold = th
                best_pred_adj = pred_adj_temp
                best_gt_adj = gt_adj_temp

        # Set final predictions to the optimal found
        pred_adj = best_pred_adj
        gt_adj = best_gt_adj
        threshold = best_threshold

        # Calculate Basic Metrics based on the best adjusted arrays
        accuracy = accuracy_score(gt_adj, pred_adj)
        precision, recall, f_score, _ = precision_recall_fscore_support(
            gt_adj, pred_adj, average='binary', zero_division=0 # Changed y_test_np to gt_adj
        )
        
        # Get Advanced metrics using the adjusted predictions
        ad_metrics = get_metrics_pred(score=scores, labels=gt_adj, pred=pred_adj, slidingWindow=args.sliding_window)
        
        # Update with threshold-independent metrics safely
        try:
            ad_metrics.update(get_metrics(
                score=scores, 
                labels=gt_adj, 
                pred=None, 
                slidingWindow=args.sliding_window 
            ))  
        except Exception as e:
            # Prevent the pipeline from crashing if the metric library fails on a single building
            tqdm.write(f"\n[Warning] AUC/VUS metrics failed for building {b_id}. Reason: {e}")

        # LOG SUMMARY RESULTS
        building_metrics = {
            "building_id": b_id,
            "best_threshold": best_threshold, 
            **ad_metrics,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F-score": f_score,
        }
        
        all_metrics.append(building_metrics)

        # Store predictions
        pred_df = pd.DataFrame({
            "building_id": b_id,
            "timestamp": test_timestamps,
            "meter_reading": test_readings,
            "score": scores,
            "label": gt_adj,
            "pred": pred_adj
        })
        all_predictions.append(pred_df)
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(args.metrics_csv) or ".", exist_ok=True)
        
        # Append Predictions
        if not pred_df.empty:
            pred_df.to_csv(args.output_csv, mode='a', header=not os.path.exists(args.output_csv), index=False)

        # Append Metrics
        metric_df = pd.DataFrame([building_metrics])
        cols = ['building_id'] + [c for c in metric_df.columns if c != 'building_id']
        metric_df = metric_df[cols]
        metric_df.to_csv(args.metrics_csv, mode='a', header=not os.path.exists(args.metrics_csv), index=False)

        torch.cuda.empty_cache()
        gc.collect()

    print(f"Fine-tuning evaluation complete.")
    print(f"Predictions incrementally saved to: {args.output_csv}")
    print(f"Metrics incrementally saved to: {args.metrics_csv}")
