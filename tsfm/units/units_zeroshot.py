import os
import sys
import argparse
import random
import gc
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

# Scikit-learn core metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.exceptions import UndefinedMetricWarning
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ===========================
# Pathing & Imports
# ===========================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the UniTS Model architecture & Pipeline modules
from UniTS import Model 
from utils.data_preprocessing import preprocess, train_test_split
from evaluation.metrics import get_metrics, get_metrics_pred
from utils.tools import adjustment

# ===========================================================
# Determinism Configuration
# ===========================================================
def fix_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Disable cuDNN backend descriptors to completely avoid version mismatch crashes
        torch.backends.cudnn.enabled = False  
        
    print(f"Global seed set to: {seed}")

# =========================================================
# Dataset: Windowing with Local Normalization
# =========================================================
class BuildingDataset(Dataset):
    def __init__(self, df, building_id, win_size, local_normalize=False):
        """
        Custom Dataset for managing sliding window data extraction.
        """
        self.win_size = win_size
        self.local_normalize = local_normalize
        
        X, y = preprocess(df, building_id)
        _, _, X_test, _, _, y_test = train_test_split(X, y)
        
        self.data = np.asarray(X_test).astype(np.float32)
        self.labels = np.asarray(y_test).astype(np.int32).flatten()

    def __len__(self):
        return len(self.data) - self.win_size + 1

    def __getitem__(self, idx):
        window = self.data[idx : idx + self.win_size].copy()
        
        if self.local_normalize:
            mean = window.mean()
            std = window.std() + 1e-6
            window = (window - mean) / std
        
        return (
            torch.tensor(window, dtype=torch.float32),
            torch.tensor(self.labels[idx + self.win_size - 1], dtype=torch.float32)
        )

# =========================================================
# Scoring Logic (Reconstruction Error)
# =========================================================
def units_score_fn(model, loader, device):
    model.eval()
    mse = nn.MSELoss(reduction="none")
    scores_all = []

    with torch.no_grad():
        for batch_x, _ in loader:
            if batch_x.dim() == 2:
                batch_x = batch_x.unsqueeze(-1)
            batch_x = batch_x.to(device)

            outputs = model(
                x_enc=batch_x,
                x_mark_enc=None,
                task_id=0,
                task_name="anomaly_detection"
            )

            error = mse(outputs, batch_x).mean(dim=(1, 2))
            scores_all.append(error.cpu().numpy())

    scores = np.concatenate(scores_all).flatten()
    return gaussian_filter1d(scores, sigma=1.5)

# =========================================================
# Main Evaluation Loop
# =========================================================
def main():
    parser = argparse.ArgumentParser("UniTS Zero-Shot Pretrained Evaluation Pipeline")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--ckpt_path", type=str, default="./tsfm/units/units_x32_pretrain_checkpoint.pth", help="Model checkpoint path")
    parser.add_argument("--metrics_csv", type=str, default="results/UniTS/zeroshot/metrics/UniTS_ZeroShot.csv", help="Path to output summary metrics CSV file")
    parser.add_argument("--predictions_csv", type=str, default="results/UniTS/zeroshot/predictions/UniTS_ZeroShot_Predictions.csv", help="Path to aggregated predictions CSV file")
    parser.add_argument("--win_size", type=int, default=168, help="Window frame context size") 
    parser.add_argument("--sliding_window", type=int, default=168, help="Metric evaluation shift scope")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch allocation dimension")
    parser.add_argument("--seed", type=int, default=42, help="Seed value for deterministic execution")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Target hardware execution layer")
    
    # Architecture settings matching 'units_x128' pretraining specification
    parser.add_argument("--prompt_num", type=int, default=10)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--e_layers", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--patch_len", type=int, default=1) 
    parser.add_argument("--stride", type=int, default=1) 
    parser.add_argument("--dropout", type=float, default=0.01)
    
    args = parser.parse_args()
    fix_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    # Directory layout initialization
    os.makedirs(os.path.dirname(args.metrics_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.predictions_csv) or ".", exist_ok=True)

    # Clean old run tracks
    if os.path.exists(args.metrics_csv):
        os.remove(args.metrics_csv)
    if os.path.exists(args.predictions_csv):
        os.remove(args.predictions_csv)

    # Define the task configuration context for UniTS
    configs_list = [
        ("AD_Task", {
            "task_name": "anomaly_detection",
            "dataset": "building_data",
            "seq_len": args.win_size,
            "enc_in": 1
        })
    ]

    print(f"Loading UniTS model architecture on environment: [{device}]...")
    model = Model(args, configs_list, pretrain=False).to(device)
    
    print(f"Loading weights from {args.ckpt_path}...")
    
    # --- ALLOWLIST NUMPY STRUCTS FOR PYTORCH 2.6+ SECURE UNPICKLING ---
    import numpy
    torch.serialization.add_safe_globals([
        numpy.core.multiarray.scalar, 
        numpy.dtype
    ])
    # -------------------------------------------------------------------
    
    checkpoint = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Weights Load Status: {msg}")
    model.eval()

    # Data ingestion filtering
    df = pd.read_csv(args.input_csv)

    df1 = pd.read_csv('dataset/train_features.csv')
    valid_buildings = df1['building_id'].unique()
    df = df[df['building_id'].isin(valid_buildings)]

    all_rows = []

    for building_id in tqdm(df.building_id.unique(), desc="Evaluating buildings"):
        try:
            test_ds = BuildingDataset(df, building_id, args.win_size)
            if len(test_ds) <= 0:
                tqdm.write(f"Skipping Building {building_id}: Sequence length insufficient for window size.")
                continue
                
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

            # Extract continuous anomaly scores
            scores = units_score_fn(model, test_loader, device)
            
            # Ground Truth Alignment
            y_true = np.concatenate([y.numpy() for _, y in test_loader]).flatten()
            y_true = (y_true[:len(scores)] > 0).astype(int) 

            if len(y_true) == 0:
                continue

            # =========================================================
            # Test Set Oracle Threshold Optimization (Maximizing Point-Adjusted F1)
            # =========================================================
            best_test_f1 = -1.0
            best_th = scores[0] if len(scores) > 0 else 0.0
            
            threshold_candidates = np.linspace(np.min(scores), np.max(scores), 100)
            
            for thresh in tqdm(threshold_candidates, desc="Optimizing thresholds (Oracle F1 maximization)", leave=False):
                test_preds_candidate = (scores > thresh).astype(int)
                
                if np.sum(y_true) > 0:
                    val_gt_adj, val_pred_adj = adjustment(y_true.copy(), test_preds_candidate.copy())
                    _, _, f_score_temp, _ = precision_recall_fscore_support(
                        val_gt_adj, val_pred_adj, average='binary', zero_division=0
                    )
                    if f_score_temp > best_test_f1:
                        best_test_f1 = f_score_temp 
                        best_th = thresh

            # Finalize labels using optimized point adjustment rules
            final_preds = (scores > best_th).astype(int)
            
            if np.sum(y_true) > 0:
                gt_adj_temp, pred_adj_temp = adjustment(y_true.copy(), final_preds.copy())
            else:
                gt_adj_temp, pred_adj_temp = y_true.copy(), final_preds.copy()

            accuracy = accuracy_score(gt_adj_temp, pred_adj_temp)
            precision, recall, f_score, _ = precision_recall_fscore_support(
                gt_adj_temp, pred_adj_temp, average='binary', zero_division=0
            )
            
            ad_metrics = {}
            if np.sum(y_true) == 0:
                tqdm.write(f"[Notice] Building {building_id} has 0 true anomalies. AUC/VUS skipped.")
            else:
                try:
                    ad_metrics = get_metrics_pred(score=scores, labels=y_true, pred=final_preds, slidingWindow=args.sliding_window)
                    ad_metrics.update(get_metrics(
                        score=scores, 
                        labels=y_true, 
                        pred=None, 
                        slidingWindow=args.sliding_window 
                    ))  
                except Exception as e:
                    tqdm.write(f"[Warning] Benchmark suite failed for building {building_id}. Reason: {e}")

            # Save raw predictions using append mode
            pred_df = pd.DataFrame({
                "building_id": building_id,
                "score": scores, 
                "label": y_true, 
                "prediction": final_preds
            })
            pred_df.to_csv(args.predictions_csv, mode='a', header=not os.path.exists(args.predictions_csv), index=False)

            # Pack and append metrics record entry
            metrics_row = {
                "building_id": building_id,
                "best_threshold_value": best_th, 
                **ad_metrics,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F-score": f_score,
            }
            all_rows.append(metrics_row)
            
            metrics_df = pd.DataFrame([metrics_row])
            cols = ['building_id'] + [c for c in metrics_df.columns if c != 'building_id']
            metrics_df = metrics_df[cols]
            metrics_df.to_csv(args.metrics_csv, mode='a', header=not os.path.exists(args.metrics_csv), index=False)

            tqdm.write(f"[Building {building_id}] Saved results. Point-Adjusted F1: {f_score:.4f}")
            
        except Exception as e:
            tqdm.write(f"[Error] Execution failed for building {building_id}. Detail: {e}")
        
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if all_rows:
        summary_df = pd.DataFrame(all_rows)
        print(f"\nZero-shot pipeline execution complete. Final Average Adjusted F1: {summary_df['F-score'].mean():.4f}")
    else:
        print("\nPipeline finished execution. No matching records evaluated.")

if __name__ == "__main__":
    main()
