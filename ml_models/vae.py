import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
import copy
import itertools

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Scikit-learn metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.exceptions import UndefinedMetricWarning
import warnings
import random
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Keep your local custom imports
from utils.data_preprocessing import preprocess, train_test_split
from evaluation.metrics import get_metrics, get_metrics_pred
from utils.tools import adjustment

def fix_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False  # Bypasses cuDNN descriptor version mismatches
    print(f"Global seed set to: {seed}")

# ===========================================================
# Sliding Window Dataset
# ===========================================================
class SlidingWindowDataset(Dataset):
    """
    Creates sequences of length `win_size` for time series evaluation.
    """
    def __init__(self, X, y=None, win_size=168):
        self.X = np.asarray(X).flatten()
        self.y = np.asarray(y).flatten() if y is not None else np.zeros_like(self.X)
        self.win_size = win_size

    def __len__(self):
        return len(self.X) - self.win_size + 1

    def __getitem__(self, idx):
        X_win = self.X[idx : idx + self.win_size]
        y_win = self.y[idx : idx + self.win_size]
        return torch.tensor(X_win, dtype=torch.float32), torch.tensor(y_win, dtype=torch.float32)


# ===========================================================
# Variational Autoencoder (VarEncoderDecoder)
# ===========================================================
class VarEncoderDecoder(nn.Module):
    def __init__(self, hidden_layers, hidden_size, latent_dim=32, seq_length=168):
        super().__init__()
        
        hidden_sizes = []
        hidden_sizes.append(seq_length)
        for i in range(hidden_layers):
            hidden_sizes.append(hidden_size)
            hidden_size //= 2
        
        self.encoder = nn.ModuleList()
        for i in range(1, len(hidden_sizes)):
            linear = nn.Linear(hidden_sizes[i-1], hidden_sizes[i])
            activation = nn.ReLU()
            norm = nn.BatchNorm1d(hidden_sizes[i])
            self.encoder.append(linear)
            self.encoder.append(norm)
            self.encoder.append(activation)

        self.encoder = nn.Sequential(*self.encoder)
        
        # mu, var
        self.enc_fc_mu = nn.Linear(hidden_sizes[-1], latent_dim)
        self.enc_fc_var = nn.Linear(hidden_sizes[-1], latent_dim)

        self.decoder_in = nn.Linear(latent_dim, hidden_sizes[-1])
        self.decoder = nn.ModuleList()
        hidden_sizes = list(reversed(hidden_sizes))
        for i in range(1, len(hidden_sizes)):
            linear = nn.Linear(hidden_sizes[i-1], hidden_sizes[i])
            activation = nn.LeakyReLU()
            # No batchnorm on the final output layer
            if i < len(hidden_sizes) - 1:
                norm = nn.BatchNorm1d(hidden_sizes[i])
                self.decoder.append(linear)
                self.decoder.append(norm)
                self.decoder.append(activation)
            else:
                self.decoder.append(linear)
                
        self.decoder = nn.Sequential(*self.decoder)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, X):
        original_X = X  
        
        enc_out = self.encoder(X)
        mu = self.enc_fc_mu(enc_out)
        log_var = self.enc_fc_var(enc_out)
        
        z = self.reparameterize(mu, log_var)
        
        dec_in = self.decoder_in(z)
        out = self.decoder(dec_in)
        
        return out, original_X, mu, log_var

# VAE Loss Function (Reconstruction + KL Divergence)
def vae_loss_fn(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.numel()
    return recon_loss


# ===========================================================
# Argument Parsing & Main Execution
# ===========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Anomaly detection using Variational Autoencoder (Grid Search & Oracle)")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_csv", type=str, default="./results/VAE/vae_predictions.csv")
    parser.add_argument("--metrics_csv", type=str, default="./results/VAE/vae_metrics.csv")
    
    # Model & DataLoader hyperparams
    parser.add_argument("--win_size", type=int, default=168, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs per building")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Optimizer initial learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--seed", type=int, default=42)
    
    # Model Architecture (Static)
    parser.add_argument("--hidden_layers", type=int, default=2, help="Number of hidden layers")
    
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sliding_window", type=int, default=168, help="Tolerance window for metric evaluation")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device)
    fix_seed(args.seed)
    
    # Define Hyperparameter Grid
    hs_options = [256,192,164,128,64]
    ld_options = [16, 32, 48, 64, 92]
    param_grid = list(itertools.product(hs_options, ld_options))

    # Load and filter data
    df = pd.read_csv(args.input_csv)
    remove_bid = [32, 534, 558, 653, 693, 723, 739, 855, 910, 970, 1147, 1183, 1264, 1282]
    df = df[~df['building_id'].isin(remove_bid)]

    df1 = pd.read_csv('dataset/train_features.csv')
    valid_buildings = df1['building_id'].unique()
    df = df[df['building_id'].isin(valid_buildings)]
    df = df[df['building_id'] > 55]

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_csv) or ".", exist_ok=True)

    print(f"Executing on device: {device.type.upper()}")
    print(f"Hyperparameter grid size: {len(param_grid)} combinations per building.")

    for b_id in tqdm(df.building_id.unique(), desc="Evaluating buildings"):
       
        X, y = preprocess(df, b_id)
        X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(X, y)
        
        # Ensure sufficient data across ALL splits
        if len(X_train) < args.win_size or len(X_val) < args.win_size or len(X_test) < args.win_size:
            tqdm.write(f"\n[Skip] Building {b_id} lacks sufficient data for window size {args.win_size}.")
            continue
            
        building = df[df["building_id"] == b_id].sort_values("timestamp").reset_index(drop=True)
        timestamps = building["timestamp"].values
        
        test_timestamps = timestamps[-len(y_test):][args.win_size - 1 :]
        reading_col = "meter_reading" 
        test_readings = building[reading_col].values[-len(y_test):][args.win_size - 1 :]

        # DataLoaders
        train_dataset = SlidingWindowDataset(X_train, win_size=args.win_size)
        val_dataset = SlidingWindowDataset(X_val, y_val, win_size=args.win_size)
        test_dataset = SlidingWindowDataset(X_test, y_test, win_size=args.win_size)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # ---------------------------------------------------------
        # Hyperparameter Tuning (Grid Search)
        # ---------------------------------------------------------
        best_building_val_f1 = -1.0
        best_building_val_loss = float('inf')
        best_building_model_state = None
        best_hs = None
        best_ld = None

        for hs, ld in param_grid:
            model = VarEncoderDecoder(
                hidden_layers=args.hidden_layers, 
                hidden_size=hs, 
                latent_dim=ld, 
                seq_length=args.win_size
            ).to(device)
            
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
            # PyTorch built-in Cosine Annealing Learning Rate Scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
            
            best_val_f1 = -1.0
            best_val_loss = float('inf')
            patience_counter = 0
            best_model_state = None

            for epoch in range(args.epochs):
                # Training phase
                model.train()
                for X_batch, _ in train_loader:
                    X_batch = X_batch.to(device)
                    optimizer.zero_grad()
                    
                    recon_batch, x_batch_ret, mu, logvar = model(X_batch)
                    loss = vae_loss_fn(recon_batch, x_batch_ret, mu, logvar)
                    
                    loss.backward()
                    optimizer.step()

                # Validation phase
                model.eval()
                val_loss = 0.0
                val_scores_list = []
                val_labels_list = []
                
                with torch.no_grad():
                    for X_val_batch, y_val_batch in val_loader:
                        X_val_batch = X_val_batch.to(device)
                        recon_val, x_val_ret, mu_val, logvar_val = model(X_val_batch)
                        
                        batch_loss = vae_loss_fn(recon_val, x_val_ret, mu_val, logvar_val)
                        val_loss += batch_loss.item()
                        
                        mse_val = torch.mean((x_val_ret - recon_val) ** 2, dim=1).cpu().numpy()
                        val_scores_list.extend(mse_val)
                        val_labels_list.extend(y_val_batch[:, -1].numpy())
                
                avg_val_loss = val_loss / len(val_loader)
                val_scores = np.array(val_scores_list)
                y_val_np = np.array(val_labels_list).astype(int)
                
                has_val_anomalies = np.sum(y_val_np) > 0
                
                # Early Stopping Evaluation
                if has_val_anomalies:
                    epoch_val_f1 = 0.0
                    min_sc, max_sc = np.min(val_scores), np.max(val_scores)
                    for th in np.linspace(min_sc, max_sc, 50):
                        val_preds = (val_scores >= th).astype(int)
                        gt_adj, pred_adj = adjustment(y_val_np.copy(), val_preds.copy())
                        _, _, f1, _ = precision_recall_fscore_support(gt_adj, pred_adj, average='binary', zero_division=0)
                        if f1 > epoch_val_f1:
                            epoch_val_f1 = f1
                    
                    if epoch_val_f1 > best_val_f1:
                        best_val_f1 = epoch_val_f1
                        patience_counter = 0
                        best_model_state = copy.deepcopy(model.state_dict())
                    else:
                        patience_counter += 1
                else:
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        patience_counter = 0
                        best_model_state = copy.deepcopy(model.state_dict())
                    else:
                        patience_counter += 1
                    
                if patience_counter >= args.patience:
                    break

                # Step the built-in PyTorch scheduler at the end of each epoch
                scheduler.step()

            # --- Evaluate against global best for this building ---
            if has_val_anomalies:
                if best_val_f1 > best_building_val_f1:
                    best_building_val_f1 = best_val_f1
                    best_building_model_state = copy.deepcopy(best_model_state)
                    best_hs = hs
                    best_ld = ld
            else:
                if best_val_loss < best_building_val_loss:
                    best_building_val_loss = best_val_loss
                    best_building_model_state = copy.deepcopy(best_model_state)
                    best_hs = hs
                    best_ld = ld
                    
            del model, optimizer, scheduler
            torch.cuda.empty_cache()

        tqdm.write(f"\n[Info] Building {b_id} Optimal Params -> Hidden Size: {best_hs}, Latent Dim: {best_ld}")

        # ---------------------------------------------------------
        # Testing with Best Found Configuration
        # ---------------------------------------------------------
        best_model = VarEncoderDecoder(
            hidden_layers=args.hidden_layers, 
            hidden_size=best_hs, 
            latent_dim=best_ld, 
            seq_length=args.win_size
        ).to(device)
        best_model.load_state_dict(best_building_model_state)
        best_model.eval()
        
        recon_errors_list = []
        labels_list = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                recon_batch, x_batch_ret, _, _ = best_model(X_batch)
                
                mse = torch.mean((x_batch_ret - recon_batch) ** 2, dim=1).cpu().numpy()
                recon_errors_list.extend(mse)
                labels_list.extend(y_batch[:, -1].numpy())
            
        scores = np.array(recon_errors_list)
        y_test_np = np.array(labels_list).astype(int)

        # ---------------------------------------------------------
        # Oracle Threshold Sweep (Test Set Only)
        # ---------------------------------------------------------
        best_f1 = -1.0
        best_thresh = 0.0
        # best_pred_adj = np.zeros_like(y_test_np)
        
        min_score = np.min(scores)
        max_score = np.max(scores)
       
        thresholds = np.linspace(min_score, max_score, num=100)
        for th in thresholds:
            binary_preds = (scores >= th).astype(int)
            
            gt_adj_temp, pred_adj_temp = adjustment(y_test_np.copy(), binary_preds.copy())
            
            _, _, f_score_temp, _ = precision_recall_fscore_support(
                gt_adj_temp, pred_adj_temp, average='binary', zero_division=0
            )
            
            if f_score_temp > best_f1:
                best_f1 = f_score_temp 
                best_threshold = th
                best_pred_adj = pred_adj_temp

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
                standard_metrics = get_metrics(
                    score=scores, 
                    labels=y_test_np, 
                    pred=best_pred_adj, 
                    slidingWindow=args.sliding_window 
                )  
                pred_metrics = get_metrics_pred(
                    score=scores, 
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
            "best_hidden_size": best_hs,
            "best_latent_dim": best_ld,
            "best_threshold": round(best_threshold, 4), 
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
            "label": y_test_np,
            "pred": best_pred_adj
        })

        if not pred_df.empty:
            pred_df.to_csv(args.output_csv, mode='a', header=not os.path.exists(args.output_csv), index=False)

        metric_df = pd.DataFrame([building_metrics])
        cols = ['building_id'] + [c for c in metric_df.columns if c != 'building_id']
        metric_df = metric_df[cols]
        metric_df.to_csv(args.metrics_csv, mode='a', header=not os.path.exists(args.metrics_csv), index=False)
        
        del best_model, train_loader, val_loader, test_loader, pred_df, metric_df
        torch.cuda.empty_cache()
        gc.collect() 

    print(f"Test evaluation complete.")
    print(f"Predictions incrementally saved to: {args.output_csv}")
    print(f"Metrics incrementally saved to: {args.metrics_csv}")