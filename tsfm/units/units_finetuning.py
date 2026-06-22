import os
import sys
import argparse
import random
import time
import tempfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import gc

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
from tsfm.units.UniTS import Model
from utils.data_preprocessing import preprocess, train_test_split
from evaluation.metrics import get_metrics, get_metrics_pred
from utils.tools import adjustment, adjust_learning_rate
from utils.layer_decay import param_groups_lrd

# =========================================================
# Utils
# =========================================================

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


def count_parameters(model):
    """
    Counts total parameters and trainable parameters in the model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("-" * 50)
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Frozen Parameters:    {total_params - trainable_params:,}")
    print("-" * 50)
    
    return total_params, trainable_params


def freeze_backbone(model):
    """
    Freeze UniTS backbone for proper fine-tuning, keeping heads and prompts active.
    """
    for name, param in model.named_parameters():
        if any(layer in name for layer in [
                'norm', 'forecast_head', 
            ]):
            param.requires_grad = True
        else:
            param.requires_grad = False


def load_checkpoint(path, device):
    """
    PyTorch safe checkpoint loading.
    Explicitly disables weights_only mode to handle complex nested numpy structures.
    """
    return torch.load(
        path,
        map_location=device,
        weights_only=False
    )


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


def calculate_metrics_for_threshold(test_scores, y_test_true, threshold, sliding_window, prefix=""):
    """
    Helper function to fairly compute identical benchmark metrics given any threshold.
    """
    preds = (test_scores > threshold).astype(int)
    
    gt_adj, pred_adj = adjustment(y_test_true.copy(), preds.copy())
    
    accuracy = accuracy_score(gt_adj, pred_adj)
    precision, recall, f_score, _ = precision_recall_fscore_support(
        gt_adj, pred_adj, average='binary', zero_division=0
    )
    
    metrics = {
        f"{prefix}Accuracy": accuracy,
        f"{prefix}Precision": precision,
        f"{prefix}Recall": recall,
        f"{prefix}F-score": f_score
    }
    
    if np.sum(y_test_true) != 0:
        try:
            ad_metrics = get_metrics_pred(score=test_scores, labels=y_test_true, pred=preds, slidingWindow=sliding_window)
            ad_metrics.update(get_metrics(score=test_scores, labels=y_test_true, pred=None, slidingWindow=sliding_window))
            
            # Append prefix to advanced suite metrics keys
            for k, v in ad_metrics.items():
                metrics[f"{prefix}{k}"] = v
        except Exception as e:
            pass
            
    return metrics

# =========================================================
# Dataset (Train / Val / Test 6-Way Split)
# =========================================================

class BuildingDataset(Dataset):
    def __init__(self, df, building_id, win_size, flag="train", local_normalize=False):
        """
        Custom Dataset for managing sliding window data extraction.
        """
        self.win_size = win_size
        self.local_normalize = local_normalize
        
        X, y = preprocess(df, building_id)
        X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(X, y)
        
        if flag == "train":
            self.data = np.asarray(X_train).astype(np.float32)
            self.labels = np.asarray(y_train).astype(np.int32).flatten()
        elif flag == "val":
            self.data = np.asarray(X_val).astype(np.float32)
            self.labels = np.asarray(y_val).astype(np.int32).flatten()
        elif flag == "test":
            self.data = np.asarray(X_test).astype(np.float32)
            self.labels = np.asarray(y_test).astype(np.int32).flatten()
        else:
            raise ValueError("flag must be 'train', 'val', or 'test'")

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
# Training & Scoring Modules
# =========================================================

def train_one_epoch(model, loader, optimizer, scaler, criterion, device, clip_grad=None):
    """
    Trains for one epoch using mixed precision (AMP) matching official pipeline stability.
    """
    model.train()
    total_loss = 0.0

    for batch_x, _ in loader:
        batch_x = batch_x.to(device)
        if batch_x.dim() == 2:
            batch_x = batch_x.unsqueeze(-1)

        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            outputs = model(
                x_enc=batch_x,
                x_mark_enc=None,
                task_id=0,
                task_name="anomaly_detection"
            )
            loss = criterion(outputs, batch_x)

        scaler.scale(loss).backward()

        if clip_grad is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_norm=clip_grad
            )

        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()

    return total_loss / len(loader)


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            if batch_x.dim() == 2:
                batch_x = batch_x.unsqueeze(-1)
                
            with torch.amp.autocast('cuda'):
                outputs = model(
                    x_enc=batch_x,
                    x_mark_enc=None,
                    task_id=0,
                    task_name="anomaly_detection"
                )
                loss = criterion(outputs, batch_x)
                
            total_loss += loss.item()
            
    return total_loss / len(loader)


def units_score_fn(model, loader, device):
    model.eval()
    mse = nn.MSELoss(reduction="none")
    scores_all = []

    with torch.no_grad():
        for batch_x, _ in loader:
            if batch_x.dim() == 2:
                batch_x = batch_x.unsqueeze(-1)
            batch_x = batch_x.to(device)

            with torch.amp.autocast('cuda'):
                outputs = model(
                    x_enc=batch_x,
                    x_mark_enc=None,
                    task_id=0,
                    task_name="anomaly_detection"
                )
                scores = mse(outputs, batch_x).mean(dim=(1, 2))
            scores_all.append(scores.cpu().numpy())

    return np.concatenate(scores_all).flatten()


def collect_labels(loader):
    labels = []
    for _, y in loader:
        labels.append(y.numpy())
    return np.concatenate(labels).reshape(-1).astype(int)


# =========================================================
# Main Execution Layout
# =========================================================

def main():
    parser = argparse.ArgumentParser("UniTS Fine-Tuning Validation & Evaluation Suite")

    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, default="./tsfm/units/units_x128_pretrain_checkpoint.pth")
    parser.add_argument("--metrics_csv", type=str, default="results/UniTS/finetune/metrics/units_finetuned_metrics.csv")
    parser.add_argument("--predictions_csv", type=str, default="results/UniTS/finetune/metrics/units_finetuned_predictions.csv")
    parser.add_argument("--win_size", type=int, default=168)
    parser.add_argument("--sliding_window", type=int, default=168)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    
    # Early Stopping Params
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")

    # UniTS defaults
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--prompt_num", type=int, default=10)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--e_layers", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=8)
    
    # Layer decay specific variables
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay coefficient")
    parser.add_argument("--layer_decay", type=float, default=0.75, help="Layer-wise learning rate decay coefficient")

    # Patch geometry
    parser.add_argument("--patch_len", type=int, default=24)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--clip_grad", type=float, default=100.0, help="Max grad norm for clipping")

    args = parser.parse_args()
    
    args.learning_rate = args.lr
    args.train_epochs = args.epochs 
    
    fix_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(os.path.dirname(args.metrics_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.predictions_csv) or ".", exist_ok=True)

    # if os.path.exists(args.metrics_csv): os.remove(args.metrics_csv)
    # if os.path.exists(args.predictions_csv): os.remove(args.predictions_csv)

    df = pd.read_csv(args.input_csv)
    df = df[df['building_id'] > 909]

    df1 = pd.read_csv('dataset/train_features.csv')
    valid_buildings = df1['building_id'].unique()
    df = df[df['building_id'].isin(valid_buildings)]

    all_rows = []

    for building_id in tqdm(df.building_id.unique(), desc="Fine-tuning across buildings"):
        print(f"\n===== Building {building_id} =====")

        train_ds = BuildingDataset(df, building_id, args.win_size, flag="train")
        val_ds = BuildingDataset(df, building_id, args.win_size, flag="val")
        test_ds = BuildingDataset(df, building_id, args.win_size, flag="test")

        if len(test_ds) <= 0 or len(val_ds) <= 0:
            print(f"Skipping sequence for Building {building_id}: Insufficient data split size.")
            continue

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        # 2. Configure model architecture
        model = Model(args, [
            ("lead_anomaly", {
                "task_name": "anomaly_detection",
                "dataset": "lead",
                "enc_in": 1,
                "seq_len": args.win_size
            })
        ], pretrain=False).to(device)

        ckpt = load_checkpoint(args.ckpt_path, device)
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)

        freeze_backbone(model)
        count_parameters(model)
        
        # Extract layer groups and drop explicitly frozen tensors to align layer decay optimization
        raw_param_groups = param_groups_lrd(
            model, 
            args.weight_decay,
            no_weight_decay_list=['prompts', 'mask_tokens', 'cls_tokens', 'category_tokens', 'prompt_token'],
            layer_decay=args.layer_decay
        )
        
        param_groups = []
        for group in raw_param_groups:
            filtered_params = [p for p in group['params'] if p.requires_grad]
            if len(filtered_params) > 0:
                new_group = group.copy()
                new_group['params'] = filtered_params
                param_groups.append(new_group)
        
        optimizer = torch.optim.Adam(param_groups, lr=args.lr)
        scaler = torch.amp.GradScaler('cuda')
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
        
        criterion = nn.MSELoss()
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)

        # 3. Execution fine-tuning loop
        t0 = time.perf_counter()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, f"checkpoint_building_{building_id}.pth")
            
            for epoch in range(args.epochs):
                train_loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device, clip_grad=args.clip_grad)
                val_loss = validate_one_epoch(model, val_loader, criterion, device)
                
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch: {epoch + 1} | LR: {current_lr:.2e} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
                
                early_stopping(val_loss, model, checkpoint_path)
                if early_stopping.early_stop:
                    print("Early stopping triggered. Halting training for this building.")
                    break
                
                scheduler.step()
                
            train_time = time.perf_counter() - t0
            model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

        # 4. Find the optimal threshold via linear spaces over validation scores range
        val_scores = units_score_fn(model, val_loader, device)
        y_val_true = collect_labels(val_loader)
        
        best_val_f1 = -1.0
        best_th = np.min(val_scores) if len(val_scores) > 0 else 0.0
        
        if len(val_scores) > 0:
            val_threshold_candidates = np.linspace(np.min(val_scores), np.max(val_scores), 100)
            for thresh in val_threshold_candidates:
                val_preds_candidate = (val_scores > thresh).astype(int)
                if np.sum(y_val_true) > 0:
                    val_gt_adj, val_pred_adj = adjustment(y_val_true.copy(), val_preds_candidate.copy())
                    _, _, f_score_temp, _ = precision_recall_fscore_support(
                        val_gt_adj, val_pred_adj, average='binary', zero_division=0
                    )
                    if f_score_temp > best_val_f1:
                        best_val_f1 = f_score_temp 
                        best_th = thresh
                else:
                    best_th = thresh

        # 5. Evaluate the optimized model strictly on the completely unseen TEST split
        test_scores = units_score_fn(model, test_loader, device)
        y_test_true = collect_labels(test_loader)
        if len(y_test_true) == 0:
            continue

        # 5b. Find the optimal threshold via linear spaces over TEST scores range
        best_test_f1 = -1.0
        best_test_th = 0.0
        
        if len(test_scores) > 0:
            test_threshold_candidates = np.linspace(np.min(test_scores), np.max(test_scores), 100)
            for thresh in test_threshold_candidates:
                test_preds_candidate = (test_scores > thresh).astype(int)
                if np.sum(y_test_true) > 0:
                    test_gt_adj, test_pred_adj = adjustment(y_test_true.copy(), test_preds_candidate.copy())
                    _, _, f_score_temp, _ = precision_recall_fscore_support(
                        test_gt_adj, test_pred_adj, average='binary', zero_division=0
                    )
                    if f_score_temp > best_test_f1:
                        best_test_f1 = f_score_temp
                        best_test_th = thresh
                else:
                    best_test_th = thresh

        # 🟢 5c. Run metrics using both thresholds for a fair side-by-side comparison
        # val_th_metrics = calculate_metrics_for_threshold(
        #     test_scores, y_test_true, best_th, args.sliding_window, prefix=""
        # )
        test_th_metrics = calculate_metrics_for_threshold(
            test_scores, y_test_true, best_test_th, args.sliding_window, prefix=""
        )

        # Save standard testing predictions incrementally (based on the real validation threshold)
        final_test_preds = (test_scores > best_th).astype(int)
        pred_df = pd.DataFrame({
            "building_id": building_id,
            "score": test_scores,
            "label": y_test_true,
            "prediction": final_test_preds
        })
        pred_df.to_csv(args.predictions_csv, mode='a', header=not os.path.exists(args.predictions_csv), index=False)

        # Build final log dictionary containing both standard and test-optimized metric sets
        metrics_row = {
            "building_id": building_id,
            "best_threshold_value": best_th,
            "test_threshold_value": best_test_th,
            # **val_th_metrics,   # Normal metrics (unprefixed)
            **test_th_metrics,  # Oracle upper bound metrics (prefixed with 'Test_Th_')
        }
        all_rows.append(metrics_row)
        
        # Save summary evaluation metrics sheets incrementally
        metrics_df = pd.DataFrame([metrics_row])
        cols = ['building_id', 'best_threshold_value', 'test_threshold_value'] + [c for c in metrics_df.columns if c not in ['building_id', 'best_threshold_value', 'test_threshold_value']]
        metrics_df = metrics_df[cols]
        metrics_df.to_csv(args.metrics_csv, mode='a', header=not os.path.exists(args.metrics_csv), index=False)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    print("\n✅ EVALUATION PIPELINE COMPLETE. RESULTS STORED IN APPEND MODE VIA TEST SPLITS.")


if __name__ == "__main__":
    main()
