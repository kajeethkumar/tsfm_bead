# # 1. Without Hyperparameter tuning

# import os, sys
# import argparse
# import numpy as np
# import pandas as pd
# import csv
# import warnings
# import torch
# from torch import nn, optim
# from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import (
#     precision_recall_fscore_support,
#     accuracy_score,
#     roc_auc_score,
#     precision_recall_curve,
#     auc
# )
# warnings.filterwarnings('ignore')

# ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if ROOT not in sys.path:
#     sys.path.append(ROOT)

# try:
#     from utils.ad_metrics import get_metrics_pred
# except ModuleNotFoundError:
#     def get_metrics_pred(**kwargs):
#         return {}

# # ===========================================================
# # Correct splits
# # ===========================================================
# NUM_TRAIN        = 182 * 24        # 4368
# NUM_VAL          =  61 * 24        # 1464
# NUM_TEST         = 122 * 24        # 2928
# NUM_TRAIN_TOTAL = NUM_TRAIN + NUM_VAL  # 5832

# class LocalForecastDataset(Dataset):
#     def __init__(self, data, window_size=168, pred_len=1):
#         self.data = data.astype(np.float32)
#         self.window_size = window_size
#         self.pred_len = pred_len

#     def __len__(self):
#         return len(self.data) - self.window_size - self.pred_len + 1

#     def __getitem__(self, idx):
#         x = self.data[idx : idx + self.window_size]
#         y = self.data[idx + self.window_size : idx + self.window_size + self.pred_len]
#         # Shape output to (window, features) and (pred_len, features)
#         return torch.tensor(x).unsqueeze(-1), torch.tensor(y).unsqueeze(-1)

# class AdaptiveConcatPool1d(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.ap = torch.nn.AdaptiveAvgPool1d(1)
#         self.mp = torch.nn.AdaptiveMaxPool1d(1)
    
#     def forward(self, x):
#         return torch.cat([self.ap(x), self.mp(x)], 1)

# class CNNModel(nn.Module):
#     def __init__(self, n_features=1, num_channel=[32, 32, 40], kernel_size=3, stride=1, predict_time_steps=1, dropout_rate=0.25):
#         super(CNNModel, self).__init__()
#         self.n_features = n_features
#         self.predict_time_steps = predict_time_steps
#         self.num_channel = num_channel
        
#         self.conv_layers = nn.Sequential()
#         prev_channels = self.n_features
#         for idx, _ in enumerate(self.num_channel[:-1]):
#             self.conv_layers.add_module(f"conv{idx}", torch.nn.Conv1d(prev_channels, self.num_channel[idx + 1], kernel_size, stride))
#             self.conv_layers.add_module(f"relu{idx}", nn.ReLU())
#             self.conv_layers.add_module(f"pool{idx}", nn.MaxPool1d(kernel_size=2))
#             prev_channels = self.num_channel[idx + 1]
            
#         self.fc = nn.Sequential(
#             AdaptiveConcatPool1d(),
#             torch.nn.Flatten(),
#             torch.nn.Linear(2 * self.num_channel[-1], self.num_channel[-1]),
#             torch.nn.ReLU(),
#             torch.nn.Dropout(dropout_rate),
#             torch.nn.Linear(self.num_channel[-1], self.n_features * self.predict_time_steps)
#         )

#     def forward(self, x):
#         b, l, c = x.shape
#         x = x.view(b, c, l)
#         x = self.conv_layers(x)
#         return self.fc(x)

# def impute_nans_with_median(df, building_id):
#     bdf = df[df["building_id"] == building_id].copy().reset_index(drop=True)
#     value_col = next((c for c in ["value", "meter_reading", "consumption"] if c in bdf.columns),
#                      [c for c in bdf.columns if c not in ["building_id", "timestamp", "anomaly", "label"]][0])
#     n_nans_before = bdf[value_col].isna().sum()
#     if n_nans_before > 0:
#         median_value = bdf[value_col].median()
#         print(f"  Imputing {n_nans_before} NaN values with median={median_value:.4f}")
#         bdf[value_col] = bdf[value_col].fillna(median_value)
#     return bdf, value_col

# def adjustment(gt, pred):
#     anomaly_state = False
#     for i in range(len(gt)):
#         if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
#             anomaly_state = True
#             for j in range(i, 0, -1):
#                 if gt[j] == 0:
#                     break
#                 pred[j] = 1
#         elif gt[i] == 0:
#             anomaly_state = False
#         if anomaly_state:
#             pred[i] = 1
#     return gt, pred

# def find_best_threshold(scores, y_true):
#     best_thresh = scores.min()
#     best_f1 = 0.0
#     for t in np.linspace(scores.min(), scores.max(), 100):
#         pred = (scores >= t).astype(int)
#         gt = y_true.astype(int)
#         _, pred_adj = adjustment(gt.copy(), pred.copy())
#         try:
#             _, _, f1, _ = precision_recall_fscore_support(gt, pred_adj, average="binary", zero_division=0)
#             if f1 > best_f1:
#                 best_f1 = f1
#                 best_thresh = t
#         except:
#             pass
#     return best_thresh, best_f1

# def safe_get_metrics(score, labels, pred, sliding_window):
#     try:
#         ad_metrics = get_metrics_pred(score=score, labels=labels, pred=pred, slidingWindow=sliding_window)
#         if isinstance(ad_metrics, dict):
#             ad_metrics = {k: (0.0 if (np.isnan(v) or np.isinf(v)) else v) for k, v in ad_metrics.items()}
#         return ad_metrics
#     except:
#         return {}

# def parse_args():
#     parser = argparse.ArgumentParser(description="CNN Anomaly Detector")
#     parser.add_argument("--input_csv", type=str, required=True)
#     parser.add_argument("--seq_len", type=int, default=168, help="Context sequence length")
#     return parser.parse_args()

# def main():
#     args = parse_args()
#     df = pd.read_csv(args.input_csv)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     os.makedirs("./results", exist_ok=True)
#     os.makedirs("./predictions/cnn", exist_ok=True)

#     csv_summary = "./results/CNN_metrics.csv"
#     master_pred_file = "./predictions/cnn/CNN_predictions.csv"
#     summary_written = os.path.isfile(csv_summary)
#     pred_written = os.path.isfile(master_pred_file)

#     for building_id in df.building_id.unique():
#         print(f"\n{'='*50}\nProcessing building: {building_id}\n{'='*50}")
#         bdf, value_col = impute_nans_with_median(df, building_id)
#         label_col = "anomaly" if "anomaly" in bdf.columns else "label"

#         X_raw = bdf[value_col].values.astype(float)
#         y_all = bdf[label_col].values.astype(int)

#         scaler = MinMaxScaler()
#         scaler.fit(X_raw[:NUM_TRAIN].reshape(-1, 1))
#         X_scaled = scaler.transform(X_raw.reshape(-1, 1)).ravel()

#         if len(X_scaled) < NUM_TRAIN_TOTAL or len(np.unique(y_all[:NUM_TRAIN])) < 2:
#             print("  Skipping — insufficient structural window bounds")
#             continue

#         # Segment arrays
#         X_train_arr = X_scaled[:NUM_TRAIN]
#         X_val_arr = X_scaled[NUM_TRAIN:NUM_TRAIN_TOTAL]
#         X_test_arr = X_scaled[NUM_TRAIN_TOTAL:]

#         # Dataloaders
#         train_dataset = LocalForecastDataset(X_train_arr, window_size=args.seq_len, pred_len=1)
#         val_dataset = LocalForecastDataset(X_scaled[:NUM_TRAIN_TOTAL], window_size=args.seq_len, pred_len=1)
#         test_dataset = LocalForecastDataset(X_scaled, window_size=args.seq_len, pred_len=1)

#         train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        
#         # Model configuration
#         model = CNNModel(n_features=1, num_channel=[32, 32, 40], predict_time_steps=1).to(device)
#         optimizer = optim.Adam(model.parameters(), lr=0.0008)
#         criterion = nn.MSELoss()

#         # Training loop
#         model.train()
#         for epoch in range(15):  # Accelerated standard execution budget
#             for x_b, y_b in train_loader:
#                 x_b, y_b = x_b.to(device), y_b.to(device)
#                 optimizer.zero_grad()
#                 out = model(x_b).view(-1, 1)
#                 loss = criterion(out, y_b.view(-1, 1))
#                 loss.backward()
#                 optimizer.step()

#         # Inference routine
#         model.eval()
        
#         # Helper to compute point scores matching target shapes
#         def get_all_squared_errors(dataset):
#             loader = DataLoader(dataset, batch_size=128, shuffle=False)
#             errs = []
#             with torch.no_grad():
#                 for x_b, y_b in loader:
#                     x_b, y_b = x_b.to(device), y_b.to(device)
#                     out = model(x_b).view(-1, 1)
#                     mse = torch.pow(out - y_b.view(-1, 1), 2).cpu().numpy().flatten()
#                     errs.extend(mse)
#             # Realign shapes back safely to point vectors via leading value pads
#             pad_len = len(dataset.data) - len(errs)
#             return np.concatenate([np.array([errs[0]] * pad_len), np.array(errs)])

#         all_scores = get_all_squared_errors(test_dataset)
        
#         # Correctly capture sequential slices matching split thresholds
#         val_scores = all_scores[NUM_TRAIN:NUM_TRAIN_TOTAL]
#         y_val = y_all[NUM_TRAIN:NUM_TRAIN_TOTAL]

#         best_thresh, best_val_f1 = find_best_threshold(val_scores, y_val)
#         print(f"  Optimized Threshold from VAL: {best_thresh:.4f}  val_f1={best_val_f1:.4f}")

#         scores = all_scores[NUM_TRAIN_TOTAL:]
#         gt = y_all[NUM_TRAIN_TOTAL:]
        
#         if len(scores) == 0:
#             continue

#         pred = (scores >= best_thresh).astype(int)
#         gt_adj, pred_adj = adjustment(gt.copy(), pred.copy())

#         accuracy = accuracy_score(gt_adj, pred_adj)
#         precision, recall, f_score, _ = precision_recall_fscore_support(gt_adj, pred_adj, average="binary", zero_division=0)

#         try:
#             roc_auc = roc_auc_score(gt_adj, scores)
#             prec_curve, rec_curve, _ = precision_recall_curve(gt_adj, scores)
#             pr_auc = auc(rec_curve, prec_curve)
#         except:
#             roc_auc, pr_auc = 0.0, 0.0

#         ad_metrics = safe_get_metrics(scores, gt_adj, pred_adj, args.seq_len)

#         row_data = {
#             "Building_ID": building_id,
#             "method": "cnn",
#             "threshold": best_thresh,
#             "best_val_f1": best_val_f1,
#             **ad_metrics,
#             "Accuracy": accuracy,
#             "Precision": precision,
#             "Recall": recall,
#             "F-score": f_score,
#             "AUC-ROC": roc_auc,
#             "PR-AUC": pr_auc,
#         }

#         pred_df = pd.DataFrame({
#             "Building_ID": building_id,
#             "Index": np.arange(len(scores)),
#             "GroundTruth": gt,
#             "AnomalyScore": scores,
#             "RawPrediction": pred,
#             "AdjustedPrediction": pred_adj,
#         })
#         pred_df.to_csv(master_pred_file, mode="a", index=False, header=not pred_written)
#         pred_written = True

#         with open(csv_summary, mode="a", newline="") as f:
#             writer = csv.DictWriter(f, fieldnames=row_data.keys())
#             if not summary_written:
#                 writer.writeheader()
#                 summary_written = True
#             writer.writerow(row_data)

#         print(f"  [TEST METRICS] Acc={accuracy:.4f}  P={precision:.4f}  R={recall:.4f}  F1={f_score:.4f}")
        
#     print(f"\nDone.\n - {csv_summary}\n - {master_pred_file}")

# if __name__ == "__main__":
#     main()

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 2, With hyperparameter tuning

import os, sys
import argparse
import numpy as np
import pandas as pd
import csv
import warnings
import traceback
import optuna
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    roc_auc_score,
    precision_recall_curve,
    auc
)
warnings.filterwarnings('ignore')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)
try:
    from utils.ad_metrics import get_metrics_pred
except ModuleNotFoundError:
    def get_metrics_pred(**kwargs):
        return {}

# ===========================================================
# Splits
# ===========================================================
NUM_TRAIN       = 183 * 24
NUM_VAL         =  61 * 24
NUM_TEST        = 122 * 24
NUM_TRAIN_TOTAL = NUM_TRAIN + NUM_VAL


# ===========================================================
# Feature engineering
# ===========================================================
def impute_nans_with_median(df, building_id):
    bdf = df[df["building_id"] == building_id].copy().reset_index(drop=True)
    value_col = next(
        (c for c in ["value", "meter_reading", "consumption"] if c in bdf.columns),
        [c for c in bdf.columns
         if c not in ["building_id", "timestamp", "anomaly", "label"]][0])
    if bdf[value_col].isna().sum() > 0:
        bdf[value_col] = bdf[value_col].fillna(bdf[value_col].median())
    return bdf, value_col


def extract_multiresolution_features(X, window_sizes=[24, 48, 168]):
    """
    FIX: Vectorised rolling stats instead of the original O(n²) Python loop.
    Uses numpy stride tricks via as_strided-based rolling window approach,
    falling back to pandas rolling for simplicity and correctness.
    Runtime drops from O(n * W * S) to O(n * S) with pandas rolling.
    """
    n        = len(X)
    features = np.zeros((n, len(window_sizes) * 6), dtype=np.float64)
    col_idx  = 0

    X_series = pd.Series(X)

    for ws in window_sizes:
        # Centered rolling window: half before, half after current point.
        # pandas rolling is left-aligned; we shift by -half to center it.
        half = ws // 2

        roll        = X_series.rolling(window=ws, min_periods=1)
        mean_val    = roll.mean().shift(-half).fillna(method='bfill').fillna(method='ffill').values
        std_val     = roll.std(ddof=0).shift(-half).fillna(method='bfill').fillna(method='ffill').values
        q25         = roll.quantile(0.25).shift(-half).fillna(method='bfill').fillna(method='ffill').values
        q75         = roll.quantile(0.75).shift(-half).fillna(method='bfill').fillna(method='ffill').values
        iqr_val     = q75 - q25
        deviation   = X - mean_val

        features[:, col_idx]     = deviation / (std_val + 1e-10)
        features[:, col_idx + 1] = (X - q25) / (iqr_val + 1e-10)
        features[:, col_idx + 2] = iqr_val
        features[:, col_idx + 3] = mean_val
        features[:, col_idx + 4] = std_val
        features[:, col_idx + 5] = X
        col_idx += 6

    return features


def extract_derivative_features(X, lags=[1, 3, 24]):
    """FIX: Vectorised lag differences instead of Python loop."""
    features = np.zeros((len(X), len(lags) * 2), dtype=np.float64)
    for col_idx, lag in enumerate(lags):
        diff = np.zeros(len(X))
        diff[lag:] = X[lag:] - X[:-lag]          # diff[i] = X[i] - X[i-lag]; 0 for i<lag
        features[:, col_idx * 2]     = diff
        features[:, col_idx * 2 + 1] = np.abs(diff)
    return features


# ===========================================================
# Threshold + adjustment
# ===========================================================
def adjustment(gt, pred):
    gt   = np.asarray(gt)
    pred = np.asarray(pred)
    adjusted_pred = pred.copy()
    diff   = np.diff(np.concatenate([[0], gt, [0]]))
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]
    for start, end in zip(starts, ends):
        if np.any(pred[start:end] == 1):
            adjusted_pred[start:end] = 1
    return gt, adjusted_pred


def find_best_threshold(scores, y_true, resolution=500):
    best_thresh    = scores.min()
    best_f1        = 0.0
    best_precision = 0.0
    best_recall    = 0.0
    min_s, max_s   = scores.min(), scores.max()
    if max_s - min_s < 1e-6:
        return min_s, 0.0, 0.0, 0.0
    for t in np.linspace(min_s, max_s, resolution):
        pred = (scores >= t).astype(int)
        _, pred_adj = adjustment(y_true, pred)
        try:
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, pred_adj, average="binary", zero_division=0)
            if f1 > best_f1 or (f1 == best_f1 and abs(prec - rec) < abs(best_precision - best_recall)):
                best_f1        = f1
                best_thresh    = t
                best_precision = prec
                best_recall    = rec
        except Exception:
            pass
    return best_thresh, best_f1, best_precision, best_recall


def safe_get_metrics(score, labels, pred, sliding_window):
    try:
        ad_metrics = get_metrics_pred(
            score=score, labels=labels, pred=pred, slidingWindow=sliding_window)
        if isinstance(ad_metrics, dict):
            ad_metrics = {k: (0.0 if (np.isnan(v) or np.isinf(v)) else v)
                          for k, v in ad_metrics.items()}
        return ad_metrics
    except Exception:
        return {}


# ===========================================================
# Optuna objective
# ===========================================================
def make_objective(X_train_scaled, X_val_scaled, y_train_all, y_val_all):
    def objective(trial):
        nu    = trial.suggest_float("nu", 0.005, 0.15)
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
        tol   = trial.suggest_categorical("tol", [1e-4, 1e-3, 1e-2])

        clean_mask    = (y_train_all == 0)
        X_train_clean = X_train_scaled[clean_mask]
        if len(X_train_clean) == 0:
            X_train_clean = X_train_scaled

        # FIX: wrap SVM fit in try/except to handle convergence failures per trial
        try:
            model = OneClassSVM(
                kernel='rbf', nu=nu, gamma=gamma,
                cache_size=500, max_iter=1500, tol=tol)
            model.fit(X_train_clean)
        except Exception:
            return 0.0

        val_scores_raw = -model.decision_function(X_val_scaled).ravel()

        # FIX: guard constant scores (all same value → scaler would produce NaN)
        if val_scores_raw.max() - val_scores_raw.min() < 1e-8:
            return 0.0

        val_scaler = MinMaxScaler(feature_range=(0, 1))
        val_scores = val_scaler.fit_transform(val_scores_raw.reshape(-1, 1)).ravel()

        _, best_val_f1, _, _ = find_best_threshold(val_scores, y_val_all, resolution=200)
        return best_val_f1

    return objective


# ===========================================================
# CLI
# ===========================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", type=str, required=True)
    p.add_argument("--seq_len",   type=int, default=168)
    p.add_argument("--trials",    type=int, default=10)
    return p.parse_args()


# ===========================================================
# Main
# ===========================================================
def main():
    args = parse_args()
    df   = pd.read_csv(args.input_csv)

    os.makedirs("./optuna_hyperparameter_results",           exist_ok=True)
    os.makedirs("./optuna_hyperparameter_predictions/ocsvm", exist_ok=True)

    csv_summary      = "./optuna_hyperparameter_results/OCSVM_metrics.csv"
    master_pred_file = "./optuna_hyperparameter_predictions/ocsvm/OCSVM_predictions.csv"
    summary_written  = os.path.isfile(csv_summary)
    pred_written     = os.path.isfile(master_pred_file)

    all_buildings = df.building_id.unique()
    print(f"Total buildings to process: {len(all_buildings)}")

    for building_id in all_buildings:
        # ── FIX: Wrap entire per-building block in try/except so one failure
        #   doesn't kill the whole run (OCSVM had no such guard at all).
        try:
            print(f"\n{'='*50}\nProcessing building: {building_id}\n{'='*50}")

            bdf, value_col = impute_nans_with_median(df, building_id)
            label_col = "anomaly" if "anomaly" in bdf.columns else "label"

            X_raw = bdf[value_col].values.astype(float)
            y_all = bdf[label_col].values.astype(int)

            if len(X_raw) < NUM_TRAIN_TOTAL:
                print(f"  Skipping: insufficient data ({len(X_raw)} < {NUM_TRAIN_TOTAL})")
                continue

            # ── FIX: Guard empty test split
            if len(X_raw) <= NUM_TRAIN_TOTAL:
                print("  Skipping — no test data available after train+val split")
                continue

            test_timestamps = bdf["timestamp"].iloc[NUM_TRAIN_TOTAL:].values

            # ── Feature extraction (now vectorised — O(n) vs original O(n²)) ──
            print("  Extracting multi-resolution temporal features...")
            features_mr    = extract_multiresolution_features(X_raw, window_sizes=[24, 48, 168])
            features_deriv = extract_derivative_features(X_raw, lags=[1, 3, 24])
            X_all_features = np.hstack([features_mr, features_deriv])

            # ── FIX: guard NaN/Inf in features (can arise from constant signals)
            if not np.isfinite(X_all_features).all():
                print("  Warning — non-finite values in features; clipping to finite range")
                X_all_features = np.nan_to_num(
                    X_all_features, nan=0.0, posinf=1e6, neginf=-1e6)

            X_train_features = X_all_features[:NUM_TRAIN]
            X_val_features   = X_all_features[NUM_TRAIN:NUM_TRAIN_TOTAL]
            X_test_features  = X_all_features[NUM_TRAIN_TOTAL:]

            y_train_all = y_all[:NUM_TRAIN]
            y_val_all   = y_all[NUM_TRAIN:NUM_TRAIN_TOTAL]
            y_test_all  = y_all[NUM_TRAIN_TOTAL:]

            # ── Scaling ──────────────────────────────────────────────────────────
            train_scaler   = StandardScaler()
            X_train_scaled = train_scaler.fit_transform(X_train_features)
            X_val_scaled   = train_scaler.transform(X_val_features)
            X_test_scaled  = train_scaler.transform(X_test_features)

            # ── Optuna ───────────────────────────────────────────────────────────
            print(f"--> Initializing Optuna search optimization pipeline ({args.trials} trials)...")
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(direction="maximize")
            study.optimize(
                make_objective(X_train_scaled, X_val_scaled, y_train_all, y_val_all),
                n_trials=args.trials)

            best_params = study.best_params
            print(f"--> Optuna Tuning Complete. Selected Parameters: {best_params}")

            # ── Re-train production model ─────────────────────────────────────────
            clean_mask    = (y_train_all == 0)
            X_train_clean = X_train_scaled[clean_mask]
            if len(X_train_clean) == 0:
                X_train_clean = X_train_scaled

            # FIX: wrap production fit in try/except
            try:
                production_model = OneClassSVM(
                    kernel='rbf',
                    nu=best_params["nu"],
                    gamma=best_params["gamma"],
                    cache_size=500,
                    max_iter=2000,
                    tol=best_params["tol"])
                production_model.fit(X_train_clean)
            except Exception as fit_err:
                print(f"  Production model fit failed: {fit_err} — skipping building")
                continue

            # ── Validation threshold ──────────────────────────────────────────────
            val_scores_raw = -production_model.decision_function(X_val_scaled).ravel()

            # FIX: guard constant val scores
            if val_scores_raw.max() - val_scores_raw.min() < 1e-8:
                print("  Warning — constant validation scores; defaulting threshold to 0")
                best_thresh, best_val_f1, val_prec, val_recall = 0.0, 0.0, 0.0, 0.0
                val_scaler = MinMaxScaler(feature_range=(0, 1))
                val_scaler.fit([[0], [1]])          # dummy fit so transform works later
                val_scores = np.zeros_like(val_scores_raw)
            else:
                val_scaler = MinMaxScaler(feature_range=(0, 1))
                val_scores = val_scaler.fit_transform(val_scores_raw.reshape(-1, 1)).ravel()
                best_thresh, best_val_f1, val_prec, val_recall = find_best_threshold(
                    val_scores, y_val_all, resolution=500)

            print(f"  Production Threshold from VAL: {best_thresh:.4f}  val_f1={best_val_f1:.4f}")

            # ── Test scoring ──────────────────────────────────────────────────────
            test_scores_raw = -production_model.decision_function(X_test_scaled).ravel()

            # FIX: use the val_scaler to normalise test scores consistently;
            # clip out-of-range values so they don't break downstream metrics
            test_scores = val_scaler.transform(test_scores_raw.reshape(-1, 1)).ravel()
            test_scores = np.clip(test_scores, 0.0, 1.0)

            if test_scores.max() - test_scores.min() < 1e-8:
                pred = np.zeros_like(test_scores, dtype=int)
            else:
                pred = (test_scores >= best_thresh).astype(int)

            gt_adj, pred_adj = adjustment(y_test_all, pred)

            # ── Metrics ───────────────────────────────────────────────────────────
            accuracy = accuracy_score(gt_adj, pred_adj)
            precision, recall, f_score, _ = precision_recall_fscore_support(
                gt_adj, pred_adj, average="binary", zero_division=0)
            try:
                roc_auc = roc_auc_score(gt_adj, test_scores)
                pc, rc, _ = precision_recall_curve(gt_adj, test_scores)
                pr_auc = auc(rc, pc)
            except Exception:
                roc_auc, pr_auc = 0.0, 0.0

            ad_metrics = safe_get_metrics(test_scores, gt_adj, pred_adj, args.seq_len)

            print(f"  [TEST METRICS] Acc={accuracy:.4f}  P={precision:.4f}"
                  f"  R={recall:.4f}  F1={f_score:.4f}")

            row_data = {
                "Building_ID":  building_id,
                "method":       "ocsvm_optuna",
                "nu":           best_params["nu"],
                "gamma":        best_params["gamma"],
                "tol":          best_params["tol"],
                "threshold":    best_thresh,
                "best_val_f1":  best_val_f1,
                **ad_metrics,
                "Accuracy":     accuracy,
                "Precision":    precision,
                "Recall":       recall,
                "F-score":      f_score,
                "AUC-ROC":      roc_auc,
                "PR-AUC":       pr_auc,
            }

            pd.DataFrame({
                "Building_ID":        building_id,
                "Timestamp":          test_timestamps,
                "GroundTruth":        y_test_all,
                "AnomalyScore":       test_scores,
                "RawPrediction":      pred,
                "AdjustedPrediction": pred_adj,
            }).to_csv(master_pred_file, mode="a", index=False, header=not pred_written)
            pred_written = True

            with open(csv_summary, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row_data.keys())
                if not summary_written:
                    writer.writeheader()
                    summary_written = True
                writer.writerow(row_data)

        except Exception as e:
            print("\n" + "=" * 80)
            print(f"ERROR PROCESSING BUILDING {building_id}: {e}")
            print("=" * 80)
            traceback.print_exc()
            continue

    print(f"\nDone.\n - {csv_summary}\n - {master_pred_file}")


if __name__ == "__main__":
    main()