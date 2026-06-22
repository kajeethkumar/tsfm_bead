import os, sys
import argparse
import numpy as np
import pandas as pd
import csv
import warnings
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
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

NUM_TRAIN        = 182 * 24        
NUM_VAL          =  61 * 24        
NUM_TEST         = 122 * 24        
NUM_TRAIN_TOTAL = NUM_TRAIN + NUM_VAL  

class LocalForecastDataset(Dataset):
    def __init__(self, data, window_size=168, pred_len=1):
        self.data = data.astype(np.float32)
        self.window_size = window_size
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.window_size - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.window_size]
        y = self.data[idx + self.window_size : idx + self.window_size + self.pred_len]
        return torch.tensor(x).unsqueeze(-1), torch.tensor(y).unsqueeze(-1)

class LSTMModel(nn.Module):
    def __init__(self, feats=1, hidden_dim=20, pred_len=1, num_layers=2, device='cpu') -> None:
        super().__init__()
        self.pred_len = pred_len
        self.feats = feats
        self.device = device
        
        self.lstm_encoder = nn.LSTM(input_size=feats, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.lstm_decoder = nn.LSTM(input_size=feats, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.relu = nn.GELU()
        self.fc = nn.Linear(hidden_dim, feats)
        
    def forward(self, src):
        _, decoder_hidden = self.lstm_encoder(src)
        cur_batch = src.shape[0]
        
        decoder_input = torch.zeros(cur_batch, 1, self.feats).to(self.device)
        outputs = torch.zeros(self.pred_len, cur_batch, self.feats).to(self.device)
        
        for t in range(self.pred_len):
            decoder_output, decoder_hidden = self.lstm_decoder(decoder_input, decoder_hidden)
            decoder_output = self.relu(decoder_output)
            decoder_input = self.fc(decoder_output)
            outputs[t] = torch.squeeze(decoder_input, dim=-2)
            
        return outputs

def impute_nans_with_median(df, building_id):
    bdf = df[df["building_id"] == building_id].copy().reset_index(drop=True)
    value_col = next((c for c in ["value", "meter_reading", "consumption"] if c in bdf.columns),
                     [c for c in bdf.columns if c not in ["building_id", "timestamp", "anomaly", "label"]][0])
    if bdf[value_col].isna().sum() > 0:
        bdf[value_col] = bdf[value_col].fillna(bdf[value_col].median())
    return bdf, value_col

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0: break
                pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state: pred[i] = 1
    return gt, pred

def find_best_threshold(scores, y_true):
    best_thresh = scores.min()
    best_f1 = 0.0
    for t in np.linspace(scores.min(), scores.max(), 100):
        pred = (scores >= t).astype(int)
        gt = y_true.astype(int)
        _, pred_adj = adjustment(gt.copy(), pred.copy())
        try:
            _, _, f1, _ = precision_recall_fscore_support(gt, pred_adj, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
        except: pass
    return best_thresh, best_f1

def safe_get_metrics(score, labels, pred, sliding_window):
    try:
        ad_metrics = get_metrics_pred(score=score, labels=labels, pred=pred, slidingWindow=sliding_window)
        if isinstance(ad_metrics, dict):
            ad_metrics = {k: (0.0 if (np.isnan(v) or np.isinf(v)) else v) for k, v in ad_metrics.items()}
        return ad_metrics
    except:
        return {}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=168)
    args = parser.parse_args()
    
    df = pd.read_csv(args.input_csv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("./results", exist_ok=True)
    os.makedirs("./predictions/lstmad", exist_ok=True)

    csv_summary = "./results/LSTMAD_metrics.csv"
    master_pred_file = "./predictions/lstmad/LSTMAD_predictions.csv"
    summary_written = os.path.isfile(csv_summary)
    pred_written = os.path.isfile(master_pred_file)

    for building_id in df.building_id.unique():
        print(f"\nProcessing building: {building_id}")
        bdf, value_col = impute_nans_with_median(df, building_id)
        label_col = "anomaly" if "anomaly" in bdf.columns else "label"

        X_raw = bdf[value_col].values.astype(float)
        y_all = bdf[label_col].values.astype(int)

        scaler = MinMaxScaler()
        scaler.fit(X_raw[:NUM_TRAIN].reshape(-1, 1))
        X_scaled = scaler.transform(X_raw.reshape(-1, 1)).ravel()

        if len(X_scaled) < NUM_TRAIN_TOTAL or len(np.unique(y_all[:NUM_TRAIN])) < 2:
            continue

        train_dataset = LocalForecastDataset(X_scaled[:NUM_TRAIN], window_size=args.seq_len, pred_len=1)
        test_dataset = LocalForecastDataset(X_scaled, window_size=args.seq_len, pred_len=1)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

        model = LSTMModel(feats=1, hidden_dim=20, pred_len=1, num_layers=2, device=device).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0008)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(10):
            for x_b, y_b in train_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                out = model(x_b).view(1, -1, 1)  
                loss = criterion(out, y_b.view(1, -1, 1))
                loss.backward()
                optimizer.step()

        model.eval()
        errs = []
        with torch.no_grad():
            for x_b, y_b in DataLoader(test_dataset, batch_size=128, shuffle=False):
                x_b, y_b = x_b.to(device), y_b.to(device)
                out = model(x_b).view(-1, 1)
                mse = torch.pow(out - y_b.view(-1, 1), 2).cpu().numpy().flatten()
                errs.extend(mse)

        pad_len = len(X_scaled) - len(errs)
        all_scores = np.concatenate([np.array([errs[0]] * pad_len), np.array(errs)])

        val_scores = all_scores[NUM_TRAIN:NUM_TRAIN_TOTAL]
        y_val = y_all[NUM_TRAIN:NUM_TRAIN_TOTAL]
        best_thresh, best_val_f1 = find_best_threshold(val_scores, y_val)

        scores = all_scores[NUM_TRAIN_TOTAL:]
        gt = y_all[NUM_TRAIN_TOTAL:]
        pred = (scores >= best_thresh).astype(int)
        gt_adj, pred_adj = adjustment(gt.copy(), pred.copy())

        accuracy = accuracy_score(gt_adj, pred_adj)
        precision, recall, f_score, _ = precision_recall_fscore_support(gt_adj, pred_adj, average="binary", zero_division=0)
        
        try:
            roc_auc = roc_auc_score(gt_adj, scores)
            prec_curve, rec_curve, _ = precision_recall_curve(gt_adj, scores)
            pr_auc = auc(rec_curve, prec_curve)
        except: roc_auc, pr_auc = 0.0, 0.0

        ad_metrics = safe_get_metrics(scores, gt_adj, pred_adj, args.seq_len)

        row_data = {
            "Building_ID": building_id,
            "method": "lstmad",
            "threshold": best_thresh,
            "best_val_f1": best_val_f1,
            **ad_metrics,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F-score": f_score,
            "AUC-ROC": roc_auc,
            "PR-AUC": pr_auc,
        }

        pd.DataFrame({
            "Building_ID": building_id,
            "Index": np.arange(len(scores)),
            "GroundTruth": gt,
            "AnomalyScore": scores,
            "RawPrediction": pred,
            "AdjustedPrediction": pred_adj,
        }).to_csv(master_pred_file, mode="a", index=False, header=not pred_written)
        pred_written = True

        with open(csv_summary, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row_data.keys())
            if not summary_written:
                writer.writeheader()
                summary_written = True
            writer.writerow(row_data)
        
        print(f"  [TEST METRICS] Acc={accuracy:.4f}  P={precision:.4f}  R={recall:.4f}  F1={f_score:.4f}")
    
        print(f"\nDone.\n - {csv_summary}\n - {master_pred_file}")

if __name__ == "__main__":
    main()
