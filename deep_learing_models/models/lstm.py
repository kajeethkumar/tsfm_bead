import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
import torchinfo

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# Scikit-learn validation & core metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.exceptions import UndefinedMetricWarning
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Project path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..")) 
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Pipeline data preprocessing and metrics modules
from utils.data_preprocessing import preprocess, train_test_split
from evaluation.metrics import get_metrics, get_metrics_pred

# Custom model dependencies
from utils.utility import get_activation_by_name
from utils.torch_utility import EarlyStoppingTorch, get_gpu
from utils.dataset import ForecastDataset

# ===========================================================
# Model Component Architectures
# ===========================================================
class LSTMModel(nn.Module):
    def __init__(self, window_size, feats, 
                 hidden_dim, pred_len, num_layers, batch_size, device) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.batch_size = batch_size
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

class LSTMAD():
    def __init__(self,
                 window_size=168,
                 pred_len=1,
                 batch_size=128,
                 epochs=50,
                 lr=0.0008,
                 feats=1,
                 hidden_dim=20,
                 num_layer=2,
                 validation_size=0.2):
        super().__init__()
        self.__anomaly_score = None
        
        cuda = True
        self.y_hats = None
        self.cuda = cuda
        self.device = get_gpu(self.cuda)
        
        self.window_size = window_size
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.feats = feats
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.lr = lr
        self.validation_size = validation_size

        print('self.device: ', self.device)
        
        self.model = LSTMModel(self.window_size, feats, hidden_dim, self.pred_len, num_layer, batch_size=self.batch_size, device=self.device).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.75)
        self.loss = nn.MSELoss()
        self.save_path = None
        self.early_stopping = EarlyStoppingTorch(save_path=self.save_path, patience=3)
        
        self.mu = None
        self.sigma = None
        self.eps = 1e-10
        
    def fit(self, data):
        tsTrain = data[:int((1-self.validation_size)*len(data))]
        tsValid = data[int((1-self.validation_size)*len(data)):]

        train_loader = DataLoader(
            ForecastDataset(tsTrain, window_size=self.window_size, pred_len=self.pred_len),
            batch_size=self.batch_size,
            shuffle=True)
        
        valid_loader = DataLoader(
            ForecastDataset(tsValid, window_size=self.window_size, pred_len=self.pred_len),
            batch_size=self.batch_size,
            shuffle=False)
        
        for epoch in range(1, self.epochs + 1):
            self.model.train(mode=True)
            avg_loss = 0
            loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
            for idx, (x, target) in loop:
                x, target = x.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(x)
                
                # Align output structure from (pred_len, bs, feat) to flat flattened metrics
                output = output.view(-1, self.feats*self.pred_len)
                target = target.view(-1, self.feats*self.pred_len)

                loss = self.loss(output, target)
                loss.backward()
                self.optimizer.step()
                
                avg_loss += loss.cpu().item()
                loop.set_description(f'Training Epoch [{epoch}/{self.epochs}]')
                loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
            
            self.model.eval()
            scores = []
            avg_loss = 0
            loop = tqdm(enumerate(valid_loader), total=len(valid_loader), leave=True)
            with torch.no_grad():
                for idx, (x, target) in loop:
                    x, target = x.to(self.device), target.to(self.device)

                    output = self.model(x)
                    output = output.view(-1, self.feats*self.pred_len)
                    target = target.view(-1, self.feats*self.pred_len)
                    
                    loss = self.loss(output, target)
                    avg_loss += loss.cpu().item()
                    loop.set_description(f'Validation Epoch [{epoch}/{self.epochs}]')
                    loop.set_postfix(loss=loss.item(), avg_loss=avg_loss/(idx+1))
                    
                    mse = torch.sub(output, target).pow(2)
                    scores.append(mse.cpu())
            
            valid_loss = avg_loss/max(len(valid_loader), 1)
            self.scheduler.step()
            
            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop or epoch == self.epochs:
                if len(scores) > 0:
                    scores = torch.cat(scores, dim=0)
                    self.mu = torch.mean(scores)
                    self.sigma = torch.var(scores)
                if self.early_stopping.early_stop:
                    print("   Early stopping<<<")
                break

    def decision_function(self, data):
        test_loader = DataLoader(
            ForecastDataset(data, window_size=self.window_size, pred_len=self.pred_len),
            batch_size=self.batch_size,
            shuffle=False
        )
        
        self.model.eval()
        scores = []
        y_hats = []
        loop = tqdm(enumerate(test_loader), total=len(test_loader), leave=True)
        with torch.no_grad():
            for idx, (x, target) in loop:
                x, target = x.to(self.device), target.to(self.device)
                output = self.model(x)
                
                output = output.view(-1, self.feats*self.pred_len)
                target = target.view(-1, self.feats*self.pred_len)

                mse = torch.sub(output, target).pow(2)
                y_hats.append(output.cpu())
                scores.append(mse.cpu())
                loop.set_description(f'Testing: ')

        scores = torch.cat(scores, dim=0)
        scores = scores.numpy()
        scores = np.mean(scores, axis=1)
        
        y_hats = torch.cat(y_hats, dim=0).numpy()
        
        if scores.shape[0] < len(data):
            padded_decision_scores_ = np.zeros(len(data))
            padded_decision_scores_[: self.window_size+self.pred_len-1] = scores[0]
            padded_decision_scores_[self.window_size+self.pred_len-1 : ] = scores
        else:
            padded_decision_scores_ = scores

        self.__anomaly_score = padded_decision_scores_
        return padded_decision_scores_

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score
    
    def get_y_hat(self) -> np.ndarray:
        return self.y_hats


# ===========================================================
# Pipeline Execution Setup
# ===========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Anomaly detection using Forecasting LSTM Model")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_csv", type=str, default="./results/LSTM/lstm_predictions.csv")
    parser.add_argument("--metrics_csv", type=str, default="./results/LSTM/lstm_metrics.csv")
    parser.add_argument("--win_size", type=int, default=168, help="Window frame context size")
    parser.add_argument("--epochs", type=int, default=20, help="Training parameter iterations")
    parser.add_argument("--sliding_window", type=int, default=168, help="Metric evaluation shift scope")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    df = pd.read_csv(args.input_csv)

    df1 = pd.read_csv('../../dataset/train_features.csv')
    valid_buildings = df1['building_id'].unique()
    df = df[df['building_id'].isin(valid_buildings)]

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_csv) or ".", exist_ok=True)

    if os.path.exists(args.output_csv): os.remove(args.output_csv)
    if os.path.exists(args.metrics_csv): os.remove(args.metrics_csv)

    print(f"Initializing Framework Execution Scope utilizing Recurrent LSTM Networks...")

    for b_id in tqdm(df.building_id.unique(), desc="Evaluating buildings"):
       
        X, y = preprocess(df, b_id)
        X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(X, y)
        
        # Explicit 2D Reshaping to guarantee `.shape[1]` compatibility inside ForecastDataset
        X_train_np = np.asarray(X_train).astype(float).reshape(-1, 1)
        X_val_np = np.asarray(X_val).astype(float).reshape(-1, 1)
        X_test_np = np.asarray(X_test).astype(float).reshape(-1, 1)
        
        y_val_np = np.asarray(y_val).astype(int).flatten()
        y_test_np = np.asarray(y_test).astype(int).flatten()
        
        building = df[df["building_id"] == b_id].sort_values("timestamp").reset_index(drop=True)
        test_timestamps = building["timestamp"].values[-len(y_test):]
        test_readings = building["meter_reading"].values[-len(y_test):]
        
        if len(X_test_np) == 0 or len(X_train_np) == 0 or len(X_val_np) == 0:
            continue

        # =========================================================
        # 1. TRAIN
        # =========================================================
        pipeline = LSTMAD(window_size=args.win_size, pred_len=1, epochs=args.epochs, feats=1)
        pipeline.fit(X_train_np)
        
        # =========================================================
        # 2. VALIDATION (Threshold Tuning)
        # =========================================================
        val_scores = pipeline.decision_function(X_val_np)
        
        if len(val_scores) != len(y_val_np):
            min_len = min(len(val_scores), len(y_val_np))
            val_scores = val_scores[:min_len]
            current_y_val = y_val_np[:min_len]
        else:
            current_y_val = y_val_np

        best_val_f1 = -1.0
        best_threshold_value = np.percentile(val_scores, 99)
        
        for pct in np.linspace(90, 99.9, 50):
            candidate_thresh = np.percentile(val_scores, pct)
            val_preds = (val_scores > candidate_thresh).astype(int)
            
            if np.sum(current_y_val) > 0:
                _, _, f_score_temp, _ = precision_recall_fscore_support(
                    current_y_val, val_preds, average='binary', zero_division=0
                )
                if f_score_temp > best_val_f1:
                    best_val_f1 = f_score_temp 
                    best_threshold_value = candidate_thresh

        # =========================================================
        # 3. TEST
        # =========================================================
        test_scores = pipeline.decision_function(X_test_np)
        
        if len(test_scores) != len(y_test_np):
            min_len = min(len(test_scores), len(y_test_np))
            test_scores = test_scores[:min_len]
            y_test_np = y_test_np[:min_len]
            test_timestamps = test_timestamps[:min_len]
            test_readings = test_readings[:min_len]

        test_preds = (test_scores > best_threshold_value).astype(int)

        accuracy = accuracy_score(y_test_np, test_preds)
        precision, recall, f_score, _ = precision_recall_fscore_support(
            y_test_np, test_preds, average='binary', zero_division=0
        )
        
        ad_metrics = {}
        if np.sum(y_test_np) == 0:
            tqdm.write(f"\n[Notice] Building {b_id} has 0 true anomalies. AUC/VUS skipped.")
        else:
            try:
                ad_metrics = get_metrics_pred(score=test_scores, labels=y_test_np, pred=test_preds, slidingWindow=args.sliding_window)
                ad_metrics.update(get_metrics(
                    score=test_scores, 
                    labels=y_test_np, 
                    pred=None, 
                    slidingWindow=args.sliding_window 
                ))  
            except Exception as e:
                tqdm.write(f"\n[Warning] Benchmark suite failed for building {b_id}. Reason: {e}")

        building_metrics = {
            "building_id": b_id,
            "best_threshold": best_threshold_value, 
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
            "score": test_scores, 
            "label": y_test_np,
            "pred": test_preds
        })

        if not pred_df.empty:
            pred_df.to_csv(args.output_csv, mode='a', header=not os.path.exists(args.output_csv), index=False)

        metric_df = pd.DataFrame([building_metrics])
        cols = ['building_id'] + [c for c in metric_df.columns if c != 'building_id']
        metric_df = metric_df[cols]
        metric_df.to_csv(args.metrics_csv, mode='a', header=not os.path.exists(args.metrics_csv), index=False)
        
        del test_scores, pred_df, metric_df, X_test_np, X_train_np, y_test
        torch.cuda.empty_cache()
        gc.collect() 

    print(f"LSTM evaluation complete.")
    print(f"Predictions incrementally saved to: {args.output_csv}")
    print(f"Metrics incrementally saved to: {args.metrics_csv}")
