from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from utils.ad_metrics import get_metrics_pred
import csv
import torch
import torch.nn as nn
from torch import optim
import torch.multiprocessing
import os
import time
import warnings
import pandas as pd
import numpy as np
import torch.nn as nn

warnings.filterwarnings('ignore')
torch.multiprocessing.set_sharing_strategy('file_system')


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)

                # Anomaly detection in TSlib usually skips time marks
                outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach()
                true = batch_x.detach()

                loss = criterion(pred, true)
                total_loss.append(loss.item())
        
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                loss = criterion(outputs, batch_x)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        self.model.eval()
        anomaly_criterion = nn.MSELoss(reduction='none')

        test_energy, test_labels, test_timestamps = [], [], []

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x, None, None, None)
                
                score = torch.mean(anomaly_criterion(batch_x, outputs), dim=-1)
                score = score[:, -1]
                
                test_energy.append(score.detach().cpu().numpy())
                test_labels.append(batch_y[:, -1].detach().cpu().numpy())
                
                # Align timestamps with the last point of the window
                for j in range(len(batch_x)):
                    idx = i * self.args.batch_size + j + self.args.seq_len - 1
                    if idx < len(test_data.timestamps):
                        test_timestamps.append(test_data.timestamps[idx])

        test_energy = np.concatenate(test_energy).reshape(-1)
        test_labels = np.concatenate(test_labels).reshape(-1)
        test_timestamps = test_timestamps[:len(test_energy)]

        # Thresholding and Adjustment
        threshold = np.percentile(test_energy, 100 - self.args.anomaly_ratio)
        pred = (test_energy > threshold).astype(int)
        gt = test_labels.astype(int)
        gt_adj, pred_adj = adjustment(gt.copy(), pred.copy())

        # --- NEW: APPEND TO MASTER PREDICTION CSV ---
        prediction_dir = 'predictions'
        os.makedirs(prediction_dir, exist_ok=True)
        # One file per model, containing all buildings
        master_pred_file = f"{prediction_dir}/{self.args.model}_predictions.csv"
        
        output_df = pd.DataFrame({
            'Building_ID': self.args.building_id, # Identifier column
            'Timestamp': test_timestamps,
            'GroundTruth': gt,
            'AnomalyScore': test_energy,
            'RawPrediction': pred,
            'AdjustedPrediction': pred_adj
        })

        # Mode 'a' (append). Write header only if file doesn't exist
        file_exists = os.path.isfile(master_pred_file)
        output_df.to_csv(master_pred_file, mode='a', index=False, header=not file_exists)
        
        print(f"Predictions for Building {self.args.building_id} appended to {master_pred_file}")

        # --- Metrics Calculation (Keep your existing logic below) ---
        accuracy = accuracy_score(gt_adj, pred_adj)
        precision, recall, f_score, _ = precision_recall_fscore_support(gt_adj, pred_adj, average='binary', zero_division=0)
    
        try:
            roc_auc = roc_auc_score(gt_adj, test_energy)
            precision_curve, recall_curve, _ = precision_recall_curve(gt_adj, test_energy)
            pr_auc = auc(recall_curve, precision_curve)
        except:
            roc_auc, pr_auc = 0.0, 0.0

        try:
            ad_metrics = get_metrics_pred(
                score=test_energy,
                labels=gt_adj,
                pred=pred_adj,
                slidingWindow=self.args.seq_len
            )
        except:
            ad_metrics = {}

        # 7. LOG SUMMARY RESULTS
        row_data = {
            "Building_ID": self.args.building_id,
            "Setting": setting,
            **ad_metrics,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F-score": f_score,
            "AUC-ROC": roc_auc,
            "PR-AUC": pr_auc
        }
        
        os.makedirs('results', exist_ok=True)
        csv_summary = f"results/{self.args.model}.csv"
        file_exists = os.path.isfile(csv_summary)

        with open(csv_summary, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_data)

        return pred_adj
    
    # def test(self, setting, test=0):
    #     test_data, test_loader = self._get_data(flag='test')
    #     train_data, train_loader = self._get_data(flag='train')

    #     if test:
    #         print('loading model')
    #         self.model.load_state_dict(torch.load(
    #             os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
    #         ))

    #     self.model.eval()
    #     anomaly_criterion = nn.MSELoss(reduction='none')

    #     # 1. TRAIN ENERGY
    #     train_energy = []
    #     with torch.no_grad():
    #         for batch_x, _ in train_loader:
    #             batch_x = batch_x.float().to(self.device)
    #             outputs = self.model(batch_x, None, None, None)
    #             score = torch.mean(anomaly_criterion(batch_x, outputs), dim=-1)
    #             score = score[:, -1]  # Last timestep only
    #             train_energy.append(score.detach().cpu().numpy())

    #     train_energy = np.concatenate(train_energy).reshape(-1)

    #     # 2. TEST ENERGY + LABELS
    #     test_energy = []
    #     test_labels = []

    #     with torch.no_grad():
    #         for batch_x, batch_y in test_loader:
    #             batch_x = batch_x.float().to(self.device)
    #             outputs = self.model(batch_x, None, None, None)
                
    #             score = torch.mean(anomaly_criterion(batch_x, outputs), dim=-1)
    #             score = score[:, -1]
    #             test_energy.append(score.detach().cpu().numpy())
    #             test_labels.append(batch_y[:, -1].detach().cpu().numpy())

    #     test_energy = np.concatenate(test_energy).reshape(-1)
    #     test_labels = np.concatenate(test_labels).reshape(-1)

    #     print("Test samples:", len(test_energy))
    #     print("Actual anomalies:", int(test_labels.sum()))
    #     print("gt shape:", test_labels.shape)

    #     # 4. THRESHOLD (Fixed to use args)
    #     # TSlib expects anomaly_ratio as a percentage (e.g. 5 for 5%)
    #     # So we use 100 - anomaly_ratio to get the percentile threshold
    #     percentile_target = 100 - self.args.anomaly_ratio
    #     threshold = np.percentile(test_energy, percentile_target)
    #     print(f"Threshold (Percentile {percentile_target}):", threshold)

    #     # 5. PREDICTION
    #     pred = (test_energy > threshold).astype(int)
    #     gt = test_labels.astype(int)

    #     gt = gt.reshape(-1)
    #     pred = pred.reshape(-1)

    #     print("Predicted anomalies:", int(pred.sum()))
    #     print("pred shape:", pred.shape)

    #     # 6. ADJUSTMENT
    #     gt, pred = adjustment(gt, pred)

    #     # 7. CLASSIC METRICS
    #     accuracy = accuracy_score(gt, pred)
    #     precision, recall, f_score, _ = precision_recall_fscore_support(
    #         gt, pred, average='binary', zero_division=0
    #     )

    #     # 8. AUC METRICS
    #     try:
    #         roc_auc = roc_auc_score(gt, test_energy)
    #         precision_curve, recall_curve, _ = precision_recall_curve(gt, test_energy)
    #         pr_auc = auc(recall_curve, precision_curve)
    #     except Exception as e:
    #         print("AUC computation failed:", e)
    #         roc_auc, pr_auc = 0.0, 0.0

    #     # 9. AD METRICS (SAFE)
    #     try:
    #         ad_metrics = get_metrics_pred(
    #             score=test_energy,
    #             labels=gt,
    #             pred=pred,
    #             slidingWindow=self.args.seq_len
    #         )
    #     except Exception as e:
    #         print("VUS metric failed:", e)
    #         ad_metrics = {}

    #     # 10. PRINT + SAVE
    #     result_line = (
    #         " | ".join([f"{k}: {v:.4f}" for k, v in ad_metrics.items()]) +
    #         f" | Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
    #         f"Recall: {recall:.4f}, F-score: {f_score:.4f}, "
    #         f"AUC-ROC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}"
    #     )

    #     print(result_line)

    #     # with open("result_anomaly_detection.txt", "a") as f:
    #     #     # Add the building ID to the top line of the log entry
    #     #     f.write(f"Building ID: {self.args.building_id} | Setting: {setting}\n")
    #     #     f.write(result_line + "\n\n")
        
    #     row_data = {
    #         "Building_ID": self.args.building_id,
    #         "Setting": setting,
    #         **ad_metrics,  # Unpacks all the metrics from get_metrics_pred dynamically
    #         "Accuracy": accuracy,
    #         "Precision": precision,
    #         "Recall": recall,
    #         "F-score": f_score,
    #         "AUC-ROC": roc_auc,
    #         "PR-AUC": pr_auc
    #     }
    #     os.makedirs('results', exist_ok=True)
    #     csv_filename = f"results/{self.args.model}_results.csv"
        
    #     # Check if file exists so we know whether to write the header row
    #     file_exists = os.path.isfile(csv_filename)

    #     with open(csv_filename, mode="a", newline="") as f:
    #         # We use DictWriter which automatically maps dictionary keys to columns
    #         writer = csv.DictWriter(f, fieldnames=row_data.keys())
            
    #         if not file_exists:
    #             writer.writeheader()  # Write column names on the very first run
                
    #         writer.writerow(row_data)

    #     return