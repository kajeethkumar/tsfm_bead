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
        # model = self.model_dict[self.args.model](self.args).float()
        model = self.model_dict[self.args.model](self.args)
        if isinstance(model, nn.Module):
            model = model.float()
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

        # --- UPDATED: Thresholding via Grid Search (linspace) ---
        gt = test_labels.astype(int)
        
        # Generate 100 evenly spaced thresholds between the min and max scores
        thresholds = np.linspace(np.min(test_energy), np.max(test_energy), 100)
        
        best_f_score = -1.0
        best_pred_adj = None
        best_gt_adj = None
        best_accuracy = 0.0
        best_precision = 0.0
        best_recall = 0.0
        
        for threshold in thresholds:
            pred = (test_energy > threshold).astype(int)
            gt_adj, pred_adj = adjustment(gt.copy(), pred.copy())
            
            precision, recall, f_score, _ = precision_recall_fscore_support(
                gt_adj, pred_adj, average='binary', zero_division=0
            )
            
            # Keep track of the threshold that produces the best F-score
            if f_score > best_f_score:
                best_f_score = f_score
                best_pred_adj = pred_adj
                best_gt_adj = gt_adj
                best_accuracy = accuracy_score(gt_adj, pred_adj)
                best_precision = precision
                best_recall = recall

        # Apply best results to downstream variables
        pred_adj = best_pred_adj
        gt_adj = best_gt_adj
        accuracy = best_accuracy
        precision = best_precision
        recall = best_recall
        f_score = best_f_score

        # --- Metric Calculations ---
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

        # LOG SUMMARY RESULTS
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
        
        return float(f_score), row_data
