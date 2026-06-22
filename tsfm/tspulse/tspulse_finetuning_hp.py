import os
import sys
import math
import random
import tempfile
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import gc

# Limit CPU threading to prevent CPU memory spikes
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
torch.set_num_threads(4)
torch.backends.cudnn.enabled = False

# Scikit-learn metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.exceptions import UndefinedMetricWarning
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Silence Hugging Face sequential pipeline warnings
import transformers
transformers.logging.set_verbosity_error()

# TSPulse / TSFM specific imports
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
from tsfm_public.models.tspulse.modeling_tspulse import TSPulseForReconstruction
from tsfm_public.models.tspulse.utils.helpers import PatchMaskingDatasetWrapper
from tsfm_public.toolkit.time_series_anomaly_detection_pipeline import TimeSeriesAnomalyDetectionPipeline
from tsfm_public.toolkit.ad_helpers import AnomalyScoreMethods


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utils.dataset import TSPulseFinetuneDataset
from utils.torch_utility import get_gpu
from utils.tools import adjustment  
from utils.data_preprocessing import preprocess, train_test_split
from evaluation.metrics import get_metrics, get_metrics_pred

# ===========================================================
# TSPulse Fine-Tuning Class
# ===========================================================
class TSPulseFineTuner:
    def __init__(
        self,
        model_path: str = "ibm-granite/granite-timeseries-tspulse-r1",
        batch_size: int = 256,
        aggr_win_size: int = 168,
        num_input_channels: int = 1,
        smoothing_window: int = 24,
        prediction_mode: str = "time", 
        decoder_mode: str = "common_channel",
        finetune_epochs: int = 20,
        finetune_lr: float = 1e-4,
        finetune_seed: int = 42,
        finetune_freeze_backbone: bool = False,
    ):
        self._batch_size = batch_size
        self._headers = [f"x{i + 1}" for i in range(num_input_channels)]
        self._finetune_params = {
            "finetune_epochs": finetune_epochs,
            "finetune_lr": finetune_lr,
            "finetune_seed": finetune_seed,
            "finetune_freeze_backbone": finetune_freeze_backbone,
        }

        if num_input_channels == 1:
            self.decoder_mode = "common_channel"
        else:
            self.decoder_mode = decoder_mode
            
        random.seed(finetune_seed)
        np.random.seed(finetune_seed)
        torch.manual_seed(finetune_seed)
            
        self._model = TSPulseForReconstruction.from_pretrained(
            model_path,
            num_input_channels=num_input_channels,
            decoder_mode=self.decoder_mode,
            scaling="revin",
            mask_type="user",
        )
        
        self.base_state_dict = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
        
        p_length = self._model.config.patch_length
        # print(f"Model loaded with patch_length={p_length} and decoder_mode={self.decoder_mode}")
        if (aggr_win_size < p_length) or (aggr_win_size % p_length != 0):
            raise ValueError(f"Error: aggregation window must be greater than and multiple of patch_length={p_length}")
            
        prediction_mode_array = [s_.strip() for s_ in str(prediction_mode).split("+")]
        
        self._pipeline_config = {
            "target_columns": self._headers.copy(),
            "prediction_mode": [
                    AnomalyScoreMethods.PREDICTIVE.value,
                    AnomalyScoreMethods.TIME_RECONSTRUCTION.value,
                    AnomalyScoreMethods.FREQUENCY_RECONSTRUCTION.value,
            ],
            "aggr_function": "mean",
            "aggregation_length": 96,
            "smoothing_window": smoothing_window,
            "least_significant_scale": 0.01,
            "least_significant_score": 0.2,
        }
        
        self._setup_scorer(prediction_mode_array, "mean")

    def _setup_scorer(self, mode_array, aggr_func):
        """Helper to re-initialize the pipeline with new hyperparameters."""
        self._scorer = TimeSeriesAnomalyDetectionPipeline(
            self._model,
            target_columns=self._pipeline_config.get("target_columns"),
            prediction_mode=mode_array,
            aggr_function=aggr_func,
            aggregation_length=self._pipeline_config.get("aggregation_length"),
            smoothing_window=self._pipeline_config.get("smoothing_window"),
            least_significant_scale=self._pipeline_config.get("least_significant_scale"),
            least_significant_score=self._pipeline_config.get("least_significant_score"),
        )

    def reset_weights(self):
        """Restores the model to its original pre-trained state."""
        self._model.load_state_dict(self.base_state_dict)
        # Reset to default scorer state before tuning
        self._setup_scorer(self._pipeline_config["prediction_mode"], self._pipeline_config["aggr_function"])

    def fit(self, X_train, X_val):
        """Fine-tunes the model on the explicitly provided train and validation sets."""
        try:
            X_train = np.asarray(X_train)
            if X_train.ndim == 1:
                X_train = X_train.reshape(-1, 1)
                
            if X_val is not None:
                X_val = np.asarray(X_val)
                if X_val.ndim == 1:
                    X_val = X_val.reshape(-1, 1)

            context_length = self._model.config.context_length

            if len(X_train) < context_length:
                tqdm.write("Skipping fine-tuning due to very short training sequence.")
                return

            max_finetune_samples = 4392  # 6 months of hourly data, which is a reasonable upper bound for fine-tuning without overfitting  
            if len(X_train) > max_finetune_samples:
                X_train = X_train[-max_finetune_samples:]
                tqdm.write(f"Training sequence truncated to the most recent {max_finetune_samples} steps.")

            train_dataset = PatchMaskingDatasetWrapper(
                TSPulseFinetuneDataset(X_train, window_size=context_length, return_dict=True),
                window_length=self._pipeline_config.get("aggregation_length"),
                patch_length=self._model.config.patch_length,
                window_position="last",
            )
            
            if len(train_dataset) < 100:
                tqdm.write("Skipping fine-tuning due to very few training samples.")
                return

            create_valid = False
            if X_val is not None and len(X_val) >= context_length:
                create_valid = True
                valid_dataset = PatchMaskingDatasetWrapper(
                    TSPulseFinetuneDataset(X_val, window_size=context_length, return_dict=True),
                    window_length=self._pipeline_config.get("aggregation_length"),
                    patch_length=self._model.config.patch_length,
                    window_position="last",
                )
            else:
                tqdm.write("Validation set too small or not provided. Training without validation evaluation.")
                valid_dataset = train_dataset

            freeze_backbone = self._finetune_params.get("finetune_freeze_backbone")
            if freeze_backbone:
                for param in self._model.backbone.parameters():
                    param.requires_grad = False

            temp_dir = tempfile.mkdtemp()
            suggested_lr = self._finetune_params.get("finetune_lr", 1e-4)
            finetune_num_epochs = int(self._finetune_params.get("finetune_epochs", 20))
            if not create_valid:
                finetune_num_epochs = min(5, finetune_num_epochs)

            finetune_batch_size = self._batch_size
            if len(train_dataset) < 500:
                finetune_batch_size = 8
                
            num_workers = 4
            num_gpus = 1 if torch.cuda.is_available() else 0

            finetune_args = TrainingArguments(
                output_dir=temp_dir,
                overwrite_output_dir=True,
                learning_rate=suggested_lr,
                num_train_epochs=finetune_num_epochs,
                do_eval=True,
                eval_strategy="epoch",
                per_device_train_batch_size=finetune_batch_size,
                per_device_eval_batch_size=finetune_batch_size * 10,
                dataloader_num_workers=num_workers,
                report_to="none", 
                save_strategy="epoch",
                logging_strategy="epoch",
                save_total_limit=1,
                logging_dir=temp_dir, 
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False, 
            )

            early_stopping_callback = EarlyStoppingCallback(
                early_stopping_patience=3, 
                early_stopping_threshold=1e-5,
            )

            optimizer = AdamW(filter(lambda p: p.requires_grad, self._model.parameters()), lr=suggested_lr)
            scheduler = OneCycleLR(
                optimizer,
                suggested_lr,
                epochs=finetune_num_epochs,
                steps_per_epoch=math.ceil(len(train_dataset) / (finetune_batch_size * max(num_gpus, 1))),
            )

            finetune_trainer = Trainer(
                model=self._model,
                args=finetune_args,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                callbacks=[early_stopping_callback],
                optimizers=(optimizer, scheduler),
            )

            finetune_trainer.train()

        except Exception as e:
            tqdm.write(f"Error occurred in finetune. Error = {e}")

    def tune_pipeline(self, X_val, y_val, val_timestamps):
        """Sweeps prediction modes and aggregation functions to find the optimal combination."""
        best_f1 = -1.0
        best_config = None
        y_val_np = np.asarray(y_val).astype(int).flatten()

        if np.sum(y_val_np) == 0:
            tqdm.write("No anomalies in validation set. Skipping hyperparameter tuning.")
            return

        # Define candidate configurations
        candidates = [
            {"mode": [AnomalyScoreMethods.PREDICTIVE.value], "aggr": "mean"},
            {"mode": [AnomalyScoreMethods.TIME_RECONSTRUCTION.value], "aggr": "mean"},
            {"mode": [AnomalyScoreMethods.FREQUENCY_RECONSTRUCTION.value], "aggr": "mean"},
            {"mode": [AnomalyScoreMethods.PREDICTIVE.value, AnomalyScoreMethods.TIME_RECONSTRUCTION.value], "aggr": "mean"},
            {"mode": [AnomalyScoreMethods.PREDICTIVE.value, AnomalyScoreMethods.TIME_RECONSTRUCTION.value], "aggr": "max"},
            {"mode": [AnomalyScoreMethods.PREDICTIVE.value, AnomalyScoreMethods.TIME_RECONSTRUCTION.value, AnomalyScoreMethods.FREQUENCY_RECONSTRUCTION.value], "aggr": "mean"},
            {"mode": [AnomalyScoreMethods.PREDICTIVE.value, AnomalyScoreMethods.TIME_RECONSTRUCTION.value, AnomalyScoreMethods.FREQUENCY_RECONSTRUCTION.value], "aggr": "max"},
        ]

        for config in candidates:
            # Inject new params
            self._setup_scorer(config["mode"], config["aggr"])
            
            # Predict
            try:
                val_scores = self.decision_function(X_val, val_timestamps)
                val_scores = np.asarray(val_scores).astype(float).flatten()
                
                # Fast sweep for F1 (50 steps instead of 100 to save time)
                current_best_f1 = 0
                candidate_thresholds = np.linspace(np.min(val_scores), np.max(val_scores), 100)
                
                for th in candidate_thresholds:
                    binary_preds = (val_scores >= th).astype(int)
                    gt_adj_temp, pred_adj_temp = adjustment(y_val_np.copy(), binary_preds.copy())
                    _, _, f_score_temp, _ = precision_recall_fscore_support(
                        gt_adj_temp, pred_adj_temp, average='binary', zero_division=0
                    )
                    if f_score_temp > current_best_f1:
                        current_best_f1 = f_score_temp
                
                if current_best_f1 > best_f1:
                    best_f1 = current_best_f1
                    best_config = config
            except Exception as e:
                tqdm.write(f"Tuning failed for config {config}. Error: {e}")
                continue

        if best_config:
            tqdm.write(f"Optimized pipeline for this building: {best_config['mode']} | Aggregation: {best_config['aggr']} (Val F1: {best_f1:.4f})")
            self._setup_scorer(best_config["mode"], best_config["aggr"])


    def decision_function(self, X, timestamps):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        data = pd.DataFrame(X, columns=self._headers)
        data["timestamp"] = timestamps 
        
        score = self._scorer(data, batch_size=self._batch_size)
        
        if not isinstance(score, pd.DataFrame) or ("anomaly_score" not in score):
            raise ValueError("Error: expect anomaly_score column in the output!")

        score = score["anomaly_score"].values.ravel()
        
        norm_value = np.nanmax(np.asarray(score), axis=0, keepdims=True) + 1e-5
        anomaly_score = score / norm_value
        return anomaly_score


# ===========================================================
# Argument Parsing & Main Execution
# ===========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuned anomaly detection using TSPulse")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_csv", type=str, default="./results/TSPulse/finetuned/tspulse_finetuned_predictions.csv", help="Output CSV filename for predictions")
    parser.add_argument("--metrics_csv", type=str, default="./results/TSPulse/finetuned/tspulse_finetuned.csv", help="Output CSV filename for evaluation metrics")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    
    parser.add_argument("--win_size", type=int, default=168, help="Aggregation window size (must be multiple of patch_length)")
    parser.add_argument("--smoothing_window", type=int, default=8, help="Smoothing window for TSPulse")
    parser.add_argument("--sliding_window", type=int, default=168, help="Tolerance window for metric evaluation")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for data loader processing")
    parser.add_argument("--epochs", type=int, default=20, help="Number of fine-tuning epochs per building")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for fine-tuning")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    use_cuda = (args.device == "cuda") and torch.cuda.is_available()
    if args.device == "cuda" and not use_cuda:
        print("CUDA requested but not available. Falling back to CPU.")

    df = pd.read_csv(args.input_csv)

    df1 = pd.read_csv('dataset/train_features.csv')

    valid_buildings = df1['building_id'].unique()
    df = df[df['building_id'].isin(valid_buildings)]
    df = df[df['building_id'] > 1225]

    print("Loading TSPulse Model...")
    detector = TSPulseFineTuner(
        model_path="ibm-granite/granite-timeseries-tspulse-r1",
        batch_size=args.batch_size,
        aggr_win_size=args.win_size,
        smoothing_window=args.smoothing_window,
        num_input_channels=1, 
        finetune_epochs=args.epochs,
        finetune_lr=args.lr
    )

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_csv) or ".", exist_ok=True)

    for b_id in tqdm(df.building_id.unique(), desc="Evaluating buildings"):
       
        X, y = preprocess(df, b_id)
        
        X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(X, y)
        y_test_np = np.asarray(y_test).astype(int).flatten()
        
        building = df[df["building_id"] == b_id].sort_values("timestamp").reset_index(drop=True)
        timestamps = building["timestamp"].values
        
        train_len = len(y_train)
        val_len = len(y_val)
        
        # Ensure proper timestamp alignment for validation and test splits
        val_timestamps = timestamps[train_len : train_len + val_len]
        test_timestamps = timestamps[-len(y_test):]
        
        reading_col = "meter_reading" 
        test_readings = building[reading_col].values[-len(y_test):]

        # ---------------------------------------------------------
        # Per-Building Fine-Tuning & Pipeline Optimization
        # ---------------------------------------------------------
        detector.reset_weights() 
        
        tqdm.write(f"\n--- Fine-tuning Building {b_id} ---")
        detector.fit(X_train, X_val)
        
        if len(X_val) > 0 and len(y_val) > 0:
            detector.tune_pipeline(X_val, y_val, val_timestamps)
        
        # ---------------------------------------------------------
        # Testing
        # ---------------------------------------------------------
        scores = detector.decision_function(X_test, timestamps=test_timestamps)
        scores = np.asarray(scores).astype(float).flatten()

        best_f1 = -1.0
        best_threshold = np.min(scores)  
        best_pred_adj = None
        best_gt_adj = None
    
        candidate_thresholds = np.linspace(np.min(scores), np.max(scores), 100)
            
        for th in candidate_thresholds:
            binary_preds = (scores >= th).astype(int)
            
            gt_adj_temp, pred_adj_temp = adjustment(y_test_np.copy(), binary_preds.copy())
            
            _, _, f_score_temp, _ = precision_recall_fscore_support(
                gt_adj_temp, pred_adj_temp, average='binary', zero_division=0
            )
            
            if f_score_temp > best_f1:
                best_f1 = f_score_temp 
                best_threshold = th
                best_pred_adj = pred_adj_temp
                best_gt_adj = gt_adj_temp

        pred_adj = best_pred_adj
        gt_adj = best_gt_adj
        threshold = best_threshold

        gt_adj = np.asarray(gt_adj).flatten()
        pred_adj = np.asarray(pred_adj).flatten()
        scores = np.asarray(scores).flatten()

        accuracy = accuracy_score(gt_adj, pred_adj)
        precision, recall, f_score, _ = precision_recall_fscore_support(
            y_test_np, pred_adj, average='binary', zero_division=0
        )
        
        ad_metrics = {}
        if np.sum(gt_adj) == 0:
            tqdm.write(f"\n[Notice] Building {b_id} has 0 true anomalies. AUC/VUS skipped.")
        else:
            try:
                ad_metrics = get_metrics_pred(score=scores, labels=gt_adj, pred=pred_adj, slidingWindow=args.sliding_window)
                ad_metrics.update(get_metrics(
                    score=scores, 
                    labels=gt_adj, 
                    pred=None, 
                    slidingWindow=args.sliding_window 
                ))  
            except Exception as e:
                tqdm.write(f"\n[Warning] AUC/VUS metrics failed for building {b_id}. Reason: {e}")

        building_metrics = {
            "building_id": b_id,
            "best_threshold": best_threshold, 
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
            "label": gt_adj,
            "pred": pred_adj
        })

        if not pred_df.empty:
            pred_df.to_csv(args.output_csv, mode='a', header=not os.path.exists(args.output_csv), index=False)

        metric_df = pd.DataFrame([building_metrics])
        cols = ['building_id'] + [c for c in metric_df.columns if c != 'building_id']
        metric_df = metric_df[cols]
        metric_df.to_csv(args.metrics_csv, mode='a', header=not os.path.exists(args.metrics_csv), index=False)
        
        del scores, pred_df, metric_df, X_test, y_test
        if use_cuda:
            torch.cuda.empty_cache()
        gc.collect() 

    print(f"Fine-tuning evaluation complete.")
    print(f"Predictions incrementally saved to: {args.output_csv}")
    print(f"Metrics incrementally saved to: {args.metrics_csv}")
