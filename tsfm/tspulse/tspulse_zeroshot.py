import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
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

# TSPulse / TSFM specific imports
from tsfm_public.models.tspulse.modeling_tspulse import TSPulseForReconstruction
from tsfm_public.toolkit.time_series_anomaly_detection_pipeline import TimeSeriesAnomalyDetectionPipeline
from tsfm_public.toolkit.ad_helpers import AnomalyScoreMethods

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.torch_utility import get_gpu
from utils.tools import adjustment  
from utils.data_preprocessing import preprocess, train_test_split
from evaluation.metrics import get_metrics, get_metrics_pred


# ===========================================================
# TSPulse Zero-Shot Pipeline Helper & Class
# ===========================================================
class TSPulseZeroShot:
    def __init__(
        self,
        model_path: str = "ibm-granite/granite-timeseries-tspulse-r1",
        batch_size: int = 256,
        aggr_win_size: int = 168,
        num_input_channels: int = 1,
        smoothing_window: int = 24,
        prediction_mode: str = "time", 
        decoder_mode: str = "common_channel",
    ):
        self._batch_size = batch_size
        self._headers = [f"x{i + 1}" for i in range(num_input_channels)]

        if num_input_channels == 1:
            self.decoder_mode = "common_channel"
        else:
            self.decoder_mode = decoder_mode
            
        # Loading the model to memory
        self._model = TSPulseForReconstruction.from_pretrained(
            model_path,
            num_input_channels=num_input_channels,
            decoder_mode=self.decoder_mode,
            scaling="revin",
            mask_type="user",
        )
        
        # Reading patch length from the loaded model instance
        p_length = self._model.config.patch_length
        print(f"Model loaded with patch_length={p_length} and decoder_mode={self.decoder_mode}")
        if (aggr_win_size < p_length) or (aggr_win_size % p_length != 0):
            raise ValueError(f"Error: aggregation window must be greater than and multiple of patch_length={p_length}")
            
        prediction_mode_array = [s_.strip() for s_ in str(prediction_mode).split("+")]
        
        # Storing pipeline configuration parameters
        self._pipeline_config = {
            "target_columns": self._headers.copy(),
            "prediction_mode": [
                    AnomalyScoreMethods.PREDICTIVE.value,
                    AnomalyScoreMethods.TIME_RECONSTRUCTION.value,
                    AnomalyScoreMethods.FREQUENCY_RECONSTRUCTION.value,
            ],
            "aggr_function": "mean",
            "aggregation_length": aggr_win_size,
            "smoothing_window": smoothing_window,
            "least_significant_scale": 0.01,
            "least_significant_score": 0.2,
        }
        
        self._scorer = TimeSeriesAnomalyDetectionPipeline(
            self._model,
            target_columns=self._pipeline_config.get("target_columns"),
            prediction_mode=prediction_mode_array,
            aggregation_length=96,
            smoothing_window=self._pipeline_config.get("smoothing_window"),
            least_significant_scale=self._pipeline_config.get("least_significant_scale"),
            least_significant_score=self._pipeline_config.get("least_significant_score"),
        )

    def decision_function(self, X, timestamps):
        """
        Calculates the zero-shot anomaly score for the input data X using actual timestamps.
        """
        # Ensure X is 2D before putting it in a DataFrame
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        data = pd.DataFrame(X, columns=self._headers)
        data["timestamp"] = timestamps  # Map the actual timestamps from the dataset
        
        score = self._scorer(data, batch_size=self._batch_size)
        
        if not isinstance(score, pd.DataFrame) or ("anomaly_score" not in score):
            raise ValueError("Error: expect anomaly_score column in the output!")

        score = score["anomaly_score"].values.ravel()
        
        # Normalize the scores
        norm_value = np.nanmax(np.asarray(score), axis=0, keepdims=True) + 1e-5
        anomaly_score = score / norm_value
        return anomaly_score


# ===========================================================
# Argument Parsing & Main Execution
# ===========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot anomaly detection using TSPulse")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_csv", type=str, default="./results/TSPulse/zeroshot/tspulse_zeroshot_predictions.csv", help="Output CSV filename for predictions")
    parser.add_argument("--metrics_csv", type=str, default="./results/TSPulse/zeroshot/tspulse_zeroshot.csv", help="Output CSV filename for evaluation metrics")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    
    parser.add_argument("--win_size", type=int, default=168, help="Aggregation window size (must be multiple of patch_length)")
    parser.add_argument("--smoothing_window", type=int, default=8, help="Smoothing window for TSPulse")
    parser.add_argument("--sliding_window", type=int, default=168, help="Tolerance window for metric evaluation")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for data loader processing")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    use_cuda = (args.device == "cuda") and torch.cuda.is_available()
    if args.device == "cuda" and not use_cuda:
        print("CUDA requested but not available. Falling back to CPU.")

    df = pd.read_csv(args.input_csv)

    df1 = pd.read_csv('../../dataset/train_features.csv')

    valid_buildings = df1['building_id'].unique()
    df = df[df['building_id'].isin(valid_buildings)]

    df = df[df['building_id'] > 136]

    print("Loading TSPulse Model...")
    detector = TSPulseZeroShot(
        model_path="ibm-granite/granite-timeseries-tspulse-r1",
        batch_size=args.batch_size,
        aggr_win_size=args.win_size,
        smoothing_window=args.smoothing_window,
        num_input_channels=1, 
    )

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_csv) or ".", exist_ok=True)

    for b_id in tqdm(df.building_id.unique(), desc="Evaluating buildings"):
       
        X, y = preprocess(df, b_id)
        
        _, _, X_test, _, _, y_test = train_test_split(X, y)
        y_test_np = np.asarray(y_test).astype(int).flatten()
        
        # Extract actual timestamps BEFORE calling the decision function
        building = df[df["building_id"] == b_id].sort_values("timestamp").reset_index(drop=True)
        timestamps = building["timestamp"].values
        test_timestamps = timestamps[-len(y_test):]
        reading_col = "meter_reading" 
        test_readings = building[reading_col].values[-len(y_test):]

        # Pass the real timestamps to the model
        scores = detector.decision_function(X_test, timestamps=test_timestamps)
        scores = np.asarray(scores).astype(float).flatten()

        best_f1 = -1.0
        best_threshold = np.min(scores)  # Default to min score if all thresholds fail
        best_pred_adj = None
        best_gt_adj = None
    
        candidate_thresholds = np.linspace(np.min(scores), np.max(scores), 100)
            
        for th in candidate_thresholds:
            binary_preds = (scores >= th).astype(int)
            
            # 1. Apply point adjustment
            gt_adj_temp, pred_adj_temp = adjustment(y_test_np.copy(), binary_preds.copy())
            
            # 2. Calculate F1-score on the ADJUSTED arrays
            _, _, f_score_temp, _ = precision_recall_fscore_support(
                gt_adj_temp, pred_adj_temp, average='binary', zero_division=0
            )
            
            # 3. Keep the best threshold and corresponding adjusted arrays
            if f_score_temp > best_f1:
                best_f1 = f_score_temp # Fixed array-scalar bug here
                best_threshold = th
                best_pred_adj = pred_adj_temp
                best_gt_adj = gt_adj_temp

        # Set final predictions to the optimal found
        pred_adj = best_pred_adj
        gt_adj = best_gt_adj
        threshold = best_threshold

        # Force strict 1D arrays to prevent dimensionality crash
        gt_adj = np.asarray(gt_adj).flatten()
        pred_adj = np.asarray(pred_adj).flatten()
        scores = np.asarray(scores).flatten()

        # Calculate Basic Metrics based on the best adjusted arrays
        accuracy = accuracy_score(gt_adj, pred_adj)
        precision, recall, f_score, _ = precision_recall_fscore_support(
            y_test_np, pred_adj, average='binary', zero_division=0
        )
        
        ad_metrics = get_metrics_pred(score=scores, labels=gt_adj, pred=pred_adj, slidingWindow=args.sliding_window)
        
        try:
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

    print(f"Zero-shot evaluation complete.")
    print(f"Predictions incrementally saved to: {args.output_csv}")
    print(f"Metrics incrementally saved to: {args.metrics_csv}")
