"""
Advanced Metrics for Time Series Anomaly Detection
===================================================

Provides advanced evaluation metrics beyond standard classification metrics,
specifically designed for time series anomaly detection:

1. AUC-PR (Area Under Precision-Recall Curve)
2. AUC-ROC (Area Under ROC Curve)
3. VUS-PR (Volume Under Surface for PR)
4. VUS-ROC (Volume Under Surface for ROC)
5. Range-based F-score (considers contiguous anomaly segments)
6. Windowed F-score (F1 computed per window then averaged)

References:
- "Precision and Recall for Time Series" (NeurIPS 2018)
- "VUS: Volume Under Surface for Anomaly Detection" (VLDB 2022)
"""

import numpy as np
from typing import List, Sequence, Dict, Any, Optional, Tuple
from sklearn.metrics import (
    precision_recall_curve, 
    auc, 
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)
import logging


logger = logging.getLogger(__name__)


def compute_auc_pr(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """
    Compute Area Under Precision-Recall Curve.
    
    For binary predictions (not probabilities), this approximates AUC-PR
    using the single operating point.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        
    Returns:
        AUC-PR score (0 to 1)
    """
    try:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Handle edge cases
        if len(y_true) == 0 or len(y_pred) == 0:
            return 0.0
        if len(np.unique(y_true)) < 2:
            return 1.0 if np.sum(y_true) == 0 and np.sum(y_pred) == 0 else 0.0
        
        # For binary predictions, compute precision and recall
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        # Approximate AUC-PR as the area of the rectangle formed by P and R
        # This is a simplified version for binary predictions
        return precision * recall if (precision + recall) > 0 else 0.0
        
    except Exception as e:
        logger.warning(f"Error computing AUC-PR: {e}")
        return 0.0


def compute_auc_roc(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """
    Compute Area Under ROC Curve.
    
    For binary predictions, this uses the single operating point.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        
    Returns:
        AUC-ROC score (0 to 1)
    """
    try:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Handle edge cases
        if len(y_true) == 0 or len(y_pred) == 0:
            return 0.5
        if len(np.unique(y_true)) < 2:
            return 1.0 if np.sum(y_pred) == 0 else 0.5
        
        return roc_auc_score(y_true, y_pred)
        
    except Exception as e:
        logger.warning(f"Error computing AUC-ROC: {e}")
        return 0.5


def _range_convers(label: np.ndarray) -> List[Tuple[int, int]]:
    """
    Convert binary label array to list of anomaly segment ranges.
    
    Based on TSB-UAD benchmark implementation.
    
    Args:
        label: Binary array of labels
        
    Returns:
        List of (start, end) tuples for each anomaly segment
    """
    label = np.asarray(label).astype(float)
    anomaly_starts = np.where(np.diff(label) == 1)[0] + 1
    anomaly_ends = np.where(np.diff(label) == -1)[0]
    
    if len(anomaly_ends):
        if not len(anomaly_starts) or anomaly_ends[0] < anomaly_starts[0]:
            anomaly_starts = np.insert(anomaly_starts, 0, 0)
    if len(anomaly_starts):
        if not len(anomaly_ends) or anomaly_ends[-1] < anomaly_starts[-1]:
            anomaly_ends = np.append(anomaly_ends, len(label) - 1)
    
    return list(zip(anomaly_starts.astype(int), anomaly_ends.astype(int)))


def _extend_positive_range(label: np.ndarray, window: int) -> np.ndarray:
    """
    Extend positive (anomaly) regions by window size with decay.
    
    Based on TSB-UAD benchmark: extends anomaly segments with sqrt decay.
    
    Args:
        label: Binary label array
        window: Window size for extension
        
    Returns:
        Extended label array (float, 0-1)
    """
    label = label.copy().astype(float)
    L = _range_convers(label)
    length = len(label)
    
    for s, e in L:
        # Extend after end of anomaly
        x1 = np.arange(e + 1, min(e + window // 2 + 1, length))
        if len(x1) > 0:
            label[x1] += np.sqrt(1 - (x1 - e) / window)
        
        # Extend before start of anomaly
        x2 = np.arange(max(s - window // 2, 0), s)
        if len(x2) > 0:
            label[x2] += np.sqrt(1 - (s - x2) / window)
    
    label = np.minimum(np.ones(length), label)
    return label


def _compute_existence_reward(ranges: List[Tuple[int, int]], preds: np.ndarray) -> float:
    """
    Compute existence reward: fraction of anomaly segments with at least one detection.
    """
    if len(ranges) == 0:
        return 0.0
    
    score = 0
    for start, end in ranges:
        if np.any(preds[start:end + 1]):
            score += 1
    return score / len(ranges)


def _compute_range_tpr_fpr_precision(
    labels_extended: np.ndarray, 
    preds: np.ndarray, 
    P: int, 
    ranges: List[Tuple[int, int]]
) -> Tuple[float, float, float]:
    """
    Compute Range-based TPR, FPR, and Precision.
    
    Based on TSB-UAD benchmark TPR_FPR_RangeAUC function.
    
    Args:
        labels_extended: Extended labels (with window expansion)
        preds: Binary predictions
        P: Original number of anomaly points
        ranges: List of anomaly segment ranges
        
    Returns:
        Tuple of (TPR_Range, FPR_Range, Precision_Range)
    """
    product = labels_extended * preds
    TP = np.sum(product)
    
    # Adjusted P for TPR calculation (matches official TSB-AD implementation)
    # P_new = average of original positives and extended positives
    P_new = (P + np.sum(labels_extended >= 0.5)) / 2
    recall = min(TP / P_new, 1) if P_new > 0 else 0.0
    
    # Existence reward: fraction of segments detected
    existence_ratio = _compute_existence_reward(ranges, preds)
    
    # Range-based TPR combines recall with existence
    TPR_Range = recall * existence_ratio
    
    # FPR calculation
    FP = np.sum(preds) - TP
    N_new = len(labels_extended) - P_new
    FPR_Range = FP / N_new if N_new > 0 else 0.0
    
    # Precision
    Precision_Range = TP / np.sum(preds) if np.sum(preds) > 0 else 0.0
    
    return TPR_Range, FPR_Range, Precision_Range


def compute_vus_pr(y_true: Sequence[int], y_pred: Sequence[int], window_size: Optional[int] = None) -> float:
    """
    Compute Volume Under Surface for Precision-Recall (VUS-PR).
    
    Implements the VUS metric from TSB-UAD benchmark, adapted for binary predictions.
    VUS extends Range-AUC by integrating over different window sizes (0 to window_size),
    creating a 3D surface. VUS-PR averages Range-AP across all window levels.
    
    Reference: "Volume Under the Surface: A New Accuracy Evaluation Measure for 
               Time-Series Anomaly Detection" (VLDB 2022)
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        window_size: Maximum window for label extension (default: len/10, min 2)
        
    Returns:
        VUS-PR score (0 to 1)
    """
    try:
        y_true = np.array(y_true).astype(int)
        y_pred = np.array(y_pred).astype(int)
        
        if len(y_true) == 0 or len(y_pred) == 0:
            return np.nan
        
        P = np.sum(y_true)  # Original number of anomaly points
        
        # VUS is undefined when there are no anomalies (nothing to rank)
        # Return NaN to exclude from averaging (consistent with F1 handling)
        if P == 0:
            return np.nan
        
        # Get anomaly segment ranges
        ranges = _range_convers(y_true)
        if len(ranges) == 0:
            return 0.0
        
        # Set window size (typically 10% of sequence length)
        if window_size is None:
            window_size = max(2, len(y_true) // 10)
        
        # Compute Range-AP at each window level
        ap_values = []
        
        for window in range(window_size + 1):
            # Extend labels by current window
            labels_extended = _extend_positive_range(y_true, window)
            
            # Compute Range-based precision for binary prediction
            _, _, precision = _compute_range_tpr_fpr_precision(
                labels_extended, y_pred, P, ranges
            )
            
            # For binary predictions, Range-AP ≈ precision (single threshold)
            ap_values.append(precision)
        
        # VUS-PR is average Range-AP across all window levels
        vus_pr = np.mean(ap_values)
        return float(vus_pr)
        
    except Exception as e:
        logger.warning(f"Error computing VUS-PR: {e}")
        return 0.0


def compute_vus_roc(y_true: Sequence[int], y_pred: Sequence[int], window_size: Optional[int] = None) -> float:
    """
    Compute Volume Under Surface for ROC (VUS-ROC).
    
    Implements the VUS metric from TSB-UAD benchmark, adapted for binary predictions.
    VUS extends Range-AUC by integrating over different window sizes (0 to window_size),
    creating a 3D surface. VUS-ROC averages Range-AUC across all window levels.
    
    Reference: "Volume Under the Surface: A New Accuracy Evaluation Measure for 
               Time-Series Anomaly Detection" (VLDB 2022)
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        window_size: Maximum window for label extension (default: len/10, min 2)
        
    Returns:
        VUS-ROC score (0 to 1)
    """
    try:
        y_true = np.array(y_true).astype(int)
        y_pred = np.array(y_pred).astype(int)
        
        if len(y_true) == 0 or len(y_pred) == 0:
            return np.nan
        
        P = np.sum(y_true)  # Original number of anomaly points
        
        # VUS is undefined when there are no anomalies (nothing to rank)
        # Return NaN to exclude from averaging (consistent with F1 handling)
        if P == 0:
            return np.nan
        
        # Get anomaly segment ranges
        ranges = _range_convers(y_true)
        if len(ranges) == 0:
            return 0.5
        
        # Set window size (typically 10% of sequence length)
        if window_size is None:
            window_size = max(2, len(y_true) // 10)
        
        # Compute Range-AUC at each window level
        auc_values = []
        
        for window in range(window_size + 1):
            # Extend labels by current window
            labels_extended = _extend_positive_range(y_true, window)
            
            # Compute Range-based TPR and FPR for binary prediction
            tpr, fpr, _ = _compute_range_tpr_fpr_precision(
                labels_extended, y_pred, P, ranges
            )
            
            # For binary predictions with single threshold, Range-AUC ≈ 
            # area of trapezoid from (0,0) to (fpr,tpr) to (1,1)
            # Simplified: we use (TPR + (1-FPR)) / 2 as the operating point quality
            range_auc = (tpr + (1 - fpr)) / 2
            auc_values.append(range_auc)
        
        # VUS-ROC is average Range-AUC across all window levels
        vus_roc = np.mean(auc_values)
        return float(vus_roc)
        
    except Exception as e:
        logger.warning(f"Error computing VUS-ROC: {e}")
        return 0.5


def compute_range_f_score(
    y_true: Sequence[int], 
    y_pred: Sequence[int],
    alpha: float = 0.5
) -> Dict[str, float]:
    """
    Compute Range-based Precision, Recall, and F-score.
    
    Range-based metrics consider contiguous anomaly segments rather than
    individual points. A predicted segment that overlaps with a true
    segment is considered a (partial) match.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        alpha: Weight for combining overlap and cardinality (0 to 1)
        
    Returns:
        Dict with 'range_precision', 'range_recall', 'range_f_score'
    """
    try:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Find contiguous ranges
        true_ranges = _find_ranges(y_true)
        pred_ranges = _find_ranges(y_pred)
        
        if len(true_ranges) == 0 and len(pred_ranges) == 0:
            return {'range_precision': 1.0, 'range_recall': 1.0, 'range_f_score': 1.0}
        
        if len(true_ranges) == 0:
            return {'range_precision': 0.0, 'range_recall': 1.0, 'range_f_score': 0.0}
        
        if len(pred_ranges) == 0:
            return {'range_precision': 1.0, 'range_recall': 0.0, 'range_f_score': 0.0}
        
        # Compute range-based metrics
        range_precision = _compute_range_precision(true_ranges, pred_ranges, alpha)
        range_recall = _compute_range_recall(true_ranges, pred_ranges, alpha)
        
        if range_precision + range_recall > 0:
            range_f = 2 * (range_precision * range_recall) / (range_precision + range_recall)
        else:
            range_f = 0.0
        
        return {
            'range_precision': range_precision,
            'range_recall': range_recall,
            'range_f_score': range_f
        }
        
    except Exception as e:
        logger.warning(f"Error computing range F-score: {e}")
        return {'range_precision': 0.0, 'range_recall': 0.0, 'range_f_score': 0.0}


def _find_ranges(labels: np.ndarray) -> List[Tuple[int, int]]:
    """Find contiguous ranges of 1s in a binary array."""
    ranges = []
    in_range = False
    start = 0
    
    for i, val in enumerate(labels):
        if val == 1 and not in_range:
            in_range = True
            start = i
        elif val == 0 and in_range:
            in_range = False
            ranges.append((start, i - 1))
    
    if in_range:
        ranges.append((start, len(labels) - 1))
    
    return ranges


def _compute_range_precision(
    true_ranges: List[Tuple[int, int]], 
    pred_ranges: List[Tuple[int, int]],
    alpha: float
) -> float:
    """Compute range-based precision."""
    if len(pred_ranges) == 0:
        return 1.0
    
    total_score = 0.0
    for pred_start, pred_end in pred_ranges:
        pred_len = pred_end - pred_start + 1
        
        # Find overlapping true ranges
        overlap = 0
        for true_start, true_end in true_ranges:
            overlap_start = max(pred_start, true_start)
            overlap_end = min(pred_end, true_end)
            if overlap_start <= overlap_end:
                overlap += overlap_end - overlap_start + 1
        
        # Score based on overlap ratio
        total_score += overlap / pred_len if pred_len > 0 else 0
    
    return total_score / len(pred_ranges)


def _compute_range_recall(
    true_ranges: List[Tuple[int, int]], 
    pred_ranges: List[Tuple[int, int]],
    alpha: float
) -> float:
    """Compute range-based recall."""
    if len(true_ranges) == 0:
        return 1.0
    
    total_score = 0.0
    for true_start, true_end in true_ranges:
        true_len = true_end - true_start + 1
        
        # Find overlapping predicted ranges
        overlap = 0
        for pred_start, pred_end in pred_ranges:
            overlap_start = max(true_start, pred_start)
            overlap_end = min(true_end, pred_end)
            if overlap_start <= overlap_end:
                overlap += overlap_end - overlap_start + 1
        
        # Score based on overlap ratio
        total_score += overlap / true_len if true_len > 0 else 0
    
    return total_score / len(true_ranges)


def compute_avg_windowed_f1(window_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute average windowed F1 from pre-computed window results.
    
    Args:
        window_results: List of dicts with 'f1_score', 'was_skipped', 'true_anomaly_count'
        
    Returns:
        Dict with 'avg_windowed_f1' and 'avg_windowed_f1_weighted'
    """
    try:
        # Filter to processed windows with anomalies
        processed = [w for w in window_results 
                    if not w.get('was_skipped', True) and w.get('true_anomaly_count', 0) > 0]
        
        if not processed:
            return {'avg_windowed_f1': 0.0, 'avg_windowed_f1_weighted': 0.0}
        
        f1_scores = [w.get('f1_score', 0.0) for w in processed]
        weights = [w.get('true_anomaly_count', 1) for w in processed]
        
        avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
        
        # Weighted average by anomaly count
        total_weight = sum(weights)
        if total_weight > 0:
            avg_f1_weighted = sum(f * w for f, w in zip(f1_scores, weights)) / total_weight
        else:
            avg_f1_weighted = 0.0
        
        return {
            'avg_windowed_f1': float(avg_f1),
            'avg_windowed_f1_weighted': float(avg_f1_weighted)
        }
        
    except Exception as e:
        logger.warning(f"Error computing average windowed F1: {e}")
        return {'avg_windowed_f1': 0.0, 'avg_windowed_f1_weighted': 0.0}


def compute_windowed_f_score(
    y_true: Sequence[int], 
    y_pred: Sequence[int],
    window_size: int = 24
) -> Dict[str, float]:
    """
    Compute F1 score for each window and average.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels  
        window_size: Size of each window
        
    Returns:
        Dict with 'windowed_f1', 'windowed_precision', 'windowed_recall'
    """
    try:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        n_windows = len(y_true) // window_size
        
        if n_windows == 0:
            # Single window case
            return {
                'windowed_f1': f1_score(y_true, y_pred, zero_division=0),
                'windowed_precision': precision_score(y_true, y_pred, zero_division=0),
                'windowed_recall': recall_score(y_true, y_pred, zero_division=0)
            }
        
        f1_scores = []
        precision_scores = []
        recall_scores = []
        
        for i in range(n_windows):
            start = i * window_size
            end = start + window_size
            
            window_true = y_true[start:end]
            window_pred = y_pred[start:end]
            
            f1_scores.append(f1_score(window_true, window_pred, zero_division=0))
            precision_scores.append(precision_score(window_true, window_pred, zero_division=0))
            recall_scores.append(recall_score(window_true, window_pred, zero_division=0))
        
        return {
            'windowed_f1': np.mean(f1_scores),
            'windowed_precision': np.mean(precision_scores),
            'windowed_recall': np.mean(recall_scores)
        }
        
    except Exception as e:
        logger.warning(f"Error computing windowed F-score: {e}")
        return {'windowed_f1': 0.0, 'windowed_precision': 0.0, 'windowed_recall': 0.0}


def compute_all_advanced_metrics(
    y_true: Sequence[int], 
    y_pred: Sequence[int],
    window_size: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute all advanced metrics in one call.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        window_size: Optional window size for windowed metrics
        
    Returns:
        Dict with all metric values
    """
    results = {
        'auc_pr': compute_auc_pr(y_true, y_pred),
        'auc_roc': compute_auc_roc(y_true, y_pred),
        'vus_pr': compute_vus_pr(y_true, y_pred),
        'vus_roc': compute_vus_roc(y_true, y_pred),
    }
    
    range_metrics = compute_range_f_score(y_true, y_pred)
    results.update(range_metrics)
    
    if window_size:
        windowed = compute_windowed_f_score(y_true, y_pred, window_size)
        results.update(windowed)
    
    return results