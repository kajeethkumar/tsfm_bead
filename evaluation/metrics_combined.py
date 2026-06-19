from .basic_metrics import basic_metricor, generate_curve

def calculate_all_metrics(score, labels, pred=None, slidingWindow=100, version='opt', thre=250):
    """
    Integrates threshold-independent curve metrics and threshold-dependent F1 metrics.
    """
    metrics = {}
    grader = basic_metricor()

    # ---------------------------------------------------------
    # 1. Threshold-Independent Metrics (Relies only on continuous 'score')
    # ---------------------------------------------------------
    AUC_ROC = grader.metric_ROC(labels, score)
    AUC_PR = grader.metric_PR(labels, score)

    _, _, _, _, _, _, VUS_ROC, VUS_PR = generate_curve(
        labels.astype(int), score, slidingWindow, version, thre
    )

    metrics['AUC-PR'] = AUC_PR
    metrics['AUC-ROC'] = AUC_ROC
    metrics['VUS-PR'] = VUS_PR
    metrics['VUS-ROC'] = VUS_ROC

    # ---------------------------------------------------------
    # 2. Threshold-Dependent Metrics 
    # If pred is None --> basic_metricor uses the oracle/optimal threshold
    # ---------------------------------------------------------
    metrics['Standard-F1'] = grader.metric_PointF1(labels, score, preds=pred)
    metrics['PA-F1'] = grader.metric_PointF1PA(labels, score, preds=pred)
    metrics['Event-based-F1'] = grader.metric_EventF1PA(labels, score, preds=pred)
    metrics['R-based-F1'] = grader.metric_RF1(labels, score, preds=pred)
    metrics['Affiliation-F'] = grader.metric_Affiliation(labels, score, preds=pred)

    # ---------------------------------------------------------
    # 3. VUS Prediction Metrics (Relies on discrete 'pred')
    # ---------------------------------------------------------
    # We use a try-except block in case `metric_VUS_pred` explicitly 
    # requires `pred` to be an array and fails when `pred` is None.
    try:
        VUS_R, VUS_P, VUS_F = grader.metric_VUS_pred(labels, preds=pred, windowSize=slidingWindow)
        metrics['VUS-Recall'] = VUS_R
        metrics['VUS-Precision'] = VUS_P
        metrics['VUS-F'] = VUS_F
    except Exception:
        # Fails gracefully if 'pred' is None and the grader doesn't support optimal searching for VUS-pred
        metrics['VUS-Recall'] = None
        metrics['VUS-Precision'] = None
        metrics['VUS-F'] = None

    return metrics