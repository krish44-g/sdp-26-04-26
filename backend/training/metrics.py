"""
Evaluation metrics for SpineAI: PCKh@0.5, F1, AUC-ROC, Severity MAE.
"""
import numpy as np
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix, average_precision_score
)
from typing import Dict


def compute_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    probs: (N, C) sigmoid probabilities
    labels: (N, C) multi-hot ground truth
    """
    preds = (probs >= threshold).astype(int)

    metrics = {}

    # Per-class and macro F1
    metrics["macro_f1"] = float(f1_score(labels, preds, average="macro", zero_division=0))
    metrics["weighted_f1"] = float(f1_score(labels, preds, average="weighted", zero_division=0))
    metrics["per_class_f1"] = f1_score(labels, preds, average=None, zero_division=0).tolist()

    # Precision and Recall
    metrics["macro_precision"] = float(precision_score(labels, preds, average="macro", zero_division=0))
    metrics["macro_recall"] = float(recall_score(labels, preds, average="macro", zero_division=0))

    # AUC-ROC (macro)
    try:
        metrics["macro_auc"] = float(roc_auc_score(labels, probs, average="macro"))
        metrics["per_class_auc"] = roc_auc_score(labels, probs, average=None).tolist()
    except ValueError:
        metrics["macro_auc"] = 0.0
        metrics["per_class_auc"] = [0.0] * labels.shape[1]

    # Average Precision
    try:
        metrics["map"] = float(average_precision_score(labels, probs, average="macro"))
    except ValueError:
        metrics["map"] = 0.0

    return metrics


def pckh_at_05(
    pred_kps: np.ndarray,
    gt_kps: np.ndarray,
    head_size: float = 0.1,
) -> float:
    """
    Percentage of Correct Keypoints with threshold 0.5 * head_size.
    pred_kps, gt_kps: (N, K, 2) normalized [0,1]
    head_size: normalized head diameter estimate (default 10% of image)
    """
    threshold = 0.5 * head_size
    dist = np.linalg.norm(pred_kps - gt_kps, axis=-1)  # (N, K)
    correct = (dist < threshold).astype(float)
    return float(correct.mean())


def severity_mae(pred_probs: np.ndarray, gt_severity: np.ndarray, labels: np.ndarray) -> float:
    """Mean Absolute Error of severity prediction for positive classes only."""
    mask = labels > 0
    if not mask.any():
        return 0.0
    return float(np.abs(pred_probs[mask] - gt_severity[mask]).mean())
