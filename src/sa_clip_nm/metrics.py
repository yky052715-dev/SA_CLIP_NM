from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


@dataclass
class BinaryMetrics:
    auroc: float
    aupr: float
    calibrated_f1: float
    calibrated_iou: float
    oracle_f1: float
    oracle_threshold: float


def _safe_metric(function, labels: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return float("nan")
    return float(function(labels, scores))


def binary_iou(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = labels.astype(bool)
    predictions = predictions.astype(bool)
    intersection = np.logical_and(labels, predictions).sum()
    union = np.logical_or(labels, predictions).sum()
    return float(intersection / union) if union > 0 else 1.0


def oracle_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        return 0.0, float("nan")
    f1 = 2.0 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    index = int(np.nanargmax(f1))
    return float(f1[index]), float(thresholds[index])


def evaluate_binary_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    calibrated_threshold: float,
    compute_oracle: bool,
) -> BinaryMetrics:
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    predictions = scores >= calibrated_threshold
    calibrated_f1 = float(f1_score(labels, predictions, zero_division=0))
    calibrated_iou = binary_iou(labels, predictions)
    if compute_oracle:
        best_f1, best_threshold = oracle_f1_threshold(labels, scores)
    else:
        best_f1, best_threshold = float("nan"), float("nan")
    return BinaryMetrics(
        auroc=_safe_metric(roc_auc_score, labels, scores),
        aupr=_safe_metric(average_precision_score, labels, scores),
        calibrated_f1=calibrated_f1,
        calibrated_iou=calibrated_iou,
        oracle_f1=best_f1,
        oracle_threshold=best_threshold,
    )


def flatten_metrics(prefix: str, metrics: BinaryMetrics) -> dict[str, float]:
    return {
        f"{prefix}_AUROC": metrics.auroc,
        f"{prefix}_AUPR": metrics.aupr,
        f"{prefix}_F1_calibrated": metrics.calibrated_f1,
        f"{prefix}_IoU_calibrated": metrics.calibrated_iou,
        f"{prefix}_F1_oracle": metrics.oracle_f1,
        f"{prefix}_threshold_oracle": metrics.oracle_threshold,
    }

