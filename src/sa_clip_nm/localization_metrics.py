from __future__ import annotations

import numpy as np
from scipy.ndimage import label as connected_component_label


def defect_size_group(
    ground_truth_fraction: float,
    small_max_fraction: float,
    medium_max_fraction: float,
) -> str:
    if not 0.0 <= small_max_fraction <= medium_max_fraction <= 1.0:
        raise ValueError(
            "Defect-size boundaries must satisfy "
            "0 <= small_max_fraction <= medium_max_fraction <= 1"
        )
    if ground_truth_fraction <= small_max_fraction:
        return "small"
    if ground_truth_fraction <= medium_max_fraction:
        return "medium"
    return "large"


def filter_small_components(
    prediction: np.ndarray,
    min_component_area_pixels: int,
) -> np.ndarray:
    prediction_array = np.asarray(prediction).astype(bool)
    if prediction_array.ndim != 2:
        raise ValueError("prediction must be two-dimensional")
    if min_component_area_pixels <= 1:
        return prediction_array.copy()
    labels, component_count = connected_component_label(prediction_array)
    filtered = np.zeros_like(prediction_array, dtype=bool)
    for component_id in range(1, int(component_count) + 1):
        component = labels == component_id
        if int(component.sum()) >= int(min_component_area_pixels):
            filtered |= component
    return filtered


def prediction_from_anomaly_map(
    anomaly_map: np.ndarray,
    threshold: float,
    min_component_area_pixels: int = 0,
) -> np.ndarray:
    anomaly_array = np.asarray(anomaly_map, dtype=np.float64)
    if anomaly_array.ndim != 2:
        raise ValueError("anomaly_map must be two-dimensional")
    prediction = anomaly_array >= float(threshold)
    return filter_small_components(prediction, min_component_area_pixels)


def evaluate_localization_image(
    ground_truth: np.ndarray,
    anomaly_map: np.ndarray,
    threshold: float,
    small_max_fraction: float = 0.005,
    medium_max_fraction: float = 0.02,
    min_component_area_pixels: int = 0,
) -> dict[str, float | int | str | None]:
    ground_truth_array = np.asarray(ground_truth).astype(bool)
    anomaly_array = np.asarray(anomaly_map, dtype=np.float64)
    if ground_truth_array.ndim != 2 or anomaly_array.ndim != 2:
        raise ValueError("ground_truth and anomaly_map must be two-dimensional")
    if ground_truth_array.shape != anomaly_array.shape:
        raise ValueError("ground_truth and anomaly_map must have the same shape")

    prediction = prediction_from_anomaly_map(
        anomaly_array,
        threshold=float(threshold),
        min_component_area_pixels=min_component_area_pixels,
    )
    total_pixels = int(ground_truth_array.size)
    ground_truth_area = int(ground_truth_array.sum())
    prediction_area = int(prediction.sum())
    true_positive = int(np.logical_and(ground_truth_array, prediction).sum())
    false_positive = int(np.logical_and(~ground_truth_array, prediction).sum())
    false_negative = int(np.logical_and(ground_truth_array, ~prediction).sum())
    union = true_positive + false_positive + false_negative
    precision = (
        float(true_positive / (true_positive + false_positive))
        if true_positive + false_positive > 0
        else 0.0
    )
    recall = (
        float(true_positive / (true_positive + false_negative))
        if true_positive + false_negative > 0
        else 0.0
    )
    f1 = (
        float(2.0 * precision * recall / (precision + recall))
        if precision + recall > 0
        else 0.0
    )
    iou = float(true_positive / union) if union > 0 else 1.0
    _, component_count = connected_component_label(prediction)
    positive_fraction = float(prediction_area / total_pixels)
    ground_truth_fraction = float(ground_truth_area / total_pixels)

    result: dict[str, float | int | str | None] = {
        "ground_truth_area": ground_truth_area,
        "ground_truth_fraction": ground_truth_fraction,
        "prediction_area": prediction_area,
        "prediction_positive_fraction": positive_fraction,
        "connected_component_count": int(component_count),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "area_ratio": None,
        "overseg": None,
        "underseg": None,
        "defect_size_group": "normal",
    }
    if ground_truth_area > 0:
        result.update(
            {
                "area_ratio": float(prediction_area / ground_truth_area),
                "overseg": float(false_positive / ground_truth_area),
                "underseg": float(false_negative / ground_truth_area),
                "defect_size_group": defect_size_group(
                    ground_truth_fraction,
                    small_max_fraction=small_max_fraction,
                    medium_max_fraction=medium_max_fraction,
                ),
            }
        )
    return result


def _mean_not_none(rows: list[dict[str, object]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else float("nan")


def summarize_localization_rows(
    rows: list[dict[str, object]],
) -> dict[str, float | int]:
    anomaly_rows = [row for row in rows if int(row["label"]) == 1]
    normal_rows = [row for row in rows if int(row["label"]) == 0]
    summary: dict[str, float | int] = {
        "anomaly_images": len(anomaly_rows),
        "normal_images": len(normal_rows),
        "area_ratio_anomaly_macro": _mean_not_none(anomaly_rows, "area_ratio"),
        "overseg_anomaly_macro": _mean_not_none(anomaly_rows, "overseg"),
        "underseg_anomaly_macro": _mean_not_none(anomaly_rows, "underseg"),
        "precision_anomaly_macro": _mean_not_none(anomaly_rows, "precision"),
        "recall_anomaly_macro": _mean_not_none(anomaly_rows, "recall"),
        "f1_anomaly_macro": _mean_not_none(anomaly_rows, "f1"),
        "iou_anomaly_macro": _mean_not_none(anomaly_rows, "iou"),
        "connected_components_anomaly_macro": _mean_not_none(
            anomaly_rows,
            "connected_component_count",
        ),
        "test_normal_pixel_positive_rate": _mean_not_none(
            normal_rows,
            "prediction_positive_fraction",
        ),
        "test_normal_image_positive_rate": float(
            np.mean([float(row["prediction_area"]) > 0 for row in normal_rows])
        )
        if normal_rows
        else float("nan"),
    }
    for group in ("small", "medium", "large"):
        group_rows = [
            row for row in anomaly_rows if row["defect_size_group"] == group
        ]
        summary[f"{group}_defect_images"] = len(group_rows)
        summary[f"{group}_defect_f1_macro"] = _mean_not_none(group_rows, "f1")
        summary[f"{group}_defect_recall_macro"] = _mean_not_none(
            group_rows,
            "recall",
        )
    return summary
