from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .config import save_json


@dataclass
class LayerCalibration:
    median: float
    mad: float
    reliability: float
    spatial_weight: float


@dataclass
class CategoryCalibration:
    category: str
    layers: dict[int, LayerCalibration]
    pixel_threshold: float
    image_threshold: float
    pixel_threshold_method: str
    pixel_quantile: float
    pixel_image_quantile: float
    pixel_topk_fraction: float
    image_quantile: float
    image_score_method: str
    image_topk_fraction: float
    normal_pixel_positive_rate: float
    normal_image_positive_rate: float
    threshold_fit_images: int
    normal_validation_images: int
    normalization_fit_images: int
    normal_diagnostic_mode: str
    adaptive_pixel_image_quantiles: list[float] | None = None
    adaptive_selected_pixel_image_quantile: float | None = None
    adaptive_max_normal_image_positive_rate: float | None = None
    adaptive_max_normal_pixel_positive_rate: float | None = None
    adaptive_threshold_candidates: list[dict[str, float]] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "category": self.category,
            "layers": {
                str(layer): asdict(values) for layer, values in self.layers.items()
            },
            "pixel_threshold": self.pixel_threshold,
            "image_threshold": self.image_threshold,
            "pixel_threshold_method": self.pixel_threshold_method,
            "pixel_quantile": self.pixel_quantile,
            "pixel_image_quantile": self.pixel_image_quantile,
            "pixel_topk_fraction": self.pixel_topk_fraction,
            "image_quantile": self.image_quantile,
            "image_score_method": self.image_score_method,
            "image_topk_fraction": self.image_topk_fraction,
            "normal_pixel_positive_rate": self.normal_pixel_positive_rate,
            "normal_image_positive_rate": self.normal_image_positive_rate,
            "threshold_fit_images": self.threshold_fit_images,
            "normal_validation_images": self.normal_validation_images,
            "normalization_fit_images": self.normalization_fit_images,
            "normal_diagnostic_mode": self.normal_diagnostic_mode,
            "adaptive_pixel_image_quantiles": self.adaptive_pixel_image_quantiles,
            "adaptive_selected_pixel_image_quantile": (
                self.adaptive_selected_pixel_image_quantile
            ),
            "adaptive_max_normal_image_positive_rate": (
                self.adaptive_max_normal_image_positive_rate
            ),
            "adaptive_max_normal_pixel_positive_rate": (
                self.adaptive_max_normal_pixel_positive_rate
            ),
            "adaptive_threshold_candidates": (
                self.adaptive_threshold_candidates
            ),
        }

    def save(self, path: str | Path) -> None:
        save_json(self.to_dict(), path)


def _upper_triangle_mean(matrix: torch.Tensor) -> torch.Tensor:
    size = matrix.shape[-1]
    if size < 2:
        return torch.zeros(matrix.shape[:-2], dtype=matrix.dtype, device=matrix.device)
    upper = torch.triu_indices(size, size, offset=1, device=matrix.device)
    return matrix[..., upper[0], upper[1]].mean(dim=-1)


@torch.inference_mode()
def estimate_spatial_reliability(calibration_features: torch.Tensor) -> dict[str, float]:
    """Compute robust same-position and cross-position cosine-distance statistics.

    Args:
        calibration_features: L2-normalized tensor with shape [N, P, D].
    """
    if calibration_features.ndim != 3:
        raise ValueError("calibration_features must have shape [N, P, D]")
    if calibration_features.shape[0] < 2:
        raise ValueError("At least two calibration images are required")
    features = F.normalize(calibration_features.float(), dim=-1)

    by_position = features.permute(1, 0, 2)
    same_similarity = torch.bmm(by_position, by_position.transpose(1, 2))
    same_distance = 1.0 - same_similarity
    same_per_position = _upper_triangle_mean(same_distance)
    same = torch.median(same_per_position)

    cross_similarity = torch.bmm(features, features.transpose(1, 2))
    cross_distance = 1.0 - cross_similarity
    cross_per_image = _upper_triangle_mean(cross_distance)
    cross = torch.median(cross_per_image)

    epsilon = 1e-8
    reliability = cross / (same + epsilon)
    return {
        "same_position_distance": float(same.item()),
        "cross_position_distance": float(cross.item()),
        "reliability": float(reliability.item()),
    }


def compute_layer_tau(
    category_reliabilities: dict[str, dict[int, float]],
    quantile: float,
) -> dict[int, float]:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    layer_values: dict[int, list[float]] = {}
    for reliabilities in category_reliabilities.values():
        for layer, value in reliabilities.items():
            layer_values.setdefault(int(layer), []).append(float(value))
    return {
        layer: float(np.quantile(values, quantile))
        for layer, values in layer_values.items()
    }


def reliability_to_weight(
    reliability: float,
    tau: float,
    lambda_max: float,
) -> float:
    if lambda_max < 0:
        raise ValueError("lambda_max must be non-negative")
    denominator = reliability + tau
    if denominator <= 0:
        return 0.0
    return float(lambda_max * reliability / denominator)


def robust_location_scale(values: torch.Tensor, epsilon: float) -> tuple[float, float]:
    flat = values.detach().float().reshape(-1)
    median = torch.median(flat)
    mad = torch.median(torch.abs(flat - median))
    return float(median.item()), max(float(mad.item()), epsilon)


def calibrate_scores(
    scores: torch.Tensor,
    median: float,
    mad: float,
    clamp_min_zero: bool,
) -> torch.Tensor:
    calibrated = (scores - median) / mad
    if clamp_min_zero:
        calibrated = calibrated.clamp_min(0.0)
    return calibrated


def image_score_from_map(
    anomaly_map: torch.Tensor,
    method: str,
    topk_fraction: float,
) -> torch.Tensor:
    if anomaly_map.ndim < 2:
        raise ValueError("anomaly_map must have at least two dimensions")
    if anomaly_map.ndim == 2:
        flattened = anomaly_map.reshape(1, -1)
    else:
        flattened = anomaly_map.reshape(anomaly_map.shape[0], -1)
    if method == "max":
        return flattened.max(dim=1).values
    if method == "q99":
        return torch.quantile(flattened, 0.99, dim=1)
    if method == "topk_mean":
        if not 0.0 < topk_fraction <= 1.0:
            raise ValueError("topk_fraction must be in (0, 1]")
        count = max(1, int(round(flattened.shape[1] * topk_fraction)))
        return torch.topk(flattened, count, dim=1).values.mean(dim=1)
    raise ValueError(f"Unsupported image score method: {method}")


def quantile_threshold(values: Iterable[float] | np.ndarray, quantile: float) -> float:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values)
    if array.size == 0:
        raise ValueError("Cannot calculate a threshold from empty values")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    return float(np.quantile(array, quantile))


def pixel_threshold_from_maps(
    maps: torch.Tensor,
    method: str,
    pixel_quantile: float,
    image_quantile: float,
    topk_fraction: float,
) -> float:
    """Estimate a pixel decision threshold using only normal calibration maps."""
    if maps.ndim != 3:
        raise ValueError("maps must have shape [N, H, W]")
    if maps.shape[0] == 0:
        raise ValueError("At least one calibration map is required")

    flattened = maps.detach().float().reshape(maps.shape[0], -1)
    if method == "global_quantile":
        values = flattened.reshape(-1)
        quantile = pixel_quantile
    elif method == "image_max_quantile":
        values = flattened.max(dim=1).values
        quantile = image_quantile
    elif method == "image_topk_quantile":
        if not 0.0 < topk_fraction <= 1.0:
            raise ValueError("topk_fraction must be in (0, 1]")
        count = max(1, int(round(flattened.shape[1] * topk_fraction)))
        values = torch.topk(flattened, count, dim=1).values.mean(dim=1)
        quantile = image_quantile
    else:
        raise ValueError(
            "pixel threshold method must be one of: "
            "global_quantile, image_max_quantile, image_topk_quantile"
        )
    return quantile_threshold(values.cpu().numpy(), quantile)


def adaptive_pixel_threshold_from_maps(
    threshold_fit_maps: torch.Tensor,
    normal_validation_maps: torch.Tensor,
    method: str,
    pixel_quantile: float,
    image_quantiles: Sequence[float],
    topk_fraction: float,
    max_normal_image_positive_rate: float,
    max_normal_pixel_positive_rate: float | None = None,
) -> tuple[float, float, list[dict[str, float]]]:
    """Select a normal-only pixel threshold from candidate image quantiles.

    Candidates are evaluated on held-out normal calibration maps only. The
    first candidate whose normal image-level and optional pixel-level false
    positive rates are within bounds is selected. Candidates should therefore
    be passed from permissive to conservative, e.g. [0.90, 0.925, 0.95].
    """
    if not image_quantiles:
        raise ValueError("image_quantiles must not be empty")
    if not 0.0 <= max_normal_image_positive_rate <= 1.0:
        raise ValueError("max_normal_image_positive_rate must be in [0, 1]")
    if max_normal_pixel_positive_rate is not None and not (
        0.0 <= max_normal_pixel_positive_rate <= 1.0
    ):
        raise ValueError("max_normal_pixel_positive_rate must be in [0, 1]")

    candidates: list[dict[str, float]] = []
    selected: dict[str, float] | None = None
    for quantile in image_quantiles:
        threshold = pixel_threshold_from_maps(
            threshold_fit_maps,
            method=method,
            pixel_quantile=pixel_quantile,
            image_quantile=float(quantile),
            topk_fraction=topk_fraction,
        )
        diagnostics = normal_threshold_diagnostics(
            normal_validation_maps,
            threshold,
        )
        candidate = {
            "pixel_image_quantile": float(quantile),
            "pixel_threshold": float(threshold),
            "normal_pixel_positive_rate": float(
                diagnostics["normal_pixel_positive_rate"]
            ),
            "normal_image_positive_rate": float(
                diagnostics["normal_image_positive_rate"]
            ),
        }
        candidates.append(candidate)
        pixel_ok = (
            max_normal_pixel_positive_rate is None
            or candidate["normal_pixel_positive_rate"]
            <= max_normal_pixel_positive_rate
        )
        image_ok = (
            candidate["normal_image_positive_rate"]
            <= max_normal_image_positive_rate
        )
        if selected is None and image_ok and pixel_ok:
            selected = candidate

    if selected is None:
        selected = candidates[-1]
    return (
        float(selected["pixel_threshold"]),
        float(selected["pixel_image_quantile"]),
        candidates,
    )


def normal_threshold_diagnostics(
    maps: torch.Tensor,
    threshold: float,
) -> dict[str, float]:
    if maps.ndim != 3:
        raise ValueError("maps must have shape [N, H, W]")
    predictions = maps.detach().float() >= float(threshold)
    return {
        "normal_pixel_positive_rate": float(predictions.float().mean().item()),
        "normal_image_positive_rate": float(
            predictions.reshape(predictions.shape[0], -1).any(dim=1).float().mean().item()
        ),
    }
