from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .config import save_json


@dataclass
class PositionCalibration:
    rho: float
    quantile: float
    scalar_threshold: float
    threshold_map: torch.Tensor
    normal_pixel_positive_rate: float
    normal_image_positive_rate: float
    threshold_fit_images: int
    normal_validation_images: int

    def summary(self) -> dict[str, object]:
        values = self.threshold_map.detach().float()
        payload = asdict(self)
        payload.pop("threshold_map")
        payload.update(
            {
                "threshold_map_shape": list(values.shape),
                "threshold_map_min": float(values.min().item()),
                "threshold_map_median": float(values.median().item()),
                "threshold_map_max": float(values.max().item()),
            }
        )
        return payload

    def save(self, artifact_dir: str | Path) -> None:
        directory = Path(artifact_dir)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            self.threshold_map.detach().float().cpu(),
            directory / "position_threshold_map.pt",
        )
        save_json(self.summary(), directory / "position_calibration.json")


def fit_position_threshold_map(
    normal_maps: torch.Tensor,
    scalar_threshold: float,
    quantile: float,
    rho: float,
) -> torch.Tensor:
    """Fit a shrunk per-position threshold map using normal maps only."""
    if normal_maps.ndim != 3:
        raise ValueError("normal_maps must have shape [N, H, W]")
    if normal_maps.shape[0] == 0:
        raise ValueError("At least one normal map is required")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    if not torch.isfinite(normal_maps).all():
        raise ValueError("normal_maps contain non-finite values")
    position_threshold = torch.quantile(
        normal_maps.detach().float(),
        float(quantile),
        dim=0,
    )
    scalar = torch.full_like(position_threshold, float(scalar_threshold))
    return (1.0 - float(rho)) * scalar + float(rho) * position_threshold


def adjust_maps_by_position_threshold(
    anomaly_maps: torch.Tensor,
    threshold_map: torch.Tensor,
    scalar_threshold: float,
    clamp_min_zero: bool,
) -> torch.Tensor:
    """Convert a threshold map decision into scalar-threshold score space.

    ``adjusted >= scalar_threshold`` is exactly equivalent to
    ``anomaly_maps >= threshold_map`` before optional zero clamping.
    """
    if anomaly_maps.ndim != 3:
        raise ValueError("anomaly_maps must have shape [N, H, W]")
    if threshold_map.shape != anomaly_maps.shape[-2:]:
        raise ValueError("threshold_map shape does not match anomaly_maps")
    adjusted = (
        anomaly_maps.detach().float()
        - threshold_map.detach().float()[None]
        + float(scalar_threshold)
    )
    if clamp_min_zero:
        adjusted = adjusted.clamp_min(0.0)
    return adjusted


def position_threshold_diagnostics(
    normal_maps: torch.Tensor,
    threshold_map: torch.Tensor,
) -> dict[str, float]:
    if normal_maps.ndim != 3:
        raise ValueError("normal_maps must have shape [N, H, W]")
    if threshold_map.shape != normal_maps.shape[-2:]:
        raise ValueError("threshold_map shape does not match normal_maps")
    predictions = normal_maps.detach().float() >= threshold_map.float()[None]
    return {
        "normal_pixel_positive_rate": float(predictions.float().mean().item()),
        "normal_image_positive_rate": float(
            predictions.reshape(predictions.shape[0], -1)
            .any(dim=1)
            .float()
            .mean()
            .item()
        ),
    }
