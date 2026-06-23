from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from .calibration import LayerCalibration, calibrate_scores


@dataclass(frozen=True)
class AnomalyMapOutputs:
    layer_patch_maps: dict[int, torch.Tensor]
    fused_patch_map: torch.Tensor
    anomaly_maps: torch.Tensor


def calibrate_layer_maps(
    raw_scores: Mapping[int, torch.Tensor],
    layer_calibrations: Mapping[int, LayerCalibration],
    clamp_min_zero: bool,
) -> dict[int, torch.Tensor]:
    calibrated: dict[int, torch.Tensor] = {}
    for layer in sorted(raw_scores):
        if layer not in layer_calibrations:
            raise KeyError(f"Missing calibration for layer {layer}")
        parameters = layer_calibrations[layer]
        calibrated[layer] = calibrate_scores(
            raw_scores[layer],
            median=parameters.median,
            mad=parameters.mad,
            clamp_min_zero=clamp_min_zero,
        )
    if not calibrated:
        raise ValueError("At least one layer map is required")
    shapes = {tuple(values.shape) for values in calibrated.values()}
    if len(shapes) != 1:
        raise ValueError("All calibrated layer maps must have the same shape")
    return calibrated


def _normalized_layer_weights(
    layers: list[int],
    layer_weights: Mapping[int | str, float] | None,
) -> torch.Tensor:
    if layer_weights is None:
        raise ValueError("weighted_mean requires layer_weights")
    values = []
    for layer in layers:
        if layer in layer_weights:
            value = layer_weights[layer]
        elif str(layer) in layer_weights:
            value = layer_weights[str(layer)]
        else:
            raise KeyError(f"Missing fusion weight for layer {layer}")
        values.append(float(value))
    weights = torch.tensor(values, dtype=torch.float32)
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Layer weights must be finite and non-negative")
    total = float(weights.sum().item())
    if total <= 0:
        raise ValueError("Layer weights must have a positive sum")
    return weights / total


def fuse_patch_maps(
    layer_maps: Mapping[int, torch.Tensor],
    method: str = "mean",
    epsilon: float = 1.0e-6,
    layer_weights: Mapping[int | str, float] | None = None,
) -> torch.Tensor:
    if epsilon <= 0:
        raise ValueError("Fusion epsilon must be positive")
    layers = sorted(layer_maps)
    if not layers:
        raise ValueError("At least one layer map is required")
    stacked = torch.stack([layer_maps[layer] for layer in layers], dim=0)
    if method == "mean":
        return stacked.mean(dim=0)
    if method == "minimum":
        return stacked.min(dim=0).values
    if method == "geometric":
        if (stacked < 0).any():
            raise ValueError("Geometric fusion requires non-negative layer maps")
        return torch.exp(torch.log(stacked + epsilon).mean(dim=0))
    if method == "weighted_mean":
        weights = _normalized_layer_weights(layers, layer_weights).to(
            device=stacked.device,
            dtype=stacked.dtype,
        )
        view_shape = (len(layers),) + (1,) * (stacked.ndim - 1)
        return (stacked * weights.reshape(view_shape)).sum(dim=0)
    raise ValueError(
        "Layer fusion must be one of: mean, geometric, minimum, weighted_mean"
    )


def reshape_patch_scores(patch_scores: torch.Tensor) -> torch.Tensor:
    if patch_scores.ndim != 2:
        raise ValueError("patch_scores must have shape [batch, patches]")
    side = int(round(math.sqrt(patch_scores.shape[1])))
    if side * side != patch_scores.shape[1]:
        raise ValueError("Patch scores do not form a square map")
    return patch_scores.reshape(patch_scores.shape[0], side, side)


def upsample_anomaly_maps(
    patch_maps: torch.Tensor,
    output_size: int,
    mode: str = "bilinear",
    gaussian_sigma: float = 0.0,
) -> torch.Tensor:
    if patch_maps.ndim != 3:
        raise ValueError("patch_maps must have shape [batch, height, width]")
    if output_size <= 0:
        raise ValueError("output_size must be positive")
    if mode not in {"bilinear", "nearest"}:
        raise ValueError("Upsample mode must be bilinear or nearest")
    kwargs: dict[str, object] = {
        "size": (output_size, output_size),
        "mode": mode,
    }
    if mode == "bilinear":
        kwargs["align_corners"] = False
    maps = F.interpolate(patch_maps[:, None], **kwargs)[:, 0]
    if gaussian_sigma > 0:
        smoothed = [
            gaussian_filter(anomaly_map.numpy(), sigma=gaussian_sigma)
            for anomaly_map in maps.detach().cpu()
        ]
        maps = torch.from_numpy(np.stack(smoothed)).float()
    return maps.cpu()


def build_anomaly_map_outputs(
    raw_scores: Mapping[int, torch.Tensor],
    layer_calibrations: Mapping[int, LayerCalibration],
    output_size: int,
    clamp_min_zero: bool,
    gaussian_sigma: float,
    layer_fusion: str = "mean",
    layer_fusion_epsilon: float = 1.0e-6,
    layer_weights: Mapping[int | str, float] | None = None,
    upsample_mode: str = "bilinear",
) -> AnomalyMapOutputs:
    calibrated = calibrate_layer_maps(
        raw_scores,
        layer_calibrations,
        clamp_min_zero=clamp_min_zero,
    )
    layer_patch_maps = {
        layer: reshape_patch_scores(values) for layer, values in calibrated.items()
    }
    fused_scores = fuse_patch_maps(
        calibrated,
        method=layer_fusion,
        epsilon=layer_fusion_epsilon,
        layer_weights=layer_weights,
    )
    fused_patch_map = reshape_patch_scores(fused_scores)
    anomaly_maps = upsample_anomaly_maps(
        fused_patch_map,
        output_size=output_size,
        mode=upsample_mode,
        gaussian_sigma=gaussian_sigma,
    )
    return AnomalyMapOutputs(
        layer_patch_maps={
            layer: values.cpu() for layer, values in layer_patch_maps.items()
        },
        fused_patch_map=fused_patch_map.cpu(),
        anomaly_maps=anomaly_maps,
    )
