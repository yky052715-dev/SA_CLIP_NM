from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


def multiscale_key(layer: int, scale: int) -> int:
    if scale <= 0:
        raise ValueError("scale must be positive")
    if scale >= 100:
        raise ValueError("scale must be smaller than 100")
    return int(layer) * 100 + int(scale)


def parse_multiscale_key(key: int) -> tuple[int, int]:
    layer = int(key) // 100
    scale = int(key) % 100
    if scale <= 0:
        raise ValueError(f"Invalid multiscale key: {key}")
    return layer, scale


def multiscale_key_name(key: int) -> str:
    layer, scale = parse_multiscale_key(key)
    return f"L{layer}_r{scale}"


def validate_scales(scales: Sequence[int]) -> list[int]:
    resolved = [int(scale) for scale in scales]
    if not resolved:
        raise ValueError("At least one multiscale scale is required")
    for scale in resolved:
        if scale <= 0:
            raise ValueError("Multiscale scales must be positive")
        if scale % 2 != 1:
            raise ValueError("Multiscale scales must be odd")
    return resolved


def normalized_scale_weights(
    scales: Sequence[int],
    weights: Sequence[float] | Mapping[int | str, float] | None,
) -> dict[int, float]:
    resolved_scales = validate_scales(scales)
    if weights is None:
        raw = [1.0 for _ in resolved_scales]
    elif isinstance(weights, Mapping):
        raw = []
        for scale in resolved_scales:
            if scale in weights:
                raw.append(float(weights[scale]))
            elif str(scale) in weights:
                raw.append(float(weights[str(scale)]))
            else:
                raise KeyError(f"Missing weight for scale {scale}")
    else:
        raw = [float(value) for value in weights]
        if len(raw) != len(resolved_scales):
            raise ValueError("scale_weights length must match scales")
    tensor = torch.tensor(raw, dtype=torch.float32)
    if not torch.isfinite(tensor).all() or (tensor < 0).any():
        raise ValueError("scale_weights must be finite and non-negative")
    total = float(tensor.sum().item())
    if total <= 0:
        raise ValueError("scale_weights must have a positive sum")
    normalized = (tensor / total).tolist()
    return {
        int(scale): float(weight)
        for scale, weight in zip(resolved_scales, normalized, strict=True)
    }


def aggregate_patch_features(
    image_features: torch.Tensor,
    scale: int,
    padding_mode: str = "replicate",
) -> torch.Tensor:
    """Apply feature-space neighborhood averaging to square patch features.

    Args:
        image_features: Tensor with shape [images, patches, dim].
        scale: Odd pooling kernel size. scale=1 returns the input unchanged.
        padding_mode: Padding mode for scale > 1. Use replicate by default to
            avoid zero-padding artifacts on the small 14x14 CLIP grid.
    """
    if image_features.ndim != 3:
        raise ValueError("image_features must have shape [images, patches, dim]")
    scale = int(scale)
    validate_scales([scale])
    if scale == 1:
        return image_features
    image_count, patch_count, dimension = image_features.shape
    side = int(round(math.sqrt(patch_count)))
    if side * side != patch_count:
        raise ValueError("Patch features do not form a square grid")
    if scale > side:
        raise ValueError("scale cannot exceed the patch grid side")
    if padding_mode != "replicate":
        raise ValueError("Only replicate padding is supported for multiscale pooling")

    grid = image_features.reshape(image_count, side, side, dimension).permute(
        0, 3, 1, 2
    )
    pad = scale // 2
    pooled = F.avg_pool2d(
        F.pad(grid, (pad, pad, pad, pad), mode=padding_mode),
        kernel_size=scale,
        stride=1,
        padding=0,
    )
    return pooled.permute(0, 2, 3, 1).reshape(
        image_count,
        patch_count,
        dimension,
    )


def aggregate_features_by_scale(
    features_by_layer: Mapping[int, torch.Tensor],
    scales: Sequence[int],
    padding_mode: str = "replicate",
) -> dict[int, torch.Tensor]:
    resolved_scales = validate_scales(scales)
    outputs: dict[int, torch.Tensor] = {}
    for layer, features in features_by_layer.items():
        for scale in resolved_scales:
            outputs[multiscale_key(int(layer), scale)] = aggregate_patch_features(
                features,
                scale=scale,
                padding_mode=padding_mode,
            )
    return outputs


def fuse_multiscale_raw_scores(
    calibrated_scores: Mapping[int, torch.Tensor],
    scales: Sequence[int],
    scale_weights: Mapping[int, float],
    layer_weights: Mapping[int | str, float] | None = None,
) -> torch.Tensor:
    """Fuse calibrated [batch, patches] scores across scales then layers."""
    resolved_scales = validate_scales(scales)
    grouped: dict[int, list[torch.Tensor]] = {}
    for key, values in calibrated_scores.items():
        layer, scale = parse_multiscale_key(int(key))
        if scale not in resolved_scales:
            continue
        grouped.setdefault(layer, []).append(values * float(scale_weights[scale]))
    if not grouped:
        raise ValueError("No calibrated multiscale scores to fuse")

    layer_maps: dict[int, torch.Tensor] = {
        layer: torch.stack(values, dim=0).sum(dim=0)
        for layer, values in grouped.items()
    }
    layers = sorted(layer_maps)
    stacked = torch.stack([layer_maps[layer] for layer in layers], dim=0)
    if layer_weights is None:
        return stacked.mean(dim=0)
    raw = []
    for layer in layers:
        if layer in layer_weights:
            raw.append(float(layer_weights[layer]))
        elif str(layer) in layer_weights:
            raw.append(float(layer_weights[str(layer)]))
        else:
            raise KeyError(f"Missing layer weight for layer {layer}")
    weights = torch.tensor(raw, dtype=stacked.dtype, device=stacked.device)
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("layer weights must be finite and non-negative")
    total = float(weights.sum().item())
    if total <= 0:
        raise ValueError("layer weights must have a positive sum")
    weights = weights / total
    return (stacked * weights.reshape(-1, 1, 1)).sum(dim=0)
