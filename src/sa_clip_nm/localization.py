from __future__ import annotations

from collections.abc import Sequence

import torch


WindowBox = tuple[int, int, int, int]


def _axis_starts(canvas_size: int, window_size: int, stride: int) -> list[int]:
    if canvas_size <= 0 or window_size <= 0 or stride <= 0:
        raise ValueError("canvas_size, window_size, and stride must be positive")
    if window_size > canvas_size:
        raise ValueError("window_size cannot exceed canvas_size")
    last = canvas_size - window_size
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def generate_window_boxes(
    canvas_size: int,
    window_size: int,
    stride: int,
) -> list[WindowBox]:
    """Generate fully covering square windows in row-major order."""
    starts = _axis_starts(canvas_size, window_size, stride)
    boxes = [
        (left, top, left + window_size, top + window_size)
        for top in starts
        for left in starts
    ]
    validate_window_coverage(boxes, canvas_size)
    return boxes


def validate_window_coverage(
    boxes: Sequence[WindowBox],
    canvas_size: int,
) -> torch.Tensor:
    """Validate boxes and return an integer per-pixel coverage map."""
    if canvas_size <= 0:
        raise ValueError("canvas_size must be positive")
    if not boxes:
        raise ValueError("At least one window box is required")
    coverage = torch.zeros((canvas_size, canvas_size), dtype=torch.int32)
    for box in boxes:
        if len(box) != 4:
            raise ValueError("Each window box must contain four coordinates")
        left, top, right, bottom = (int(value) for value in box)
        if not (0 <= left < right <= canvas_size):
            raise ValueError(f"Invalid horizontal window bounds: {box}")
        if not (0 <= top < bottom <= canvas_size):
            raise ValueError(f"Invalid vertical window bounds: {box}")
        coverage[top:bottom, left:right] += 1
    if torch.any(coverage == 0):
        raise ValueError("Window boxes do not cover every canvas pixel")
    return coverage


def extract_image_windows(
    images: torch.Tensor,
    boxes: Sequence[WindowBox],
) -> torch.Tensor:
    """Extract windows from ``[B, C, H, W]`` images as ``[B, N, C, h, w]``."""
    if images.ndim != 4:
        raise ValueError("images must have shape [B, C, H, W]")
    if images.shape[-2] != images.shape[-1]:
        raise ValueError("Only square canvases are supported")
    validate_window_coverage(boxes, int(images.shape[-1]))
    shapes = {
        (int(bottom - top), int(right - left))
        for left, top, right, bottom in boxes
    }
    if len(shapes) != 1:
        raise ValueError("All windows must have the same spatial shape")
    return torch.stack(
        [images[:, :, top:bottom, left:right] for left, top, right, bottom in boxes],
        dim=1,
    )


def build_window_weight(
    window_size: int,
    merge: str = "hann",
    min_weight: float = 0.05,
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Build a finite two-dimensional window merge weight."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if not 0.0 < min_weight <= 1.0:
        raise ValueError("min_weight must be in (0, 1]")
    if merge == "mean":
        weight = torch.ones((window_size, window_size), dtype=dtype, device=device)
    elif merge == "hann":
        axis = torch.hann_window(
            window_size,
            periodic=False,
            dtype=dtype,
            device=device,
        )
        weight = torch.outer(axis, axis).clamp_min(float(min_weight))
    else:
        raise ValueError("merge must be one of: mean, hann")
    if not torch.isfinite(weight).all() or torch.any(weight <= 0):
        raise RuntimeError("Window merge weight must be finite and positive")
    return weight


def merge_window_maps(
    window_maps: torch.Tensor,
    boxes: Sequence[WindowBox],
    canvas_size: int,
    merge: str = "hann",
    min_weight: float = 0.05,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Merge ``[B, N, h, w]`` window maps onto a square canvas."""
    if window_maps.ndim != 4:
        raise ValueError("window_maps must have shape [B, N, h, w]")
    if window_maps.shape[1] != len(boxes):
        raise ValueError("window_maps count does not match boxes")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    validate_window_coverage(boxes, canvas_size)
    window_height, window_width = window_maps.shape[-2:]
    if window_height != window_width:
        raise ValueError("Only square window maps are supported")
    for left, top, right, bottom in boxes:
        if right - left != window_width or bottom - top != window_height:
            raise ValueError("Window map size does not match its box")

    weight = build_window_weight(
        window_width,
        merge=merge,
        min_weight=min_weight,
        dtype=window_maps.dtype,
        device=window_maps.device,
    )
    output = torch.zeros(
        (window_maps.shape[0], canvas_size, canvas_size),
        dtype=window_maps.dtype,
        device=window_maps.device,
    )
    denominator = torch.zeros_like(output)
    for index, (left, top, right, bottom) in enumerate(boxes):
        output[:, top:bottom, left:right] += window_maps[:, index] * weight
        denominator[:, top:bottom, left:right] += weight
    merged = output / denominator.clamp_min(float(epsilon))
    if not torch.isfinite(merged).all():
        raise RuntimeError("Merged window maps contain non-finite values")
    return merged


def combine_image_scores(
    global_scores: torch.Tensor,
    local_scores: torch.Tensor,
    method: str = "max",
) -> torch.Tensor:
    if global_scores.shape != local_scores.shape:
        raise ValueError("global_scores and local_scores must have the same shape")
    if method == "max":
        return torch.maximum(global_scores, local_scores)
    if method == "mean":
        return 0.5 * (global_scores + local_scores)
    raise ValueError("image score merge must be one of: max, mean")
