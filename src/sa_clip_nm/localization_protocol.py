from __future__ import annotations

from typing import Any


def localization_mode(config: dict[str, Any]) -> str:
    localization = config.get("localization", {})
    if not bool(localization.get("enabled", False)):
        return "global"
    mode = str(localization.get("mode", "global"))
    if mode not in {"global", "tiled"}:
        raise ValueError("localization.mode must be one of: global, tiled")
    return mode


def validate_tiled_config(config: dict[str, Any]) -> dict[str, Any]:
    localization = config.get("localization", {})
    required = [
        "canvas_size",
        "window_size",
        "stride",
        "merge",
        "min_window_weight",
        "local_memory_ratio",
        "local_memory_max_entries",
        "window_batch_size",
        "image_score_merge",
    ]
    missing = [key for key in required if key not in localization]
    if missing:
        raise KeyError(f"Missing tiled localization fields: {missing}")
    canvas_size = int(localization["canvas_size"])
    window_size = int(localization["window_size"])
    stride = int(localization["stride"])
    if window_size != int(config["data"]["image_size"]):
        raise ValueError("localization.window_size must equal data.image_size")
    if not (0 < stride <= window_size <= canvas_size):
        raise ValueError("Expected 0 < stride <= window_size <= canvas_size")
    if str(localization["merge"]) not in {"hann", "mean"}:
        raise ValueError("localization.merge must be one of: hann, mean")
    if str(localization["image_score_merge"]) not in {"max", "mean"}:
        raise ValueError("localization.image_score_merge must be max or mean")
    ratio = float(localization["local_memory_ratio"])
    if not 0.0 < ratio <= 1.0:
        raise ValueError("local_memory_ratio must be in (0, 1]")
    if int(localization["local_memory_max_entries"]) <= 0:
        raise ValueError("local_memory_max_entries must be positive")
    if int(localization["window_batch_size"]) <= 0:
        raise ValueError("window_batch_size must be positive")
    return localization


def effective_local_memory_ratio(
    candidate_count: int,
    requested_ratio: float,
    max_entries: int,
) -> float:
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    if not 0.0 < requested_ratio <= 1.0:
        raise ValueError("requested_ratio must be in (0, 1]")
    if max_entries <= 0:
        raise ValueError("max_entries must be positive")
    requested_entries = max(1, int(round(candidate_count * requested_ratio)))
    retained_entries = min(requested_entries, int(max_entries))
    return min(1.0, retained_entries / candidate_count)
