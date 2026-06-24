from __future__ import annotations

import copy
from typing import Any

from .protocol import payload_fingerprint


REQUIRED_PARAMETERS = {
    "active_layers",
    "spatial_mode",
    "global_memory_ratio",
    "layer_fusion",
    "layer_fusion_epsilon",
    "upsample_mode",
    "localization_mode",
    "canvas_size",
    "window_size",
    "stride",
    "window_merge",
    "min_window_weight",
    "local_memory_ratio",
    "local_memory_max_entries",
    "image_score_merge",
    "position_rho",
    "position_quantile",
    "position_clamp_min_zero",
    "pixel_threshold_method",
    "pixel_image_quantile",
    "threshold_split_seed",
    "model_seed",
}


def validate_selected_method(selection: dict[str, Any]) -> None:
    if selection.get("status") != "locked":
        raise ValueError("Selected localization method is not locked")
    if selection.get("selected_on") != "mvtec_dev5":
        raise ValueError("Localization method must be selected on mvtec_dev5")
    parameters = selection.get("parameters")
    if not isinstance(parameters, dict):
        raise TypeError("selection.parameters must be a mapping")
    missing = sorted(REQUIRED_PARAMETERS - set(parameters))
    if missing:
        raise KeyError(f"Locked method is missing parameters: {missing}")
    development = {str(value) for value in selection["development_categories"]}
    validation = {str(value) for value in selection["validation_categories"]}
    overlap = sorted(development & validation)
    if overlap:
        raise ValueError(
            f"Development and validation categories overlap: {overlap}"
        )
    if len(validation) != 10:
        raise ValueError("Locked Validation10 split must contain 10 categories")


def apply_locked_method(
    config: dict[str, Any],
    selection: dict[str, Any],
) -> dict[str, Any]:
    """Return a validation config whose method fields come only from the lock."""
    validate_selected_method(selection)
    if config.get("experiment", {}).get("protocol_stage") != "validation":
        raise ValueError("Locked method can only be applied to validation config")
    expected_categories = [
        str(value) for value in selection["validation_categories"]
    ]
    actual_categories = [
        str(value) for value in config.get("data", {}).get("categories", [])
    ]
    if actual_categories != expected_categories:
        raise ValueError(
            "Validation config categories must exactly match locked Validation10"
        )

    resolved = copy.deepcopy(config)
    parameters = selection["parameters"]
    resolved["experiment"]["seed"] = int(parameters["model_seed"])
    resolved["experiment"]["locked_method_id"] = str(
        selection["method_id"]
    )
    resolved["model"]["active_layers"] = [
        int(value) for value in parameters["active_layers"]
    ]
    resolved["memory"]["ratio"] = float(
        parameters["global_memory_ratio"]
    )
    resolved["retrieval"]["spatial_mode"] = str(
        parameters["spatial_mode"]
    )
    resolved["calibration"]["pixel_threshold_method"] = str(
        parameters["pixel_threshold_method"]
    )
    resolved["calibration"]["pixel_image_quantile"] = float(
        parameters["pixel_image_quantile"]
    )
    resolved["calibration"]["threshold_split_seed"] = int(
        parameters["threshold_split_seed"]
    )
    resolved["inference"]["layer_fusion"] = str(
        parameters["layer_fusion"]
    )
    resolved["inference"]["layer_fusion_epsilon"] = float(
        parameters["layer_fusion_epsilon"]
    )
    resolved["inference"]["upsample_mode"] = str(
        parameters["upsample_mode"]
    )
    resolved["localization"] = {
        **resolved.get("localization", {}),
        "enabled": True,
        "mode": str(parameters["localization_mode"]),
        "canvas_size": int(parameters["canvas_size"]),
        "window_size": int(parameters["window_size"]),
        "stride": int(parameters["stride"]),
        "merge": str(parameters["window_merge"]),
        "min_window_weight": float(parameters["min_window_weight"]),
        "local_memory_ratio": float(parameters["local_memory_ratio"]),
        "local_memory_max_entries": int(
            parameters["local_memory_max_entries"]
        ),
        "image_score_merge": str(parameters["image_score_merge"]),
    }
    resolved["position_calibration"] = {
        "enabled": True,
        "rho": float(parameters["position_rho"]),
        "quantile": float(parameters["position_quantile"]),
        "clamp_min_zero": bool(parameters["position_clamp_min_zero"]),
    }
    return resolved


def locked_method_fingerprint(selection: dict[str, Any]) -> str:
    validate_selected_method(selection)
    return payload_fingerprint(selection)
