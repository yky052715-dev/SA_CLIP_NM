from __future__ import annotations

import copy
from typing import Any

from .protocol import payload_fingerprint


REQUIRED_PARAMETERS = {
    "checkpoint",
    "input_image_size",
    "evaluation_size",
    "active_layers",
    "token_norm",
    "spatial_mode",
    "global_memory_ratio",
    "layer_fusion",
    "layer_fusion_epsilon",
    "upsample_mode",
    "gaussian_sigma",
    "image_score",
    "image_topk_fraction",
    "threshold_fit_fraction",
    "pixel_threshold_method",
    "pixel_quantile",
    "pixel_image_quantile",
    "pixel_topk_fraction",
    "adaptive_pixel_image_quantiles",
    "adaptive_max_normal_image_positive_rate",
    "adaptive_max_normal_pixel_positive_rate",
    "adaptive_max_normal_positive_area_p95_fraction",
    "adaptive_max_normal_positive_area_max_fraction",
    "postprocess_min_component_area_fraction",
    "threshold_split_seed",
    "model_seed",
}


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def validate_selected_highres_method(selection: dict[str, Any]) -> None:
    if selection.get("status") != "locked":
        raise ValueError("Selected highres localization method is not locked")
    if selection.get("selected_on") != "mvtec_dev5":
        raise ValueError("Highres localization method must be selected on mvtec_dev5")
    parameters = selection.get("parameters")
    if not isinstance(parameters, dict):
        raise TypeError("selection.parameters must be a mapping")
    missing = sorted(REQUIRED_PARAMETERS - set(parameters))
    if missing:
        raise KeyError(f"Locked highres method is missing parameters: {missing}")
    if int(parameters["input_image_size"]) <= 0:
        raise ValueError("input_image_size must be positive")
    if int(parameters["evaluation_size"]) <= 0:
        raise ValueError("evaluation_size must be positive")
    if not 0.0 < float(parameters["global_memory_ratio"]) <= 1.0:
        raise ValueError("global_memory_ratio must be in (0, 1]")
    if float(parameters["postprocess_min_component_area_fraction"]) < 0.0:
        raise ValueError("postprocess_min_component_area_fraction must be non-negative")
    if not parameters["adaptive_pixel_image_quantiles"]:
        raise ValueError("adaptive_pixel_image_quantiles must not be empty")


def apply_locked_highres_method(
    config: dict[str, Any],
    selection: dict[str, Any],
) -> dict[str, Any]:
    """Return a config whose highres localization method fields are locked."""
    validate_selected_highres_method(selection)
    stage = str(config.get("experiment", {}).get("protocol_stage", ""))
    if stage not in {"validation", "external_validation"}:
        raise ValueError(
            "Locked highres method can only be applied to validation or "
            "external_validation configs"
        )

    resolved = copy.deepcopy(config)
    parameters = selection["parameters"]
    resolved["experiment"]["seed"] = int(parameters["model_seed"])
    resolved["experiment"]["locked_method_id"] = str(selection["method_id"])
    resolved["data"]["image_size"] = int(parameters["input_image_size"])
    resolved.setdefault("high_resolution", {})["evaluation_size"] = int(
        parameters["evaluation_size"]
    )
    resolved["model"]["checkpoint"] = str(parameters["checkpoint"])
    resolved["model"]["active_layers"] = [
        int(value) for value in parameters["active_layers"]
    ]
    resolved["model"]["token_norm"] = str(parameters["token_norm"])
    resolved["memory"]["ratio"] = float(parameters["global_memory_ratio"])
    resolved["retrieval"]["spatial_mode"] = str(parameters["spatial_mode"])

    calibration = resolved["calibration"]
    calibration["threshold_fit_fraction"] = float(
        parameters["threshold_fit_fraction"]
    )
    calibration["threshold_split_seed"] = int(
        parameters["threshold_split_seed"]
    )
    calibration["pixel_threshold_method"] = str(
        parameters["pixel_threshold_method"]
    )
    calibration["pixel_quantile"] = float(parameters["pixel_quantile"])
    calibration["pixel_image_quantile"] = float(
        parameters["pixel_image_quantile"]
    )
    calibration["pixel_topk_fraction"] = float(
        parameters["pixel_topk_fraction"]
    )
    calibration["adaptive_pixel_image_quantiles"] = [
        float(value) for value in parameters["adaptive_pixel_image_quantiles"]
    ]
    calibration["adaptive_max_normal_image_positive_rate"] = _optional_float(
        parameters["adaptive_max_normal_image_positive_rate"]
    )
    calibration["adaptive_max_normal_pixel_positive_rate"] = _optional_float(
        parameters["adaptive_max_normal_pixel_positive_rate"]
    )
    calibration["adaptive_max_normal_positive_area_p95_fraction"] = _optional_float(
        parameters["adaptive_max_normal_positive_area_p95_fraction"]
    )
    calibration["adaptive_max_normal_positive_area_max_fraction"] = _optional_float(
        parameters["adaptive_max_normal_positive_area_max_fraction"]
    )

    inference = resolved["inference"]
    inference["gaussian_sigma"] = float(parameters["gaussian_sigma"])
    inference["image_score"] = str(parameters["image_score"])
    inference["image_topk_fraction"] = float(
        parameters["image_topk_fraction"]
    )
    inference["layer_fusion"] = str(parameters["layer_fusion"])
    inference["layer_fusion_epsilon"] = float(
        parameters["layer_fusion_epsilon"]
    )
    if "layer_weights" in parameters:
        inference["layer_weights"] = {
            int(layer): float(weight)
            for layer, weight in parameters["layer_weights"].items()
        }
    inference["upsample_mode"] = str(parameters["upsample_mode"])

    resolved.setdefault("postprocess", {})[
        "min_component_area_fraction"
    ] = float(parameters["postprocess_min_component_area_fraction"])
    resolved["localization"] = {
        **resolved.get("localization", {}),
        "enabled": False,
        "mode": "global",
    }
    return resolved


def locked_highres_method_fingerprint(selection: dict[str, Any]) -> str:
    validate_selected_highres_method(selection)
    return payload_fingerprint(selection["parameters"])
