from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sa_clip_nm.locked_highres_localization import (
    apply_locked_highres_method,
    locked_highres_method_fingerprint,
    validate_selected_highres_method,
)


ROOT = Path(__file__).resolve().parents[1]


def _selection() -> dict:
    return json.loads(
        (ROOT / "configs/selected_highres_localization_method.json").read_text(
            encoding="utf-8"
        )
    )


def _config() -> dict:
    return {
        "experiment": {"protocol_stage": "external_validation", "seed": 999},
        "data": {"image_size": 224, "categories": ["dummy"]},
        "high_resolution": {"evaluation_size": 224},
        "model": {
            "checkpoint": "wrong/checkpoint",
            "active_layers": [12],
            "token_norm": "layernorm",
        },
        "memory": {"ratio": 0.9},
        "retrieval": {"spatial_mode": "adaptive"},
        "calibration": {
            "threshold_fit_fraction": 0.5,
            "threshold_split_seed": 999,
            "pixel_threshold_method": "global_quantile",
            "pixel_quantile": 0.1,
            "pixel_image_quantile": 0.1,
            "pixel_topk_fraction": 0.5,
        },
        "inference": {
            "gaussian_sigma": 3.0,
            "image_score": "max",
            "image_topk_fraction": 0.5,
            "layer_fusion": "minimum",
            "layer_fusion_epsilon": 1.0,
            "upsample_mode": "nearest",
        },
        "localization": {"enabled": True, "mode": "tiled"},
        "postprocess": {"min_component_area_fraction": 0.0},
    }


def test_locked_highres_method_overrides_selectable_fields() -> None:
    selection = _selection()
    resolved = apply_locked_highres_method(_config(), selection)
    parameters = selection["parameters"]
    assert resolved["experiment"]["seed"] == parameters["model_seed"]
    assert resolved["data"]["image_size"] == 336
    assert resolved["high_resolution"]["evaluation_size"] == 448
    assert resolved["model"]["checkpoint"] == parameters["checkpoint"]
    assert resolved["model"]["active_layers"] == [3, 6]
    assert resolved["memory"]["ratio"] == pytest.approx(0.0444444444)
    assert resolved["calibration"]["pixel_threshold_method"] == (
        "adaptive_area_image_max_quantile"
    )
    assert resolved["calibration"]["adaptive_pixel_image_quantiles"] == [
        0.875,
        0.9,
        0.925,
        0.95,
    ]
    assert resolved["calibration"]["adaptive_max_normal_image_positive_rate"] is None
    assert resolved["postprocess"]["min_component_area_fraction"] == pytest.approx(
        0.0005
    )
    assert resolved["localization"]["enabled"] is False
    assert resolved["localization"]["mode"] == "global"


def test_locked_highres_rejects_unlocked_selection() -> None:
    selection = copy.deepcopy(_selection())
    selection["status"] = "draft"
    with pytest.raises(ValueError, match="not locked"):
        validate_selected_highres_method(selection)


def test_locked_highres_requires_validation_stage() -> None:
    config = _config()
    config["experiment"]["protocol_stage"] = "development"
    with pytest.raises(ValueError, match="validation"):
        apply_locked_highres_method(config, _selection())


def test_locked_highres_fingerprint_is_stable() -> None:
    selection = _selection()
    assert locked_highres_method_fingerprint(selection) == locked_highres_method_fingerprint(
        copy.deepcopy(selection)
    )


def test_locked_highres_fingerprint_ignores_documentation_metadata() -> None:
    selection = _selection()
    changed = copy.deepcopy(selection)
    changed["selection_note"] = "updated note"
    changed["known_dev5_risks"] = ["new documentation-only risk note"]
    changed["dev5_three_seed_reference"]["pixel_AUROC_delta_mean"] = 123.0

    assert locked_highres_method_fingerprint(selection) == locked_highres_method_fingerprint(
        changed
    )


def test_locked_highres_fingerprint_changes_when_parameters_change() -> None:
    selection = _selection()
    changed = copy.deepcopy(selection)
    changed["parameters"]["pixel_image_quantile"] = 0.9

    assert locked_highres_method_fingerprint(selection) != locked_highres_method_fingerprint(
        changed
    )
