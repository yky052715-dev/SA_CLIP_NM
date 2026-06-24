from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sa_clip_nm.locked_localization import (
    apply_locked_method,
    locked_method_fingerprint,
    validate_selected_method,
)


ROOT = Path(__file__).resolve().parents[1]


def _selection() -> dict:
    return json.loads(
        (ROOT / "configs/selected_localization_method.json").read_text(
            encoding="utf-8"
        )
    )


def _config() -> dict:
    return {
        "experiment": {"protocol_stage": "validation", "seed": 999},
        "data": {
            "categories": list(_selection()["validation_categories"])
        },
        "model": {"active_layers": [12]},
        "memory": {"ratio": 0.9},
        "retrieval": {"spatial_mode": "adaptive"},
        "calibration": {
            "pixel_threshold_method": "global_quantile",
            "pixel_image_quantile": 0.1,
            "threshold_split_seed": 999,
        },
        "inference": {
            "layer_fusion": "minimum",
            "layer_fusion_epsilon": 1.0,
            "upsample_mode": "nearest",
        },
        "localization": {},
        "position_calibration": {},
    }


def test_locked_method_overrides_every_selectable_method_field() -> None:
    selection = _selection()
    resolved = apply_locked_method(_config(), selection)
    parameters = selection["parameters"]
    assert resolved["experiment"]["seed"] == 42
    assert resolved["model"]["active_layers"] == [3, 6]
    assert resolved["memory"]["ratio"] == 0.1
    assert resolved["retrieval"]["spatial_mode"] == "none"
    assert resolved["inference"]["layer_fusion"] == "mean"
    assert resolved["localization"]["local_memory_ratio"] == 0.01
    assert resolved["position_calibration"]["rho"] == 0.1
    assert resolved["position_calibration"]["quantile"] == 0.95
    assert resolved["calibration"]["pixel_threshold_method"] == (
        parameters["pixel_threshold_method"]
    )


def test_unlocked_selection_is_rejected() -> None:
    selection = copy.deepcopy(_selection())
    selection["status"] = "draft"
    with pytest.raises(ValueError, match="not locked"):
        validate_selected_method(selection)


def test_validation_categories_must_match_lock_exactly() -> None:
    config = _config()
    config["data"]["categories"] = config["data"]["categories"][:-1]
    with pytest.raises(ValueError, match="exactly match"):
        apply_locked_method(config, _selection())


def test_development_and_validation_categories_must_be_disjoint() -> None:
    selection = copy.deepcopy(_selection())
    selection["validation_categories"][0] = selection[
        "development_categories"
    ][0]
    with pytest.raises(ValueError, match="overlap"):
        validate_selected_method(selection)


def test_locked_fingerprint_is_stable() -> None:
    selection = _selection()
    assert locked_method_fingerprint(selection) == locked_method_fingerprint(
        copy.deepcopy(selection)
    )
