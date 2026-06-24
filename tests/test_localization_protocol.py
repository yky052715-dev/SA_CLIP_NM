from __future__ import annotations

import pytest

from sa_clip_nm.localization_protocol import (
    effective_local_memory_ratio,
    localization_mode,
    validate_tiled_config,
)


def _config() -> dict:
    return {
        "data": {"image_size": 224},
        "localization": {
            "enabled": True,
            "mode": "tiled",
            "canvas_size": 448,
            "window_size": 224,
            "stride": 112,
            "merge": "hann",
            "min_window_weight": 0.05,
            "local_memory_ratio": 0.03,
            "local_memory_max_entries": 50000,
            "window_batch_size": 16,
            "image_score_merge": "max",
        },
    }


def test_disabled_localization_preserves_global_mode() -> None:
    config = _config()
    config["localization"]["enabled"] = False
    assert localization_mode(config) == "global"


def test_tiled_protocol_accepts_locked_p2_geometry() -> None:
    localization = validate_tiled_config(_config())
    assert localization["canvas_size"] == 448
    assert localization["stride"] == 112


def test_tiled_protocol_rejects_non_native_window_size() -> None:
    config = _config()
    config["localization"]["window_size"] = 192
    with pytest.raises(ValueError, match="window_size"):
        validate_tiled_config(config)


def test_local_memory_ratio_is_capped_by_max_entries() -> None:
    ratio = effective_local_memory_ratio(
        candidate_count=2_000_000,
        requested_ratio=0.03,
        max_entries=50_000,
    )
    assert ratio == 0.025
