from __future__ import annotations

import json

import pytest
import torch

from sa_clip_nm.multiscale import (
    aggregate_features_by_scale,
    aggregate_patch_features,
    fuse_multiscale_raw_scores,
    multiscale_key,
    normalized_scale_weights,
    parse_multiscale_key,
)


def test_multiscale_key_roundtrip() -> None:
    key = multiscale_key(6, 3)
    assert parse_multiscale_key(key) == (6, 3)


def test_aggregate_scale_one_preserves_features() -> None:
    features = torch.randn(2, 196, 4)
    assert torch.equal(aggregate_patch_features(features, scale=1), features)


def test_aggregate_patch_features_uses_replicate_padding() -> None:
    features = torch.arange(9, dtype=torch.float32).reshape(1, 9, 1)
    pooled = aggregate_patch_features(features, scale=3)
    assert pooled.shape == features.shape
    # Top-left 3x3 window after replicate padding:
    # 0 0 1
    # 0 0 1
    # 3 3 4
    assert pooled[0, 0, 0].item() == pytest.approx(12.0 / 9.0)
    assert pooled[0, 4, 0].item() == pytest.approx(4.0)


def test_aggregate_features_by_scale_uses_composite_keys() -> None:
    features = {3: torch.randn(1, 196, 8), 6: torch.randn(1, 196, 8)}
    aggregated = aggregate_features_by_scale(features, scales=[1, 3])
    assert set(aggregated) == {
        multiscale_key(3, 1),
        multiscale_key(3, 3),
        multiscale_key(6, 1),
        multiscale_key(6, 3),
    }


def test_normalized_scale_weights_accepts_list_and_mapping() -> None:
    assert normalized_scale_weights([1, 3], [0.7, 0.3]) == {
        1: pytest.approx(0.7),
        3: pytest.approx(0.3),
    }
    assert normalized_scale_weights([1, 3], {"1": 7, "3": 3}) == {
        1: pytest.approx(0.7),
        3: pytest.approx(0.3),
    }


def test_fuse_multiscale_raw_scores_fuses_scales_then_layers() -> None:
    scores = {
        multiscale_key(3, 1): torch.ones(2, 4) * 1,
        multiscale_key(3, 3): torch.ones(2, 4) * 3,
        multiscale_key(6, 1): torch.ones(2, 4) * 10,
        multiscale_key(6, 3): torch.ones(2, 4) * 30,
    }
    fused = fuse_multiscale_raw_scores(
        scores,
        scales=[1, 3],
        scale_weights={1: 0.75, 3: 0.25},
        layer_weights={3: 0.5, 6: 0.5},
    )
    assert fused.shape == (2, 4)
    assert torch.allclose(fused, torch.ones(2, 4) * 8.25)


def test_summarize_multiscale_marks_m25_trigger(tmp_path) -> None:
    from summarize_multiscale_ablation import add_selection_checks, load_case

    root = tmp_path
    for name, auroc in [("M0_r1", 0.99), ("M2_r5", 0.985)]:
        case = root / name
        case.mkdir()
        (case / "experiment_complete.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "multiscale_scales": [5] if name.startswith("M2") else [1],
                    "multiscale_scale_weights": {"5": 1.0}
                    if name.startswith("M2")
                    else {"1": 1.0},
                }
            ),
            encoding="utf-8",
        )
        (case / "metrics_summary.json").write_text(
            json.dumps(
                {
                    "mean": {
                        "pixel_AUROC": auroc,
                        "pixel_F1_calibrated": 0.5,
                        "pixel_F1_oracle": 0.6,
                        "localization_overseg_anomaly_macro": 1.0,
                        "localization_recall_anomaly_macro": 0.8,
                        "localization_small_defect_f1_macro": 0.3,
                        "localization_small_defect_recall_macro": 0.7,
                        "localization_test_normal_image_positive_rate": 0.1,
                        "inference_ms_per_image": 10.0,
                        "memory_entries": 100,
                    }
                }
            ),
            encoding="utf-8",
        )

    rows = [load_case(root / "M0_r1"), load_case(root / "M2_r5")]
    assert all(row is not None for row in rows)
    add_selection_checks(rows)  # type: ignore[arg-type]
    assert rows[1]["m25_recommended_if_m2"] is True  # type: ignore[index]

