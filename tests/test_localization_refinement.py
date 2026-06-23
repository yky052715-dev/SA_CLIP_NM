from __future__ import annotations

import numpy as np
import pytest
import torch

from sa_clip_nm.calibration import LayerCalibration
from sa_clip_nm.localization_metrics import (
    evaluate_localization_image,
    summarize_localization_rows,
)
from sa_clip_nm.map_refinement import (
    build_anomaly_map_outputs,
    fuse_patch_maps,
    reshape_patch_scores,
    upsample_anomaly_maps,
)
from sa_clip_nm.pipeline import _fused_maps


def _calibration() -> LayerCalibration:
    return LayerCalibration(
        median=0.0,
        mad=1.0,
        reliability=1.0,
        spatial_weight=0.0,
    )


def test_mean_refinement_matches_legacy_fused_maps() -> None:
    generator = torch.Generator().manual_seed(42)
    raw_scores = {
        3: torch.rand((2, 196), generator=generator),
        6: torch.rand((2, 196), generator=generator),
    }
    calibrations = {3: _calibration(), 6: _calibration()}
    legacy = _fused_maps(
        raw_scores,
        calibrations,
        image_size=224,
        clamp_min_zero=True,
        gaussian_sigma=0.0,
    )
    refined = build_anomaly_map_outputs(
        raw_scores,
        calibrations,
        output_size=224,
        clamp_min_zero=True,
        gaussian_sigma=0.0,
        layer_fusion="mean",
        upsample_mode="bilinear",
    ).anomaly_maps
    assert torch.allclose(legacy, refined, atol=1.0e-6, rtol=0.0)


def test_layer_fusion_variants_are_correct() -> None:
    maps = {
        3: torch.tensor([[1.0, 4.0]]),
        6: torch.tensor([[9.0, 16.0]]),
    }
    geometric = fuse_patch_maps(maps, method="geometric", epsilon=1.0e-12)
    minimum = fuse_patch_maps(maps, method="minimum")
    weighted = fuse_patch_maps(
        maps,
        method="weighted_mean",
        layer_weights={3: 0.25, 6: 0.75},
    )
    assert torch.allclose(geometric, torch.tensor([[3.0, 8.0]]), atol=1.0e-6)
    assert torch.equal(minimum, torch.tensor([[1.0, 4.0]]))
    assert torch.allclose(weighted, torch.tensor([[7.0, 13.0]]))


def test_geometric_fusion_rejects_negative_maps() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        fuse_patch_maps(
            {3: torch.tensor([[-1.0]]), 6: torch.tensor([[1.0]])},
            method="geometric",
        )


def test_patch_reshape_and_nearest_upsample() -> None:
    patch_map = reshape_patch_scores(torch.arange(4).reshape(1, 4).float())
    upsampled = upsample_anomaly_maps(
        patch_map,
        output_size=4,
        mode="nearest",
    )
    assert patch_map.shape == (1, 2, 2)
    assert upsampled.shape == (1, 4, 4)
    assert torch.equal(upsampled[0, :2, :2], torch.zeros((2, 2)))


def test_localization_metrics_measure_oversegmentation() -> None:
    ground_truth = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    anomaly_map = np.array([[1.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    metrics = evaluate_localization_image(
        ground_truth,
        anomaly_map,
        threshold=0.5,
        small_max_fraction=0.5,
        medium_max_fraction=0.75,
    )
    assert metrics["area_ratio"] == 1.5
    assert metrics["overseg"] == 0.5
    assert metrics["underseg"] == 0.0
    assert np.isclose(float(metrics["precision"]), 2.0 / 3.0)
    assert metrics["recall"] == 1.0
    assert metrics["connected_component_count"] == 1
    assert metrics["defect_size_group"] == "small"


def test_normal_image_uses_positive_fraction_not_area_ratio() -> None:
    metrics = evaluate_localization_image(
        np.zeros((2, 2), dtype=np.uint8),
        np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        threshold=0.5,
    )
    assert metrics["area_ratio"] is None
    assert metrics["prediction_positive_fraction"] == 0.25
    assert metrics["defect_size_group"] == "normal"


def test_localization_summary_is_macro_averaged() -> None:
    rows = [
        {
            "label": 1,
            "area_ratio": 2.0,
            "overseg": 1.0,
            "underseg": 0.0,
            "precision": 0.5,
            "recall": 1.0,
            "f1": 2.0 / 3.0,
            "iou": 0.5,
            "connected_component_count": 1,
            "prediction_positive_fraction": 0.2,
            "prediction_area": 20,
            "defect_size_group": "small",
        },
        {
            "label": 0,
            "area_ratio": None,
            "overseg": None,
            "underseg": None,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "iou": 0.0,
            "connected_component_count": 0,
            "prediction_positive_fraction": 0.0,
            "prediction_area": 0,
            "defect_size_group": "normal",
        },
    ]
    summary = summarize_localization_rows(rows)
    assert summary["area_ratio_anomaly_macro"] == 2.0
    assert summary["small_defect_images"] == 1
    assert summary["test_normal_pixel_positive_rate"] == 0.0
    assert summary["test_normal_image_positive_rate"] == 0.0
