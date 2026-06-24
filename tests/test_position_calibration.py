from __future__ import annotations

import torch

from sa_clip_nm.position_calibration import (
    adjust_maps_by_position_threshold,
    fit_position_threshold_map,
    position_threshold_diagnostics,
)


def test_rho_zero_reproduces_scalar_threshold_and_original_maps() -> None:
    normal_maps = torch.rand((5, 4, 4))
    threshold = 3.0
    threshold_map = fit_position_threshold_map(
        normal_maps,
        scalar_threshold=threshold,
        quantile=0.95,
        rho=0.0,
    )
    assert torch.equal(threshold_map, torch.full((4, 4), threshold))
    anomaly_maps = torch.rand((2, 4, 4)) * 5.0
    adjusted = adjust_maps_by_position_threshold(
        anomaly_maps,
        threshold_map,
        scalar_threshold=threshold,
        clamp_min_zero=False,
    )
    assert torch.allclose(adjusted, anomaly_maps, atol=1.0e-7)


def test_shrinkage_interpolates_scalar_and_position_thresholds() -> None:
    normal_maps = torch.tensor(
        [
            [[0.0, 2.0], [4.0, 6.0]],
            [[2.0, 4.0], [6.0, 8.0]],
        ]
    )
    position = torch.quantile(normal_maps, 0.5, dim=0)
    threshold_map = fit_position_threshold_map(
        normal_maps,
        scalar_threshold=10.0,
        quantile=0.5,
        rho=0.25,
    )
    assert torch.allclose(threshold_map, 0.75 * 10.0 + 0.25 * position)


def test_adjusted_scalar_decision_matches_threshold_map_decision() -> None:
    anomaly_maps = torch.tensor(
        [[[1.0, 3.0], [5.0, 7.0]], [[8.0, 6.0], [4.0, 2.0]]]
    )
    threshold_map = torch.tensor([[2.0, 3.0], [4.0, 8.0]])
    scalar_threshold = 5.0
    adjusted = adjust_maps_by_position_threshold(
        anomaly_maps,
        threshold_map,
        scalar_threshold,
        clamp_min_zero=True,
    )
    assert torch.equal(
        adjusted >= scalar_threshold,
        anomaly_maps >= threshold_map[None],
    )


def test_position_diagnostics_use_only_normal_map_decisions() -> None:
    normal_maps = torch.tensor(
        [[[0.0, 2.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]
    )
    diagnostics = position_threshold_diagnostics(
        normal_maps,
        torch.ones((2, 2)),
    )
    assert diagnostics["normal_pixel_positive_rate"] == 0.125
    assert diagnostics["normal_image_positive_rate"] == 0.5
