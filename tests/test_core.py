from __future__ import annotations

import numpy as np
import torch

from sa_clip_nm.backbone import make_patch_coordinates
from sa_clip_nm.calibration import (
    adaptive_pixel_threshold_from_maps,
    calibrate_scores,
    compute_layer_tau,
    estimate_spatial_reliability,
    image_score_from_map,
    normal_threshold_diagnostics,
    pixel_threshold_from_maps,
    reliability_to_weight,
    robust_location_scale,
)
from sa_clip_nm.memory import LayerMemoryBank, build_layer_memory
from sa_clip_nm.metrics import evaluate_binary_scores
from sa_clip_nm.retrieval import exact_spatial_nearest_neighbor


def test_patch_coordinates_form_normalized_grid() -> None:
    coordinates = make_patch_coordinates(4)
    expected = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    )
    assert torch.allclose(coordinates, expected)


def test_memory_bank_random_sampling_keeps_metadata_aligned() -> None:
    features = torch.randn(3, 4, 8)
    features = torch.nn.functional.normalize(features, dim=-1)
    coordinates = make_patch_coordinates(4)
    bank = build_layer_memory(
        image_features=features,
        patch_coordinates=coordinates,
        ratio=0.5,
        method="random",
        seed=42,
        projection_dim=4,
        max_candidates=100,
    )
    assert bank.features.shape == (6, 8)
    assert bank.coordinates.shape == (6, 2)
    assert bank.source_ids.shape == (6,)


def test_spatial_weight_changes_ambiguous_nearest_neighbor() -> None:
    bank = LayerMemoryBank(
        features=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        coordinates=torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
        source_ids=torch.tensor([0, 1]),
    )
    query = torch.tensor([[1.0, 0.0]])
    coordinate = torch.tensor([[1.0, 1.0]])
    _, no_spatial_index = exact_spatial_nearest_neighbor(
        query,
        coordinate,
        bank,
        spatial_weight=0.0,
        device="cpu",
    )
    _, spatial_index = exact_spatial_nearest_neighbor(
        query,
        coordinate,
        bank,
        spatial_weight=0.1,
        device="cpu",
    )
    assert int(no_spatial_index.item()) == 0
    assert int(spatial_index.item()) == 1


def test_reliability_is_higher_for_stable_position_specific_features() -> None:
    base = torch.eye(4).repeat(4, 1, 1)
    stable = torch.nn.functional.normalize(base + 0.001 * torch.randn_like(base), dim=-1)
    repeated = torch.ones(4, 4, 4)
    repeated = torch.nn.functional.normalize(
        repeated + 0.001 * torch.randn_like(repeated), dim=-1
    )
    stable_value = estimate_spatial_reliability(stable)["reliability"]
    repeated_value = estimate_spatial_reliability(repeated)["reliability"]
    assert stable_value > repeated_value


def test_layer_tau_and_weight_are_deterministic() -> None:
    reliabilities = {
        "a": {3: 1.0, 6: 2.0},
        "b": {3: 3.0, 6: 4.0},
    }
    tau = compute_layer_tau(reliabilities, quantile=0.5)
    assert tau == {3: 2.0, 6: 3.0}
    weight = reliability_to_weight(2.0, tau=2.0, lambda_max=0.1)
    assert np.isclose(weight, 0.05)


def test_mad_calibration_and_image_score() -> None:
    values = torch.tensor([1.0, 2.0, 3.0, 100.0])
    median, mad = robust_location_scale(values, epsilon=1e-6)
    calibrated = calibrate_scores(values, median, mad, clamp_min_zero=True)
    assert median == 2.0
    assert mad == 1.0
    assert calibrated.tolist() == [0.0, 0.0, 1.0, 98.0]

    maps = torch.tensor([[[0.0, 1.0], [2.0, 3.0]]])
    score = image_score_from_map(maps, method="topk_mean", topk_fraction=0.5)
    assert torch.allclose(score, torch.tensor([2.5]))


def test_metrics_keep_calibrated_and_oracle_f1_separate() -> None:
    labels = np.array([0, 0, 1, 1], dtype=np.uint8)
    scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    metrics = evaluate_binary_scores(
        labels,
        scores,
        calibrated_threshold=0.85,
        compute_oracle=True,
    )
    assert np.isclose(metrics.auroc, 1.0)
    assert metrics.oracle_f1 >= metrics.calibrated_f1
    assert metrics.calibrated_iou < 1.0


def test_normal_only_pixel_threshold_strategies() -> None:
    maps = torch.tensor(
        [
            [[0.0, 1.0], [2.0, 3.0]],
            [[0.0, 2.0], [4.0, 6.0]],
        ]
    )
    global_threshold = pixel_threshold_from_maps(
        maps,
        method="global_quantile",
        pixel_quantile=0.5,
        image_quantile=0.5,
        topk_fraction=0.5,
    )
    max_threshold = pixel_threshold_from_maps(
        maps,
        method="image_max_quantile",
        pixel_quantile=0.5,
        image_quantile=0.5,
        topk_fraction=0.5,
    )
    topk_threshold = pixel_threshold_from_maps(
        maps,
        method="image_topk_quantile",
        pixel_quantile=0.5,
        image_quantile=0.5,
        topk_fraction=0.5,
    )
    assert np.isclose(global_threshold, 2.0)
    assert np.isclose(max_threshold, 4.5)
    assert np.isclose(topk_threshold, 3.75)

    diagnostics = normal_threshold_diagnostics(maps, threshold=4.0)
    assert np.isclose(diagnostics["normal_pixel_positive_rate"], 0.25)
    assert np.isclose(diagnostics["normal_image_positive_rate"], 0.5)


def test_adaptive_pixel_threshold_selects_first_normal_safe_quantile() -> None:
    threshold_fit_maps = torch.tensor(
        [
            [[1.0]],
            [[2.0]],
            [[3.0]],
            [[4.0]],
        ]
    )
    normal_validation_maps = torch.tensor([[[2.0]], [[5.0]]])

    threshold, selected_quantile, candidates = adaptive_pixel_threshold_from_maps(
        threshold_fit_maps=threshold_fit_maps,
        normal_validation_maps=normal_validation_maps,
        method="image_max_quantile",
        pixel_quantile=0.5,
        image_quantiles=[0.0, 0.5, 1.0],
        topk_fraction=1.0,
        max_normal_image_positive_rate=0.5,
    )

    assert np.isclose(threshold, 2.5)
    assert np.isclose(selected_quantile, 0.5)
    assert [item["pixel_image_quantile"] for item in candidates] == [
        0.0,
        0.5,
        1.0,
    ]
    assert np.isclose(candidates[0]["normal_image_positive_rate"], 1.0)
    assert np.isclose(candidates[1]["normal_image_positive_rate"], 0.5)


def test_adaptive_pixel_threshold_falls_back_to_most_conservative_candidate() -> None:
    threshold_fit_maps = torch.tensor(
        [
            [[1.0]],
            [[2.0]],
            [[3.0]],
            [[4.0]],
        ]
    )
    normal_validation_maps = torch.tensor([[[2.0]], [[5.0]]])

    threshold, selected_quantile, _ = adaptive_pixel_threshold_from_maps(
        threshold_fit_maps=threshold_fit_maps,
        normal_validation_maps=normal_validation_maps,
        method="image_max_quantile",
        pixel_quantile=0.5,
        image_quantiles=[0.0, 0.5],
        topk_fraction=1.0,
        max_normal_image_positive_rate=0.0,
    )

    assert np.isclose(threshold, 2.5)
    assert np.isclose(selected_quantile, 0.5)


def test_binary_metrics_can_use_postprocessed_calibrated_predictions() -> None:
    labels = np.array([0, 1, 1], dtype=np.uint8)
    scores = np.array([0.9, 0.8, 0.1], dtype=np.float64)
    metrics = evaluate_binary_scores(
        labels,
        scores,
        calibrated_threshold=0.5,
        compute_oracle=True,
        calibrated_predictions=np.array([0, 1, 0], dtype=np.uint8),
    )

    assert np.isclose(metrics.calibrated_f1, 2.0 / 3.0)
    assert np.isclose(metrics.calibrated_iou, 0.5)
    assert metrics.oracle_f1 >= metrics.calibrated_f1
