import math

import pytest

from summarize_highres_seed_stability import (
    METRICS,
    aggregate_categories,
    aggregate_pairs,
    category_pair_rows,
    finite_stats,
    pair_runs,
)


def _category(name, f1, overseg, recall=0.75, normal_fp=0.10):
    return {
        "category": name,
        "pixel_F1_calibrated": f1,
        "localization_overseg_anomaly_macro": overseg,
        "localization_recall_anomaly_macro": recall,
        "localization_test_normal_image_positive_rate": normal_fp,
    }


def _run(seed, method, *, f1, overseg, normal_fp, categories):
    values = {
        "pixel_AUROC": 0.984 if method == "H0_clip224" else 0.988,
        "pixel_F1_calibrated": f1,
        "pixel_F1_oracle": f1 + 0.10,
        "localization_overseg_anomaly_macro": overseg,
        "localization_recall_anomaly_macro": 0.75,
        "localization_small_defect_f1_macro": 0.30,
        "localization_small_defect_recall_macro": 0.80,
        "localization_test_normal_image_positive_rate": normal_fp,
        "inference_ms_per_image": 16.0 if method == "H0_clip224" else 30.0,
        "gpu_peak_memory_mb": 460.0 if method == "H0_clip224" else 619.0,
        "memory_entries": 7888.0,
    }
    return {
        "seed": seed,
        "method": method,
        "input_image_size": 224 if method == "H0_clip224" else 336,
        "patch_grid_side": 14 if method == "H0_clip224" else 21,
        "memory_ratio": 0.10 if method == "H0_clip224" else 0.0444444444,
        "categories": categories,
        **values,
    }


def _paired_runs():
    runs = []
    for index, seed in enumerate((42, 3407, 2024)):
        runs.extend(
            [
                _run(
                    seed,
                    "H0_clip224",
                    f1=0.47 + index * 0.01,
                    overseg=4.30 - index * 0.10,
                    normal_fp=0.15,
                    categories=[
                        _category("bottle", 0.70 + index * 0.01, 1.0),
                        _category("screw", 0.38 + index * 0.01, 0.55),
                    ],
                ),
                _run(
                    seed,
                    "H1_clip336_matched",
                    f1=0.53 + index * 0.01,
                    overseg=2.55 - index * 0.10,
                    normal_fp=0.16,
                    categories=[
                        _category("bottle", 0.76 + index * 0.01, 0.30),
                        _category("screw", 0.49 + index * 0.01, 0.46),
                    ],
                ),
            ]
        )
    return runs


def test_finite_stats_ignores_non_finite_values():
    stats = finite_stats([1.0, float("nan"), 3.0, float("inf")])

    assert stats["count"] == 2
    assert stats["mean"] == pytest.approx(2.0)
    assert stats["min"] == pytest.approx(1.0)
    assert stats["max"] == pytest.approx(3.0)
    assert math.isfinite(stats["std"])


def test_pair_runs_requires_a_complete_h0_h1_pair_for_every_seed():
    runs = _paired_runs()
    incomplete = [row for row in runs if not (
        row["seed"] == 3407 and row["method"] == "H1_clip336_matched"
    )]

    with pytest.raises(RuntimeError, match="3407"):
        pair_runs(incomplete)


def test_paired_aggregate_preserves_delta_direction_and_passes():
    pairs = pair_runs(_paired_runs())
    aggregate = aggregate_pairs(pairs)

    assert len(pairs) == 3
    assert all(pair["meets_accuracy_criteria"] for pair in pairs)
    assert aggregate["all_seeds_meet_accuracy_criteria"] is True
    assert aggregate["deltas"]["pixel_F1_calibrated"]["mean"] == pytest.approx(0.06)
    assert aggregate["deltas"]["localization_overseg_anomaly_macro"]["mean"] == pytest.approx(-1.75)


def test_category_summary_requires_consistent_improvement_across_seeds():
    pairs = pair_runs(_paired_runs())
    category_summary = aggregate_categories(category_pair_rows(_paired_runs()))
    by_category = {row["category"]: row for row in category_summary}

    assert by_category["bottle"]["f1_improved_all_seeds"] is True
    assert by_category["bottle"]["overseg_improved_all_seeds"] is True
    assert by_category["screw"]["f1_improved_seed_count"] == 3
    assert by_category["screw"]["overseg_improved_seed_count"] == 3
    assert set(METRICS) >= {
        "pixel_AUROC",
        "pixel_F1_calibrated",
        "localization_overseg_anomaly_macro",
    }
