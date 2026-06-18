from __future__ import annotations

import json
import sys
from pathlib import Path

import summarize_threshold_ablation
from summarize_threshold_ablation import normal_only_selection_key
from sa_clip_nm.config import config_fingerprint


METHODS = [
    "global_quantile",
    "image_max_quantile",
    "image_topk_quantile",
]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def config(method: str, split_seed: int) -> dict:
    return {
        "experiment": {
            "seed": 42,
            "output_dir": "unused",
            "protocol_stage": "development",
        },
        "data": {
            "dataset": "mvtec",
            "categories": ["bottle", "grid"],
            "image_size": 224,
            "resize_mode": "resize",
            "calibration_fraction": 0.2,
        },
        "model": {
            "checkpoint": "clip",
            "active_layers": [3, 6],
            "token_norm": "none",
        },
        "memory": {
            "sampling": "kcenter",
            "ratio": 0.1,
            "projection_dim": 128,
            "max_candidates": 50000,
        },
        "retrieval": {
            "spatial_mode": "none",
            "fixed_lambda": 0.05,
            "query_chunk_size": 256,
            "bank_chunk_size": 4096,
            "lambda_max": 0.1,
            "tau_quantile": 0.8,
        },
        "calibration": {
            "mad_epsilon": 1e-6,
            "threshold_fit_fraction": 0.5,
            "threshold_split_seed": split_seed,
            "pixel_threshold_method": method,
            "pixel_quantile": 0.995,
            "pixel_image_quantile": 0.95,
            "pixel_topk_fraction": 0.01,
            "image_quantile": 0.99,
            "clamp_min_zero": True,
        },
        "inference": {
            "gaussian_sigma": 0.0,
            "image_score": "topk_mean",
            "image_topk_fraction": 0.01,
        },
        "evaluation": {"compute_oracle_f1": True},
    }


def metric_row(category: str, calibrated: float) -> dict:
    return {
        "category": category,
        "pixel_F1_calibrated": calibrated,
        "pixel_IoU_calibrated": calibrated - 0.1,
        "pixel_F1_oracle": 0.7 if category == "bottle" else 0.6,
        "normal_pixel_positive_rate": 0.01 + calibrated / 100,
        "normal_image_positive_rate": 0.1 + calibrated / 10,
        "threshold_fit_images": 10,
        "normal_validation_images": 10,
        "normalization_fit_images": 20,
        "normal_diagnostic_mode": "held_out_from_threshold_fit",
    }


def write_case(
    root: Path,
    method: str,
    split_seed: int,
    method_index: int,
) -> None:
    case_dir = root / f"{method}_split{split_seed}"
    case_config = config(method, split_seed)
    rows = [
        metric_row("bottle", 0.5 + 0.01 * method_index),
        metric_row("grid", 0.4 + 0.001 * (split_seed - 42)),
    ]
    write_json(case_dir / "resolved_config.json", case_config)
    write_json(
        case_dir / "metrics_summary.json",
        {"categories": rows, "mean": {}},
    )
    write_json(
        case_dir / "experiment_complete.json",
        {
            "status": "complete",
            "expected_categories": ["bottle", "grid"],
            "completed_categories": ["bottle", "grid"],
            "expected_count": 2,
            "completed_count": 2,
            "seed": 42,
            "threshold_split_seed": split_seed,
            "pixel_threshold_method": method,
            "active_layers": [3, 6],
            "spatial_mode": "none",
            "config_fingerprint": config_fingerprint(case_config),
        },
    )


def test_development_summary_locks_one_method(
    tmp_path: Path,
    monkeypatch,
) -> None:
    for method_index, method in enumerate(METHODS):
        for split_seed in [42, 43]:
            write_case(
                tmp_path,
                method,
                split_seed,
                method_index,
            )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_threshold_ablation.py",
            "--experiment-dir",
            str(tmp_path),
            "--mode",
            "development",
        ],
    )
    summarize_threshold_ablation.main()
    summary = json.loads(
        (tmp_path / "threshold_summary.json").read_text(encoding="utf-8")
    )
    selection = json.loads(
        (tmp_path / "selected_threshold_method.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["model_seed"] == 42
    assert summary["threshold_split_seeds"] == [42, 43]
    assert len(summary["split_stability"]) == 3
    assert summary["oracle_baseline"]["maximum_case_difference"] == 0.0
    by_method = {
        row["method"]: row for row in summary["split_stability"]
    }
    assert (
        by_method["image_topk_quantile"][
            "pixel_F1_calibrated_mean_across_splits"
        ]
        > by_method["global_quantile"][
            "pixel_F1_calibrated_mean_across_splits"
        ]
    )
    assert selection["selected_method"] == "global_quantile"
    assert selection["uses_anomaly_labels_for_selection"] is False
    assert selection["pixel_f1_used_for_selection"] is False
    assert "Pixel F1" not in selection["selection_rule"]
    assert selection["selected_protocol_fingerprint"]


def test_normal_only_selection_ignores_higher_pixel_f1() -> None:
    rows = [
        {
            "method": "global_quantile",
            "pixel_F1_calibrated_mean_across_splits": 0.20,
            "normal_image_positive_rate_mean_across_splits": 0.02,
            "normal_image_positive_rate_std_across_splits": 0.01,
            "normal_pixel_positive_rate_mean_across_splits": 0.001,
            "normal_pixel_positive_rate_std_across_splits": 0.0001,
        },
        {
            "method": "image_topk_quantile",
            "pixel_F1_calibrated_mean_across_splits": 0.90,
            "normal_image_positive_rate_mean_across_splits": 0.10,
            "normal_image_positive_rate_std_across_splits": 0.02,
            "normal_pixel_positive_rate_mean_across_splits": 0.010,
            "normal_pixel_positive_rate_std_across_splits": 0.001,
        },
    ]
    order = {"global_quantile": 0, "image_topk_quantile": 1}
    selected = min(
        rows,
        key=lambda row: normal_only_selection_key(row, order),
    )
    assert selected["method"] == "global_quantile"
