from __future__ import annotations

import json
import sys
from pathlib import Path

from summarize_layer_ablation import main


CASES = {
    "block3": [3],
    "block6": [6],
    "block9": [9],
    "block12": [12],
    "fusion_3_6_9_12": [3, 6, 9, 12],
}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_layer_ablation_summary(tmp_path: Path, monkeypatch) -> None:
    categories = ["bottle", "screw"]
    for case_index, (case, layers) in enumerate(CASES.items()):
        case_dir = tmp_path / case
        write_json(
            case_dir / "resolved_config.json",
            {
                "experiment": {"seed": 42},
                "data": {
                    "image_size": 224,
                    "calibration_fraction": 0.2,
                },
                "model": {
                    "checkpoint": "clip",
                    "active_layers": layers,
                    "token_norm": "none",
                },
                "memory": {"sampling": "kcenter", "ratio": 0.1},
                "retrieval": {"spatial_mode": "adaptive", "lambda_max": 0.1},
            },
        )
        rows = []
        for category_index, category in enumerate(categories):
            value = 0.8 + 0.01 * case_index + 0.001 * category_index
            rows.append(
                {
                    "category": category,
                    "image_AUROC": value,
                    "image_F1_calibrated": value,
                    "image_F1_oracle": value,
                    "pixel_AUROC": value,
                    "pixel_F1_calibrated": value,
                    "pixel_F1_oracle": value,
                }
            )
        write_json(
            case_dir / "metrics_summary.json",
            {"categories": rows, "mean": {}},
        )

    monkeypatch.setattr(
        sys,
        "argv",
        ["summarize_layer_ablation.py", "--experiment-dir", str(tmp_path)],
    )
    main()

    summary = json.loads(
        (tmp_path / "layer_ablation_summary.json").read_text(encoding="utf-8")
    )
    assert len(summary["mean_metrics"]) == 5
    assert summary["best_single_layer_counts_by_pixel_AUROC"] == {"block12": 2}
    assert (tmp_path / "layer_ablation_means.md").is_file()
    assert (tmp_path / "layer_ablation_per_category.csv").is_file()
