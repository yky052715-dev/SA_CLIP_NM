from __future__ import annotations

import json

from summarize_highres_clip_ablation import (
    add_selection_checks,
    load_case,
)


def _metrics(**overrides) -> dict:
    values = {
        "pixel_AUROC": 0.98,
        "pixel_F1_calibrated": 0.50,
        "pixel_F1_oracle": 0.60,
        "localization_overseg_anomaly_macro": 1.0,
        "localization_recall_anomaly_macro": 0.75,
        "localization_small_defect_f1_macro": 0.30,
        "localization_small_defect_recall_macro": 0.70,
        "localization_test_normal_image_positive_rate": 0.10,
        "inference_ms_per_image": 5.0,
        "gpu_peak_memory_mb": 500.0,
        "memory_entries": 6000.0,
    }
    values.update(overrides)
    return values


def _write_case(root, name: str, size: int, metrics: dict) -> None:
    case = root / name
    case.mkdir(parents=True)
    side = size // 16
    (case / "experiment_complete.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "input_image_size": size,
                "evaluation_size": 448,
                "patch_grid_side": side,
                "patch_count": side * side,
                "position_interpolation": size != 224,
            }
        ),
        encoding="utf-8",
    )
    (case / "metrics_summary.json").write_text(
        json.dumps({"mean": metrics}), encoding="utf-8"
    )


def test_highres_candidate_selection_uses_h0_as_baseline(tmp_path) -> None:
    _write_case(tmp_path, "H0_clip224", 224, _metrics())
    _write_case(
        tmp_path,
        "H1_clip336",
        336,
        _metrics(
            pixel_AUROC=0.979,
            pixel_F1_calibrated=0.53,
            localization_overseg_anomaly_macro=0.8,
            localization_small_defect_recall_macro=0.71,
            localization_test_normal_image_positive_rate=0.11,
        ),
    )
    rows = [
        load_case(tmp_path / "H0_clip224"),
        load_case(tmp_path / "H1_clip336"),
    ]
    checked = [row for row in rows if row]
    add_selection_checks(checked)
    candidate = next(row for row in checked if row["case"] == "H1_clip336")
    assert candidate["patch_grid_side"] == 21
    assert candidate["position_interpolation"] is True
    assert candidate["meets_dev5_accuracy_criteria"] is True
