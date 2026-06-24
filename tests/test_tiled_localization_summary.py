from __future__ import annotations

import json

from summarize_tiled_localization_ablation import (
    add_selection_diagnostics,
    load_completed_case,
)


def _write_case(root, name: str, metrics: dict, mode: str) -> None:
    case = root / name
    case.mkdir(parents=True)
    (case / "experiment_complete.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    (case / "metrics_summary.json").write_text(
        json.dumps({"mean": metrics}), encoding="utf-8"
    )
    (case / "resolved_config.json").write_text(
        json.dumps(
            {
                "localization": {
                    "enabled": mode == "tiled",
                    "mode": mode,
                },
                "inference": {"layer_fusion": "mean"},
            }
        ),
        encoding="utf-8",
    )


def _metrics(**overrides) -> dict:
    values = {
        "pixel_AUROC": 0.98,
        "pixel_F1_calibrated": 0.50,
        "pixel_IoU_calibrated": 0.40,
        "localization_area_ratio_anomaly_macro": 2.0,
        "localization_overseg_anomaly_macro": 1.0,
        "localization_underseg_anomaly_macro": 0.2,
        "localization_recall_anomaly_macro": 0.80,
        "localization_small_defect_f1_macro": 0.30,
        "localization_small_defect_recall_macro": 0.70,
        "localization_test_normal_pixel_positive_rate": 0.01,
        "localization_test_normal_image_positive_rate": 0.10,
        "inference_ms_per_image": 5.0,
    }
    values.update(overrides)
    return values


def test_selection_diagnostics_apply_mandatory_criteria(tmp_path) -> None:
    _write_case(tmp_path, "L0_global_mean", _metrics(), "global")
    _write_case(
        tmp_path,
        "L2_tiled_mean",
        _metrics(
            pixel_AUROC=0.979,
            pixel_F1_calibrated=0.52,
            localization_overseg_anomaly_macro=0.8,
            localization_small_defect_recall_macro=0.69,
            localization_test_normal_image_positive_rate=0.11,
        ),
        "tiled",
    )
    rows = [
        load_completed_case(tmp_path / "L0_global_mean"),
        load_completed_case(tmp_path / "L2_tiled_mean"),
    ]
    checked = add_selection_diagnostics([row for row in rows if row])
    candidate = next(row for row in checked if row["case"] == "L2_tiled_mean")
    assert candidate["localization_mode"] == "tiled"
    assert candidate["meets_mandatory_criteria"] is True
