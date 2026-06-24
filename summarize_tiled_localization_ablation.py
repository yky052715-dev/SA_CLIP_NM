from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SUMMARY_KEYS = [
    "pixel_AUROC",
    "pixel_F1_calibrated",
    "pixel_IoU_calibrated",
    "localization_area_ratio_anomaly_macro",
    "localization_overseg_anomaly_macro",
    "localization_underseg_anomaly_macro",
    "localization_recall_anomaly_macro",
    "localization_small_defect_f1_macro",
    "localization_small_defect_recall_macro",
    "localization_test_normal_pixel_positive_rate",
    "localization_test_normal_image_positive_rate",
    "inference_ms_per_image",
    "gpu_peak_memory_mb",
    "local_memory_entries",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the P2 L0-L3 tiled localization ablation"
    )
    parser.add_argument(
        "--root",
        default="outputs/localization_refinement/dev5_p2",
    )
    return parser.parse_args()


def load_completed_case(case_dir: Path) -> dict[str, object] | None:
    completion_path = case_dir / "experiment_complete.json"
    summary_path = case_dir / "metrics_summary.json"
    config_path = case_dir / "resolved_config.json"
    if not all(path.is_file() for path in [completion_path, summary_path, config_path]):
        return None
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "complete":
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    localization = config.get("localization", {})
    mode = (
        str(localization.get("mode", "global"))
        if bool(localization.get("enabled", False))
        else "global"
    )
    row: dict[str, object] = {
        "case": case_dir.name,
        "localization_mode": mode,
        "layer_fusion": config.get("inference", {}).get(
            "layer_fusion", "mean"
        ),
    }
    means = summary["mean"]
    row.update({key: means.get(key) for key in SUMMARY_KEYS})
    return row


def add_selection_diagnostics(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    baseline = next((row for row in rows if row["case"] == "L0_global_mean"), None)
    if baseline is None:
        return rows
    for row in rows:
        auroc_drop = float(baseline["pixel_AUROC"]) - float(row["pixel_AUROC"])
        f1_delta = float(row["pixel_F1_calibrated"]) - float(
            baseline["pixel_F1_calibrated"]
        )
        overseg_delta = float(
            row["localization_overseg_anomaly_macro"]
        ) - float(baseline["localization_overseg_anomaly_macro"])
        small_recall_delta = float(
            row["localization_small_defect_recall_macro"]
        ) - float(baseline["localization_small_defect_recall_macro"])
        normal_fp_delta = float(
            row["localization_test_normal_image_positive_rate"]
        ) - float(
            baseline["localization_test_normal_image_positive_rate"]
        )
        row.update(
            {
                "pixel_AUROC_drop": auroc_drop,
                "pixel_F1_delta": f1_delta,
                "overseg_delta": overseg_delta,
                "small_recall_delta": small_recall_delta,
                "normal_image_fp_delta": normal_fp_delta,
                "meets_mandatory_criteria": (
                    auroc_drop <= 0.002
                    and f1_delta >= 0.0
                    and overseg_delta < 0.0
                    and small_recall_delta >= -0.02
                    and normal_fp_delta <= 0.02
                ),
            }
        )
    return rows


def format_markdown(rows: list[dict[str, object]]) -> str:
    columns = [
        "case",
        "localization_mode",
        "layer_fusion",
        "pixel_AUROC",
        "pixel_F1_calibrated",
        "localization_overseg_anomaly_macro",
        "localization_recall_anomaly_macro",
        "localization_small_defect_recall_macro",
        "localization_test_normal_image_positive_rate",
        "inference_ms_per_image",
        "meets_mandatory_criteria",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(columns) - 1)) + "|",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    rows = [
        row
        for case_dir in sorted(path for path in root.glob("L*") if path.is_dir())
        if (row := load_completed_case(case_dir)) is not None
    ]
    if not rows:
        raise RuntimeError(f"No completed P2 localization cases found under {root}")
    rows = add_selection_diagnostics(rows)
    fieldnames = list(rows[0].keys())
    with (root / "tiled_localization_ablation_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (root / "tiled_localization_ablation_summary.md").write_text(
        format_markdown(rows), encoding="utf-8"
    )
    eligible = [
        str(row["case"])
        for row in rows
        if bool(row.get("meets_mandatory_criteria", False))
    ]
    (root / "tiled_localization_selection_diagnostics.json").write_text(
        json.dumps(
            {
                "baseline": "L0_global_mean",
                "eligible_cases": eligible,
                "selection_locked": False,
                "note": "Lock exactly one method only after reviewing Dev5 metrics.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
