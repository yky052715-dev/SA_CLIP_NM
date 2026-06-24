from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRIC_KEYS = [
    "pixel_AUROC",
    "pixel_F1_calibrated",
    "pixel_F1_oracle",
    "localization_overseg_anomaly_macro",
    "localization_recall_anomaly_macro",
    "localization_small_defect_f1_macro",
    "localization_small_defect_recall_macro",
    "localization_test_normal_image_positive_rate",
    "inference_ms_per_image",
    "gpu_peak_memory_mb",
    "memory_entries",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize 224/336/448 CLIP localization ablations"
    )
    parser.add_argument(
        "--root",
        default="outputs/localization_refinement/highres_clip_dev5",
    )
    return parser.parse_args()


def load_case(case_dir: Path) -> dict[str, object] | None:
    completion_path = case_dir / "experiment_complete.json"
    summary_path = case_dir / "metrics_summary.json"
    resource_path = case_dir / "resource_summary.json"
    if not completion_path.is_file() or not summary_path.is_file():
        return None
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "complete":
        return None
    means = json.loads(summary_path.read_text(encoding="utf-8"))["mean"]
    resources = (
        json.loads(resource_path.read_text(encoding="utf-8"))
        if resource_path.is_file()
        else {}
    )
    row: dict[str, object] = {
        "case": case_dir.name,
        "input_image_size": int(completion["input_image_size"]),
        "evaluation_size": int(completion["evaluation_size"]),
        "patch_grid_side": int(completion["patch_grid_side"]),
        "patch_count": int(completion["patch_count"]),
        "position_interpolation": bool(completion["position_interpolation"]),
    }
    row.update({key: means.get(key) for key in METRIC_KEYS})
    row["experiment_gpu_peak_memory_mb"] = resources.get(
        "gpu_peak_memory_mb"
    )
    return row


def add_selection_checks(rows: list[dict[str, object]]) -> None:
    baseline = next((row for row in rows if row["case"] == "H0_clip224"), None)
    if baseline is None:
        return
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
                "meets_dev5_accuracy_criteria": (
                    auroc_drop <= 0.002
                    and f1_delta >= 0.0
                    and overseg_delta < 0.0
                    and small_recall_delta >= -0.02
                    and normal_fp_delta <= 0.02
                ),
            }
        )


def markdown(rows: list[dict[str, object]]) -> str:
    columns = [
        "case",
        "input_image_size",
        "patch_grid_side",
        "pixel_AUROC",
        "pixel_F1_calibrated",
        "localization_overseg_anomaly_macro",
        "localization_recall_anomaly_macro",
        "localization_small_defect_recall_macro",
        "localization_test_normal_image_positive_rate",
        "inference_ms_per_image",
        "gpu_peak_memory_mb",
        "memory_entries",
        "meets_dev5_accuracy_criteria",
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
        for case_dir in sorted(path for path in root.glob("H*") if path.is_dir())
        if (row := load_case(case_dir)) is not None
    ]
    if not rows:
        raise RuntimeError(f"No completed high-resolution cases found under {root}")
    add_selection_checks(rows)
    with (root / "highres_clip_ablation_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (root / "highres_clip_ablation_summary.md").write_text(
        markdown(rows), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
