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
    "memory_entries",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize feature-space multiscale aggregation ablations"
    )
    parser.add_argument(
        "--root",
        default="outputs/localization_refinement/multiscale_dev5",
    )
    return parser.parse_args()


def load_case(case_dir: Path) -> dict[str, object] | None:
    completion_path = case_dir / "experiment_complete.json"
    summary_path = case_dir / "metrics_summary.json"
    if not completion_path.is_file() or not summary_path.is_file():
        return None
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "complete":
        return None
    means = json.loads(summary_path.read_text(encoding="utf-8"))["mean"]
    scales = completion.get("multiscale_scales", [])
    weights = completion.get("multiscale_scale_weights", {})
    row: dict[str, object] = {
        "case": case_dir.name,
        "scales": ",".join(str(value) for value in scales),
        "scale_weights": ",".join(
            f"{float(weights[str(scale)] if str(scale) in weights else weights[scale]):.6g}"
            if isinstance(weights, dict)
            else str(weights)
            for scale in scales
        ),
    }
    row.update({key: means.get(key) for key in METRIC_KEYS})
    return row


def add_selection_checks(rows: list[dict[str, object]]) -> None:
    baseline = next((row for row in rows if row["case"].startswith("M0")), None)
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
        ) - float(baseline["localization_test_normal_image_positive_rate"])
        row.update(
            {
                "pixel_AUROC_drop": auroc_drop,
                "pixel_F1_delta": f1_delta,
                "overseg_delta": overseg_delta,
                "small_recall_delta": small_recall_delta,
                "normal_image_fp_delta": normal_fp_delta,
                "passes_dev5_gate": (
                    auroc_drop <= 0.002
                    and f1_delta >= 0.0
                    and overseg_delta <= -0.15
                    * float(baseline["localization_overseg_anomaly_macro"])
                    and small_recall_delta >= -0.02
                    and normal_fp_delta <= 0.03
                ),
                "m25_recommended_if_m2": (
                    row["case"].startswith("M2")
                    and auroc_drop < 0.01
                ),
            }
        )


def markdown(rows: list[dict[str, object]]) -> str:
    columns = [
        "case",
        "scales",
        "scale_weights",
        "pixel_AUROC",
        "pixel_F1_calibrated",
        "pixel_F1_oracle",
        "localization_overseg_anomaly_macro",
        "localization_recall_anomaly_macro",
        "localization_small_defect_recall_macro",
        "localization_test_normal_image_positive_rate",
        "inference_ms_per_image",
        "memory_entries",
        "pixel_AUROC_drop",
        "pixel_F1_delta",
        "overseg_delta",
        "small_recall_delta",
        "normal_image_fp_delta",
        "passes_dev5_gate",
        "m25_recommended_if_m2",
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
        for case_dir in sorted(path for path in root.glob("M*") if path.is_dir())
        if (row := load_case(case_dir)) is not None
    ]
    if not rows:
        raise RuntimeError(f"No completed multiscale cases found under {root}")
    add_selection_checks(rows)
    with (root / "multiscale_ablation_summary.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (root / "multiscale_ablation_summary.md").write_text(
        markdown(rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
