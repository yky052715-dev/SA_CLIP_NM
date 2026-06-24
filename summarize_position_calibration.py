from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


KEYS = [
    "position_calibration_rho",
    "pixel_AUROC",
    "pixel_F1_calibrated",
    "pixel_F1_oracle",
    "localization_overseg_anomaly_macro",
    "localization_recall_anomaly_macro",
    "localization_small_defect_f1_macro",
    "localization_small_defect_recall_macro",
    "localization_test_normal_image_positive_rate",
    "normal_image_positive_rate",
    "position_threshold_min",
    "position_threshold_median",
    "position_threshold_max",
    "inference_ms_per_image",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize position-calibration ablations"
    )
    parser.add_argument(
        "--root",
        default="outputs/localization_refinement/p3_bottle",
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
    row: dict[str, object] = {"case": case_dir.name}
    row.update({key: means.get(key) for key in KEYS})
    return row


def markdown(rows: list[dict[str, object]]) -> str:
    columns = [
        "case",
        "position_calibration_rho",
        "pixel_AUROC",
        "pixel_F1_calibrated",
        "localization_overseg_anomaly_macro",
        "localization_recall_anomaly_macro",
        "localization_test_normal_image_positive_rate",
        "inference_ms_per_image",
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
        for case_dir in sorted(path for path in root.glob("P3*") if path.is_dir())
        if (row := load_case(case_dir)) is not None
    ]
    if not rows:
        raise RuntimeError(f"No completed position cases found under {root}")
    with (root / "position_calibration_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (root / "position_calibration_summary.md").write_text(
        markdown(rows), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
