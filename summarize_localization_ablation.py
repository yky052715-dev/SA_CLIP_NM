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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize SA-CLIP-NM localization ablations"
    )
    parser.add_argument(
        "--root",
        default="outputs/localization_refinement/dev5",
        help="Directory containing F0/F1/F2/F3 experiment directories",
    )
    return parser.parse_args()


def load_completed_case(case_dir: Path) -> dict[str, object] | None:
    completion_path = case_dir / "experiment_complete.json"
    summary_path = case_dir / "metrics_summary.json"
    config_path = case_dir / "resolved_config.json"
    if not completion_path.is_file() or not summary_path.is_file():
        return None
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "complete":
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.is_file()
        else {}
    )
    means = summary["mean"]
    inference = config.get("inference", {})
    row: dict[str, object] = {
        "case": case_dir.name,
        "layer_fusion": inference.get("layer_fusion", "mean"),
        "upsample_mode": inference.get("upsample_mode", "bilinear"),
        "eligible_for_final_selection": (
            inference.get("layer_fusion", "mean") in {"mean", "geometric"}
            and inference.get("upsample_mode", "bilinear") == "bilinear"
        ),
    }
    row.update({key: means.get(key) for key in SUMMARY_KEYS})
    return row


def format_markdown(rows: list[dict[str, object]]) -> str:
    columns = [
        "case",
        "layer_fusion",
        "upsample_mode",
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
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    rows = [
        row
        for case_dir in sorted(path for path in root.glob("F*") if path.is_dir())
        if (row := load_completed_case(case_dir)) is not None
    ]
    if not rows:
        raise RuntimeError(f"No completed localization cases found under {root}")

    fieldnames = list(rows[0].keys())
    with (root / "localization_ablation_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (root / "localization_ablation_summary.md").write_text(
        format_markdown(rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
