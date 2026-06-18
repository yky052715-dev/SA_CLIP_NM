from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


METRICS = [
    "image_AUROC",
    "image_F1_calibrated",
    "image_F1_oracle",
    "pixel_AUROC",
    "pixel_F1_calibrated",
    "pixel_F1_oracle",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize SA-CLIP-NM single-layer ablation runs."
    )
    parser.add_argument(
        "--experiment-dir",
        required=True,
        help="Directory containing block3, block6, block9, block12 and fusion runs.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def case_label(config: dict[str, Any]) -> str:
    layers = [int(layer) for layer in config["model"]["active_layers"]]
    if len(layers) == 1:
        return f"block{layers[0]}"
    return "fusion_" + "_".join(str(layer) for layer in layers)


def discover_cases(experiment_dir: Path) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for config_path in sorted(experiment_dir.glob("*/resolved_config.json")):
        case_dir = config_path.parent
        metrics_path = case_dir / "metrics_summary.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Missing metrics summary: {metrics_path}")
        config = load_json(config_path)
        label = case_label(config)
        if label in cases:
            raise ValueError(f"Duplicate layer case: {label}")
        cases[label] = {
            "directory": case_dir,
            "config": config,
            "metrics": load_json(metrics_path),
        }
    if not cases:
        raise FileNotFoundError(
            f"No completed layer runs found under {experiment_dir}"
        )
    return cases


def validate_cases(cases: dict[str, dict[str, Any]]) -> list[str]:
    expected = {"block3", "block6", "block9", "block12", "fusion_3_6_9_12"}
    missing = sorted(expected.difference(cases))
    if missing:
        raise ValueError(f"Missing expected cases: {', '.join(missing)}")

    category_lists = [
        [row["category"] for row in case["metrics"]["categories"]]
        for case in cases.values()
    ]
    reference = category_lists[0]
    for categories in category_lists[1:]:
        if categories != reference:
            raise ValueError("Category order differs between layer runs")

    protocol_fields = [
        ("experiment", "seed"),
        ("data", "image_size"),
        ("data", "calibration_fraction"),
        ("model", "checkpoint"),
        ("model", "token_norm"),
        ("memory", "sampling"),
        ("memory", "ratio"),
        ("retrieval", "spatial_mode"),
        ("retrieval", "lambda_max"),
    ]
    reference_config = next(iter(cases.values()))["config"]
    for label, case in cases.items():
        config = case["config"]
        for section, key in protocol_fields:
            if config[section][key] != reference_config[section][key]:
                raise ValueError(
                    f"Protocol mismatch for {label}: {section}.{key}"
                )
    return reference


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(
    rows: list[dict[str, Any]],
    fieldnames: list[str],
    float_fields: set[str],
) -> str:
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "|"
        + "|".join(
            "---:" if field in float_fields else "---" for field in fieldnames
        )
        + "|",
    ]
    for row in rows:
        values = []
        for field in fieldnames:
            value = row[field]
            values.append(
                f"{float(value):.6f}" if field in float_fields else str(value)
            )
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    cases = discover_cases(experiment_dir)
    categories = validate_cases(cases)
    ordered_labels = ["block3", "block6", "block9", "block12", "fusion_3_6_9_12"]

    mean_rows: list[dict[str, Any]] = []
    per_category_rows: list[dict[str, Any]] = []
    single_layer_wins: Counter[str] = Counter()

    category_metrics = {
        label: {
            row["category"]: row
            for row in cases[label]["metrics"]["categories"]
        }
        for label in ordered_labels
    }

    for label in ordered_labels:
        row: dict[str, Any] = {"case": label}
        for metric in METRICS:
            values = [
                float(category_metrics[label][category][metric])
                for category in categories
            ]
            row[metric] = float(np.nanmean(values))
        mean_rows.append(row)

    for category in categories:
        single_labels = ordered_labels[:4]
        best_pixel_auc = max(
            single_labels,
            key=lambda label: float(
                category_metrics[label][category]["pixel_AUROC"]
            ),
        )
        single_layer_wins[best_pixel_auc] += 1
        fusion_pixel_auc = float(
            category_metrics["fusion_3_6_9_12"][category]["pixel_AUROC"]
        )
        row = {
            "category": category,
            "best_single_layer": best_pixel_auc,
            "best_single_pixel_AUROC": float(
                category_metrics[best_pixel_auc][category]["pixel_AUROC"]
            ),
            "fusion_pixel_AUROC": fusion_pixel_auc,
            "fusion_minus_best_single": fusion_pixel_auc
            - float(category_metrics[best_pixel_auc][category]["pixel_AUROC"]),
        }
        for label in ordered_labels:
            row[f"{label}_pixel_AUROC"] = float(
                category_metrics[label][category]["pixel_AUROC"]
            )
            row[f"{label}_pixel_F1_oracle"] = float(
                category_metrics[label][category]["pixel_F1_oracle"]
            )
        per_category_rows.append(row)

    mean_fields = ["case", *METRICS]
    per_category_fields = list(per_category_rows[0].keys())
    write_csv(experiment_dir / "layer_ablation_means.csv", mean_rows, mean_fields)
    write_csv(
        experiment_dir / "layer_ablation_per_category.csv",
        per_category_rows,
        per_category_fields,
    )

    (experiment_dir / "layer_ablation_means.md").write_text(
        markdown_table(mean_rows, mean_fields, set(METRICS)),
        encoding="utf-8",
    )
    category_float_fields = set(per_category_fields).difference(
        {"category", "best_single_layer"}
    )
    (experiment_dir / "layer_ablation_per_category.md").write_text(
        markdown_table(
            per_category_rows,
            per_category_fields,
            category_float_fields,
        ),
        encoding="utf-8",
    )

    summary = {
        "categories": categories,
        "case_order": ordered_labels,
        "best_single_layer_counts_by_pixel_AUROC": dict(single_layer_wins),
        "mean_metrics": mean_rows,
        "per_category": per_category_rows,
    }
    with (experiment_dir / "layer_ablation_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Summarized {len(ordered_labels)} cases in {experiment_dir}")
    print((experiment_dir / "layer_ablation_means.md").read_text(encoding="utf-8"))
    print("Best single-layer counts by pixel AUROC:")
    for label in ordered_labels[:4]:
        print(f"  {label}: {single_layer_wins[label]}")


if __name__ == "__main__":
    main()
