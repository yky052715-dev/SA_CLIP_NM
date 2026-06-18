from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


METRICS = [
    "pixel_F1_calibrated",
    "pixel_IoU_calibrated",
    "pixel_F1_oracle",
    "oracle_gap",
    "normal_pixel_positive_rate",
    "normal_image_positive_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine dev5 selection and locked validation10 results."
        )
    )
    parser.add_argument("--dev5-dir", required=True)
    parser.add_argument("--validation10-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def describe(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.mean(array)), float(np.std(array, ddof=0))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]]) -> str:
    fields = list(rows[0].keys())
    text_fields = {"group", "method", "source"}
    lines = [
        "| " + " | ".join(fields) + " |",
        "|"
        + "|".join("---" if field in text_fields else "---:" for field in fields)
        + "|",
    ]
    for row in rows:
        values = [
            str(row[field])
            if field in text_fields or isinstance(row[field], int)
            else f"{float(row[field]):.6f}"
            for field in fields
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def summary_row(
    group: str,
    method: str,
    categories: int,
    splits: int,
    values: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    return {
        "group": group,
        "method": method,
        "source": source,
        "categories": categories,
        "threshold_splits": splits,
        "pixel_F1_mean": values[
            "pixel_F1_calibrated_mean_across_splits"
        ],
        "pixel_F1_split_std": values[
            "pixel_F1_calibrated_std_across_splits"
        ],
        "pixel_IoU_mean": values[
            "pixel_IoU_calibrated_mean_across_splits"
        ],
        "pixel_IoU_split_std": values[
            "pixel_IoU_calibrated_std_across_splits"
        ],
        "pixel_F1_oracle": values[
            "pixel_F1_oracle_mean_across_splits"
        ],
        "oracle_gap_mean": values["oracle_gap_mean_across_splits"],
        "normal_pixel_rate_mean": values[
            "normal_pixel_positive_rate_mean_across_splits"
        ],
        "normal_pixel_rate_split_std": values[
            "normal_pixel_positive_rate_std_across_splits"
        ],
        "normal_image_rate_mean": values[
            "normal_image_positive_rate_mean_across_splits"
        ],
        "normal_image_rate_split_std": values[
            "normal_image_positive_rate_std_across_splits"
        ],
    }


def main() -> None:
    args = parse_args()
    dev_dir = Path(args.dev5_dir)
    val_dir = Path(args.validation10_dir)
    dev = load_json(dev_dir / "threshold_summary.json")
    validation = load_json(val_dir / "threshold_summary.json")
    selection = load_json(dev_dir / "selected_threshold_method.json")
    validation_selection = load_json(val_dir / "selection_manifest.json")

    if dev["stage"] != "development":
        raise ValueError("dev5 summary is not a development-stage result")
    if validation["stage"] != "validation":
        raise ValueError("validation10 summary is not a validation result")
    if validation_selection != selection:
        raise ValueError("Validation did not preserve the selection manifest")

    selected_method = str(selection["selected_method"])
    if validation["methods"] != [selected_method]:
        raise ValueError("Validation contains unselected threshold methods")
    if dev["model_seed"] != validation["model_seed"]:
        raise ValueError("Model seed differs between dev5 and validation10")
    if dev["threshold_split_seeds"] != validation[
        "threshold_split_seeds"
    ]:
        raise ValueError("Threshold split seeds differ between stages")
    if dev["base_protocol_fingerprint"] != validation[
        "base_protocol_fingerprint"
    ]:
        raise ValueError("Base metric protocols are incompatible")
    if selection["selected_protocol_fingerprint"] != validation[
        "method_protocol_fingerprints"
    ][selected_method]:
        raise ValueError("Selected metric protocol fingerprint mismatch")
    if selection["selected_parameters"] != validation[
        "threshold_parameters"
    ][selected_method]:
        raise ValueError("Locked threshold parameters changed")
    if dev["normal_diagnostic_mode"] != "held_out_from_threshold_fit":
        raise ValueError("Invalid dev5 diagnostic protocol")
    if validation["normal_diagnostic_mode"] != (
        "held_out_from_threshold_fit"
    ):
        raise ValueError("Invalid validation10 diagnostic protocol")

    dev_categories = set(dev["categories"])
    validation_categories = set(validation["categories"])
    if dev_categories & validation_categories:
        raise ValueError("dev5 and validation10 must be disjoint")
    if (len(dev_categories), len(validation_categories)) != (5, 10):
        raise ValueError("Expected category counts 5 and 10")

    dev_stability = {
        row["method"]: row for row in dev["split_stability"]
    }[selected_method]
    val_stability = validation["split_stability"][0]
    rows = [
        summary_row(
            "dev5",
            selected_method,
            5,
            len(dev["threshold_split_seeds"]),
            dev_stability,
            "selection_stage",
        ),
        summary_row(
            "validation10",
            selected_method,
            10,
            len(validation["threshold_split_seeds"]),
            val_stability,
            "locked_validation",
        ),
    ]

    dev_by_split = {
        int(row["threshold_split_seed"]): row
        for row in dev["by_split"]
        if row["method"] == selected_method
    }
    val_by_split = {
        int(row["threshold_split_seed"]): row
        for row in validation["by_split"]
    }
    full15_by_split: list[dict[str, Any]] = []
    for split_seed in dev["threshold_split_seeds"]:
        dev_row = dev_by_split[int(split_seed)]
        val_row = val_by_split[int(split_seed)]
        combined: dict[str, Any] = {
            "method": selected_method,
            "threshold_split_seed": int(split_seed),
        }
        for metric in METRICS:
            combined[metric] = (
                5.0 * float(dev_row[metric])
                + 10.0 * float(val_row[metric])
            ) / 15.0
        full15_by_split.append(combined)

    full_values: dict[str, float] = {}
    for metric in METRICS:
        mean, std = describe(row[metric] for row in full15_by_split)
        full_values[f"{metric}_mean_across_splits"] = mean
        full_values[f"{metric}_std_across_splits"] = std
    rows.append(
        summary_row(
            "full15",
            selected_method,
            15,
            len(full15_by_split),
            full_values,
            "derived_dev5_plus_locked_validation10",
        )
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "threshold_protocol_comparison.csv", rows)
    write_csv(
        output_dir / "threshold_protocol_full15_by_split.csv",
        full15_by_split,
    )
    (output_dir / "threshold_protocol_comparison.md").write_text(
        markdown_table(rows),
        encoding="utf-8",
    )
    with (output_dir / "threshold_protocol_comparison.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "selected_method": selected_method,
                "selection_manifest": selection,
                "base_protocol_fingerprint": dev[
                    "base_protocol_fingerprint"
                ],
                "rows": rows,
                "full15_by_split": full15_by_split,
                "oracle_note": (
                    "Oracle Pixel F1 values are recomputed under the current "
                    "fixed model protocol; the historical 61.99% is not "
                    "assumed to be directly comparable."
                ),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(
        (output_dir / "threshold_protocol_comparison.md").read_text(
            encoding="utf-8"
        )
    )


if __name__ == "__main__":
    main()
