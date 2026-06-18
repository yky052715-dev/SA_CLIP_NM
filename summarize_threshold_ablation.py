from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from sa_clip_nm.config import config_fingerprint
from sa_clip_nm.protocol import (
    metric_protocol,
    metric_protocol_fingerprint,
    threshold_parameters,
)


DEVELOPMENT_METHODS = [
    "global_quantile",
    "image_max_quantile",
    "image_topk_quantile",
]

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
        description="Summarize threshold-split stability experiments."
    )
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument(
        "--mode",
        choices=["development", "validation"],
        required=True,
    )
    parser.add_argument("--selection-manifest")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def verify_completion(
    case_dir: Path,
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    marker_path = case_dir / "experiment_complete.json"
    if not marker_path.is_file():
        raise ValueError(f"Incomplete experiment: {marker_path}")
    marker = load_json(marker_path)
    categories = [str(value) for value in config["data"]["categories"]]
    completed = [str(row["category"]) for row in metrics["categories"]]
    expected_fingerprint = config_fingerprint(config)
    required = {
        "status": "complete",
        "expected_categories": categories,
        "completed_categories": categories,
        "expected_count": len(categories),
        "completed_count": len(categories),
        "seed": int(config["experiment"]["seed"]),
        "threshold_split_seed": int(
            config["calibration"]["threshold_split_seed"]
        ),
        "pixel_threshold_method": str(
            config["calibration"]["pixel_threshold_method"]
        ),
        "active_layers": [
            int(value) for value in config["model"]["active_layers"]
        ],
        "spatial_mode": str(config["retrieval"]["spatial_mode"]),
        "config_fingerprint": expected_fingerprint,
    }
    for key, expected in required.items():
        if marker.get(key) != expected:
            raise ValueError(
                f"Completion marker mismatch for {key}: {marker_path}"
            )
    if completed != categories:
        raise ValueError(f"Metrics category mismatch: {case_dir}")


def discover_cases(
    experiment_dir: Path,
) -> dict[str, dict[int, dict[str, Any]]]:
    cases: dict[str, dict[int, dict[str, Any]]] = {}
    for config_path in sorted(experiment_dir.glob("*/resolved_config.json")):
        case_dir = config_path.parent
        metrics_path = case_dir / "metrics_summary.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Missing metrics summary: {metrics_path}")
        config = load_json(config_path)
        metrics = load_json(metrics_path)
        verify_completion(case_dir, config, metrics)
        method = str(config["calibration"]["pixel_threshold_method"])
        split_seed = int(config["calibration"]["threshold_split_seed"])
        if split_seed in cases.setdefault(method, {}):
            raise ValueError(
                f"Duplicate case for {method}, split seed {split_seed}"
            )
        cases[method][split_seed] = {
            "directory": case_dir,
            "config": config,
            "metrics": metrics,
        }
    if not cases:
        raise ValueError(f"No completed cases found in {experiment_dir}")
    return cases


def validate_protocol(
    cases: dict[str, dict[int, dict[str, Any]]],
    expected_methods: list[str],
) -> tuple[list[str], list[int], dict[str, Any], str]:
    if sorted(cases) != sorted(expected_methods):
        raise ValueError(
            f"Expected methods {expected_methods}, found {sorted(cases)}"
        )
    split_sets = {method: set(by_split) for method, by_split in cases.items()}
    reference_splits = split_sets[expected_methods[0]]
    for method in expected_methods[1:]:
        if split_sets[method] != reference_splits:
            raise ValueError(f"Threshold split seeds differ for {method}")

    first_split = min(reference_splits)
    reference = cases[expected_methods[0]][first_split]["config"]
    categories = [str(value) for value in reference["data"]["categories"]]
    base_protocol = metric_protocol(
        reference,
        include_threshold_method=False,
    )
    base_fingerprint = metric_protocol_fingerprint(
        reference,
        include_threshold_method=False,
    )
    model_seed = int(reference["experiment"]["seed"])

    for method in expected_methods:
        reference_parameters = threshold_parameters(
            cases[method][first_split]["config"]
        )
        for split_seed, case in cases[method].items():
            config = case["config"]
            if str(config["experiment"].get("protocol_stage", "")) != (
                "development"
                if len(expected_methods) > 1
                else "validation"
            ):
                raise ValueError("Experiment protocol stage mismatch")
            if int(config["experiment"]["seed"]) != model_seed:
                raise ValueError("Model seed changed across threshold splits")
            if threshold_parameters(config) != reference_parameters:
                raise ValueError(
                    f"Threshold parameters changed for {method}"
                )
            if metric_protocol_fingerprint(
                config,
                include_threshold_method=False,
            ) != base_fingerprint:
                raise ValueError(
                    f"Metric protocol mismatch for {method}, "
                    f"split {split_seed}"
                )
            rows = case["metrics"]["categories"]
            if [str(row["category"]) for row in rows] != categories:
                raise ValueError(
                    f"Category mismatch for {method}, split {split_seed}"
                )
            for row in rows:
                if row.get("normal_diagnostic_mode") != (
                    "held_out_from_threshold_fit"
                ):
                    raise ValueError(
                        "Normal diagnostics are not held out from "
                        "threshold fitting"
                    )
                if int(row.get("normal_validation_images", 0)) < 1:
                    raise ValueError("No held-out normal validation images")
    return categories, sorted(reference_splits), base_protocol, base_fingerprint


def verify_shared_oracle(
    indexed: dict[str, dict[int, dict[str, dict[str, Any]]]],
    methods: list[str],
    split_seeds: list[int],
    categories: list[str],
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    per_category: dict[str, float] = {}
    maximum_difference = 0.0
    for category in categories:
        values = [
            float(indexed[method][split_seed][category]["pixel_F1_oracle"])
            for method in methods
            for split_seed in split_seeds
        ]
        difference = max(values) - min(values)
        maximum_difference = max(maximum_difference, difference)
        if difference > tolerance:
            raise ValueError(
                f"Oracle Pixel F1 changed across threshold-only cases for "
                f"{category}: range={difference}"
            )
        per_category[category] = float(np.mean(values))
    return {
        "pixel_F1_oracle_mean": float(
            np.mean(list(per_category.values()))
        ),
        "pixel_F1_oracle_by_category": per_category,
        "maximum_case_difference": maximum_difference,
        "comparison_note": (
            "Recomputed under the current fixed model protocol. Do not "
            "reuse the historical 61.99% value unless this result matches."
        ),
    }


def describe(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=0)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "range": float(np.max(array) - np.min(array)),
    }


def enriched_row(row: dict[str, Any]) -> dict[str, float]:
    calibrated = float(row["pixel_F1_calibrated"])
    oracle = float(row["pixel_F1_oracle"])
    return {
        "pixel_F1_calibrated": calibrated,
        "pixel_IoU_calibrated": float(row["pixel_IoU_calibrated"]),
        "pixel_F1_oracle": oracle,
        "oracle_gap": oracle - calibrated,
        "normal_pixel_positive_rate": float(
            row["normal_pixel_positive_rate"]
        ),
        "normal_image_positive_rate": float(
            row["normal_image_positive_rate"]
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(
    rows: list[dict[str, Any]],
    text_fields: set[str],
) -> str:
    fields = list(rows[0].keys())
    lines = [
        "| " + " | ".join(fields) + " |",
        "|"
        + "|".join("---" if field in text_fields else "---:" for field in fields)
        + "|",
    ]
    for row in rows:
        values = []
        for field in fields:
            value = row[field]
            if field in text_fields:
                values.append(str(value))
            elif isinstance(value, int):
                values.append(str(value))
            else:
                values.append(f"{float(value):.6f}")
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def load_selection(path: str | None) -> dict[str, Any]:
    if not path:
        raise ValueError("validation mode requires --selection-manifest")
    return load_json(Path(path))


def normal_only_selection_key(
    row: dict[str, Any],
    method_order: dict[str, int],
) -> tuple[float, float, float, float, int]:
    image_mean = float(
        row["normal_image_positive_rate_mean_across_splits"]
    )
    image_std = float(
        row["normal_image_positive_rate_std_across_splits"]
    )
    pixel_mean = float(
        row["normal_pixel_positive_rate_mean_across_splits"]
    )
    pixel_std = float(
        row["normal_pixel_positive_rate_std_across_splits"]
    )
    return (
        image_mean + image_std,
        pixel_mean + pixel_std,
        image_std,
        pixel_std,
        method_order[str(row["method"])],
    )


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    cases = discover_cases(experiment_dir)

    selection: dict[str, Any] | None = None
    if args.mode == "development":
        methods = DEVELOPMENT_METHODS
    else:
        selection = load_selection(args.selection_manifest)
        methods = [str(selection["selected_method"])]

    categories, split_seeds, base_protocol, base_fingerprint = (
        validate_protocol(cases, methods)
    )
    indexed = {
        method: {
            split_seed: {
                row["category"]: row
                for row in cases[method][split_seed]["metrics"]["categories"]
            }
            for split_seed in split_seeds
        }
        for method in methods
    }
    oracle_baseline = verify_shared_oracle(
        indexed,
        methods,
        split_seeds,
        categories,
    )

    stability_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    for method in methods:
        observations = [
            enriched_row(indexed[method][split_seed][category])
            for split_seed in split_seeds
            for category in categories
        ]
        stability_row: dict[str, Any] = {
            "method": method,
            "threshold_splits": len(split_seeds),
            "categories": len(categories),
        }
        for metric in METRICS:
            for statistic, value in describe(
                item[metric] for item in observations
            ).items():
                stability_row[f"{metric}_{statistic}"] = value
        stability_rows.append(stability_row)

        for split_seed in split_seeds:
            values = [
                enriched_row(indexed[method][split_seed][category])
                for category in categories
            ]
            split_row: dict[str, Any] = {
                "method": method,
                "threshold_split_seed": split_seed,
            }
            for metric in METRICS:
                split_row[metric] = float(
                    np.mean([item[metric] for item in values])
                )
            split_rows.append(split_row)

        for category in categories:
            values = [
                enriched_row(indexed[method][split_seed][category])
                for split_seed in split_seeds
            ]
            category_row: dict[str, Any] = {
                "category": category,
                "method": method,
            }
            for metric in METRICS:
                statistics = describe(item[metric] for item in values)
                category_row[f"{metric}_mean"] = statistics["mean"]
                category_row[f"{metric}_std"] = statistics["std"]
                category_row[f"{metric}_max"] = statistics["max"]
            category_rows.append(category_row)

    split_stability_rows: list[dict[str, Any]] = []
    for method in methods:
        method_splits = [
            row for row in split_rows if row["method"] == method
        ]
        row: dict[str, Any] = {"method": method}
        for metric in METRICS:
            statistics = describe(item[metric] for item in method_splits)
            row[f"{metric}_mean_across_splits"] = statistics["mean"]
            row[f"{metric}_std_across_splits"] = statistics["std"]
            row[f"{metric}_range_across_splits"] = statistics["range"]
        split_stability_rows.append(row)

    representative_configs = {
        method: cases[method][split_seeds[0]]["config"]
        for method in methods
    }
    method_protocol_fingerprints = {
        method: metric_protocol_fingerprint(
            representative_configs[method],
            include_threshold_method=True,
        )
        for method in methods
    }

    if args.mode == "development":
        by_method = {
            row["method"]: row for row in split_stability_rows
        }
        method_order = {
            method: index for index, method in enumerate(methods)
        }
        selected_method = min(
            methods,
            key=lambda method: normal_only_selection_key(
                by_method[method],
                method_order,
            ),
        )
        selected_config = representative_configs[selected_method]
        selected_stability = by_method[selected_method]
        selection = {
            "schema_version": 1,
            "selection_stage": "dev5",
            "selection_rule": (
                "Use held-out normal data only. Minimize normal image "
                "positive-rate mean plus split standard deviation; then "
                "minimize normal pixel positive-rate mean plus split "
                "standard deviation; then minimize the two standard "
                "deviations. Remaining ties use the predefined method order."
            ),
            "uses_anomaly_labels_for_selection": False,
            "pixel_f1_used_for_selection": False,
            "selection_metrics": {
                "normal_image_positive_rate_mean": selected_stability[
                    "normal_image_positive_rate_mean_across_splits"
                ],
                "normal_image_positive_rate_std": selected_stability[
                    "normal_image_positive_rate_std_across_splits"
                ],
                "normal_image_positive_rate_risk": (
                    selected_stability[
                        "normal_image_positive_rate_mean_across_splits"
                    ]
                    + selected_stability[
                        "normal_image_positive_rate_std_across_splits"
                    ]
                ),
                "normal_pixel_positive_rate_mean": selected_stability[
                    "normal_pixel_positive_rate_mean_across_splits"
                ],
                "normal_pixel_positive_rate_std": selected_stability[
                    "normal_pixel_positive_rate_std_across_splits"
                ],
                "normal_pixel_positive_rate_risk": (
                    selected_stability[
                        "normal_pixel_positive_rate_mean_across_splits"
                    ]
                    + selected_stability[
                        "normal_pixel_positive_rate_std_across_splits"
                    ]
                ),
            },
            "selected_method": selected_method,
            "selected_parameters": threshold_parameters(selected_config),
            "model_seed": int(selected_config["experiment"]["seed"]),
            "threshold_split_seeds": split_seeds,
            "development_categories": categories,
            "base_protocol": base_protocol,
            "base_protocol_fingerprint": base_fingerprint,
            "selected_protocol_fingerprint": (
                method_protocol_fingerprints[selected_method]
            ),
            "oracle_baseline": oracle_baseline,
        }
        with (
            experiment_dir / "selected_threshold_method.json"
        ).open("w", encoding="utf-8") as handle:
            json.dump(selection, handle, ensure_ascii=False, indent=2)
    else:
        assert selection is not None
        method = methods[0]
        config = representative_configs[method]
        if threshold_parameters(config) != selection["selected_parameters"]:
            raise ValueError(
                "Validation threshold parameters differ from selection manifest"
            )
        if int(config["experiment"]["seed"]) != int(
            selection["model_seed"]
        ):
            raise ValueError("Validation model seed differs from manifest")
        if base_fingerprint != selection["base_protocol_fingerprint"]:
            raise ValueError(
                "Validation base protocol differs from development protocol"
            )
        if method_protocol_fingerprints[method] != selection[
            "selected_protocol_fingerprint"
        ]:
            raise ValueError(
                "Validation selected protocol fingerprint mismatch"
            )
        if split_seeds != [
            int(value) for value in selection["threshold_split_seeds"]
        ]:
            raise ValueError(
                "Validation threshold split seeds differ from manifest"
            )

    outputs = [
        ("threshold_stability", stability_rows, {"method"}),
        ("threshold_by_split", split_rows, {"method"}),
        (
            "threshold_split_stability",
            split_stability_rows,
            {"method"},
        ),
        ("threshold_per_category", category_rows, {"category", "method"}),
    ]
    for stem, rows, text_fields in outputs:
        write_csv(experiment_dir / f"{stem}.csv", rows)
        (experiment_dir / f"{stem}.md").write_text(
            markdown_table(rows, text_fields),
            encoding="utf-8",
        )

    summary = {
        "stage": args.mode,
        "categories": categories,
        "category_count": len(categories),
        "model_seed": int(
            representative_configs[methods[0]]["experiment"]["seed"]
        ),
        "threshold_split_seeds": split_seeds,
        "methods": methods,
        "normal_diagnostic_mode": "held_out_from_threshold_fit",
        "base_protocol": base_protocol,
        "base_protocol_fingerprint": base_fingerprint,
        "method_protocol_fingerprints": method_protocol_fingerprints,
        "threshold_parameters": {
            method: threshold_parameters(representative_configs[method])
            for method in methods
        },
        "oracle_baseline": oracle_baseline,
        "stability": stability_rows,
        "by_split": split_rows,
        "split_stability": split_stability_rows,
        "per_category": category_rows,
    }
    if selection is not None:
        summary["selection_manifest"] = selection
    with (experiment_dir / "threshold_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Summarized {args.mode} threshold experiment in {experiment_dir}")
    print(
        (experiment_dir / "threshold_split_stability.md").read_text(
            encoding="utf-8"
        )
    )


if __name__ == "__main__":
    main()
