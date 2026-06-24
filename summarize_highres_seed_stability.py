from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = [
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
        description="Summarize paired H0/H1 seed stability"
    )
    parser.add_argument(
        "--root",
        default="outputs/localization_refinement/highres_clip_stability",
    )
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    return parser.parse_args()


def load_run(root: Path, seed: int, method: str) -> dict[str, object]:
    directory = root / f"seed_{seed}" / method
    required = {
        "completion": directory / "experiment_complete.json",
        "summary": directory / "metrics_summary.json",
        "config": directory / "resolved_config.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete seed run: {missing}")
    completion = json.loads(required["completion"].read_text(encoding="utf-8"))
    summary = json.loads(required["summary"].read_text(encoding="utf-8"))
    config = json.loads(required["config"].read_text(encoding="utf-8"))
    if completion.get("status") != "complete":
        raise RuntimeError(f"Seed run is not complete: {directory}")
    if int(config["experiment"]["seed"]) != seed:
        raise RuntimeError(f"Model seed mismatch: {directory}")
    if int(config["calibration"]["threshold_split_seed"]) != seed:
        raise RuntimeError(f"Threshold seed mismatch: {directory}")
    means = summary["mean"]
    row: dict[str, object] = {
        "seed": seed,
        "method": method,
        "input_image_size": int(completion["input_image_size"]),
        "patch_grid_side": int(completion["patch_grid_side"]),
        "memory_ratio": float(config["memory"]["ratio"]),
        "categories": summary["categories"],
    }
    row.update({key: means.get(key) for key in METRICS})
    return row


def pair_runs(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[int, dict[str, dict[str, object]]] = defaultdict(dict)
    for run in runs:
        grouped[int(run["seed"])][str(run["method"])] = run
    output: list[dict[str, object]] = []
    for seed in sorted(grouped):
        if set(grouped[seed]) != {"H0_clip224", "H1_clip336_matched"}:
            raise RuntimeError(f"Seed {seed} does not contain an exact H0/H1 pair")
        h0 = grouped[seed]["H0_clip224"]
        h1 = grouped[seed]["H1_clip336_matched"]
        row: dict[str, object] = {"seed": seed}
        for key in METRICS:
            old = float(h0[key])
            new = float(h1[key])
            row[f"H0_{key}"] = old
            row[f"H1_{key}"] = new
            row[f"delta_{key}"] = new - old
        row["speed_ratio"] = float(h1["inference_ms_per_image"]) / float(
            h0["inference_ms_per_image"]
        )
        row["meets_accuracy_criteria"] = (
            float(row["delta_pixel_AUROC"]) >= -0.002
            and float(row["delta_pixel_F1_calibrated"]) >= 0.0
            and float(row["delta_localization_overseg_anomaly_macro"]) < 0.0
            and float(row["delta_localization_small_defect_recall_macro"])
            >= -0.02
            and float(
                row["delta_localization_test_normal_image_positive_rate"]
            )
            <= 0.02
        )
        output.append(row)
    return output


def finite_stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0,
            **{
                key: float("nan")
                for key in ["mean", "std", "min", "max"]
            },
        }
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def aggregate_pairs(pairs: list[dict[str, object]]) -> dict[str, object]:
    return {
        "seed_count": len(pairs),
        "all_seeds_meet_accuracy_criteria": all(
            bool(row["meets_accuracy_criteria"]) for row in pairs
        ),
        "deltas": {
            key: finite_stats(
                [float(row[f"delta_{key}"]) for row in pairs]
            )
            for key in METRICS
        },
        "speed_ratio": finite_stats(
            [float(row["speed_ratio"]) for row in pairs]
        ),
    }


def category_pair_rows(
    runs: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[int, dict[str, dict[str, object]]] = defaultdict(dict)
    for run in runs:
        grouped[int(run["seed"])][str(run["method"])] = run
    keys = [
        "pixel_AUROC",
        "pixel_F1_calibrated",
        "localization_overseg_anomaly_macro",
        "localization_recall_anomaly_macro",
        "localization_test_normal_image_positive_rate",
    ]
    output: list[dict[str, object]] = []
    for seed in sorted(grouped):
        h0 = {
            str(row["category"]): row
            for row in grouped[seed]["H0_clip224"]["categories"]
        }
        h1 = {
            str(row["category"]): row
            for row in grouped[seed]["H1_clip336_matched"]["categories"]
        }
        if set(h0) != set(h1):
            raise RuntimeError(f"Category mismatch for seed {seed}")
        for category in sorted(h0):
            row: dict[str, object] = {"seed": seed, "category": category}
            for key in keys:
                row[f"delta_{key}"] = float(h1[category][key]) - float(
                    h0[category][key]
                )
            output.append(row)
    return output


def aggregate_categories(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["category"])].append(row)
    delta_keys = [key for key in rows[0] if key.startswith("delta_")]
    output: list[dict[str, object]] = []
    for category in sorted(grouped):
        result: dict[str, object] = {
            "category": category,
            "seed_count": len(grouped[category]),
        }
        for key in delta_keys:
            stats = finite_stats(
                [float(row[key]) for row in grouped[category]]
            )
            result[f"{key}_mean"] = stats["mean"]
            result[f"{key}_std"] = stats["std"]
        result["f1_improved_all_seeds"] = all(
            float(row["delta_pixel_F1_calibrated"]) >= 0.0
            for row in grouped[category]
        )
        result["overseg_improved_all_seeds"] = all(
            float(row["delta_localization_overseg_anomaly_macro"]) < 0.0
            for row in grouped[category]
        )
        result["f1_improved_seed_count"] = sum(
            float(row["delta_pixel_F1_calibrated"]) >= 0.0
            for row in grouped[category]
        )
        result["overseg_improved_seed_count"] = sum(
            float(row["delta_localization_overseg_anomaly_macro"]) < 0.0
            for row in grouped[category]
        )
        output.append(result)
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown(
    pairs: list[dict[str, object]],
    aggregate: dict[str, object],
    categories: list[dict[str, object]],
) -> str:
    lines = [
        "# High-resolution CLIP paired-seed stability",
        "",
        "## Per-seed deltas (H1 - H0)",
        "",
        "| seed | AUROC | F1 | OverSeg | Recall | Small F1 | Small Recall | Normal FP | speed | pass |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pairs:
        lines.append(
            "| {seed} | {auroc:.6f} | {f1:.6f} | {overseg:.6f} | "
            "{recall:.6f} | {small_f1:.6f} | {small_recall:.6f} | "
            "{normal_fp:.6f} | {speed:.3f} | {passed} |".format(
                seed=row["seed"],
                auroc=row["delta_pixel_AUROC"],
                f1=row["delta_pixel_F1_calibrated"],
                overseg=row["delta_localization_overseg_anomaly_macro"],
                recall=row["delta_localization_recall_anomaly_macro"],
                small_f1=row["delta_localization_small_defect_f1_macro"],
                small_recall=row[
                    "delta_localization_small_defect_recall_macro"
                ],
                normal_fp=row[
                    "delta_localization_test_normal_image_positive_rate"
                ],
                speed=row["speed_ratio"],
                passed=row["meets_accuracy_criteria"],
            )
        )
    lines.extend(
        [
            "",
            "## Aggregate deltas",
            "",
            "| metric | mean | std | min | max |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key in METRICS:
        stats = aggregate["deltas"][key]
        lines.append(
            f"| {key} | {stats['mean']:.6f} | {stats['std']:.6f} | "
            f"{stats['min']:.6f} | {stats['max']:.6f} |"
        )
    speed = aggregate["speed_ratio"]
    lines.append(
        f"| speed_ratio | {speed['mean']:.6f} | {speed['std']:.6f} | "
        f"{speed['min']:.6f} | {speed['max']:.6f} |"
    )
    lines.extend(
        [
            "",
            "## Per-category mean deltas",
            "",
            "| category | F1 | OverSeg | Recall | Normal FP | F1 improved | OverSeg improved |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in categories:
        lines.append(
            "| {category} | {f1:.6f} | {overseg:.6f} | {recall:.6f} | "
            "{normal_fp:.6f} | {f1_all} | {overseg_all} |".format(
                category=row["category"],
                f1=row["delta_pixel_F1_calibrated_mean"],
                overseg=row[
                    "delta_localization_overseg_anomaly_macro_mean"
                ],
                recall=row[
                    "delta_localization_recall_anomaly_macro_mean"
                ],
                normal_fp=row[
                    "delta_localization_test_normal_image_positive_rate_mean"
                ],
                f1_all=f'{row["f1_improved_seed_count"]}/{row["seed_count"]}',
                overseg_all=f'{row["overseg_improved_seed_count"]}/{row["seed_count"]}',
            )
        )
    lines.extend(
        [
            "",
            f"All seeds pass: {aggregate['all_seeds_meet_accuracy_criteria']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    runs = [
        load_run(root, seed, method)
        for seed in args.seeds
        for method in ["H0_clip224", "H1_clip336_matched"]
    ]
    run_rows = [
        {key: value for key, value in run.items() if key != "categories"}
        for run in runs
    ]
    pairs = pair_runs(runs)
    aggregate = aggregate_pairs(pairs)
    category_rows = category_pair_rows(runs)
    category_summary = aggregate_categories(category_rows)
    root.mkdir(parents=True, exist_ok=True)
    write_csv(root / "highres_seed_runs.csv", run_rows)
    write_csv(root / "highres_seed_paired_deltas.csv", pairs)
    write_csv(root / "highres_seed_category_deltas.csv", category_rows)
    write_csv(root / "highres_seed_category_summary.csv", category_summary)
    (root / "highres_seed_stability_summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (root / "highres_seed_stability_summary.md").write_text(
        markdown(pairs, aggregate, category_summary),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
