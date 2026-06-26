from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sa_clip_nm.calibration import (
    adaptive_area_pixel_threshold_from_maps,
    image_score_from_map,
    normal_threshold_diagnostics,
    pixel_threshold_from_maps,
    quantile_threshold,
)
from sa_clip_nm.config import save_json
from sa_clip_nm.localization_metrics import (
    evaluate_localization_image,
    prediction_from_anomaly_map,
    summarize_localization_rows,
)
from sa_clip_nm.metrics import evaluate_binary_scores, flatten_metrics


DEFAULT_DEV5 = ["bottle", "metal_nut", "grid", "leather", "screw"]


def _load_npz(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = np.load(path, allow_pickle=True)
    return {key: payload[key] for key in payload.files}


def _path_index(paths: np.ndarray) -> dict[str, int]:
    return {str(path): index for index, path in enumerate(paths.tolist())}


def _align_payload(reference: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    reference_paths = [str(path) for path in reference["paths"].tolist()]
    other_index = _path_index(other["paths"])
    missing = [path for path in reference_paths if path not in other_index]
    if missing:
        raise ValueError(f"Missing {len(missing)} paths in secondary map payload; first={missing[0]}")
    indices = np.asarray([other_index[path] for path in reference_paths], dtype=np.int64)
    aligned = dict(other)
    for key in ("maps", "masks", "labels", "defect_types"):
        if key in aligned:
            aligned[key] = aligned[key][indices]
    aligned["paths"] = aligned["paths"][indices]
    return aligned


def _threshold_from_maps(
    threshold_fit_maps: torch.Tensor,
    normal_validation_maps: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[float, float | None, list[dict[str, float]] | None]:
    method = str(args.pixel_threshold_method)
    if method.startswith("adaptive_area_"):
        base_methods = {
            "adaptive_area_image_max_quantile": "image_max_quantile",
            "adaptive_area_image_topk_quantile": "image_topk_quantile",
        }
        if method not in base_methods:
            raise ValueError(f"Unsupported adaptive area threshold method: {method}")
        return adaptive_area_pixel_threshold_from_maps(
            threshold_fit_maps=threshold_fit_maps,
            normal_validation_maps=normal_validation_maps,
            method=base_methods[method],
            pixel_quantile=float(args.pixel_quantile),
            image_quantiles=[float(value) for value in args.adaptive_pixel_image_quantiles],
            topk_fraction=float(args.pixel_topk_fraction),
            max_normal_image_positive_rate=args.adaptive_max_normal_image_positive_rate,
            max_normal_pixel_positive_rate=args.adaptive_max_normal_pixel_positive_rate,
            max_normal_positive_area_p95_fraction=args.adaptive_max_normal_positive_area_p95_fraction,
            max_normal_positive_area_max_fraction=args.adaptive_max_normal_positive_area_max_fraction,
        )
    threshold = pixel_threshold_from_maps(
        threshold_fit_maps,
        method=method,
        pixel_quantile=float(args.pixel_quantile),
        image_quantile=float(args.pixel_image_quantile),
        topk_fraction=float(args.pixel_topk_fraction),
    )
    return threshold, None, None


def _fuse_maps(clip_maps: np.ndarray, dino_maps: np.ndarray, alpha: float) -> np.ndarray:
    if clip_maps.shape != dino_maps.shape:
        raise ValueError(f"Map shape mismatch: CLIP={clip_maps.shape}, DINO={dino_maps.shape}")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    return (float(alpha) * clip_maps.astype(np.float32) + (1.0 - float(alpha)) * dino_maps.astype(np.float32)).astype(np.float32)


def run_category(args: argparse.Namespace, category: str, alpha: float) -> dict[str, Any]:
    clip_artifact = Path(args.clip_root) / category / "artifacts"
    dino_artifact = Path(args.dino_root) / category / "artifacts"
    clip_cal = _load_npz(clip_artifact / "calibration_maps.npz")
    dino_cal = _align_payload(clip_cal, _load_npz(dino_artifact / "calibration_maps.npz"))
    clip_test = _load_npz(clip_artifact / "test_maps.npz")
    dino_test = _align_payload(clip_test, _load_npz(dino_artifact / "test_maps.npz"))

    fused_cal_maps_np = _fuse_maps(clip_cal["maps"], dino_cal["maps"], alpha)
    fused_test_maps_np = _fuse_maps(clip_test["maps"], dino_test["maps"], alpha)
    threshold_fit_indices = clip_cal["threshold_fit_indices"].astype(np.int64)
    normal_validation_indices = clip_cal["normal_validation_indices"].astype(np.int64)
    fused_cal_maps = torch.from_numpy(fused_cal_maps_np)
    threshold_fit_maps = fused_cal_maps.index_select(0, torch.from_numpy(threshold_fit_indices))
    normal_validation_maps = fused_cal_maps.index_select(0, torch.from_numpy(normal_validation_indices))
    pixel_threshold, selected_quantile, adaptive_candidates = _threshold_from_maps(
        threshold_fit_maps,
        normal_validation_maps,
        args,
    )
    image_scores_for_threshold = image_score_from_map(
        threshold_fit_maps,
        method=str(args.image_score),
        topk_fraction=float(args.image_topk_fraction),
    )
    image_threshold = quantile_threshold(
        image_scores_for_threshold.numpy(),
        float(args.image_quantile),
    )
    normal_diag = normal_threshold_diagnostics(normal_validation_maps, pixel_threshold)

    masks = clip_test["masks"].astype(np.uint8)
    labels = clip_test["labels"].astype(np.uint8)
    paths = [str(path) for path in clip_test["paths"].tolist()]
    defect_types = [str(value) for value in clip_test["defect_types"].tolist()]
    fused_test_maps = torch.from_numpy(fused_test_maps_np)
    image_scores = image_score_from_map(
        fused_test_maps,
        method=str(args.image_score),
        topk_fraction=float(args.image_topk_fraction),
    ).numpy()
    min_component_area_pixels = int(round(float(args.min_component_area_fraction) * fused_test_maps_np.shape[-1] * fused_test_maps_np.shape[-2]))

    pixel_predictions: list[np.ndarray] = []
    localization_rows: list[dict[str, object]] = []
    for index, anomaly_map in enumerate(fused_test_maps_np):
        prediction = prediction_from_anomaly_map(
            anomaly_map,
            threshold=pixel_threshold,
            min_component_area_pixels=min_component_area_pixels,
        )
        pixel_predictions.append(prediction.reshape(-1))
        row = evaluate_localization_image(
            ground_truth=masks[index],
            anomaly_map=anomaly_map,
            threshold=pixel_threshold,
            small_max_fraction=float(args.small_max_fraction),
            medium_max_fraction=float(args.medium_max_fraction),
            min_component_area_pixels=min_component_area_pixels,
        )
        localization_rows.append(
            {
                "path": paths[index],
                "defect_type": defect_types[index],
                "label": int(labels[index]),
                **row,
            }
        )

    image_metrics = evaluate_binary_scores(
        labels,
        image_scores,
        calibrated_threshold=image_threshold,
        compute_oracle=True,
    )
    pixel_metrics = evaluate_binary_scores(
        masks.reshape(-1),
        fused_test_maps_np.reshape(-1),
        calibrated_threshold=pixel_threshold,
        compute_oracle=True,
        calibrated_predictions=np.concatenate(pixel_predictions),
    )
    localization_summary = summarize_localization_rows(localization_rows)

    category_dir = Path(args.output_dir) / f"alpha_{alpha:.3f}" / category
    artifact_dir = category_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    save_json(localization_summary, category_dir / "localization_metrics.json")
    if localization_rows:
        with (category_dir / "per_image_localization_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(localization_rows[0].keys()))
            writer.writeheader()
            writer.writerows(localization_rows)
    np.savez_compressed(
        artifact_dir / "calibration_maps.npz",
        maps=fused_cal_maps_np,
        paths=clip_cal["paths"],
        threshold_fit_indices=threshold_fit_indices,
        normal_validation_indices=normal_validation_indices,
    )
    np.savez_compressed(
        artifact_dir / "test_maps.npz",
        maps=fused_test_maps_np,
        masks=masks,
        labels=labels,
        paths=clip_test["paths"],
        defect_types=clip_test["defect_types"],
    )
    save_json(
        {
            "category": category,
            "alpha": float(alpha),
            "pixel_threshold": float(pixel_threshold),
            "image_threshold": float(image_threshold),
            "pixel_threshold_method": str(args.pixel_threshold_method),
            "adaptive_selected_pixel_image_quantile": selected_quantile,
            "adaptive_threshold_candidates": adaptive_candidates,
            "normal_pixel_positive_rate": normal_diag["normal_pixel_positive_rate"],
            "normal_image_positive_rate": normal_diag["normal_image_positive_rate"],
            "postprocess_min_component_area_pixels": min_component_area_pixels,
        },
        artifact_dir / "calibration.json",
    )
    result: dict[str, Any] = {
        "category": category,
        "alpha": float(alpha),
        **{f"localization_{key}": value for key, value in localization_summary.items()},
        **flatten_metrics("image", image_metrics),
        **flatten_metrics("pixel", pixel_metrics),
        "image_threshold_calibrated": float(image_threshold),
        "pixel_threshold_calibrated": float(pixel_threshold),
        "pixel_threshold_method": str(args.pixel_threshold_method),
        "adaptive_selected_pixel_image_quantile": selected_quantile,
        "normal_pixel_positive_rate": normal_diag["normal_pixel_positive_rate"],
        "normal_image_positive_rate": normal_diag["normal_image_positive_rate"],
        "postprocess_min_component_area_fraction": float(args.min_component_area_fraction),
        "postprocess_min_component_area_pixels": min_component_area_pixels,
        "test_images": int(len(labels)),
    }
    save_json(result, category_dir / "metrics.json")
    return result


def write_summary(results: list[dict[str, Any]], output_dir: Path) -> None:
    if not results:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    numeric_keys = [key for key, value in results[0].items() if key != "category" and isinstance(value, (int, float))]
    means = {key: float(np.nanmean([float(row[key]) for row in results])) for key in numeric_keys}
    save_json({"categories": results, "mean": means}, output_dir / "metrics_summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse CLIP and DINOv2 anomaly maps")
    parser.add_argument("--clip-root", required=True, help="Root containing category/artifacts/*_maps.npz for CLIP")
    parser.add_argument("--dino-root", required=True, help="Root containing category/artifacts/*_maps.npz for DINOv2")
    parser.add_argument("--output-dir", default="outputs/dual_backbone_fusion/dev5")
    parser.add_argument("--categories", nargs="+", default=DEFAULT_DEV5)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.5, 0.7, 0.3])
    parser.add_argument("--pixel-threshold-method", default="adaptive_area_image_max_quantile")
    parser.add_argument("--pixel-quantile", type=float, default=0.995)
    parser.add_argument("--pixel-image-quantile", type=float, default=0.95)
    parser.add_argument("--pixel-topk-fraction", type=float, default=0.01)
    parser.add_argument("--adaptive-pixel-image-quantiles", nargs="+", type=float, default=[0.875, 0.90, 0.925, 0.95])
    parser.add_argument("--adaptive-max-normal-image-positive-rate", type=float, default=None)
    parser.add_argument("--adaptive-max-normal-pixel-positive-rate", type=float, default=None)
    parser.add_argument("--adaptive-max-normal-positive-area-p95-fraction", type=float, default=0.001)
    parser.add_argument("--adaptive-max-normal-positive-area-max-fraction", type=float, default=None)
    parser.add_argument("--image-quantile", type=float, default=0.99)
    parser.add_argument("--image-score", default="topk_mean")
    parser.add_argument("--image-topk-fraction", type=float, default=0.01)
    parser.add_argument("--min-component-area-fraction", type=float, default=0.0005)
    parser.add_argument("--small-max-fraction", type=float, default=0.005)
    parser.add_argument("--medium-max-fraction", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    save_json(vars(args), root / "resolved_args.json")
    for alpha in args.alphas:
        results = []
        alpha_dir = root / f"alpha_{alpha:.3f}"
        for category in args.categories:
            print(f"[fusion alpha={alpha:.3f}] {category}", flush=True)
            results.append(run_category(args, category, alpha))
            write_summary(results, alpha_dir)
        save_json(
            {
                "status": "complete",
                "alpha": float(alpha),
                "expected_categories": list(args.categories),
                "completed_categories": [row["category"] for row in results],
                "expected_count": len(args.categories),
                "completed_count": len(results),
            },
            alpha_dir / "experiment_complete.json",
        )


if __name__ == "__main__":
    main()
