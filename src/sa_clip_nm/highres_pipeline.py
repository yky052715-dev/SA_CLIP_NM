from __future__ import annotations

import copy
import csv
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .backbone import make_patch_coordinates
from .calibration import (
    CategoryCalibration,
    compute_layer_tau,
    image_score_from_map,
)
from .config import config_fingerprint, resolve_output_dir, save_json
from .data import build_mvtec_records
from .highres_backbone import (
    HighResolutionCLIPVisionFeatureExtractor,
    clip_patch_grid,
)
from .highres_data import (
    HighResolutionMVTecDataset,
    collate_highres_records,
)
from .localization_diagnostics import save_localization_diagnostics
from .localization_metrics import (
    evaluate_localization_image,
    summarize_localization_rows,
)
from .map_refinement import upsample_anomaly_maps
from .memory import load_memory_banks
from .metrics import evaluate_binary_scores, flatten_metrics
from .pipeline import (
    _configured_map_outputs,
    _extract_batch_features,
    _raw_layer_scores,
    calibrate_category,
    prepare_category,
    write_summary,
)
from .visualization import save_result_figure


def _evaluation_size(config: dict[str, Any]) -> int:
    size = int(config.get("high_resolution", {}).get("evaluation_size", 448))
    if size <= 0:
        raise ValueError("high_resolution.evaluation_size must be positive")
    return size


def _map_config(config: dict[str, Any]) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    resolved["data"]["image_size"] = _evaluation_size(config)
    return resolved


def _highres_loader(
    records,
    config: dict[str, Any],
    include_mask: bool,
) -> DataLoader:
    dataset = HighResolutionMVTecDataset(
        records=records,
        input_size=int(config["data"]["image_size"]),
        evaluation_size=_evaluation_size(config),
        resize_mode=str(config["data"]["resize_mode"]),
        include_mask=include_mask,
    )
    return DataLoader(
        dataset,
        batch_size=int(config["model"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_highres_records,
    )


def _category_dir(output_dir: Path, category: str) -> Path:
    path = output_dir / category
    path.mkdir(parents=True, exist_ok=True)
    return path


def evaluate_highres_category(
    category: str,
    config: dict[str, Any],
    extractor: HighResolutionCLIPVisionFeatureExtractor,
    output_dir: Path,
    calibration: CategoryCalibration,
    device: str | torch.device,
) -> dict[str, Any]:
    category_dir = _category_dir(output_dir, category)
    artifact_dir = category_dir / "artifacts"
    banks, _ = load_memory_banks(artifact_dir / "memory_bank.pt")
    _, test_records = build_mvtec_records(config["data"]["root"], category)
    loader = _highres_loader(test_records, config, include_mask=True)
    map_config = _map_config(config)
    input_size = int(config["data"]["image_size"])
    grid_side, patch_count = clip_patch_grid(input_size, extractor.patch_size)

    image_labels: list[int] = []
    image_scores_all: list[float] = []
    pixel_labels: list[np.ndarray] = []
    pixel_scores: list[np.ndarray] = []
    localization_rows: list[dict[str, object]] = []
    samples: list[dict[str, object]] = []
    inference_seconds = 0.0
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch.device(device))

    for batch in loader:
        start = time.perf_counter()
        features_by_layer = _extract_batch_features(extractor, batch)
        actual_patch_count = int(next(iter(features_by_layer.values())).shape[1])
        if actual_patch_count != patch_count:
            raise RuntimeError(
                f"Expected {patch_count} patches, got {actual_patch_count}"
            )
        patch_coordinates = make_patch_coordinates(actual_patch_count)
        raw_scores = _raw_layer_scores(
            features_by_layer,
            patch_coordinates,
            banks,
            {
                layer: calibration.layers[layer].spatial_weight
                for layer in features_by_layer
            },
            config,
            device,
        )
        map_outputs = _configured_map_outputs(
            raw_scores,
            calibration.layers,
            map_config,
        )
        maps = map_outputs.anomaly_maps
        batch_image_scores = image_score_from_map(
            maps,
            method=calibration.image_score_method,
            topk_fraction=calibration.image_topk_fraction,
        )
        inference_seconds += time.perf_counter() - start

        masks = batch["mask"].numpy().astype(np.uint8)
        labels = batch["label"].numpy().astype(np.uint8)
        image_labels.extend(labels.tolist())
        image_scores_all.extend(batch_image_scores.numpy().tolist())
        pixel_labels.extend([mask.reshape(-1) for mask in masks])
        pixel_scores.extend([score.numpy().reshape(-1) for score in maps])
        for index in range(maps.shape[0]):
            row = evaluate_localization_image(
                ground_truth=masks[index],
                anomaly_map=maps[index].numpy(),
                threshold=calibration.pixel_threshold,
                small_max_fraction=float(
                    config["evaluation"].get("small_max_fraction", 0.005)
                ),
                medium_max_fraction=float(
                    config["evaluation"].get("medium_max_fraction", 0.02)
                ),
            )
            localization_rows.append(
                {
                    "path": str(batch["path"][index]),
                    "defect_type": str(batch["defect_type"][index]),
                    "label": int(labels[index]),
                    **row,
                }
            )
            if bool(config["inference"]["save_visualizations"]) or bool(
                config.get("diagnostics", {}).get("enabled", False)
            ):
                samples.append(
                    {
                        "path": batch["path"][index],
                        "defect_type": batch["defect_type"][index],
                        "image": batch["display_image"][index],
                        "mask": masks[index],
                        "map": maps[index].numpy(),
                        "image_score": float(batch_image_scores[index].item()),
                        "layer_patch_maps": {
                            layer: values[index].numpy()
                            for layer, values in map_outputs.layer_patch_maps.items()
                        },
                        "fused_patch_map": map_outputs.fused_patch_map[
                            index
                        ].numpy(),
                    }
                )

    image_metrics = evaluate_binary_scores(
        np.asarray(image_labels),
        np.asarray(image_scores_all),
        calibrated_threshold=calibration.image_threshold,
        compute_oracle=bool(config["evaluation"]["compute_oracle_f1"]),
    )
    pixel_metrics = evaluate_binary_scores(
        np.concatenate(pixel_labels),
        np.concatenate(pixel_scores),
        calibrated_threshold=calibration.pixel_threshold,
        compute_oracle=bool(config["evaluation"]["compute_oracle_f1"]),
    )
    localization_summary = summarize_localization_rows(localization_rows)
    save_json(localization_summary, category_dir / "localization_metrics.json")
    if localization_rows:
        with (category_dir / "per_image_localization_metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(localization_rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(localization_rows)

    peak_memory_mb = 0.0
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        peak_memory_mb = float(
            torch.cuda.max_memory_allocated(torch.device(device)) / (1024**2)
        )
    result: dict[str, Any] = {
        "category": category,
        **{
            f"localization_{key}": value
            for key, value in localization_summary.items()
        },
        "input_image_size": input_size,
        "evaluation_size": _evaluation_size(config),
        "patch_grid_side": grid_side,
        "patch_count": patch_count,
        "position_interpolation": input_size != extractor.pretrained_image_size,
        "layer_fusion": str(config["inference"].get("layer_fusion", "mean")),
        **flatten_metrics("image", image_metrics),
        **flatten_metrics("pixel", pixel_metrics),
        "image_threshold_calibrated": calibration.image_threshold,
        "pixel_threshold_calibrated": calibration.pixel_threshold,
        "pixel_threshold_method": calibration.pixel_threshold_method,
        "pixel_image_quantile_calibrated": calibration.pixel_image_quantile,
        "adaptive_selected_pixel_image_quantile": (
            calibration.adaptive_selected_pixel_image_quantile
        ),
        "adaptive_max_normal_image_positive_rate": (
            calibration.adaptive_max_normal_image_positive_rate
        ),
        "normal_pixel_positive_rate": calibration.normal_pixel_positive_rate,
        "normal_image_positive_rate": calibration.normal_image_positive_rate,
        "threshold_fit_images": calibration.threshold_fit_images,
        "normal_validation_images": calibration.normal_validation_images,
        "normalization_fit_images": calibration.normalization_fit_images,
        "normal_diagnostic_mode": calibration.normal_diagnostic_mode,
        "inference_seconds_total": inference_seconds,
        "inference_ms_per_image": 1000.0 * inference_seconds / len(test_records),
        "gpu_peak_memory_mb": peak_memory_mb,
        "test_images": len(test_records),
        "memory_entries": int(
            sum(bank.features.shape[0] for bank in banks.values())
        ),
    }
    save_json(result, category_dir / "metrics.json")

    ranked = sorted(
        samples, key=lambda item: float(item["image_score"]), reverse=True
    )
    color_max = max(calibration.pixel_threshold * 2.5, 1.0e-6)
    if bool(config["inference"]["save_visualizations"]):
        limit = int(config["inference"]["max_visualizations_per_category"])
        for index, sample in enumerate(ranked[:limit]):
            stem = Path(str(sample["path"])).stem
            anomaly_map = np.asarray(sample["map"])
            save_result_figure(
                image=np.asarray(sample["image"]),
                ground_truth=np.asarray(sample["mask"]),
                anomaly_map=anomaly_map,
                prediction=anomaly_map >= calibration.pixel_threshold,
                output_path=category_dir
                / "visualizations"
                / f"{index:03d}_{sample['defect_type']}_{stem}.png",
                title=(
                    f"{category}/{sample['defect_type']} CLIP {input_size} "
                    f"score={sample['image_score']:.3f}"
                ),
                color_max=color_max,
            )
    if bool(config.get("diagnostics", {}).get("enabled", False)):
        limit = int(config["diagnostics"].get("max_samples_per_category", 10))
        for index, sample in enumerate(ranked[:limit]):
            stem = Path(str(sample["path"])).stem
            anomaly_map = np.asarray(sample["map"])
            nearest_map = upsample_anomaly_maps(
                torch.from_numpy(np.asarray(sample["fused_patch_map"]))[None],
                output_size=_evaluation_size(config),
                mode="nearest",
                gaussian_sigma=0.0,
            )[0].numpy()
            save_localization_diagnostics(
                output_dir=category_dir / "diagnostic_maps",
                prefix=f"{index:03d}_{sample['defect_type']}_{stem}",
                image=np.asarray(sample["image"]),
                ground_truth=np.asarray(sample["mask"]),
                anomaly_map=anomaly_map,
                nearest_map=nearest_map,
                prediction=anomaly_map >= calibration.pixel_threshold,
                layer_patch_maps=sample["layer_patch_maps"],
                fused_patch_map=np.asarray(sample["fused_patch_map"]),
                threshold=calibration.pixel_threshold,
                color_max=color_max,
            )
    return result


def run_highres_experiment(
    config: dict[str, Any],
    device: str | torch.device,
) -> list[dict[str, Any]]:
    input_size = int(config["data"]["image_size"])
    evaluation_size = _evaluation_size(config)
    output_dir = resolve_output_dir(config)
    completion_path = output_dir / "experiment_complete.json"
    if completion_path.exists():
        completion_path.unlink()
    save_json(config, output_dir / "resolved_config.json")

    extractor = HighResolutionCLIPVisionFeatureExtractor(
        checkpoint=str(config["model"]["checkpoint"]),
        layers=config["model"]["active_layers"],
        token_norm=str(config["model"]["token_norm"]),
        device=device,
    )
    grid_side, patch_count = clip_patch_grid(input_size, extractor.patch_size)
    try:
        import transformers

        transformers_version = transformers.__version__
    except ImportError:
        transformers_version = "not installed"
    save_json(
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "transformers": transformers_version,
            "checkpoint": config["model"]["checkpoint"],
            "layers": list(extractor.layers),
            "token_norm": extractor.token_norm,
            "pretrained_image_size": extractor.pretrained_image_size,
            "input_image_size": input_size,
            "evaluation_size": evaluation_size,
            "patch_size": extractor.patch_size,
            "patch_grid_side": grid_side,
            "patch_count": patch_count,
            "position_interpolation": (
                input_size != extractor.pretrained_image_size
            ),
            "supports_position_interpolation": (
                extractor.supports_position_interpolation
            ),
        },
        output_dir / "environment_and_feature_protocol.json",
    )

    category_reliabilities: dict[str, dict[int, float]] = {}
    for category in config["data"]["categories"]:
        print(f"[prepare-highres-{input_size}] {category}", flush=True)
        category_reliabilities[category] = prepare_category(
            category, config, extractor, output_dir
        )
    layer_tau = compute_layer_tau(
        category_reliabilities,
        quantile=float(config["retrieval"]["tau_quantile"]),
    )

    map_config = _map_config(config)
    results: list[dict[str, Any]] = []
    for category in config["data"]["categories"]:
        print(f"[calibrate-highres-{input_size}] {category}", flush=True)
        calibration = calibrate_category(
            category,
            map_config,
            output_dir,
            layer_tau,
            device,
        )
        print(f"[evaluate-highres-{input_size}] {category}", flush=True)
        results.append(
            evaluate_highres_category(
                category,
                config,
                extractor,
                output_dir,
                calibration,
                device,
            )
        )
        write_summary(results, output_dir)

    save_json(
        {
            "input_image_size": input_size,
            "evaluation_size": evaluation_size,
            "patch_grid_side": grid_side,
            "patch_count": patch_count,
            "gpu_peak_memory_mb": max(
                float(row["gpu_peak_memory_mb"]) for row in results
            ),
            "mean_inference_ms_per_image": float(
                np.mean([row["inference_ms_per_image"] for row in results])
            ),
            "mean_memory_entries": float(
                np.mean([row["memory_entries"] for row in results])
            ),
        },
        output_dir / "resource_summary.json",
    )
    expected = [str(value) for value in config["data"]["categories"]]
    completed = [str(row["category"]) for row in results]
    save_json(
        {
            "status": "complete",
            "expected_categories": expected,
            "completed_categories": completed,
            "expected_count": len(expected),
            "completed_count": len(completed),
            "input_image_size": input_size,
            "evaluation_size": evaluation_size,
            "patch_grid_side": grid_side,
            "patch_count": patch_count,
            "position_interpolation": (
                input_size != extractor.pretrained_image_size
            ),
            "layer_fusion": str(
                config["inference"].get("layer_fusion", "mean")
            ),
            "config_fingerprint": config_fingerprint(config),
        },
        completion_path,
    )
    return results
