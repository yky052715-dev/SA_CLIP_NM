from __future__ import annotations

import csv
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader

from .backbone import CLIPVisionFeatureExtractor, make_patch_coordinates
from .calibration import (
    CategoryCalibration,
    LayerCalibration,
    adaptive_area_pixel_threshold_from_maps,
    adaptive_pixel_threshold_from_maps,
    calibrate_scores,
    compute_layer_tau,
    estimate_spatial_reliability,
    image_score_from_map,
    normal_threshold_diagnostics,
    pixel_threshold_from_maps,
    quantile_threshold,
    reliability_to_weight,
    robust_location_scale,
)
from .config import config_fingerprint, resolve_output_dir, save_json
from .data import (
    MVTecImageDataset,
    build_mvtec_records,
    collate_records,
    split_calibration_records,
    split_normal_records,
)
from .memory import (
    LayerMemoryBank,
    build_layer_memory,
    load_memory_banks,
    save_memory_banks,
)
from .metrics import evaluate_binary_scores, flatten_metrics
from .localization_diagnostics import save_localization_diagnostics
from .localization_metrics import (
    evaluate_localization_image,
    summarize_localization_rows,
)
from .map_refinement import (
    build_anomaly_map_outputs,
    upsample_anomaly_maps,
)
from .retrieval import batched_layer_scores
from .visualization import save_result_figure


def _loader(
    records,
    config: dict[str, Any],
    include_mask: bool,
    shuffle: bool = False,
) -> DataLoader:
    dataset = MVTecImageDataset(
        records,
        image_size=int(config["data"]["image_size"]),
        resize_mode=str(config["data"]["resize_mode"]),
        include_mask=include_mask,
    )
    return DataLoader(
        dataset,
        batch_size=int(config["model"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_records,
    )


def _extract_features(
    extractor: CLIPVisionFeatureExtractor,
    loader: DataLoader,
) -> tuple[dict[int, torch.Tensor], list[str]]:
    per_layer: dict[int, list[torch.Tensor]] = {
        layer: [] for layer in extractor.layers
    }
    paths: list[str] = []
    for batch in loader:
        features = extractor.extract(batch["image"])
        for layer, values in features.items():
            per_layer[layer].append(values.cpu())
        paths.extend(batch["path"])
    return {
        layer: torch.cat(chunks, dim=0)
        for layer, chunks in per_layer.items()
    }, paths


def _category_dir(output_dir: Path, category: str) -> Path:
    path = output_dir / category
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_category(
    category: str,
    config: dict[str, Any],
    extractor: CLIPVisionFeatureExtractor,
    output_dir: Path,
) -> dict[int, float]:
    category_dir = _category_dir(output_dir, category)
    artifact_dir = category_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    train_records, _ = build_mvtec_records(config["data"]["root"], category)
    memory_records, calibration_records = split_normal_records(
        train_records,
        calibration_fraction=float(config["data"]["calibration_fraction"]),
        seed=int(config["experiment"]["seed"]),
    )
    threshold_fit_records, normal_validation_records = split_calibration_records(
        calibration_records,
        threshold_fit_fraction=float(
            config["calibration"].get("threshold_fit_fraction", 0.5)
        ),
        seed=int(config["calibration"]["threshold_split_seed"]),
    )
    save_json(
        {
            "category": category,
            "seed": int(config["experiment"]["seed"]),
            "threshold_split_seed": int(
                config["calibration"]["threshold_split_seed"]
            ),
            "memory": [str(record.path) for record in memory_records],
            "calibration": [str(record.path) for record in calibration_records],
            "threshold_fit": [
                str(record.path) for record in threshold_fit_records
            ],
            "normal_validation": [
                str(record.path) for record in normal_validation_records
            ],
        },
        artifact_dir / "split_manifest.json",
    )

    memory_features, memory_paths = _extract_features(
        extractor,
        _loader(memory_records, config, include_mask=False),
    )
    calibration_features, calibration_paths = _extract_features(
        extractor,
        _loader(calibration_records, config, include_mask=False),
    )
    threshold_fit_path_set = {
        str(record.path) for record in threshold_fit_records
    }
    normal_validation_path_set = {
        str(record.path) for record in normal_validation_records
    }
    threshold_fit_indices = [
        index
        for index, path in enumerate(calibration_paths)
        if path in threshold_fit_path_set
    ]
    normal_validation_indices = [
        index
        for index, path in enumerate(calibration_paths)
        if path in normal_validation_path_set
    ]
    if (
        len(threshold_fit_indices) + len(normal_validation_indices)
        != len(calibration_paths)
    ):
        raise RuntimeError("Threshold split does not cover calibration paths")
    first_layer = next(iter(memory_features))
    patch_count = memory_features[first_layer].shape[1]
    patch_coordinates = make_patch_coordinates(patch_count)

    banks: dict[int, LayerMemoryBank] = {}
    reliabilities: dict[int, float] = {}
    reliability_details: dict[str, object] = {}
    for layer in extractor.layers:
        banks[layer] = build_layer_memory(
            image_features=memory_features[layer],
            patch_coordinates=patch_coordinates,
            ratio=float(config["memory"]["ratio"]),
            method=str(config["memory"]["sampling"]),
            seed=int(config["experiment"]["seed"]) + layer,
            projection_dim=int(config["memory"]["projection_dim"]),
            max_candidates=int(config["memory"]["max_candidates"]),
        )
        detail = estimate_spatial_reliability(calibration_features[layer])
        reliabilities[layer] = float(detail["reliability"])
        reliability_details[str(layer)] = detail

    save_memory_banks(
        banks,
        metadata={
            "category": category,
            "memory_paths": memory_paths,
            "checkpoint": config["model"]["checkpoint"],
            "layers": list(extractor.layers),
            "token_norm": config["model"]["token_norm"],
            "image_size": int(config["data"]["image_size"]),
            "resize_mode": config["data"]["resize_mode"],
            "sampling": config["memory"]["sampling"],
            "ratio": float(config["memory"]["ratio"]),
        },
        path=artifact_dir / "memory_bank.pt",
    )
    torch.save(
        {
            "features": {
                layer: values.half()
                for layer, values in calibration_features.items()
            },
            "paths": calibration_paths,
            "threshold_fit_indices": threshold_fit_indices,
            "normal_validation_indices": normal_validation_indices,
            "patch_coordinates": patch_coordinates,
        },
        artifact_dir / "calibration_features.pt",
    )
    save_json(
        {
            "category": category,
            "layers": reliability_details,
        },
        artifact_dir / "reliability.json",
    )
    return reliabilities


def _raw_layer_scores(
    features_by_layer: dict[int, torch.Tensor],
    patch_coordinates: torch.Tensor,
    banks: dict[int, LayerMemoryBank],
    spatial_weights: dict[int, float],
    config: dict[str, Any],
    device: str | torch.device,
) -> dict[int, torch.Tensor]:
    return {
        layer: batched_layer_scores(
            image_features=features_by_layer[layer],
            patch_coordinates=patch_coordinates,
            bank=banks[layer],
            spatial_weight=spatial_weights[layer],
            query_chunk_size=int(config["retrieval"]["query_chunk_size"]),
            bank_chunk_size=int(config["retrieval"]["bank_chunk_size"]),
            device=device,
        )
        for layer in sorted(features_by_layer)
    }


def _fused_maps(
    raw_scores: dict[int, torch.Tensor],
    layer_calibrations: dict[int, LayerCalibration],
    image_size: int,
    clamp_min_zero: bool,
    gaussian_sigma: float,
) -> torch.Tensor:
    calibrated = []
    for layer in sorted(raw_scores):
        parameters = layer_calibrations[layer]
        calibrated.append(
            calibrate_scores(
                raw_scores[layer],
                median=parameters.median,
                mad=parameters.mad,
                clamp_min_zero=clamp_min_zero,
            )
        )
    fused = torch.stack(calibrated, dim=0).mean(dim=0)
    side = int(round(math.sqrt(fused.shape[1])))
    if side * side != fused.shape[1]:
        raise ValueError("Patch scores do not form a square map")
    maps = fused.reshape(fused.shape[0], 1, side, side)
    maps = F.interpolate(
        maps,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    if gaussian_sigma > 0:
        smoothed = [
            gaussian_filter(anomaly_map.numpy(), sigma=gaussian_sigma)
            for anomaly_map in maps.cpu()
        ]
        maps = torch.from_numpy(np.stack(smoothed)).float()
    return maps.cpu()


def _configured_map_outputs(
    raw_scores: dict[int, torch.Tensor],
    layer_calibrations: dict[int, LayerCalibration],
    config: dict[str, Any],
):
    inference = config["inference"]
    return build_anomaly_map_outputs(
        raw_scores=raw_scores,
        layer_calibrations=layer_calibrations,
        output_size=int(config["data"]["image_size"]),
        clamp_min_zero=bool(config["calibration"]["clamp_min_zero"]),
        gaussian_sigma=float(inference["gaussian_sigma"]),
        layer_fusion=str(inference.get("layer_fusion", "mean")),
        layer_fusion_epsilon=float(
            inference.get("layer_fusion_epsilon", 1.0e-6)
        ),
        layer_weights=inference.get("layer_weights"),
        upsample_mode=str(inference.get("upsample_mode", "bilinear")),
    )


def calibrate_category(
    category: str,
    config: dict[str, Any],
    output_dir: Path,
    layer_tau: dict[int, float],
    device: str | torch.device,
) -> CategoryCalibration:
    artifact_dir = _category_dir(output_dir, category) / "artifacts"
    banks, _ = load_memory_banks(artifact_dir / "memory_bank.pt")
    payload = torch.load(artifact_dir / "calibration_features.pt", map_location="cpu")
    calibration_features = {
        int(layer): values.float()
        for layer, values in payload["features"].items()
    }
    patch_coordinates = payload["patch_coordinates"].float()
    with (artifact_dir / "reliability.json").open("r", encoding="utf-8") as handle:
        reliability_payload = json.load(handle)
    reliabilities = {
        int(layer): float(values["reliability"])
        for layer, values in reliability_payload["layers"].items()
    }
    spatial_mode = str(config["retrieval"].get("spatial_mode", "adaptive"))
    if spatial_mode == "adaptive":
        spatial_weights = {
            layer: reliability_to_weight(
                reliability=reliabilities[layer],
                tau=float(layer_tau[layer]),
                lambda_max=float(config["retrieval"]["lambda_max"]),
            )
            for layer in calibration_features
        }
    elif spatial_mode == "fixed":
        spatial_weights = {
            layer: float(config["retrieval"]["fixed_lambda"])
            for layer in calibration_features
        }
    elif spatial_mode == "none":
        spatial_weights = {layer: 0.0 for layer in calibration_features}
    else:
        raise ValueError(
            "retrieval.spatial_mode must be one of: none, fixed, adaptive"
        )

    calibration_raw_scores = _raw_layer_scores(
        calibration_features,
        patch_coordinates,
        banks,
        spatial_weights,
        config,
        device,
    )
    layer_calibrations: dict[int, LayerCalibration] = {}
    for layer, values in calibration_raw_scores.items():
        median, mad = robust_location_scale(
            values,
            epsilon=float(config["calibration"]["mad_epsilon"]),
        )
        layer_calibrations[layer] = LayerCalibration(
            median=median,
            mad=mad,
            reliability=reliabilities[layer],
            spatial_weight=spatial_weights[layer],
        )

    calibration_outputs = _configured_map_outputs(
        calibration_raw_scores,
        layer_calibrations,
        config,
    )
    calibration_maps = calibration_outputs.anomaly_maps
    threshold_fit_indices = torch.tensor(
        payload["threshold_fit_indices"],
        dtype=torch.long,
    )
    normal_validation_indices = torch.tensor(
        payload["normal_validation_indices"],
        dtype=torch.long,
    )
    threshold_fit_maps = calibration_maps.index_select(
        0,
        threshold_fit_indices,
    )
    normal_validation_maps = calibration_maps.index_select(
        0,
        normal_validation_indices,
    )
    image_scores = image_score_from_map(
        threshold_fit_maps,
        method=str(config["inference"]["image_score"]),
        topk_fraction=float(config["inference"]["image_topk_fraction"]),
    )
    threshold_method = str(
        config["calibration"].get("pixel_threshold_method", "global_quantile")
    )
    pixel_quantile = float(config["calibration"].get("pixel_quantile", 0.995))
    pixel_image_quantile = float(
        config["calibration"].get("pixel_image_quantile", 0.95)
    )
    pixel_topk_fraction = float(
        config["calibration"].get("pixel_topk_fraction", 0.01)
    )
    adaptive_quantiles = None
    adaptive_selected_quantile = None
    adaptive_max_image_fp = None
    adaptive_max_pixel_fp = None
    adaptive_max_area_p95 = None
    adaptive_max_area_max = None
    adaptive_candidates = None
    if threshold_method.startswith("adaptive_area_"):
        adaptive_area_base_methods = {
            "adaptive_area_image_max_quantile": "image_max_quantile",
            "adaptive_area_image_topk_quantile": "image_topk_quantile",
        }
        if threshold_method not in adaptive_area_base_methods:
            raise ValueError(
                "adaptive area pixel threshold method must be one of: "
                "adaptive_area_image_max_quantile, "
                "adaptive_area_image_topk_quantile"
            )
        adaptive_quantiles = [
            float(value)
            for value in config["calibration"].get(
                "adaptive_pixel_image_quantiles",
                [0.875, 0.90, 0.925, 0.95],
            )
        ]

        def optional_float(name: str):
            value = config["calibration"].get(name, None)
            return None if value is None else float(value)

        adaptive_max_image_fp = optional_float(
            "adaptive_max_normal_image_positive_rate"
        )
        adaptive_max_pixel_fp = optional_float(
            "adaptive_max_normal_pixel_positive_rate"
        )
        adaptive_max_area_p95 = optional_float(
            "adaptive_max_normal_positive_area_p95_fraction"
        )
        adaptive_max_area_max = optional_float(
            "adaptive_max_normal_positive_area_max_fraction"
        )
        (
            pixel_threshold,
            adaptive_selected_quantile,
            adaptive_candidates,
        ) = adaptive_area_pixel_threshold_from_maps(
            threshold_fit_maps=threshold_fit_maps,
            normal_validation_maps=normal_validation_maps,
            method=adaptive_area_base_methods[threshold_method],
            pixel_quantile=pixel_quantile,
            image_quantiles=adaptive_quantiles,
            topk_fraction=pixel_topk_fraction,
            max_normal_image_positive_rate=adaptive_max_image_fp,
            max_normal_pixel_positive_rate=adaptive_max_pixel_fp,
            max_normal_positive_area_p95_fraction=adaptive_max_area_p95,
            max_normal_positive_area_max_fraction=adaptive_max_area_max,
        )
        pixel_image_quantile = adaptive_selected_quantile
    elif threshold_method.startswith("adaptive_"):
        adaptive_base_methods = {
            "adaptive_image_max_quantile": "image_max_quantile",
            "adaptive_image_topk_quantile": "image_topk_quantile",
        }
        if threshold_method not in adaptive_base_methods:
            raise ValueError(
                "adaptive pixel threshold method must be one of: "
                "adaptive_image_max_quantile, adaptive_image_topk_quantile"
            )
        adaptive_quantiles = [
            float(value)
            for value in config["calibration"].get(
                "adaptive_pixel_image_quantiles",
                [0.90, 0.925, 0.95],
            )
        ]
        adaptive_max_image_fp = float(
            config["calibration"].get(
                "adaptive_max_normal_image_positive_rate",
                0.15,
            )
        )
        raw_max_pixel_fp = config["calibration"].get(
            "adaptive_max_normal_pixel_positive_rate",
            None,
        )
        adaptive_max_pixel_fp = (
            None if raw_max_pixel_fp is None else float(raw_max_pixel_fp)
        )
        (
            pixel_threshold,
            adaptive_selected_quantile,
            adaptive_candidates,
        ) = adaptive_pixel_threshold_from_maps(
            threshold_fit_maps=threshold_fit_maps,
            normal_validation_maps=normal_validation_maps,
            method=adaptive_base_methods[threshold_method],
            pixel_quantile=pixel_quantile,
            image_quantiles=adaptive_quantiles,
            topk_fraction=pixel_topk_fraction,
            max_normal_image_positive_rate=adaptive_max_image_fp,
            max_normal_pixel_positive_rate=adaptive_max_pixel_fp,
        )
        pixel_image_quantile = adaptive_selected_quantile
    else:
        pixel_threshold = pixel_threshold_from_maps(
            threshold_fit_maps,
            method=threshold_method,
            pixel_quantile=pixel_quantile,
            image_quantile=pixel_image_quantile,
            topk_fraction=pixel_topk_fraction,
        )
    threshold_diagnostics = normal_threshold_diagnostics(
        normal_validation_maps,
        pixel_threshold,
    )
    calibration = CategoryCalibration(
        category=category,
        layers=layer_calibrations,
        pixel_threshold=pixel_threshold,
        image_threshold=quantile_threshold(
            image_scores.numpy(),
            float(config["calibration"]["image_quantile"]),
        ),
        pixel_threshold_method=threshold_method,
        pixel_quantile=pixel_quantile,
        pixel_image_quantile=pixel_image_quantile,
        pixel_topk_fraction=pixel_topk_fraction,
        image_quantile=float(config["calibration"]["image_quantile"]),
        image_score_method=str(config["inference"]["image_score"]),
        image_topk_fraction=float(config["inference"]["image_topk_fraction"]),
        normal_pixel_positive_rate=threshold_diagnostics[
            "normal_pixel_positive_rate"
        ],
        normal_image_positive_rate=threshold_diagnostics[
            "normal_image_positive_rate"
        ],
        threshold_fit_images=len(payload["threshold_fit_indices"]),
        normal_validation_images=len(payload["normal_validation_indices"]),
        normalization_fit_images=len(payload["paths"]),
        normal_diagnostic_mode="held_out_from_threshold_fit",
        adaptive_pixel_image_quantiles=adaptive_quantiles,
        adaptive_selected_pixel_image_quantile=adaptive_selected_quantile,
        adaptive_max_normal_image_positive_rate=adaptive_max_image_fp,
        adaptive_max_normal_pixel_positive_rate=adaptive_max_pixel_fp,
        adaptive_max_normal_positive_area_p95_fraction=adaptive_max_area_p95,
        adaptive_max_normal_positive_area_max_fraction=adaptive_max_area_max,
        adaptive_threshold_candidates=adaptive_candidates,
    )
    calibration.save(artifact_dir / "calibration.json")
    return calibration


def _extract_batch_features(
    extractor: CLIPVisionFeatureExtractor,
    batch: dict[str, object],
) -> dict[int, torch.Tensor]:
    return {
        layer: values.cpu()
        for layer, values in extractor.extract(batch["image"]).items()
    }


def evaluate_category(
    category: str,
    config: dict[str, Any],
    extractor: CLIPVisionFeatureExtractor,
    output_dir: Path,
    calibration: CategoryCalibration,
    device: str | torch.device,
) -> dict[str, Any]:
    category_dir = _category_dir(output_dir, category)
    artifact_dir = category_dir / "artifacts"
    banks, _ = load_memory_banks(artifact_dir / "memory_bank.pt")
    _, test_records = build_mvtec_records(config["data"]["root"], category)
    loader = _loader(test_records, config, include_mask=True)

    image_labels: list[int] = []
    image_scores_all: list[float] = []
    pixel_labels: list[np.ndarray] = []
    pixel_scores: list[np.ndarray] = []
    samples: list[dict[str, object]] = []
    localization_rows: list[dict[str, object]] = []
    inference_seconds = 0.0

    for batch in loader:
        start = time.perf_counter()
        features_by_layer = _extract_batch_features(extractor, batch)
        actual_patch_count = next(iter(features_by_layer.values())).shape[1]
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
            config,
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
            for index in range(maps.shape[0]):
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
    save_json(
        localization_summary,
        category_dir / "localization_metrics.json",
    )
    if localization_rows:
        with (category_dir / "per_image_localization_metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(localization_rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(localization_rows)
    result: dict[str, Any] = {
        "category": category,
        **{
            f"localization_{key}": value
            for key, value in localization_summary.items()
        },
        "layer_fusion": str(config["inference"].get("layer_fusion", "mean")),
        "upsample_mode": str(config["inference"].get("upsample_mode", "bilinear")),
        **flatten_metrics("image", image_metrics),
        **flatten_metrics("pixel", pixel_metrics),
        "image_threshold_calibrated": calibration.image_threshold,
        "pixel_threshold_calibrated": calibration.pixel_threshold,
        "pixel_threshold_method": calibration.pixel_threshold_method,
        "normal_pixel_positive_rate": calibration.normal_pixel_positive_rate,
        "normal_image_positive_rate": calibration.normal_image_positive_rate,
        "threshold_fit_images": calibration.threshold_fit_images,
        "normal_validation_images": calibration.normal_validation_images,
        "normalization_fit_images": calibration.normalization_fit_images,
        "normal_diagnostic_mode": calibration.normal_diagnostic_mode,
        "inference_seconds_total": inference_seconds,
        "inference_ms_per_image": 1000.0 * inference_seconds / len(test_records),
        "test_images": len(test_records),
        "memory_entries": int(sum(bank.features.shape[0] for bank in banks.values())),
    }
    save_json(result, category_dir / "metrics.json")

    if bool(config["inference"]["save_visualizations"]):
        limit = int(config["inference"]["max_visualizations_per_category"])
        ranked = sorted(samples, key=lambda item: float(item["image_score"]), reverse=True)[:limit]
        color_max = max(calibration.pixel_threshold * 2.5, 1e-6)
        for index, sample in enumerate(ranked):
            stem = Path(str(sample["path"])).stem
            prediction = np.asarray(sample["map"]) >= calibration.pixel_threshold
            save_result_figure(
                image=np.asarray(sample["image"]),
                ground_truth=np.asarray(sample["mask"]),
                anomaly_map=np.asarray(sample["map"]),
                prediction=prediction,
                output_path=category_dir
                / "visualizations"
                / f"{index:03d}_{sample['defect_type']}_{stem}.png",
                title=f"{category}/{sample['defect_type']} score={sample['image_score']:.3f}",
                color_max=color_max,
            )
    diagnostic_config = config.get("diagnostics", {})
    if bool(diagnostic_config.get("enabled", False)):
        diagnostic_limit = int(
            diagnostic_config.get(
                "max_samples_per_category",
                config["inference"]["max_visualizations_per_category"],
            )
        )
        diagnostic_samples = sorted(
            samples,
            key=lambda item: float(item["image_score"]),
            reverse=True,
        )[:diagnostic_limit]
        color_max = max(calibration.pixel_threshold * 2.5, 1e-6)
        for index, sample in enumerate(diagnostic_samples):
            stem = Path(str(sample["path"])).stem
            anomaly_map = np.asarray(sample["map"])
            prediction = anomaly_map >= calibration.pixel_threshold
            nearest_map = upsample_anomaly_maps(
                torch.from_numpy(np.asarray(sample["fused_patch_map"]))[None],
                output_size=anomaly_map.shape[0],
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
                prediction=prediction,
                layer_patch_maps=sample["layer_patch_maps"],
                fused_patch_map=np.asarray(sample["fused_patch_map"]),
                threshold=calibration.pixel_threshold,
                color_max=color_max,
            )
    return result


def write_summary(results: list[dict[str, Any]], output_dir: Path) -> None:
    if not results:
        return
    fieldnames = list(results[0].keys())
    with (output_dir / "metrics_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    numeric_keys = [
        key
        for key, value in results[0].items()
        if key != "category" and isinstance(value, (int, float))
    ]
    averages = {
        key: float(np.nanmean([float(row[key]) for row in results]))
        for key in numeric_keys
    }
    save_json({"categories": results, "mean": averages}, output_dir / "metrics_summary.json")

    table_keys = [
        "category",
        "image_AUROC",
        "image_F1_calibrated",
        "image_F1_oracle",
        "pixel_AUROC",
        "pixel_F1_calibrated",
        "pixel_F1_oracle",
        "pixel_IoU_calibrated",
        "inference_ms_per_image",
    ]
    lines = [
        "| " + " | ".join(table_keys) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(table_keys) - 1)) + "|",
    ]
    for row in results:
        values = [str(row["category"])] + [
            f"{float(row[key]):.6f}" for key in table_keys[1:]
        ]
        lines.append("| " + " | ".join(values) + " |")
    mean_values = ["mean"] + [
        f"{averages[key]:.6f}" for key in table_keys[1:]
    ]
    lines.append("| " + " | ".join(mean_values) + " |")
    (output_dir / "metrics_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run_experiment(
    config: dict[str, Any],
    device: str | torch.device,
) -> list[dict[str, Any]]:
    output_dir = resolve_output_dir(config)
    completion_path = output_dir / "experiment_complete.json"
    if completion_path.exists():
        completion_path.unlink()
    save_json(config, output_dir / "resolved_config.json")
    extractor = CLIPVisionFeatureExtractor(
        checkpoint=str(config["model"]["checkpoint"]),
        layers=config["model"]["active_layers"],
        token_norm=str(config["model"]["token_norm"]),
        device=device,
    )
    feature_spec = extractor.feature_spec(int(config["data"]["image_size"]))
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
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "transformers": transformers_version,
            "checkpoint": feature_spec.checkpoint,
            "layers": list(feature_spec.layers),
            "token_norm": feature_spec.token_norm,
            "patch_count": feature_spec.patch_count,
            "feature_dim": feature_spec.feature_dim,
            "layer_fusion": str(config["inference"].get("layer_fusion", "mean")),
            "upsample_mode": str(config["inference"].get("upsample_mode", "bilinear")),
            "hidden_state_definition": (
                "hidden_states[0] is embedding output; hidden_states[k] is output "
                "after encoder block k; CLS is removed"
            ),
        },
        output_dir / "environment_and_feature_protocol.json",
    )

    category_reliabilities: dict[str, dict[int, float]] = {}
    for category in config["data"]["categories"]:
        print(f"[prepare] {category}", flush=True)
        category_reliabilities[category] = prepare_category(
            category,
            config,
            extractor,
            output_dir,
        )
    layer_tau = compute_layer_tau(
        category_reliabilities,
        quantile=float(config["retrieval"]["tau_quantile"]),
    )
    save_json(
        {
            "quantile": float(config["retrieval"]["tau_quantile"]),
            "lambda_max": float(config["retrieval"]["lambda_max"]),
            "layer_tau": {str(layer): value for layer, value in layer_tau.items()},
            "category_reliabilities": {
                category: {
                    str(layer): value for layer, value in values.items()
                }
                for category, values in category_reliabilities.items()
            },
        },
        output_dir / "spatial_reliability_summary.json",
    )

    results = []
    for category in config["data"]["categories"]:
        print(f"[calibrate] {category}", flush=True)
        calibration = calibrate_category(
            category,
            config,
            output_dir,
            layer_tau,
            device,
        )
        print(f"[evaluate] {category}", flush=True)
        results.append(
            evaluate_category(
                category,
                config,
                extractor,
                output_dir,
                calibration,
                device,
            )
        )
        write_summary(results, output_dir)
    expected_categories = [str(value) for value in config["data"]["categories"]]
    completed_categories = [str(row["category"]) for row in results]
    save_json(
        {
            "status": "complete",
            "expected_categories": expected_categories,
            "completed_categories": completed_categories,
            "expected_count": len(expected_categories),
            "completed_count": len(completed_categories),
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
            "layer_fusion": str(config["inference"].get("layer_fusion", "mean")),
            "upsample_mode": str(config["inference"].get("upsample_mode", "bilinear")),
            "config_fingerprint": config_fingerprint(config),
        },
        completion_path,
    )
    return results
