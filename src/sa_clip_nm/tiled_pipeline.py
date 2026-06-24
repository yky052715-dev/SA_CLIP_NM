from __future__ import annotations

import csv
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .backbone import CLIPVisionFeatureExtractor, make_patch_coordinates
from .calibration import (
    CategoryCalibration,
    LayerCalibration,
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
    build_mvtec_records,
    split_calibration_records,
    split_normal_records,
)
from .localization import (
    combine_image_scores,
    generate_window_boxes,
    merge_window_maps,
)
from .localization_metrics import (
    evaluate_localization_image,
    summarize_localization_rows,
)
from .localization_protocol import (
    effective_local_memory_ratio,
    validate_tiled_config,
)
from .memory import (
    LayerMemoryBank,
    build_layer_memory,
    load_memory_banks,
    save_memory_banks,
)
from .metrics import evaluate_binary_scores, flatten_metrics
from .pipeline import (
    _configured_map_outputs,
    _extract_batch_features,
    _raw_layer_scores,
    calibrate_category,
    prepare_category,
    write_summary,
)
from .tiled_data import TiledMVTecImageDataset, collate_tiled_records
from .visualization import save_result_figure


@dataclass
class TiledCalibration:
    global_calibration: CategoryCalibration
    local_calibration: CategoryCalibration
    image_threshold: float
    image_score_merge: str
    normal_combined_image_positive_rate: float


def _category_dir(output_dir: Path, category: str) -> Path:
    path = output_dir / category
    path.mkdir(parents=True, exist_ok=True)
    return path


def _tiled_loader(
    records,
    config: dict[str, Any],
    include_mask: bool,
) -> DataLoader:
    localization = validate_tiled_config(config)
    dataset = TiledMVTecImageDataset(
        records=records,
        image_size=int(config["data"]["image_size"]),
        canvas_size=int(localization["canvas_size"]),
        window_size=int(localization["window_size"]),
        stride=int(localization["stride"]),
        resize_mode=str(config["data"]["resize_mode"]),
        include_mask=include_mask,
    )
    return DataLoader(
        dataset,
        batch_size=int(localization.get("image_batch_size", 2)),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_tiled_records,
    )


def _extract_window_batch_features(
    extractor: CLIPVisionFeatureExtractor,
    windows: torch.Tensor,
    window_batch_size: int,
) -> dict[int, torch.Tensor]:
    if windows.ndim != 5:
        raise ValueError("windows must have shape [B, N, C, H, W]")
    image_count, window_count = windows.shape[:2]
    flattened = windows.reshape(-1, *windows.shape[2:])
    chunks: dict[int, list[torch.Tensor]] = {
        layer: [] for layer in extractor.layers
    }
    for start in range(0, flattened.shape[0], window_batch_size):
        end = min(start + window_batch_size, flattened.shape[0])
        values = extractor.extract(flattened[start:end])
        for layer, features in values.items():
            chunks[layer].append(features.cpu())
    return {
        layer: torch.cat(layer_chunks, dim=0).reshape(
            image_count,
            window_count,
            *layer_chunks[0].shape[1:],
        )
        for layer, layer_chunks in chunks.items()
    }


def _extract_tiled_features(
    extractor: CLIPVisionFeatureExtractor,
    loader: DataLoader,
    window_batch_size: int,
) -> tuple[dict[int, torch.Tensor], list[str]]:
    per_layer: dict[int, list[torch.Tensor]] = {
        layer: [] for layer in extractor.layers
    }
    paths: list[str] = []
    for batch in loader:
        features = _extract_window_batch_features(
            extractor,
            batch["windows"],
            window_batch_size,
        )
        for layer, values in features.items():
            per_layer[layer].append(values)
        paths.extend(batch["path"])
    return {
        layer: torch.cat(chunks, dim=0)
        for layer, chunks in per_layer.items()
    }, paths


def _threshold_split_indices(
    paths: list[str],
    threshold_fit_records,
    normal_validation_records,
) -> tuple[list[int], list[int]]:
    threshold_paths = {str(record.path) for record in threshold_fit_records}
    validation_paths = {
        str(record.path) for record in normal_validation_records
    }
    threshold_indices = [
        index for index, path in enumerate(paths) if path in threshold_paths
    ]
    validation_indices = [
        index for index, path in enumerate(paths) if path in validation_paths
    ]
    if len(threshold_indices) + len(validation_indices) != len(paths):
        raise RuntimeError("Threshold split does not cover local calibration paths")
    return threshold_indices, validation_indices


def prepare_local_category(
    category: str,
    config: dict[str, Any],
    extractor: CLIPVisionFeatureExtractor,
    output_dir: Path,
) -> dict[int, float]:
    localization = validate_tiled_config(config)
    artifact_dir = _category_dir(output_dir, category) / "artifacts"
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
    window_batch_size = int(localization["window_batch_size"])
    memory_features, memory_paths = _extract_tiled_features(
        extractor,
        _tiled_loader(memory_records, config, include_mask=False),
        window_batch_size,
    )
    calibration_features, calibration_paths = _extract_tiled_features(
        extractor,
        _tiled_loader(calibration_records, config, include_mask=False),
        window_batch_size,
    )
    threshold_indices, validation_indices = _threshold_split_indices(
        calibration_paths,
        threshold_fit_records,
        normal_validation_records,
    )
    first_layer = next(iter(memory_features))
    window_count = int(memory_features[first_layer].shape[1])
    patch_count = int(memory_features[first_layer].shape[2])
    patch_coordinates = make_patch_coordinates(patch_count)
    boxes = generate_window_boxes(
        canvas_size=int(localization["canvas_size"]),
        window_size=int(localization["window_size"]),
        stride=int(localization["stride"]),
    )
    if window_count != len(boxes):
        raise RuntimeError("Extracted local window count does not match protocol")

    banks: dict[int, LayerMemoryBank] = {}
    reliabilities: dict[int, float] = {}
    reliability_details: dict[str, object] = {}
    effective_ratios: dict[str, float] = {}
    for layer in extractor.layers:
        values = memory_features[layer].reshape(
            -1,
            patch_count,
            memory_features[layer].shape[-1],
        )
        candidate_count = int(values.shape[0] * values.shape[1])
        effective_ratio = effective_local_memory_ratio(
            candidate_count,
            requested_ratio=float(localization["local_memory_ratio"]),
            max_entries=int(localization["local_memory_max_entries"]),
        )
        banks[layer] = build_layer_memory(
            image_features=values,
            patch_coordinates=patch_coordinates,
            ratio=effective_ratio,
            method=str(config["memory"]["sampling"]),
            seed=int(config["experiment"]["seed"]) + 1000 + layer,
            projection_dim=int(config["memory"]["projection_dim"]),
            max_candidates=int(config["memory"]["max_candidates"]),
        )
        calibration_values = calibration_features[layer].reshape(
            -1,
            patch_count,
            calibration_features[layer].shape[-1],
        )
        detail = estimate_spatial_reliability(calibration_values)
        reliabilities[layer] = float(detail["reliability"])
        reliability_details[str(layer)] = detail
        effective_ratios[str(layer)] = effective_ratio

    save_memory_banks(
        banks,
        metadata={
            "category": category,
            "scope": "local_windows_only",
            "memory_paths": memory_paths,
            "checkpoint": config["model"]["checkpoint"],
            "layers": list(extractor.layers),
            "canvas_size": int(localization["canvas_size"]),
            "window_size": int(localization["window_size"]),
            "stride": int(localization["stride"]),
            "window_count": window_count,
            "sampling": config["memory"]["sampling"],
            "requested_ratio": float(localization["local_memory_ratio"]),
            "max_entries": int(localization["local_memory_max_entries"]),
            "effective_ratios": effective_ratios,
            "retained_entries": {
                str(layer): int(bank.features.shape[0])
                for layer, bank in banks.items()
            },
        },
        path=artifact_dir / "local_memory_bank.pt",
    )
    torch.save(
        {
            "features": {
                layer: values.half()
                for layer, values in calibration_features.items()
            },
            "paths": calibration_paths,
            "threshold_fit_indices": threshold_indices,
            "normal_validation_indices": validation_indices,
            "patch_coordinates": patch_coordinates,
            "window_boxes": boxes,
        },
        artifact_dir / "local_calibration_features.pt",
    )
    save_json(
        {"category": category, "layers": reliability_details},
        artifact_dir / "local_reliability.json",
    )
    return reliabilities


def _spatial_weights(
    reliabilities: dict[int, float],
    layer_tau: dict[int, float],
    config: dict[str, Any],
) -> dict[int, float]:
    mode = str(config["retrieval"].get("spatial_mode", "adaptive"))
    if mode == "adaptive":
        return {
            layer: reliability_to_weight(
                reliability=value,
                tau=float(layer_tau[layer]),
                lambda_max=float(config["retrieval"]["lambda_max"]),
            )
            for layer, value in reliabilities.items()
        }
    if mode == "fixed":
        return {
            layer: float(config["retrieval"]["fixed_lambda"])
            for layer in reliabilities
        }
    if mode == "none":
        return {layer: 0.0 for layer in reliabilities}
    raise ValueError("retrieval.spatial_mode must be none, fixed, or adaptive")


def _local_maps_from_raw_scores(
    raw_scores: dict[int, torch.Tensor],
    layer_calibrations: dict[int, LayerCalibration],
    image_count: int,
    window_count: int,
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    localization = validate_tiled_config(config)
    outputs = _configured_map_outputs(raw_scores, layer_calibrations, config)
    window_size = int(localization["window_size"])
    window_maps = outputs.anomaly_maps.reshape(
        image_count,
        window_count,
        window_size,
        window_size,
    )
    boxes = generate_window_boxes(
        canvas_size=int(localization["canvas_size"]),
        window_size=window_size,
        stride=int(localization["stride"]),
    )
    canvas_maps = merge_window_maps(
        window_maps,
        boxes=boxes,
        canvas_size=int(localization["canvas_size"]),
        merge=str(localization["merge"]),
        min_weight=float(localization["min_window_weight"]),
    ).cpu()
    evaluation_size = int(config["data"]["image_size"])
    evaluation_maps = F.interpolate(
        canvas_maps[:, None],
        size=(evaluation_size, evaluation_size),
        mode="bilinear",
        align_corners=False,
    )[:, 0].cpu()
    return canvas_maps, evaluation_maps


def _global_calibration_maps(
    category: str,
    config: dict[str, Any],
    output_dir: Path,
    calibration: CategoryCalibration,
    device: str | torch.device,
) -> tuple[torch.Tensor, dict[str, object]]:
    artifact_dir = _category_dir(output_dir, category) / "artifacts"
    banks, _ = load_memory_banks(artifact_dir / "memory_bank.pt")
    payload = torch.load(
        artifact_dir / "calibration_features.pt", map_location="cpu"
    )
    features = {
        int(layer): values.float()
        for layer, values in payload["features"].items()
    }
    raw_scores = _raw_layer_scores(
        features,
        payload["patch_coordinates"].float(),
        banks,
        {
            layer: calibration.layers[layer].spatial_weight
            for layer in features
        },
        config,
        device,
    )
    maps = _configured_map_outputs(
        raw_scores,
        calibration.layers,
        config,
    ).anomaly_maps
    return maps, payload


def calibrate_tiled_category(
    category: str,
    config: dict[str, Any],
    output_dir: Path,
    global_calibration: CategoryCalibration,
    local_layer_tau: dict[int, float],
    device: str | torch.device,
) -> TiledCalibration:
    localization = validate_tiled_config(config)
    artifact_dir = _category_dir(output_dir, category) / "artifacts"
    banks, _ = load_memory_banks(artifact_dir / "local_memory_bank.pt")
    payload = torch.load(
        artifact_dir / "local_calibration_features.pt", map_location="cpu"
    )
    features = {
        int(layer): values.float()
        for layer, values in payload["features"].items()
    }
    with (artifact_dir / "local_reliability.json").open(
        "r", encoding="utf-8"
    ) as handle:
        reliability_payload = json.load(handle)
    reliabilities = {
        int(layer): float(values["reliability"])
        for layer, values in reliability_payload["layers"].items()
    }
    spatial_weights = _spatial_weights(reliabilities, local_layer_tau, config)
    image_count = int(next(iter(features.values())).shape[0])
    window_count = int(next(iter(features.values())).shape[1])
    flattened_features = {
        layer: values.reshape(
            image_count * window_count,
            values.shape[2],
            values.shape[3],
        )
        for layer, values in features.items()
    }
    raw_scores = _raw_layer_scores(
        flattened_features,
        payload["patch_coordinates"].float(),
        banks,
        spatial_weights,
        config,
        device,
    )
    layer_calibrations: dict[int, LayerCalibration] = {}
    for layer, values in raw_scores.items():
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
    _, local_maps = _local_maps_from_raw_scores(
        raw_scores,
        layer_calibrations,
        image_count,
        window_count,
        config,
    )
    threshold_indices = torch.tensor(
        payload["threshold_fit_indices"], dtype=torch.long
    )
    validation_indices = torch.tensor(
        payload["normal_validation_indices"], dtype=torch.long
    )
    fit_maps = local_maps.index_select(0, threshold_indices)
    validation_maps = local_maps.index_select(0, validation_indices)
    threshold_method = str(config["calibration"]["pixel_threshold_method"])
    pixel_threshold = pixel_threshold_from_maps(
        fit_maps,
        method=threshold_method,
        pixel_quantile=float(config["calibration"]["pixel_quantile"]),
        image_quantile=float(config["calibration"]["pixel_image_quantile"]),
        topk_fraction=float(config["calibration"]["pixel_topk_fraction"]),
    )
    local_image_scores = image_score_from_map(
        fit_maps,
        method=str(config["inference"]["image_score"]),
        topk_fraction=float(config["inference"]["image_topk_fraction"]),
    )
    threshold_diagnostics = normal_threshold_diagnostics(
        validation_maps,
        pixel_threshold,
    )
    local_calibration = CategoryCalibration(
        category=category,
        layers=layer_calibrations,
        pixel_threshold=pixel_threshold,
        image_threshold=quantile_threshold(
            local_image_scores.numpy(),
            float(config["calibration"]["image_quantile"]),
        ),
        pixel_threshold_method=threshold_method,
        pixel_quantile=float(config["calibration"]["pixel_quantile"]),
        pixel_image_quantile=float(
            config["calibration"]["pixel_image_quantile"]
        ),
        pixel_topk_fraction=float(
            config["calibration"]["pixel_topk_fraction"]
        ),
        image_quantile=float(config["calibration"]["image_quantile"]),
        image_score_method=str(config["inference"]["image_score"]),
        image_topk_fraction=float(
            config["inference"]["image_topk_fraction"]
        ),
        normal_pixel_positive_rate=threshold_diagnostics[
            "normal_pixel_positive_rate"
        ],
        normal_image_positive_rate=threshold_diagnostics[
            "normal_image_positive_rate"
        ],
        threshold_fit_images=len(payload["threshold_fit_indices"]),
        normal_validation_images=len(payload["normal_validation_indices"]),
        normalization_fit_images=len(payload["paths"]),
        normal_diagnostic_mode="held_out_from_threshold_fit_local_only",
    )
    local_calibration.save(artifact_dir / "local_calibration.json")

    global_maps, global_payload = _global_calibration_maps(
        category,
        config,
        output_dir,
        global_calibration,
        device,
    )
    if list(payload["paths"]) != list(global_payload["paths"]):
        raise RuntimeError("Global and local calibration image order differs")
    global_fit_maps = global_maps.index_select(0, threshold_indices)
    global_fit_scores = image_score_from_map(
        global_fit_maps,
        method=global_calibration.image_score_method,
        topk_fraction=global_calibration.image_topk_fraction,
    )
    combined_fit_scores = combine_image_scores(
        global_fit_scores,
        local_image_scores,
        method=str(localization["image_score_merge"]),
    )
    combined_threshold = quantile_threshold(
        combined_fit_scores.numpy(),
        float(config["calibration"]["image_quantile"]),
    )
    global_validation_scores = image_score_from_map(
        global_maps.index_select(0, validation_indices),
        method=global_calibration.image_score_method,
        topk_fraction=global_calibration.image_topk_fraction,
    )
    local_validation_scores = image_score_from_map(
        validation_maps,
        method=local_calibration.image_score_method,
        topk_fraction=local_calibration.image_topk_fraction,
    )
    combined_validation_scores = combine_image_scores(
        global_validation_scores,
        local_validation_scores,
        method=str(localization["image_score_merge"]),
    )
    normal_combined_rate = float(
        (combined_validation_scores >= combined_threshold).float().mean().item()
    )
    calibration = TiledCalibration(
        global_calibration=global_calibration,
        local_calibration=local_calibration,
        image_threshold=combined_threshold,
        image_score_merge=str(localization["image_score_merge"]),
        normal_combined_image_positive_rate=normal_combined_rate,
    )
    save_json(
        {
            "category": category,
            "mode": "tiled",
            "global_calibration": global_calibration.to_dict(),
            "local_calibration": local_calibration.to_dict(),
            "combined_image_threshold": combined_threshold,
            "image_score_merge": calibration.image_score_merge,
            "normal_combined_image_positive_rate": normal_combined_rate,
        },
        artifact_dir / "tiled_calibration.json",
    )
    return calibration


def evaluate_tiled_category(
    category: str,
    config: dict[str, Any],
    extractor: CLIPVisionFeatureExtractor,
    output_dir: Path,
    calibration: TiledCalibration,
    device: str | torch.device,
) -> dict[str, Any]:
    localization = validate_tiled_config(config)
    category_dir = _category_dir(output_dir, category)
    artifact_dir = category_dir / "artifacts"
    global_banks, _ = load_memory_banks(artifact_dir / "memory_bank.pt")
    local_banks, local_metadata = load_memory_banks(
        artifact_dir / "local_memory_bank.pt"
    )
    _, test_records = build_mvtec_records(config["data"]["root"], category)
    loader = _tiled_loader(test_records, config, include_mask=True)

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
        global_features = _extract_batch_features(extractor, batch)
        patch_count = int(next(iter(global_features.values())).shape[1])
        patch_coordinates = make_patch_coordinates(patch_count)
        global_raw_scores = _raw_layer_scores(
            global_features,
            patch_coordinates,
            global_banks,
            {
                layer: calibration.global_calibration.layers[
                    layer
                ].spatial_weight
                for layer in global_features
            },
            config,
            device,
        )
        global_maps = _configured_map_outputs(
            global_raw_scores,
            calibration.global_calibration.layers,
            config,
        ).anomaly_maps

        local_features = _extract_window_batch_features(
            extractor,
            batch["windows"],
            int(localization["window_batch_size"]),
        )
        image_count = int(next(iter(local_features.values())).shape[0])
        window_count = int(next(iter(local_features.values())).shape[1])
        flattened_local = {
            layer: values.reshape(
                image_count * window_count,
                values.shape[2],
                values.shape[3],
            )
            for layer, values in local_features.items()
        }
        local_raw_scores = _raw_layer_scores(
            flattened_local,
            patch_coordinates,
            local_banks,
            {
                layer: calibration.local_calibration.layers[
                    layer
                ].spatial_weight
                for layer in flattened_local
            },
            config,
            device,
        )
        local_canvas_maps, maps = _local_maps_from_raw_scores(
            local_raw_scores,
            calibration.local_calibration.layers,
            image_count,
            window_count,
            config,
        )
        global_image_scores = image_score_from_map(
            global_maps,
            method=calibration.global_calibration.image_score_method,
            topk_fraction=calibration.global_calibration.image_topk_fraction,
        )
        local_image_scores = image_score_from_map(
            maps,
            method=calibration.local_calibration.image_score_method,
            topk_fraction=calibration.local_calibration.image_topk_fraction,
        )
        batch_image_scores = combine_image_scores(
            global_image_scores,
            local_image_scores,
            method=calibration.image_score_merge,
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
                threshold=calibration.local_calibration.pixel_threshold,
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
                        "canvas_image": batch["canvas_display_image"][index],
                        "mask": masks[index],
                        "canvas_mask": batch["canvas_mask"][index].numpy(),
                        "map": maps[index].numpy(),
                        "local_canvas_map": local_canvas_maps[index].numpy(),
                        "global_map": global_maps[index].numpy(),
                        "image_score": float(batch_image_scores[index].item()),
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
        calibrated_threshold=calibration.local_calibration.pixel_threshold,
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

    global_entries = int(
        sum(bank.features.shape[0] for bank in global_banks.values())
    )
    local_entries = int(
        sum(bank.features.shape[0] for bank in local_banks.values())
    )
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
        "localization_mode": "tiled",
        "layer_fusion": str(config["inference"].get("layer_fusion", "mean")),
        "upsample_mode": str(
            config["inference"].get("upsample_mode", "bilinear")
        ),
        **flatten_metrics("image", image_metrics),
        **flatten_metrics("pixel", pixel_metrics),
        "image_threshold_calibrated": calibration.image_threshold,
        "pixel_threshold_calibrated": (
            calibration.local_calibration.pixel_threshold
        ),
        "pixel_threshold_method": (
            calibration.local_calibration.pixel_threshold_method
        ),
        "normal_pixel_positive_rate": (
            calibration.local_calibration.normal_pixel_positive_rate
        ),
        "normal_image_positive_rate": (
            calibration.local_calibration.normal_image_positive_rate
        ),
        "normal_combined_image_score_positive_rate": (
            calibration.normal_combined_image_positive_rate
        ),
        "threshold_fit_images": (
            calibration.local_calibration.threshold_fit_images
        ),
        "normal_validation_images": (
            calibration.local_calibration.normal_validation_images
        ),
        "normalization_fit_images": (
            calibration.local_calibration.normalization_fit_images
        ),
        "normal_diagnostic_mode": (
            calibration.local_calibration.normal_diagnostic_mode
        ),
        "inference_seconds_total": inference_seconds,
        "inference_ms_per_image": 1000.0 * inference_seconds / len(test_records),
        "gpu_peak_memory_mb": peak_memory_mb,
        "test_images": len(test_records),
        "memory_entries": global_entries + local_entries,
        "global_memory_entries": global_entries,
        "local_memory_entries": local_entries,
        "local_window_count": int(local_metadata["window_count"]),
    }
    save_json(result, category_dir / "metrics.json")

    ranked = sorted(
        samples, key=lambda item: float(item["image_score"]), reverse=True
    )
    if bool(config["inference"]["save_visualizations"]):
        limit = int(config["inference"]["max_visualizations_per_category"])
        color_max = max(
            calibration.local_calibration.pixel_threshold * 2.5, 1.0e-6
        )
        for index, sample in enumerate(ranked[:limit]):
            stem = Path(str(sample["path"])).stem
            prediction = (
                np.asarray(sample["map"])
                >= calibration.local_calibration.pixel_threshold
            )
            save_result_figure(
                image=np.asarray(sample["image"]),
                ground_truth=np.asarray(sample["mask"]),
                anomaly_map=np.asarray(sample["map"]),
                prediction=prediction,
                output_path=category_dir
                / "visualizations"
                / f"{index:03d}_{sample['defect_type']}_{stem}.png",
                title=(
                    f"{category}/{sample['defect_type']} tiled "
                    f"score={sample['image_score']:.3f}"
                ),
                color_max=color_max,
            )
    diagnostic_config = config.get("diagnostics", {})
    if bool(diagnostic_config.get("enabled", False)):
        limit = int(diagnostic_config.get("max_samples_per_category", 20))
        directory = category_dir / "diagnostic_maps"
        directory.mkdir(parents=True, exist_ok=True)
        color_max = max(
            calibration.local_calibration.pixel_threshold * 2.5, 1.0e-6
        )
        for index, sample in enumerate(ranked[:limit]):
            stem = Path(str(sample["path"])).stem
            prefix = f"{index:03d}_{sample['defect_type']}_{stem}"
            global_map = np.asarray(sample["global_map"], dtype=np.float32)
            local_canvas_map = np.asarray(
                sample["local_canvas_map"], dtype=np.float32
            )
            np.save(directory / f"{prefix}_global_map.npy", global_map)
            np.save(
                directory / f"{prefix}_local_canvas_map.npy",
                local_canvas_map,
            )
            canvas_prediction = (
                local_canvas_map
                >= calibration.local_calibration.pixel_threshold
            )
            save_result_figure(
                image=np.asarray(sample["canvas_image"]),
                ground_truth=np.asarray(sample["canvas_mask"]),
                anomaly_map=local_canvas_map,
                prediction=canvas_prediction,
                output_path=directory / f"{prefix}_local_diagnostic.png",
                title=f"{category}/{sample['defect_type']} local 448",
                color_max=color_max,
            )
    return result


def run_tiled_experiment(
    config: dict[str, Any],
    device: str | torch.device,
) -> list[dict[str, Any]]:
    validate_tiled_config(config)
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
    localization = config["localization"]
    boxes = generate_window_boxes(
        int(localization["canvas_size"]),
        int(localization["window_size"]),
        int(localization["stride"]),
    )
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
            "checkpoint": feature_spec.checkpoint,
            "layers": list(feature_spec.layers),
            "token_norm": feature_spec.token_norm,
            "patch_count": feature_spec.patch_count,
            "feature_dim": feature_spec.feature_dim,
            "localization_mode": "tiled",
            "canvas_size": int(localization["canvas_size"]),
            "window_size": int(localization["window_size"]),
            "stride": int(localization["stride"]),
            "window_count": len(boxes),
            "window_merge": str(localization["merge"]),
            "evaluation_size": int(config["data"]["image_size"]),
            "layer_fusion": str(
                config["inference"].get("layer_fusion", "mean")
            ),
        },
        output_dir / "environment_and_feature_protocol.json",
    )

    global_reliabilities: dict[str, dict[int, float]] = {}
    local_reliabilities: dict[str, dict[int, float]] = {}
    for category in config["data"]["categories"]:
        print(f"[prepare-global] {category}", flush=True)
        global_reliabilities[category] = prepare_category(
            category, config, extractor, output_dir
        )
        print(f"[prepare-local] {category}", flush=True)
        local_reliabilities[category] = prepare_local_category(
            category, config, extractor, output_dir
        )
    quantile = float(config["retrieval"]["tau_quantile"])
    global_tau = compute_layer_tau(global_reliabilities, quantile=quantile)
    local_tau = compute_layer_tau(local_reliabilities, quantile=quantile)
    save_json(
        {
            "quantile": quantile,
            "lambda_max": float(config["retrieval"]["lambda_max"]),
            "global_layer_tau": {
                str(layer): value for layer, value in global_tau.items()
            },
            "local_layer_tau": {
                str(layer): value for layer, value in local_tau.items()
            },
            "global_category_reliabilities": global_reliabilities,
            "local_category_reliabilities": local_reliabilities,
        },
        output_dir / "spatial_reliability_summary.json",
    )

    results: list[dict[str, Any]] = []
    for category in config["data"]["categories"]:
        print(f"[calibrate-global] {category}", flush=True)
        global_calibration = calibrate_category(
            category,
            config,
            output_dir,
            global_tau,
            device,
        )
        print(f"[calibrate-local] {category}", flush=True)
        calibration = calibrate_tiled_category(
            category,
            config,
            output_dir,
            global_calibration,
            local_tau,
            device,
        )
        print(f"[evaluate-tiled] {category}", flush=True)
        results.append(
            evaluate_tiled_category(
                category,
                config,
                extractor,
                output_dir,
                calibration,
                device,
            )
        )
        write_summary(results, output_dir)

    expected = [str(value) for value in config["data"]["categories"]]
    completed = [str(row["category"]) for row in results]
    save_json(
        {
            "status": "complete",
            "expected_categories": expected,
            "completed_categories": completed,
            "expected_count": len(expected),
            "completed_count": len(completed),
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
            "localization_mode": "tiled",
            "layer_fusion": str(
                config["inference"].get("layer_fusion", "mean")
            ),
            "window_count": len(boxes),
            "config_fingerprint": config_fingerprint(config),
        },
        completion_path,
    )
    return results
