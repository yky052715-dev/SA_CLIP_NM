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

from .backbone import CLIPVisionFeatureExtractor, make_patch_coordinates
from .calibration import (
    CategoryCalibration,
    compute_layer_tau,
    image_score_from_map,
    quantile_threshold,
)
from .config import config_fingerprint, resolve_output_dir, save_json
from .data import build_mvtec_records
from .localization import combine_image_scores, generate_window_boxes
from .localization_metrics import (
    evaluate_localization_image,
    summarize_localization_rows,
)
from .localization_protocol import validate_tiled_config
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
from .position_calibration import (
    PositionCalibration,
    adjust_maps_by_position_threshold,
    fit_position_threshold_map,
    position_threshold_diagnostics,
)
from .tiled_pipeline import (
    TiledCalibration,
    _extract_window_batch_features,
    _global_calibration_maps,
    _local_maps_from_raw_scores,
    _tiled_loader,
    calibrate_tiled_category,
    prepare_local_category,
)
from .visualization import save_result_figure


@dataclass
class PositionTiledCalibration:
    base: TiledCalibration
    position: PositionCalibration
    image_threshold: float
    normal_combined_image_positive_rate: float


def _position_config(config: dict[str, Any]) -> dict[str, Any]:
    values = config.get("position_calibration", {})
    if not bool(values.get("enabled", False)):
        raise ValueError("position_calibration.enabled must be true")
    rho = float(values.get("rho", -1.0))
    if not 0.0 <= rho <= 1.0:
        raise ValueError("position_calibration.rho must be in [0, 1]")
    quantile = float(
        values.get(
            "quantile",
            config["calibration"]["pixel_image_quantile"],
        )
    )
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("position_calibration.quantile must be in [0, 1]")
    return values


def _category_dir(output_dir: Path, category: str) -> Path:
    path = output_dir / category
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cached_local_calibration_maps(
    category: str,
    config: dict[str, Any],
    output_dir: Path,
    calibration: CategoryCalibration,
    device: str | torch.device,
) -> tuple[torch.Tensor, dict[str, object]]:
    artifact_dir = _category_dir(output_dir, category) / "artifacts"
    banks, _ = load_memory_banks(artifact_dir / "local_memory_bank.pt")
    payload = torch.load(
        artifact_dir / "local_calibration_features.pt", map_location="cpu"
    )
    features = {
        int(layer): values.float()
        for layer, values in payload["features"].items()
    }
    image_count = int(next(iter(features.values())).shape[0])
    window_count = int(next(iter(features.values())).shape[1])
    flattened = {
        layer: values.reshape(
            image_count * window_count,
            values.shape[2],
            values.shape[3],
        )
        for layer, values in features.items()
    }
    raw_scores = _raw_layer_scores(
        flattened,
        payload["patch_coordinates"].float(),
        banks,
        {
            layer: calibration.layers[layer].spatial_weight
            for layer in flattened
        },
        config,
        device,
    )
    _, evaluation_maps = _local_maps_from_raw_scores(
        raw_scores,
        calibration.layers,
        image_count,
        window_count,
        config,
    )
    return evaluation_maps, payload


def fit_position_tiled_calibration(
    category: str,
    config: dict[str, Any],
    output_dir: Path,
    base: TiledCalibration,
    device: str | torch.device,
) -> PositionTiledCalibration:
    values = _position_config(config)
    artifact_dir = _category_dir(output_dir, category) / "artifacts"
    local_maps, local_payload = _cached_local_calibration_maps(
        category,
        config,
        output_dir,
        base.local_calibration,
        device,
    )
    fit_indices = torch.tensor(
        local_payload["threshold_fit_indices"], dtype=torch.long
    )
    validation_indices = torch.tensor(
        local_payload["normal_validation_indices"], dtype=torch.long
    )
    fit_maps = local_maps.index_select(0, fit_indices)
    validation_maps = local_maps.index_select(0, validation_indices)
    scalar_threshold = float(base.local_calibration.pixel_threshold)
    threshold_map = fit_position_threshold_map(
        fit_maps,
        scalar_threshold=scalar_threshold,
        quantile=float(
            values.get(
                "quantile",
                config["calibration"]["pixel_image_quantile"],
            )
        ),
        rho=float(values["rho"]),
    )
    diagnostics = position_threshold_diagnostics(
        validation_maps,
        threshold_map,
    )
    position = PositionCalibration(
        rho=float(values["rho"]),
        quantile=float(
            values.get(
                "quantile",
                config["calibration"]["pixel_image_quantile"],
            )
        ),
        scalar_threshold=scalar_threshold,
        threshold_map=threshold_map.cpu(),
        normal_pixel_positive_rate=diagnostics[
            "normal_pixel_positive_rate"
        ],
        normal_image_positive_rate=diagnostics[
            "normal_image_positive_rate"
        ],
        threshold_fit_images=len(local_payload["threshold_fit_indices"]),
        normal_validation_images=len(
            local_payload["normal_validation_indices"]
        ),
    )
    position.save(artifact_dir)

    adjusted_fit_maps = adjust_maps_by_position_threshold(
        fit_maps,
        threshold_map,
        scalar_threshold,
        clamp_min_zero=bool(values.get("clamp_min_zero", True)),
    )
    adjusted_validation_maps = adjust_maps_by_position_threshold(
        validation_maps,
        threshold_map,
        scalar_threshold,
        clamp_min_zero=bool(values.get("clamp_min_zero", True)),
    )
    global_maps, global_payload = _global_calibration_maps(
        category,
        config,
        output_dir,
        base.global_calibration,
        device,
    )
    if list(local_payload["paths"]) != list(global_payload["paths"]):
        raise RuntimeError("Global and local calibration image order differs")
    global_fit_scores = image_score_from_map(
        global_maps.index_select(0, fit_indices),
        method=base.global_calibration.image_score_method,
        topk_fraction=base.global_calibration.image_topk_fraction,
    )
    local_fit_scores = image_score_from_map(
        adjusted_fit_maps,
        method=base.local_calibration.image_score_method,
        topk_fraction=base.local_calibration.image_topk_fraction,
    )
    combined_fit_scores = combine_image_scores(
        global_fit_scores,
        local_fit_scores,
        method=base.image_score_merge,
    )
    image_threshold = quantile_threshold(
        combined_fit_scores.numpy(),
        float(config["calibration"]["image_quantile"]),
    )
    global_validation_scores = image_score_from_map(
        global_maps.index_select(0, validation_indices),
        method=base.global_calibration.image_score_method,
        topk_fraction=base.global_calibration.image_topk_fraction,
    )
    local_validation_scores = image_score_from_map(
        adjusted_validation_maps,
        method=base.local_calibration.image_score_method,
        topk_fraction=base.local_calibration.image_topk_fraction,
    )
    combined_validation_scores = combine_image_scores(
        global_validation_scores,
        local_validation_scores,
        method=base.image_score_merge,
    )
    normal_combined_rate = float(
        (combined_validation_scores >= image_threshold).float().mean().item()
    )
    calibration = PositionTiledCalibration(
        base=base,
        position=position,
        image_threshold=image_threshold,
        normal_combined_image_positive_rate=normal_combined_rate,
    )
    save_json(
        {
            "category": category,
            "mode": "tiled_position_calibrated",
            "base_tiled_calibration": {
                "global": base.global_calibration.to_dict(),
                "local": base.local_calibration.to_dict(),
                "image_score_merge": base.image_score_merge,
            },
            "position_calibration": position.summary(),
            "combined_image_threshold": image_threshold,
            "normal_combined_image_positive_rate": normal_combined_rate,
        },
        artifact_dir / "position_tiled_calibration.json",
    )
    return calibration


def evaluate_position_tiled_category(
    category: str,
    config: dict[str, Any],
    extractor: CLIPVisionFeatureExtractor,
    output_dir: Path,
    calibration: PositionTiledCalibration,
    device: str | torch.device,
) -> dict[str, Any]:
    localization = validate_tiled_config(config)
    position_config = _position_config(config)
    category_dir = _category_dir(output_dir, category)
    artifact_dir = category_dir / "artifacts"
    global_banks, _ = load_memory_banks(artifact_dir / "memory_bank.pt")
    local_banks, local_metadata = load_memory_banks(
        artifact_dir / "local_memory_bank.pt"
    )
    _, test_records = build_mvtec_records(config["data"]["root"], category)
    loader = _tiled_loader(test_records, config, include_mask=True)
    base = calibration.base
    scalar_threshold = calibration.position.scalar_threshold
    threshold_map = calibration.position.threshold_map.float()

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
                layer: base.global_calibration.layers[layer].spatial_weight
                for layer in global_features
            },
            config,
            device,
        )
        global_maps = _configured_map_outputs(
            global_raw_scores,
            base.global_calibration.layers,
            config,
        ).anomaly_maps

        local_features = _extract_window_batch_features(
            extractor,
            batch["windows"],
            int(localization["window_batch_size"]),
        )
        image_count = int(next(iter(local_features.values())).shape[0])
        window_count = int(next(iter(local_features.values())).shape[1])
        flattened = {
            layer: values.reshape(
                image_count * window_count,
                values.shape[2],
                values.shape[3],
            )
            for layer, values in local_features.items()
        }
        local_raw_scores = _raw_layer_scores(
            flattened,
            patch_coordinates,
            local_banks,
            {
                layer: base.local_calibration.layers[layer].spatial_weight
                for layer in flattened
            },
            config,
            device,
        )
        raw_canvas_maps, raw_maps = _local_maps_from_raw_scores(
            local_raw_scores,
            base.local_calibration.layers,
            image_count,
            window_count,
            config,
        )
        maps = adjust_maps_by_position_threshold(
            raw_maps,
            threshold_map,
            scalar_threshold,
            clamp_min_zero=bool(
                position_config.get("clamp_min_zero", True)
            ),
        )
        global_image_scores = image_score_from_map(
            global_maps,
            method=base.global_calibration.image_score_method,
            topk_fraction=base.global_calibration.image_topk_fraction,
        )
        local_image_scores = image_score_from_map(
            maps,
            method=base.local_calibration.image_score_method,
            topk_fraction=base.local_calibration.image_topk_fraction,
        )
        batch_image_scores = combine_image_scores(
            global_image_scores,
            local_image_scores,
            method=base.image_score_merge,
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
                threshold=scalar_threshold,
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
                        "raw_map": raw_maps[index].numpy(),
                        "raw_canvas_map": raw_canvas_maps[index].numpy(),
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
        calibrated_threshold=scalar_threshold,
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
    position_summary = calibration.position.summary()
    result: dict[str, Any] = {
        "category": category,
        **{
            f"localization_{key}": value
            for key, value in localization_summary.items()
        },
        "localization_mode": "tiled_position_calibrated",
        "position_calibration_rho": calibration.position.rho,
        "position_calibration_quantile": calibration.position.quantile,
        "position_threshold_min": position_summary["threshold_map_min"],
        "position_threshold_median": position_summary[
            "threshold_map_median"
        ],
        "position_threshold_max": position_summary["threshold_map_max"],
        "layer_fusion": str(config["inference"].get("layer_fusion", "mean")),
        **flatten_metrics("image", image_metrics),
        **flatten_metrics("pixel", pixel_metrics),
        "image_threshold_calibrated": calibration.image_threshold,
        "pixel_threshold_calibrated": scalar_threshold,
        "pixel_threshold_method": "position_shrunk_image_quantile",
        "normal_pixel_positive_rate": (
            calibration.position.normal_pixel_positive_rate
        ),
        "normal_image_positive_rate": (
            calibration.position.normal_image_positive_rate
        ),
        "normal_combined_image_score_positive_rate": (
            calibration.normal_combined_image_positive_rate
        ),
        "threshold_fit_images": calibration.position.threshold_fit_images,
        "normal_validation_images": (
            calibration.position.normal_validation_images
        ),
        "normalization_fit_images": (
            base.local_calibration.normalization_fit_images
        ),
        "normal_diagnostic_mode": "held_out_position_threshold_validation",
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
    color_max = max(scalar_threshold * 2.5, 1.0e-6)
    if bool(config["inference"]["save_visualizations"]):
        limit = int(config["inference"]["max_visualizations_per_category"])
        for index, sample in enumerate(ranked[:limit]):
            stem = Path(str(sample["path"])).stem
            anomaly_map = np.asarray(sample["map"])
            save_result_figure(
                image=np.asarray(sample["image"]),
                ground_truth=np.asarray(sample["mask"]),
                anomaly_map=anomaly_map,
                prediction=anomaly_map >= scalar_threshold,
                output_path=category_dir
                / "visualizations"
                / f"{index:03d}_{sample['defect_type']}_{stem}.png",
                title=(
                    f"{category}/{sample['defect_type']} position "
                    f"rho={calibration.position.rho:.2f}"
                ),
                color_max=color_max,
            )
    if bool(config.get("diagnostics", {}).get("enabled", False)):
        limit = int(
            config["diagnostics"].get("max_samples_per_category", 20)
        )
        directory = category_dir / "diagnostic_maps"
        directory.mkdir(parents=True, exist_ok=True)
        canvas_size = int(localization["canvas_size"])
        canvas_threshold = F.interpolate(
            threshold_map[None, None],
            size=(canvas_size, canvas_size),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        for index, sample in enumerate(ranked[:limit]):
            stem = Path(str(sample["path"])).stem
            prefix = f"{index:03d}_{sample['defect_type']}_{stem}"
            raw_canvas = torch.from_numpy(
                np.asarray(sample["raw_canvas_map"], dtype=np.float32)
            )
            adjusted_canvas = (
                raw_canvas - canvas_threshold + scalar_threshold
            )
            if bool(position_config.get("clamp_min_zero", True)):
                adjusted_canvas = adjusted_canvas.clamp_min(0.0)
            np.save(
                directory / f"{prefix}_raw_local_canvas.npy",
                raw_canvas.numpy(),
            )
            np.save(
                directory / f"{prefix}_position_threshold.npy",
                canvas_threshold.numpy(),
            )
            np.save(
                directory / f"{prefix}_adjusted_local_canvas.npy",
                adjusted_canvas.numpy(),
            )
            save_result_figure(
                image=np.asarray(sample["canvas_image"]),
                ground_truth=np.asarray(sample["canvas_mask"]),
                anomaly_map=adjusted_canvas.numpy(),
                prediction=adjusted_canvas.numpy() >= scalar_threshold,
                output_path=directory / f"{prefix}_position_diagnostic.png",
                title=(
                    f"{category}/{sample['defect_type']} position "
                    f"rho={calibration.position.rho:.2f}"
                ),
                color_max=color_max,
            )
    return result


def run_position_tiled_experiment(
    config: dict[str, Any],
    device: str | torch.device,
) -> list[dict[str, Any]]:
    validate_tiled_config(config)
    _position_config(config)
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
            "checkpoint": str(config["model"]["checkpoint"]),
            "layers": list(config["model"]["active_layers"]),
            "localization_mode": "tiled_position_calibrated",
            "canvas_size": int(localization["canvas_size"]),
            "window_size": int(localization["window_size"]),
            "stride": int(localization["stride"]),
            "window_count": len(boxes),
            "evaluation_size": int(config["data"]["image_size"]),
            "position_calibration": config["position_calibration"],
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
    global_tau = compute_layer_tau(global_reliabilities, quantile)
    local_tau = compute_layer_tau(local_reliabilities, quantile)

    results: list[dict[str, Any]] = []
    for category in config["data"]["categories"]:
        print(f"[calibrate-global] {category}", flush=True)
        global_calibration = calibrate_category(
            category, config, output_dir, global_tau, device
        )
        print(f"[calibrate-local] {category}", flush=True)
        base = calibrate_tiled_category(
            category,
            config,
            output_dir,
            global_calibration,
            local_tau,
            device,
        )
        print(f"[calibrate-position] {category}", flush=True)
        calibration = fit_position_tiled_calibration(
            category,
            config,
            output_dir,
            base,
            device,
        )
        print(f"[evaluate-position] {category}", flush=True)
        results.append(
            evaluate_position_tiled_category(
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
            "pixel_threshold_method": "position_shrunk_image_quantile",
            "localization_mode": "tiled_position_calibrated",
            "position_calibration_rho": float(
                config["position_calibration"]["rho"]
            ),
            "position_calibration_quantile": float(
                config["position_calibration"]["quantile"]
            ),
            "layer_fusion": str(
                config["inference"].get("layer_fusion", "mean")
            ),
            "window_count": len(boxes),
            "config_fingerprint": config_fingerprint(config),
        },
        completion_path,
    )
    return results
