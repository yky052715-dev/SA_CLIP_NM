from __future__ import annotations

import csv
import json
import math
import platform
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader

from .backbone import CLIPVisionFeatureExtractor, make_patch_coordinates
from .calibration import (
    CategoryCalibration,
    LayerCalibration,
    calibrate_scores,
    image_score_from_map,
    normal_threshold_diagnostics,
    pixel_threshold_from_maps,
    quantile_threshold,
    robust_location_scale,
)
from .config import config_fingerprint, resolve_output_dir, save_json
from .data import (
    MVTecImageDataset,
    build_records,
    collate_records,
    split_calibration_records,
    split_normal_records,
)
from .localization_diagnostics import save_localization_diagnostics
from .localization_metrics import (
    evaluate_localization_image,
    summarize_localization_rows,
)
from .map_refinement import (
    AnomalyMapOutputs,
    reshape_patch_scores,
    upsample_anomaly_maps,
)
from .memory import (
    LayerMemoryBank,
    build_layer_memory,
    load_memory_banks,
    save_memory_banks,
)
from .metrics import evaluate_binary_scores, flatten_metrics
from .multiscale import (
    aggregate_features_by_scale,
    fuse_multiscale_raw_scores,
    multiscale_key_name,
    normalized_scale_weights,
    parse_multiscale_key,
    validate_scales,
)
from .retrieval import batched_layer_scores
from .visualization import save_result_figure


def _multiscale_config(config: dict[str, Any]) -> dict[str, Any]:
    multiscale = dict(config.get("multiscale", {}))
    scales = validate_scales(multiscale.get("scales", [1]))
    weights = normalized_scale_weights(scales, multiscale.get("scale_weights"))
    padding_mode = str(multiscale.get("padding_mode", "replicate"))
    if padding_mode != "replicate":
        raise ValueError("multiscale.padding_mode must be replicate")
    return {
        "enabled": bool(multiscale.get("enabled", True)),
        "scales": scales,
        "scale_weights": weights,
        "padding_mode": padding_mode,
    }


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


def _extract_batch_features(
    extractor: CLIPVisionFeatureExtractor,
    batch: dict[str, object],
) -> dict[int, torch.Tensor]:
    return {
        layer: values.cpu()
        for layer, values in extractor.extract(batch["image"]).items()
    }


def _category_dir(output_dir: Path, category: str) -> Path:
    path = output_dir / category
    path.mkdir(parents=True, exist_ok=True)
    return path


def _raw_multiscale_scores(
    features_by_key: Mapping[int, torch.Tensor],
    patch_coordinates: torch.Tensor,
    banks: Mapping[int, LayerMemoryBank],
    config: dict[str, Any],
    device: str | torch.device,
) -> dict[int, torch.Tensor]:
    return {
        key: batched_layer_scores(
            image_features=features_by_key[key],
            patch_coordinates=patch_coordinates,
            bank=banks[key],
            spatial_weight=0.0,
            query_chunk_size=int(config["retrieval"]["query_chunk_size"]),
            bank_chunk_size=int(config["retrieval"]["bank_chunk_size"]),
            device=device,
        )
        for key in sorted(features_by_key)
    }

def _scale_memory_ratio(
    config: dict[str, Any],
    scales: list[int],
    scale: int,
) -> float:
    ratios = config.get("multiscale", {}).get("scale_memory_ratios")
    if ratios is None:
        return float(config["memory"]["ratio"])
    if isinstance(ratios, Mapping):
        if scale in ratios:
            value = ratios[scale]
        elif str(scale) in ratios:
            value = ratios[str(scale)]
        else:
            raise KeyError(f"Missing memory ratio for scale {scale}")
        return float(value)
    values = [float(value) for value in ratios]
    if len(values) != len(scales):
        raise ValueError("scale_memory_ratios length must match scales")
    return float(values[scales.index(scale)])

def _multiscale_map_outputs(
    raw_scores: Mapping[int, torch.Tensor],
    layer_calibrations: Mapping[int, LayerCalibration],
    config: dict[str, Any],
) -> AnomalyMapOutputs:
    multiscale = _multiscale_config(config)
    calibrated: dict[int, torch.Tensor] = {}
    for key in sorted(raw_scores):
        if key not in layer_calibrations:
            raise KeyError(f"Missing calibration for {multiscale_key_name(key)}")
        parameters = layer_calibrations[key]
        calibrated[key] = calibrate_scores(
            raw_scores[key],
            median=parameters.median,
            mad=parameters.mad,
            clamp_min_zero=bool(config["calibration"]["clamp_min_zero"]),
        )
    fused_scores = fuse_multiscale_raw_scores(
        calibrated,
        scales=multiscale["scales"],
        scale_weights=multiscale["scale_weights"],
        layer_weights=config["inference"].get("layer_weights"),
    )
    fused_patch_map = reshape_patch_scores(fused_scores)
    anomaly_maps = upsample_anomaly_maps(
        fused_patch_map,
        output_size=int(config["data"]["image_size"]),
        mode=str(config["inference"].get("upsample_mode", "bilinear")),
        gaussian_sigma=float(config["inference"].get("gaussian_sigma", 0.0)),
    )
    if float(config["inference"].get("gaussian_sigma", 0.0)) > 0:
        # upsample_anomaly_maps already handles Gaussian smoothing. This branch
        # only exists to keep the no-op path explicit in generated artifacts.
        pass
    return AnomalyMapOutputs(
        layer_patch_maps={
            key: reshape_patch_scores(values).cpu()
            for key, values in calibrated.items()
        },
        fused_patch_map=fused_patch_map.cpu(),
        anomaly_maps=anomaly_maps,
    )


def prepare_category(
    category: str,
    config: dict[str, Any],
    extractor: CLIPVisionFeatureExtractor,
    output_dir: Path,
) -> None:
    multiscale = _multiscale_config(config)
    category_dir = _category_dir(output_dir, category)
    artifact_dir = category_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    train_records, _ = build_records(config["data"]["root"], category, config["data"])
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
    first_layer = next(iter(memory_features))
    patch_count = memory_features[first_layer].shape[1]
    patch_coordinates = make_patch_coordinates(patch_count)

    memory_by_key = aggregate_features_by_scale(
        memory_features,
        scales=multiscale["scales"],
        padding_mode=multiscale["padding_mode"],
    )
    calibration_by_key = aggregate_features_by_scale(
        calibration_features,
        scales=multiscale["scales"],
        padding_mode=multiscale["padding_mode"],
    )

    banks: dict[int, LayerMemoryBank] = {}
    for key, values in memory_by_key.items():
        layer, scale = parse_multiscale_key(key)
        ratio = _scale_memory_ratio(config, multiscale["scales"], scale)
        banks[key] = build_layer_memory(
            image_features=values,
            patch_coordinates=patch_coordinates,
            ratio=ratio,
            method=str(config["memory"]["sampling"]),
            seed=int(config["experiment"]["seed"]) + layer * 100 + scale,
            projection_dim=int(config["memory"]["projection_dim"]),
            max_candidates=int(config["memory"]["max_candidates"]),
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

    save_memory_banks(
        banks,
        metadata={
            "category": category,
            "memory_paths": memory_paths,
            "checkpoint": config["model"]["checkpoint"],
            "layers": list(extractor.layers),
            "scales": list(multiscale["scales"]),
            "scale_weights": multiscale["scale_weights"],
            "token_norm": config["model"]["token_norm"],
            "image_size": int(config["data"]["image_size"]),
            "resize_mode": config["data"]["resize_mode"],
            "sampling": config["memory"]["sampling"],
            "ratio": float(config["memory"]["ratio"]),
        },
        path=artifact_dir / "multiscale_memory_bank.pt",
    )
    torch.save(
        {
            "features": {
                int(key): values.half()
                for key, values in calibration_by_key.items()
            },
            "paths": calibration_paths,
            "threshold_fit_indices": threshold_fit_indices,
            "normal_validation_indices": normal_validation_indices,
            "patch_coordinates": patch_coordinates,
        },
        artifact_dir / "multiscale_calibration_features.pt",
    )


def calibrate_category(
    category: str,
    config: dict[str, Any],
    output_dir: Path,
    device: str | torch.device,
) -> CategoryCalibration:
    artifact_dir = _category_dir(output_dir, category) / "artifacts"
    banks, _ = load_memory_banks(artifact_dir / "multiscale_memory_bank.pt")
    payload = torch.load(
        artifact_dir / "multiscale_calibration_features.pt",
        map_location="cpu",
    )
    calibration_features = {
        int(key): values.float()
        for key, values in payload["features"].items()
    }
    patch_coordinates = payload["patch_coordinates"].float()
    raw_scores = _raw_multiscale_scores(
        calibration_features,
        patch_coordinates,
        banks,
        config,
        device,
    )
    layer_calibrations: dict[int, LayerCalibration] = {}
    for key, values in raw_scores.items():
        median, mad = robust_location_scale(
            values,
            epsilon=float(config["calibration"]["mad_epsilon"]),
        )
        layer_calibrations[key] = LayerCalibration(
            median=median,
            mad=mad,
            reliability=1.0,
            spatial_weight=0.0,
        )

    calibration_outputs = _multiscale_map_outputs(
        raw_scores,
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
    threshold_fit_maps = calibration_maps.index_select(0, threshold_fit_indices)
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
        config["calibration"].get("pixel_threshold_method", "image_max_quantile")
    )
    pixel_quantile = float(config["calibration"].get("pixel_quantile", 0.995))
    pixel_image_quantile = float(
        config["calibration"].get("pixel_image_quantile", 0.95)
    )
    pixel_topk_fraction = float(
        config["calibration"].get("pixel_topk_fraction", 0.01)
    )
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
    )
    save_json(
        {
            **calibration.to_dict(),
            "layers": {
                multiscale_key_name(int(key)): {
                    "median": value.median,
                    "mad": value.mad,
                    "reliability": value.reliability,
                    "spatial_weight": value.spatial_weight,
                }
                for key, value in calibration.layers.items()
            },
        },
        artifact_dir / "multiscale_calibration.json",
    )
    return calibration


def evaluate_category(
    category: str,
    config: dict[str, Any],
    extractor: CLIPVisionFeatureExtractor,
    output_dir: Path,
    calibration: CategoryCalibration,
    device: str | torch.device,
) -> dict[str, Any]:
    multiscale = _multiscale_config(config)
    category_dir = _category_dir(output_dir, category)
    artifact_dir = category_dir / "artifacts"
    banks, _ = load_memory_banks(artifact_dir / "multiscale_memory_bank.pt")
    _, test_records = build_records(config["data"]["root"], category, config["data"])
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
        base_features = _extract_batch_features(extractor, batch)
        features_by_key = aggregate_features_by_scale(
            base_features,
            scales=multiscale["scales"],
            padding_mode=multiscale["padding_mode"],
        )
        actual_patch_count = next(iter(base_features.values())).shape[1]
        patch_coordinates = make_patch_coordinates(actual_patch_count)
        raw_scores = _raw_multiscale_scores(
            features_by_key,
            patch_coordinates,
            banks,
            config,
            device,
        )
        map_outputs = _multiscale_map_outputs(
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
                            multiscale_key_name(layer): values[index].numpy()
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
            "w",
            encoding="utf-8",
            newline="",
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
        "multiscale_scales": ",".join(str(value) for value in multiscale["scales"]),
        "multiscale_scale_weights": ",".join(
            f"{multiscale['scale_weights'][scale]:.6g}"
            for scale in multiscale["scales"]
        ),
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
        ranked = sorted(
            samples,
            key=lambda item: float(item["image_score"]),
            reverse=True,
        )[:limit]
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
                title=(
                    f"{category}/{sample['defect_type']} "
                    f"score={sample['image_score']:.3f}"
                ),
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
        "w",
        encoding="utf-8",
        newline="",
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
        "pixel_AUROC",
        "pixel_F1_calibrated",
        "pixel_F1_oracle",
        "localization_overseg_anomaly_macro",
        "localization_recall_anomaly_macro",
        "localization_small_defect_recall_macro",
        "localization_test_normal_image_positive_rate",
        "inference_ms_per_image",
        "memory_entries",
    ]
    lines = [
        "| " + " | ".join(table_keys) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(table_keys) - 1)) + "|",
    ]
    for row in results:
        values = [str(row["category"])] + [
            (
                "nan"
                if row.get(key) is None or not np.isfinite(float(row[key]))
                else f"{float(row[key]):.6f}"
            )
            for key in table_keys[1:]
        ]
        lines.append("| " + " | ".join(values) + " |")
    mean_values = ["mean"] + [
        (
            "nan"
            if key not in averages or not np.isfinite(float(averages[key]))
            else f"{averages[key]:.6f}"
        )
        for key in table_keys[1:]
    ]
    lines.append("| " + " | ".join(mean_values) + " |")
    (output_dir / "metrics_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run_multiscale_experiment(
    config: dict[str, Any],
    device: str | torch.device,
) -> list[dict[str, Any]]:
    multiscale = _multiscale_config(config)
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
            "multiscale_scales": multiscale["scales"],
            "multiscale_scale_weights": multiscale["scale_weights"],
            "multiscale_padding_mode": multiscale["padding_mode"],
            "hidden_state_definition": (
                "hidden_states[0] is embedding output; hidden_states[k] is output "
                "after encoder block k; CLS is removed"
            ),
        },
        output_dir / "environment_and_feature_protocol.json",
    )

    for category in config["data"]["categories"]:
        print(f"[prepare-multiscale] {category}", flush=True)
        prepare_category(category, config, extractor, output_dir)

    results = []
    for category in config["data"]["categories"]:
        print(f"[calibrate-multiscale] {category}", flush=True)
        calibration = calibrate_category(category, config, output_dir, device)
        print(f"[evaluate-multiscale] {category}", flush=True)
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
            "spatial_mode": str(config["retrieval"].get("spatial_mode", "none")),
            "multiscale_scales": multiscale["scales"],
            "multiscale_scale_weights": multiscale["scale_weights"],
            "config_fingerprint": config_fingerprint(config),
        },
        completion_path,
    )
    return results


