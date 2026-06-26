from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SA_SRC = ROOT / "SA_CLIP_NM" / "src"
if str(SA_SRC) not in sys.path:
    sys.path.insert(0, str(SA_SRC))

from sa_clip_nm.calibration import (  # noqa: E402
    LayerCalibration,
    adaptive_area_pixel_threshold_from_maps,
    calibrate_scores,
    image_score_from_map,
    normal_threshold_diagnostics,
    pixel_threshold_from_maps,
    quantile_threshold,
    robust_location_scale,
)
from sa_clip_nm.config import save_json  # noqa: E402
from sa_clip_nm.data import (  # noqa: E402
    build_records,
    image_to_tensor,
    mask_to_tensor,
    split_calibration_records,
    split_normal_records,
    transform_pil,
)
from sa_clip_nm.localization_metrics import (  # noqa: E402
    evaluate_localization_image,
    prediction_from_anomaly_map,
    summarize_localization_rows,
)
from sa_clip_nm.metrics import evaluate_binary_scores, flatten_metrics  # noqa: E402


DINO_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
DINO_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
DEFAULT_DEV5 = ["bottle", "metal_nut", "grid", "leather", "screw"]


def dino_image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - DINO_MEAN) / DINO_STD


class DINOHighResolutionDataset(Dataset):
    def __init__(
        self,
        records,
        input_size: int,
        evaluation_size: int,
        resize_mode: str,
        include_mask: bool,
    ) -> None:
        self.records = list(records)
        self.input_size = int(input_size)
        self.evaluation_size = int(evaluation_size)
        self.resize_mode = str(resize_mode)
        self.include_mask = bool(include_mask)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        with Image.open(record.path) as handle:
            original = handle.convert("RGB")
        model_image = transform_pil(
            original,
            self.input_size,
            self.resize_mode,
            is_mask=False,
        )
        display_image = transform_pil(
            original,
            self.evaluation_size,
            self.resize_mode,
            is_mask=False,
        )
        item: dict[str, object] = {
            "image": dino_image_to_tensor(model_image),
            "label": int(record.label),
            "path": str(record.path),
            "defect_type": str(record.defect_type),
            "display_image": np.asarray(display_image, dtype=np.uint8),
        }
        if self.include_mask:
            if record.mask_path is None:
                mask = Image.new("L", (self.evaluation_size, self.evaluation_size), color=0)
            else:
                with Image.open(record.mask_path) as handle:
                    mask = transform_pil(
                        handle,
                        self.evaluation_size,
                        self.resize_mode,
                        is_mask=True,
                    )
            item["mask"] = mask_to_tensor(mask)
        return item


def collate_dino_records(batch: list[dict[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {
        "image": torch.stack([item["image"] for item in batch]),
        "label": torch.tensor([int(item["label"]) for item in batch], dtype=torch.int64),
        "path": [str(item["path"]) for item in batch],
        "defect_type": [str(item["defect_type"]) for item in batch],
        "display_image": [item["display_image"] for item in batch],
    }
    if "mask" in batch[0]:
        output["mask"] = torch.stack([item["mask"] for item in batch])
    return output


def loader(records, args: argparse.Namespace, include_mask: bool) -> DataLoader:
    dataset = DINOHighResolutionDataset(
        records,
        input_size=args.input_size,
        evaluation_size=args.evaluation_size,
        resize_mode=args.resize_mode,
        include_mask=include_mask,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_dino_records,
    )


def load_dinov2(model_name: str, device: torch.device, hub_dir: str | None = None):
    if hub_dir is not None:
        torch.hub.set_dir(hub_dir)
    model = torch.hub.load("facebookresearch/dinov2", model_name, trust_repo=True)
    model.eval().to(device)
    return model


@torch.inference_mode()
def extract_patch_tokens(model, images: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, int]:
    images = images.to(device, non_blocking=True)
    if device.type == "cuda":
        with torch.cuda.amp.autocast(dtype=torch.float16):
            output = model.forward_features(images)
    else:
        output = model.forward_features(images)
    tokens = output["x_norm_patchtokens"] if isinstance(output, dict) else output
    tokens = F.normalize(tokens.float(), dim=-1)
    patch_count = int(tokens.shape[1])
    grid_side = int(round(math.sqrt(patch_count)))
    if grid_side * grid_side != patch_count:
        raise RuntimeError(f"DINOv2 patch token count is not square: {patch_count}")
    return tokens, grid_side


@torch.inference_mode()
def extract_features(model, data_loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, list[str], int]:
    chunks: list[torch.Tensor] = []
    paths: list[str] = []
    grid_side = 0
    for batch in tqdm(data_loader, desc="extract features"):
        features, grid_side = extract_patch_tokens(model, batch["image"], device)
        chunks.append(features.cpu())
        paths.extend(batch["path"])
    return torch.cat(chunks, dim=0), paths, grid_side


def build_memory_bank(features: torch.Tensor, max_bank_size: int, seed: int) -> torch.Tensor:
    flat = features.reshape(-1, features.shape[-1]).float().cpu()
    if max_bank_size > 0 and flat.shape[0] > max_bank_size:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        indices = torch.randperm(flat.shape[0], generator=generator)[:max_bank_size]
        flat = flat[indices]
    return F.normalize(flat, dim=-1).contiguous()


@torch.inference_mode()
def nearest_cosine_distance(
    query_features: torch.Tensor,
    memory_bank: torch.Tensor,
    query_chunk_size: int,
    bank_chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    if query_features.ndim != 3:
        raise ValueError("query_features must have shape [B, P, C]")
    b, p, c = query_features.shape
    flat = query_features.reshape(-1, c).float()
    memory_cpu = F.normalize(memory_bank.float().cpu(), dim=-1)
    scores: list[torch.Tensor] = []
    for query_start in range(0, flat.shape[0], query_chunk_size):
        query = flat[query_start: query_start + query_chunk_size].to(device)
        best = torch.full((query.shape[0],), -float("inf"), device=device)
        for bank_start in range(0, memory_cpu.shape[0], bank_chunk_size):
            bank = memory_cpu[bank_start: bank_start + bank_chunk_size].to(device)
            similarity = query @ bank.T
            best = torch.maximum(best, similarity.max(dim=1).values)
        scores.append((1.0 - best).cpu())
    return torch.cat(scores, dim=0).reshape(b, p)


def scores_to_maps(
    patch_scores: torch.Tensor,
    grid_side: int,
    output_size: int,
    layer_calibration: LayerCalibration | None = None,
    clamp_min_zero: bool = True,
) -> torch.Tensor:
    scores = patch_scores
    if layer_calibration is not None:
        scores = calibrate_scores(
            scores,
            median=layer_calibration.median,
            mad=layer_calibration.mad,
            clamp_min_zero=clamp_min_zero,
        )
    patch_maps = scores.reshape(scores.shape[0], 1, grid_side, grid_side)
    return F.interpolate(
        patch_maps,
        size=(output_size, output_size),
        mode="bilinear",
        align_corners=False,
    )[:, 0].cpu()


def threshold_from_calibration_maps(
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
        threshold, selected, candidates = adaptive_area_pixel_threshold_from_maps(
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
        return threshold, selected, candidates
    threshold = pixel_threshold_from_maps(
        threshold_fit_maps,
        method=method,
        pixel_quantile=float(args.pixel_quantile),
        image_quantile=float(args.pixel_image_quantile),
        topk_fraction=float(args.pixel_topk_fraction),
    )
    return threshold, None, None


def save_npz(path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **kwargs)


def run_category(args: argparse.Namespace, category: str, model, device: torch.device) -> dict[str, Any]:
    category_dir = Path(args.output_dir) / category
    artifact_dir = category_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    data_config = {
        "dataset": "mvtec",
        "image_size": int(args.input_size),
        "resize_mode": args.resize_mode,
    }
    train_records, test_records = build_records(args.data_root, category, data_config)
    memory_records, calibration_records = split_normal_records(
        train_records,
        calibration_fraction=float(args.calibration_fraction),
        seed=int(args.seed),
    )
    threshold_fit_records, normal_validation_records = split_calibration_records(
        calibration_records,
        threshold_fit_fraction=float(args.threshold_fit_fraction),
        seed=int(args.threshold_split_seed),
    )
    save_json(
        {
            "category": category,
            "memory": [str(record.path) for record in memory_records],
            "calibration": [str(record.path) for record in calibration_records],
            "threshold_fit": [str(record.path) for record in threshold_fit_records],
            "normal_validation": [str(record.path) for record in normal_validation_records],
        },
        artifact_dir / "split_manifest.json",
    )

    print(f"[DINO prepare] {category}", flush=True)
    memory_features, memory_paths, grid_side = extract_features(
        model,
        loader(memory_records, args, include_mask=False),
        device,
    )
    memory_bank = build_memory_bank(
        memory_features,
        max_bank_size=int(args.max_bank_size),
        seed=int(args.seed),
    )
    torch.save(
        {
            "memory_bank": memory_bank.half(),
            "memory_paths": memory_paths,
            "grid_side": grid_side,
            "model": args.model,
            "input_size": int(args.input_size),
            "evaluation_size": int(args.evaluation_size),
        },
        artifact_dir / "memory_bank.pt",
    )

    print(f"[DINO calibrate] {category}", flush=True)
    calibration_loader = loader(calibration_records, args, include_mask=False)
    calibration_features, calibration_paths, calibration_grid = extract_features(
        model,
        calibration_loader,
        device,
    )
    if calibration_grid != grid_side:
        raise RuntimeError("Calibration grid does not match memory grid")
    calibration_scores = nearest_cosine_distance(
        calibration_features,
        memory_bank,
        query_chunk_size=int(args.query_chunk_size),
        bank_chunk_size=int(args.bank_chunk_size),
        device=device,
    )
    median, mad = robust_location_scale(calibration_scores, epsilon=float(args.mad_epsilon))
    layer_calibration = LayerCalibration(
        median=median,
        mad=mad,
        reliability=1.0,
        spatial_weight=0.0,
    )
    calibration_maps = scores_to_maps(
        calibration_scores,
        grid_side=grid_side,
        output_size=int(args.evaluation_size),
        layer_calibration=layer_calibration,
        clamp_min_zero=bool(args.clamp_min_zero),
    )
    fit_path_set = {str(record.path) for record in threshold_fit_records}
    validation_path_set = {str(record.path) for record in normal_validation_records}
    threshold_fit_indices = np.asarray(
        [index for index, path in enumerate(calibration_paths) if path in fit_path_set],
        dtype=np.int64,
    )
    normal_validation_indices = np.asarray(
        [index for index, path in enumerate(calibration_paths) if path in validation_path_set],
        dtype=np.int64,
    )
    threshold_fit_maps = calibration_maps.index_select(0, torch.from_numpy(threshold_fit_indices))
    normal_validation_maps = calibration_maps.index_select(0, torch.from_numpy(normal_validation_indices))
    pixel_threshold, selected_quantile, adaptive_candidates = threshold_from_calibration_maps(
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
    save_json(
        {
            "category": category,
            "median": median,
            "mad": mad,
            "pixel_threshold": pixel_threshold,
            "image_threshold": image_threshold,
            "pixel_threshold_method": args.pixel_threshold_method,
            "pixel_image_quantile": selected_quantile if selected_quantile is not None else float(args.pixel_image_quantile),
            "adaptive_selected_pixel_image_quantile": selected_quantile,
            "adaptive_threshold_candidates": adaptive_candidates,
            "normal_pixel_positive_rate": normal_diag["normal_pixel_positive_rate"],
            "normal_image_positive_rate": normal_diag["normal_image_positive_rate"],
            "threshold_fit_images": int(len(threshold_fit_indices)),
            "normal_validation_images": int(len(normal_validation_indices)),
            "normalization_fit_images": int(len(calibration_paths)),
        },
        artifact_dir / "calibration.json",
    )
    save_npz(
        artifact_dir / "calibration_maps.npz",
        maps=calibration_maps.numpy().astype(np.float32),
        paths=np.asarray(calibration_paths, dtype=object),
        threshold_fit_indices=threshold_fit_indices,
        normal_validation_indices=normal_validation_indices,
    )

    print(f"[DINO evaluate] {category}", flush=True)
    test_loader = loader(test_records, args, include_mask=True)
    image_labels: list[int] = []
    image_scores_all: list[float] = []
    pixel_labels: list[np.ndarray] = []
    pixel_scores: list[np.ndarray] = []
    pixel_predictions: list[np.ndarray] = []
    localization_rows: list[dict[str, object]] = []
    all_maps: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    all_paths: list[str] = []
    all_defect_types: list[str] = []
    inference_seconds = 0.0
    min_component_area_pixels = int(
        round(float(args.min_component_area_fraction) * int(args.evaluation_size) ** 2)
    )
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for batch in tqdm(test_loader, desc=f"DINO eval {category}"):
        start = time.perf_counter()
        features, test_grid = extract_patch_tokens(model, batch["image"], device)
        if test_grid != grid_side:
            raise RuntimeError("Test grid does not match memory grid")
        scores = nearest_cosine_distance(
            features.cpu(),
            memory_bank,
            query_chunk_size=int(args.query_chunk_size),
            bank_chunk_size=int(args.bank_chunk_size),
            device=device,
        )
        maps = scores_to_maps(
            scores,
            grid_side=grid_side,
            output_size=int(args.evaluation_size),
            layer_calibration=layer_calibration,
            clamp_min_zero=bool(args.clamp_min_zero),
        )
        batch_image_scores = image_score_from_map(
            maps,
            method=str(args.image_score),
            topk_fraction=float(args.image_topk_fraction),
        )
        inference_seconds += time.perf_counter() - start

        masks = batch["mask"].numpy().astype(np.uint8)
        labels = batch["label"].numpy().astype(np.uint8)
        image_labels.extend(labels.tolist())
        image_scores_all.extend(batch_image_scores.numpy().tolist())
        pixel_labels.extend([mask.reshape(-1) for mask in masks])
        pixel_scores.extend([score.numpy().reshape(-1) for score in maps])
        for index in range(maps.shape[0]):
            anomaly_map = maps[index].numpy().astype(np.float32)
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
                    "path": str(batch["path"][index]),
                    "defect_type": str(batch["defect_type"][index]),
                    "label": int(labels[index]),
                    **row,
                }
            )
            all_maps.append(anomaly_map)
            all_masks.append(masks[index])
            all_paths.append(str(batch["path"][index]))
            all_defect_types.append(str(batch["defect_type"][index]))

    image_metrics = evaluate_binary_scores(
        np.asarray(image_labels),
        np.asarray(image_scores_all),
        calibrated_threshold=image_threshold,
        compute_oracle=True,
    )
    pixel_metrics = evaluate_binary_scores(
        np.concatenate(pixel_labels),
        np.concatenate(pixel_scores),
        calibrated_threshold=pixel_threshold,
        compute_oracle=True,
        calibrated_predictions=np.concatenate(pixel_predictions),
    )
    localization_summary = summarize_localization_rows(localization_rows)
    save_json(localization_summary, category_dir / "localization_metrics.json")
    if localization_rows:
        with (category_dir / "per_image_localization_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(localization_rows[0].keys()))
            writer.writeheader()
            writer.writerows(localization_rows)

    peak_memory_mb = 0.0
    if torch.cuda.is_available() and device.type == "cuda":
        peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2))
    result: dict[str, Any] = {
        "category": category,
        "backbone": "dinov2",
        "model": args.model,
        "input_image_size": int(args.input_size),
        "evaluation_size": int(args.evaluation_size),
        "patch_grid_side": int(grid_side),
        "patch_count": int(grid_side * grid_side),
        "feature_dim": int(memory_bank.shape[1]),
        "max_bank_size": int(args.max_bank_size),
        **{f"localization_{key}": value for key, value in localization_summary.items()},
        **flatten_metrics("image", image_metrics),
        **flatten_metrics("pixel", pixel_metrics),
        "image_threshold_calibrated": image_threshold,
        "pixel_threshold_calibrated": pixel_threshold,
        "pixel_threshold_method": args.pixel_threshold_method,
        "adaptive_selected_pixel_image_quantile": selected_quantile,
        "normal_pixel_positive_rate": normal_diag["normal_pixel_positive_rate"],
        "normal_image_positive_rate": normal_diag["normal_image_positive_rate"],
        "postprocess_min_component_area_fraction": float(args.min_component_area_fraction),
        "postprocess_min_component_area_pixels": min_component_area_pixels,
        "threshold_fit_images": int(len(threshold_fit_indices)),
        "normal_validation_images": int(len(normal_validation_indices)),
        "normalization_fit_images": int(len(calibration_paths)),
        "inference_seconds_total": inference_seconds,
        "inference_ms_per_image": 1000.0 * inference_seconds / len(test_records),
        "gpu_peak_memory_mb": peak_memory_mb,
        "test_images": len(test_records),
        "memory_entries": int(memory_bank.shape[0]),
    }
    save_json(result, category_dir / "metrics.json")
    save_npz(
        artifact_dir / "test_maps.npz",
        maps=np.stack(all_maps).astype(np.float32),
        masks=np.stack(all_masks).astype(np.uint8),
        labels=np.asarray(image_labels, dtype=np.uint8),
        paths=np.asarray(all_paths, dtype=object),
        defect_types=np.asarray(all_defect_types, dtype=object),
    )
    return result


def write_summary(results: list[dict[str, Any]], output_dir: Path) -> None:
    if not results:
        return
    fieldnames = list(results[0].keys())
    with (output_dir / "metrics_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    numeric_keys = [
        key
        for key, value in results[0].items()
        if key != "category" and isinstance(value, (int, float))
    ]
    means = {
        key: float(np.nanmean([float(row[key]) for row in results]))
        for key in numeric_keys
    }
    save_json({"categories": results, "mean": means}, output_dir / "metrics_summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DINOv2 MVTec localization protocol")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--categories", nargs="+", default=DEFAULT_DEV5)
    parser.add_argument("--output-dir", default="outputs/dinov2_mvtec_localization/dev5")
    parser.add_argument("--model", default="dinov2_vits14")
    parser.add_argument("--hub-dir", default=None)
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--evaluation-size", type=int, default=448)
    parser.add_argument("--resize-mode", default="resize", choices=["resize", "pad_resize"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-bank-size", type=int, default=30000)
    parser.add_argument("--query-chunk-size", type=int, default=4096)
    parser.add_argument("--bank-chunk-size", type=int, default=8192)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--threshold-fit-fraction", type=float, default=0.80)
    parser.add_argument("--threshold-split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mad-epsilon", type=float, default=1.0e-6)
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
    parser.add_argument("--clamp-min-zero", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-component-area-fraction", type=float, default=0.0005)
    parser.add_argument("--small-max-fraction", type=float, default=0.005)
    parser.add_argument("--medium-max-fraction", type=float, default=0.02)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    save_json(vars(args), output_dir / "resolved_args.json")
    model = load_dinov2(args.model, device=device, hub_dir=args.hub_dir)
    results: list[dict[str, Any]] = []
    for category in args.categories:
        result = run_category(args, category, model, device)
        results.append(result)
        write_summary(results, output_dir)
    save_json(
        {
            "status": "complete",
            "expected_categories": list(args.categories),
            "completed_categories": [row["category"] for row in results],
            "expected_count": len(args.categories),
            "completed_count": len(results),
            "model": args.model,
            "input_image_size": int(args.input_size),
            "evaluation_size": int(args.evaluation_size),
        },
        output_dir / "experiment_complete.json",
    )


if __name__ == "__main__":
    main()
