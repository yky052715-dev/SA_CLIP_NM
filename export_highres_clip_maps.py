from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sa_clip_nm.calibration import CategoryCalibration, compute_layer_tau  # noqa: E402
from sa_clip_nm.config import load_config, save_json, set_seed  # noqa: E402
from sa_clip_nm.data import build_records, split_calibration_records, split_normal_records  # noqa: E402
from sa_clip_nm.highres_backbone import HighResolutionCLIPVisionFeatureExtractor, clip_patch_grid  # noqa: E402
from sa_clip_nm.highres_data import HighResolutionMVTecDataset, collate_highres_records  # noqa: E402
from sa_clip_nm.highres_pipeline import _evaluation_size, _map_config  # noqa: E402
from sa_clip_nm.pipeline import (  # noqa: E402
    _configured_map_outputs,
    _extract_features,
    _raw_layer_scores,
    calibrate_category,
    prepare_category,
)
from sa_clip_nm.backbone import make_patch_coordinates  # noqa: E402
from sa_clip_nm.memory import load_memory_banks  # noqa: E402


def _category_dir(output_dir: Path, category: str) -> Path:
    path = output_dir / category
    path.mkdir(parents=True, exist_ok=True)
    return path


def _highres_loader(records, config: dict[str, Any], include_mask: bool) -> DataLoader:
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


def _export_category_maps(
    category: str,
    config: dict[str, Any],
    extractor: HighResolutionCLIPVisionFeatureExtractor,
    output_dir: Path,
    calibration: CategoryCalibration,
    device: str | torch.device,
) -> None:
    category_dir = _category_dir(output_dir, category)
    artifact_dir = category_dir / "artifacts"
    banks, _ = load_memory_banks(artifact_dir / "memory_bank.pt")
    train_records, test_records = build_records(config["data"]["root"], category, config["data"])
    _, calibration_records = split_normal_records(
        train_records,
        calibration_fraction=float(config["data"]["calibration_fraction"]),
        seed=int(config["experiment"]["seed"]),
    )
    threshold_fit_records, normal_validation_records = split_calibration_records(
        calibration_records,
        threshold_fit_fraction=float(config["calibration"].get("threshold_fit_fraction", 0.5)),
        seed=int(config["calibration"]["threshold_split_seed"]),
    )
    threshold_fit_path_set = {str(record.path) for record in threshold_fit_records}
    normal_validation_path_set = {str(record.path) for record in normal_validation_records}

    map_config = _map_config(config)
    input_size = int(config["data"]["image_size"])
    grid_side, patch_count = clip_patch_grid(input_size, extractor.patch_size)

    calibration_maps: list[np.ndarray] = []
    calibration_paths: list[str] = []
    for batch in _highres_loader(calibration_records, config, include_mask=False):
        features_by_layer = {
            layer: values.cpu()
            for layer, values in extractor.extract(batch["image"]).items()
        }
        actual_patch_count = int(next(iter(features_by_layer.values())).shape[1])
        if actual_patch_count != patch_count:
            raise RuntimeError(f"Expected {patch_count} patches, got {actual_patch_count}")
        raw_scores = _raw_layer_scores(
            features_by_layer,
            make_patch_coordinates(actual_patch_count),
            banks,
            {layer: calibration.layers[layer].spatial_weight for layer in features_by_layer},
            map_config,
            device,
        )
        maps = _configured_map_outputs(raw_scores, calibration.layers, map_config).anomaly_maps
        calibration_maps.extend([value.numpy().astype(np.float32) for value in maps])
        calibration_paths.extend([str(path) for path in batch["path"]])

    threshold_fit_indices = np.asarray(
        [index for index, path in enumerate(calibration_paths) if path in threshold_fit_path_set],
        dtype=np.int64,
    )
    normal_validation_indices = np.asarray(
        [index for index, path in enumerate(calibration_paths) if path in normal_validation_path_set],
        dtype=np.int64,
    )
    np.savez_compressed(
        artifact_dir / "calibration_maps.npz",
        maps=np.stack(calibration_maps).astype(np.float32),
        paths=np.asarray(calibration_paths, dtype=object),
        threshold_fit_indices=threshold_fit_indices,
        normal_validation_indices=normal_validation_indices,
    )

    test_maps: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[str] = []
    defect_types: list[str] = []
    for batch in _highres_loader(test_records, config, include_mask=True):
        features_by_layer = {
            layer: values.cpu()
            for layer, values in extractor.extract(batch["image"]).items()
        }
        actual_patch_count = int(next(iter(features_by_layer.values())).shape[1])
        raw_scores = _raw_layer_scores(
            features_by_layer,
            make_patch_coordinates(actual_patch_count),
            banks,
            {layer: calibration.layers[layer].spatial_weight for layer in features_by_layer},
            map_config,
            device,
        )
        maps = _configured_map_outputs(raw_scores, calibration.layers, map_config).anomaly_maps
        test_maps.extend([value.numpy().astype(np.float32) for value in maps])
        masks.extend([value.numpy().astype(np.uint8) for value in batch["mask"]])
        labels.extend([int(value) for value in batch["label"].numpy().tolist()])
        paths.extend([str(path) for path in batch["path"]])
        defect_types.extend([str(value) for value in batch["defect_type"]])
    np.savez_compressed(
        artifact_dir / "test_maps.npz",
        maps=np.stack(test_maps).astype(np.float32),
        masks=np.stack(masks).astype(np.uint8),
        labels=np.asarray(labels, dtype=np.uint8),
        paths=np.asarray(paths, dtype=object),
        defect_types=np.asarray(defect_types, dtype=object),
    )
    save_json(
        {
            "category": category,
            "input_image_size": input_size,
            "evaluation_size": _evaluation_size(config),
            "patch_grid_side": grid_side,
            "calibration_maps": len(calibration_maps),
            "test_maps": len(test_maps),
        },
        artifact_dir / "map_export.json",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export high-resolution CLIP anomaly maps for map-level fusion")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.data_root:
        config["data"]["root"] = args.data_root
    if args.categories:
        config["data"]["categories"] = args.categories
    if args.batch_size is not None:
        config["model"]["batch_size"] = args.batch_size
    config["experiment"]["output_dir"] = args.output_dir
    set_seed(int(config["experiment"]["seed"]))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(config, output_dir / "resolved_config.json")
    extractor = HighResolutionCLIPVisionFeatureExtractor(
        checkpoint=str(config["model"]["checkpoint"]),
        layers=config["model"]["active_layers"],
        token_norm=str(config["model"]["token_norm"]),
        device=args.device,
    )
    category_reliabilities: dict[str, dict[int, float]] = {}
    for category in config["data"]["categories"]:
        print(f"[prepare-clip-map-export] {category}", flush=True)
        category_reliabilities[category] = prepare_category(category, config, extractor, output_dir)
    layer_tau = compute_layer_tau(
        category_reliabilities,
        quantile=float(config["retrieval"]["tau_quantile"]),
    )
    map_config = _map_config(config)
    completed = []
    for category in config["data"]["categories"]:
        print(f"[calibrate-clip-map-export] {category}", flush=True)
        calibration = calibrate_category(category, map_config, output_dir, layer_tau, args.device)
        print(f"[export-clip-maps] {category}", flush=True)
        _export_category_maps(category, config, extractor, output_dir, calibration, args.device)
        completed.append(str(category))
    save_json(
        {
            "status": "complete",
            "expected_categories": [str(value) for value in config["data"]["categories"]],
            "completed_categories": completed,
            "expected_count": len(config["data"]["categories"]),
            "completed_count": len(completed),
        },
        output_dir / "experiment_complete.json",
    )


if __name__ == "__main__":
    main()
