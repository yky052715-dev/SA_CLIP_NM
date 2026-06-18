from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sa_clip_nm.backbone import make_patch_coordinates
from sa_clip_nm.calibration import (
    calibrate_scores,
    compute_layer_tau,
    estimate_spatial_reliability,
    image_score_from_map,
    reliability_to_weight,
    robust_location_scale,
)
from sa_clip_nm.data import ImageRecord, MVTecImageDataset, split_normal_records
from sa_clip_nm.memory import LayerMemoryBank, build_layer_memory
from sa_clip_nm.metrics import evaluate_binary_scores
from sa_clip_nm.pipeline import (
    calibrate_category,
    evaluate_category,
    prepare_category,
    write_summary,
)
from sa_clip_nm.retrieval import exact_spatial_nearest_neighbor


class FakeFeatureExtractor:
    layers = (3, 6)

    def extract(self, pixel_values: torch.Tensor) -> dict[int, torch.Tensor]:
        pooled = torch.nn.functional.adaptive_avg_pool2d(
            pixel_values.mean(dim=1, keepdim=True),
            output_size=(2, 2),
        ).reshape(pixel_values.shape[0], 4, 1)
        coordinates = make_patch_coordinates(4).to(pixel_values).unsqueeze(0)
        features = torch.cat(
            [
                pooled,
                coordinates.expand(pixel_values.shape[0], -1, -1),
                torch.ones_like(pooled),
            ],
            dim=-1,
        )
        features = torch.nn.functional.normalize(features, dim=-1)
        return {3: features, 6: torch.roll(features, shifts=1, dims=-1)}


def _write_mini_mvtec(root: Path) -> None:
    for directory in [
        root / "toy" / "train" / "good",
        root / "toy" / "test" / "good",
        root / "toy" / "test" / "scratch",
        root / "toy" / "ground_truth" / "scratch",
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    for index in range(10):
        image = np.full((16, 16, 3), 100 + index % 2, dtype=np.uint8)
        Image.fromarray(image).save(root / "toy" / "train" / "good" / f"{index:03d}.png")
    for index in range(2):
        image = np.full((16, 16, 3), 100, dtype=np.uint8)
        Image.fromarray(image).save(root / "toy" / "test" / "good" / f"{index:03d}.png")
    for index in range(2):
        image = np.full((16, 16, 3), 100, dtype=np.uint8)
        image[:8, :8] = 255
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[:8, :8] = 255
        Image.fromarray(image).save(
            root / "toy" / "test" / "scratch" / f"{index:03d}.png"
        )
        Image.fromarray(mask).save(
            root / "toy" / "ground_truth" / "scratch" / f"{index:03d}_mask.png"
        )


def _mini_config(data_root: Path, output_root: Path) -> dict[str, object]:
    return {
        "experiment": {"name": "smoke", "seed": 42, "output_dir": str(output_root)},
        "data": {
            "dataset": "mvtec",
            "root": str(data_root),
            "categories": ["toy"],
            "image_size": 16,
            "resize_mode": "resize",
            "calibration_fraction": 0.2,
            "num_workers": 0,
        },
        "model": {
            "checkpoint": "fake",
            "candidate_layers": [3, 6],
            "active_layers": [3, 6],
            "token_norm": "none",
            "batch_size": 4,
        },
        "memory": {
            "sampling": "random",
            "ratio": 0.5,
            "projection_dim": 4,
            "max_candidates": 100,
        },
        "retrieval": {
            "query_chunk_size": 16,
            "bank_chunk_size": 64,
            "lambda_max": 0.1,
            "tau_quantile": 0.8,
        },
        "calibration": {
            "mad_epsilon": 1e-6,
            "pixel_threshold_method": "global_quantile",
            "pixel_quantile": 0.995,
            "pixel_image_quantile": 0.95,
            "pixel_topk_fraction": 0.01,
            "image_quantile": 0.99,
            "clamp_min_zero": True,
        },
        "inference": {
            "gaussian_sigma": 0.0,
            "image_score": "topk_mean",
            "image_topk_fraction": 0.1,
            "save_visualizations": False,
            "max_visualizations_per_category": 0,
        },
        "evaluation": {"compute_oracle_f1": True, "save_predictions": False},
    }


def main() -> None:
    coordinates = make_patch_coordinates(4)
    assert coordinates.shape == (4, 2)

    features = torch.nn.functional.normalize(torch.randn(3, 4, 8), dim=-1)
    bank = build_layer_memory(
        features,
        coordinates,
        ratio=0.5,
        method="random",
        seed=42,
        projection_dim=4,
        max_candidates=100,
    )
    assert bank.features.shape == (6, 8)

    ambiguous_bank = LayerMemoryBank(
        features=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        coordinates=torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
        source_ids=torch.tensor([0, 1]),
    )
    _, index = exact_spatial_nearest_neighbor(
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[1.0, 1.0]]),
        ambiguous_bank,
        spatial_weight=0.1,
        device="cpu",
    )
    assert int(index.item()) == 1

    stable = torch.eye(4).repeat(4, 1, 1)
    stable = torch.nn.functional.normalize(
        stable + 0.001 * torch.randn_like(stable), dim=-1
    )
    repeated = torch.nn.functional.normalize(
        torch.ones(4, 4, 4) + 0.001 * torch.randn(4, 4, 4), dim=-1
    )
    assert (
        estimate_spatial_reliability(stable)["reliability"]
        > estimate_spatial_reliability(repeated)["reliability"]
    )

    tau = compute_layer_tau(
        {"a": {3: 1.0, 6: 2.0}, "b": {3: 3.0, 6: 4.0}},
        quantile=0.5,
    )
    assert tau == {3: 2.0, 6: 3.0}
    assert np.isclose(reliability_to_weight(2.0, 2.0, 0.1), 0.05)

    values = torch.tensor([1.0, 2.0, 3.0, 100.0])
    median, mad = robust_location_scale(values, epsilon=1e-6)
    calibrated = calibrate_scores(values, median, mad, clamp_min_zero=True)
    assert median == 2.0 and mad == 1.0
    assert calibrated.tolist() == [0.0, 0.0, 1.0, 98.0]
    map_score = image_score_from_map(
        torch.tensor([[[0.0, 1.0], [2.0, 3.0]]]),
        method="topk_mean",
        topk_fraction=0.5,
    )
    assert torch.allclose(map_score, torch.tensor([2.5]))

    metrics = evaluate_binary_scores(
        np.array([0, 0, 1, 1], dtype=np.uint8),
        np.array([0.1, 0.2, 0.8, 0.9]),
        calibrated_threshold=0.85,
        compute_oracle=True,
    )
    assert np.isclose(metrics.auroc, 1.0)
    assert metrics.oracle_f1 >= metrics.calibrated_f1

    records = [
        ImageRecord(path=Path(f"{index}.png"), label=0, defect_type="good")
        for index in range(20)
    ]
    memory, calibration = split_normal_records(records, 0.2, seed=42)
    assert set(memory).isdisjoint(calibration)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        image_path = root / "image.png"
        mask_path = root / "mask.png"
        Image.new("RGB", (20, 10), color=(255, 0, 0)).save(image_path)
        Image.new("L", (20, 10), color=255).save(mask_path)
        dataset = MVTecImageDataset(
            [
                ImageRecord(
                    path=image_path,
                    label=1,
                    defect_type="defect",
                    mask_path=mask_path,
                )
            ],
            image_size=16,
            resize_mode="resize",
            include_mask=True,
        )
        item = dataset[0]
        assert item["image"].shape == (3, 16, 16)
        assert item["mask"].shape == (16, 16)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        data_root = root / "data"
        output_root = root / "outputs"
        _write_mini_mvtec(data_root)
        config = _mini_config(data_root, output_root)
        extractor = FakeFeatureExtractor()
        reliabilities = prepare_category(
            "toy",
            config,
            extractor,
            output_root,
        )
        tau = compute_layer_tau({"toy": reliabilities}, quantile=0.8)
        category_calibration = calibrate_category(
            "toy",
            config,
            output_root,
            tau,
            device="cpu",
        )
        result = evaluate_category(
            "toy",
            config,
            extractor,
            output_root,
            category_calibration,
            device="cpu",
        )
        write_summary([result], output_root)
        assert (output_root / "toy" / "metrics.json").is_file()
        assert (output_root / "metrics_summary.csv").is_file()
        assert result["test_images"] == 4

    print("SA-CLIP-NM smoke tests passed.")


if __name__ == "__main__":
    main()
