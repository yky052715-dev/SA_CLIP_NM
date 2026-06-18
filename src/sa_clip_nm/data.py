from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: int
    defect_type: str
    mask_path: Path | None = None


def list_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def build_mvtec_records(root: str | Path, category: str) -> tuple[list[ImageRecord], list[ImageRecord]]:
    category_root = Path(root) / category
    train_good = category_root / "train" / "good"
    test_root = category_root / "test"
    ground_truth_root = category_root / "ground_truth"

    if not train_good.is_dir():
        raise FileNotFoundError(f"Missing MVTec training directory: {train_good}")
    if not test_root.is_dir():
        raise FileNotFoundError(f"Missing MVTec test directory: {test_root}")

    train_records = [
        ImageRecord(path=path, label=0, defect_type="good")
        for path in list_images(train_good)
    ]
    test_records: list[ImageRecord] = []
    for defect_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
        defect_type = defect_dir.name
        for image_path in list_images(defect_dir):
            if defect_type == "good":
                mask_path = None
                label = 0
            else:
                mask_path = ground_truth_root / defect_type / f"{image_path.stem}_mask.png"
                if not mask_path.is_file():
                    raise FileNotFoundError(f"Missing mask for {image_path}: {mask_path}")
                label = 1
            test_records.append(
                ImageRecord(
                    path=image_path,
                    label=label,
                    defect_type=defect_type,
                    mask_path=mask_path,
                )
            )
    if not train_records:
        raise RuntimeError(f"No normal training images found in {train_good}")
    if not test_records:
        raise RuntimeError(f"No test images found in {test_root}")
    return train_records, test_records


def split_normal_records(
    records: list[ImageRecord],
    calibration_fraction: float,
    seed: int,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1")
    generator = np.random.default_rng(seed)
    indices = generator.permutation(len(records))
    calibration_count = max(2, int(round(len(records) * calibration_fraction)))
    calibration_count = min(calibration_count, len(records) - 1)
    calibration_indices = set(indices[:calibration_count].tolist())
    memory_records = [record for index, record in enumerate(records) if index not in calibration_indices]
    calibration_records = [record for index, record in enumerate(records) if index in calibration_indices]
    return memory_records, calibration_records


def split_calibration_records(
    records: list[ImageRecord],
    threshold_fit_fraction: float,
    seed: int,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Split the normal calibration pool into threshold-fit and held-out sets."""
    if len(records) < 2:
        raise ValueError("At least two calibration records are required")
    if not 0.0 < threshold_fit_fraction < 1.0:
        raise ValueError("threshold_fit_fraction must be between 0 and 1")
    generator = np.random.default_rng(seed)
    indices = generator.permutation(len(records))
    fit_count = int(round(len(records) * threshold_fit_fraction))
    fit_count = min(max(1, fit_count), len(records) - 1)
    fit_indices = set(indices[:fit_count].tolist())
    fit_records = [
        record for index, record in enumerate(records) if index in fit_indices
    ]
    validation_records = [
        record for index, record in enumerate(records) if index not in fit_indices
    ]
    return fit_records, validation_records


def _resize_image(image: Image.Image, size: int, is_mask: bool) -> Image.Image:
    interpolation = Image.Resampling.NEAREST if is_mask else Image.Resampling.BICUBIC
    return image.resize((size, size), interpolation)


def _pad_resize_image(image: Image.Image, size: int, is_mask: bool) -> Image.Image:
    width, height = image.size
    scale = size / max(width, height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    interpolation = Image.Resampling.NEAREST if is_mask else Image.Resampling.BICUBIC
    resized = image.resize((resized_width, resized_height), interpolation)
    fill = 0 if is_mask else (0, 0, 0)
    canvas_mode = "L" if is_mask else "RGB"
    canvas = Image.new(canvas_mode, (size, size), color=fill)
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    canvas.paste(resized, (left, top))
    return canvas


def transform_pil(image: Image.Image, size: int, mode: str, is_mask: bool) -> Image.Image:
    image = image.convert("L" if is_mask else "RGB")
    if mode == "resize":
        return _resize_image(image, size, is_mask)
    if mode == "pad_resize":
        return _pad_resize_image(image, size, is_mask)
    raise ValueError(f"Unsupported resize mode: {mode}")


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - CLIP_MEAN) / CLIP_STD


def mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    array = np.asarray(mask, dtype=np.uint8)
    return torch.from_numpy((array > 0).astype(np.uint8))


class MVTecImageDataset(Dataset):
    def __init__(
        self,
        records: Iterable[ImageRecord],
        image_size: int,
        resize_mode: str,
        include_mask: bool,
    ) -> None:
        self.records = list(records)
        self.image_size = image_size
        self.resize_mode = resize_mode
        self.include_mask = include_mask

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        with Image.open(record.path) as handle:
            original = handle.convert("RGB")
        transformed = transform_pil(original, self.image_size, self.resize_mode, is_mask=False)
        item: dict[str, object] = {
            "image": image_to_tensor(transformed),
            "label": record.label,
            "path": str(record.path),
            "defect_type": record.defect_type,
            "display_image": np.asarray(transformed, dtype=np.uint8),
        }
        if self.include_mask:
            if record.mask_path is None:
                mask = Image.new("L", (self.image_size, self.image_size), color=0)
            else:
                with Image.open(record.mask_path) as handle:
                    mask = transform_pil(handle, self.image_size, self.resize_mode, is_mask=True)
            item["mask"] = mask_to_tensor(mask)
        return item


def collate_records(batch: list[dict[str, object]]) -> dict[str, object]:
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
