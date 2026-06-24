from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .data import (
    ImageRecord,
    image_to_tensor,
    mask_to_tensor,
    transform_pil,
)
from .localization import extract_image_windows, generate_window_boxes


class TiledMVTecImageDataset(Dataset):
    """Return global CLIP inputs and local windows from the same source image."""

    def __init__(
        self,
        records: Iterable[ImageRecord],
        image_size: int,
        canvas_size: int,
        window_size: int,
        stride: int,
        resize_mode: str,
        include_mask: bool,
    ) -> None:
        if window_size != image_size:
            raise ValueError(
                "The first tiled implementation requires window_size == image_size"
            )
        self.records = list(records)
        self.image_size = int(image_size)
        self.canvas_size = int(canvas_size)
        self.resize_mode = str(resize_mode)
        self.include_mask = bool(include_mask)
        self.boxes = generate_window_boxes(
            canvas_size=self.canvas_size,
            window_size=int(window_size),
            stride=int(stride),
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        with Image.open(record.path) as handle:
            original = handle.convert("RGB")
        global_image = transform_pil(
            original,
            self.image_size,
            self.resize_mode,
            is_mask=False,
        )
        canvas = transform_pil(
            original,
            self.canvas_size,
            self.resize_mode,
            is_mask=False,
        )
        windows = extract_image_windows(
            image_to_tensor(canvas)[None],
            self.boxes,
        )[0]
        item: dict[str, object] = {
            "image": image_to_tensor(global_image),
            "windows": windows,
            "label": record.label,
            "path": str(record.path),
            "defect_type": record.defect_type,
            "display_image": np.asarray(global_image, dtype=np.uint8),
            "canvas_display_image": np.asarray(canvas, dtype=np.uint8),
        }
        if self.include_mask:
            if record.mask_path is None:
                evaluation_mask = Image.new(
                    "L", (self.image_size, self.image_size), color=0
                )
                canvas_mask = Image.new(
                    "L", (self.canvas_size, self.canvas_size), color=0
                )
            else:
                with Image.open(record.mask_path) as handle:
                    evaluation_mask = transform_pil(
                        handle,
                        self.image_size,
                        self.resize_mode,
                        is_mask=True,
                    )
                with Image.open(record.mask_path) as handle:
                    canvas_mask = transform_pil(
                        handle,
                        self.canvas_size,
                        self.resize_mode,
                        is_mask=True,
                    )
            item["mask"] = mask_to_tensor(evaluation_mask)
            item["canvas_mask"] = mask_to_tensor(canvas_mask)
        return item


def collate_tiled_records(batch: list[dict[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {
        "image": torch.stack([item["image"] for item in batch]),
        "windows": torch.stack([item["windows"] for item in batch]),
        "label": torch.tensor(
            [int(item["label"]) for item in batch], dtype=torch.int64
        ),
        "path": [str(item["path"]) for item in batch],
        "defect_type": [str(item["defect_type"]) for item in batch],
        "display_image": [item["display_image"] for item in batch],
        "canvas_display_image": [
            item["canvas_display_image"] for item in batch
        ],
    }
    if "mask" in batch[0]:
        output["mask"] = torch.stack([item["mask"] for item in batch])
        output["canvas_mask"] = torch.stack(
            [item["canvas_mask"] for item in batch]
        )
    return output
