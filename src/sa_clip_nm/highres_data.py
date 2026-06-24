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


class HighResolutionMVTecDataset(Dataset):
    """Separate CLIP input resolution from the fixed evaluation resolution."""

    def __init__(
        self,
        records: Iterable[ImageRecord],
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
            "image": image_to_tensor(model_image),
            "label": record.label,
            "path": str(record.path),
            "defect_type": record.defect_type,
            "display_image": np.asarray(display_image, dtype=np.uint8),
        }
        if self.include_mask:
            if record.mask_path is None:
                mask = Image.new(
                    "L",
                    (self.evaluation_size, self.evaluation_size),
                    color=0,
                )
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


def collate_highres_records(
    batch: list[dict[str, object]],
) -> dict[str, object]:
    output: dict[str, object] = {
        "image": torch.stack([item["image"] for item in batch]),
        "label": torch.tensor(
            [int(item["label"]) for item in batch], dtype=torch.int64
        ),
        "path": [str(item["path"]) for item in batch],
        "defect_type": [str(item["defect_type"]) for item in batch],
        "display_image": [item["display_image"] for item in batch],
    }
    if "mask" in batch[0]:
        output["mask"] = torch.stack([item["mask"] for item in batch])
    return output
