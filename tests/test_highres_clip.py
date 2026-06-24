from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from sa_clip_nm.data import ImageRecord
from sa_clip_nm.highres_backbone import (
    clip_patch_grid,
    requires_position_interpolation,
)
from sa_clip_nm.highres_data import HighResolutionMVTecDataset


@pytest.mark.parametrize(
    ("image_size", "grid_side", "patch_count"),
    [(224, 14, 196), (336, 21, 441), (448, 28, 784)],
)
def test_clip_patch_grid_sizes(
    image_size: int,
    grid_side: int,
    patch_count: int,
) -> None:
    assert clip_patch_grid(image_size, 16) == (grid_side, patch_count)


def test_clip_patch_grid_rejects_non_divisible_size() -> None:
    with pytest.raises(ValueError, match="divisible"):
        clip_patch_grid(350, 16)


def test_only_non_pretrained_sizes_require_position_interpolation() -> None:
    assert requires_position_interpolation(224, 224) is False
    assert requires_position_interpolation(336, 224) is True
    assert requires_position_interpolation(448, 224) is True


def test_dataset_separates_model_and_evaluation_resolution(tmp_path) -> None:
    image_path = tmp_path / "sample.png"
    image = np.zeros((300, 500, 3), dtype=np.uint8)
    image[..., 0] = 160
    Image.fromarray(image).save(image_path)
    dataset = HighResolutionMVTecDataset(
        records=[ImageRecord(image_path, 0, "good")],
        input_size=336,
        evaluation_size=448,
        resize_mode="resize",
        include_mask=True,
    )
    item = dataset[0]
    assert item["image"].shape == (3, 336, 336)
    assert item["display_image"].shape == (448, 448, 3)
    assert item["mask"].shape == (448, 448)
