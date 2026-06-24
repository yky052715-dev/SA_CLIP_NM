from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from sa_clip_nm.data import ImageRecord
from sa_clip_nm.localization import (
    build_window_weight,
    combine_image_scores,
    generate_window_boxes,
    merge_window_maps,
    validate_window_coverage,
)
from sa_clip_nm.tiled_data import TiledMVTecImageDataset


def test_448_canvas_generates_nine_expected_windows() -> None:
    boxes = generate_window_boxes(448, 224, 112)
    assert len(boxes) == 9
    assert boxes[0] == (0, 0, 224, 224)
    assert boxes[4] == (112, 112, 336, 336)
    assert boxes[-1] == (224, 224, 448, 448)


def test_every_canvas_pixel_is_covered() -> None:
    coverage = validate_window_coverage(
        generate_window_boxes(448, 224, 112),
        448,
    )
    assert int(coverage.min()) >= 1
    assert int(coverage.max()) == 4


def test_merging_all_one_maps_returns_all_ones() -> None:
    boxes = generate_window_boxes(448, 224, 112)
    maps = torch.ones((2, len(boxes), 224, 224))
    merged = merge_window_maps(
        maps,
        boxes,
        canvas_size=448,
        merge="hann",
        min_weight=0.05,
    )
    assert merged.shape == (2, 448, 448)
    assert torch.allclose(merged, torch.ones_like(merged), atol=1.0e-6)


def test_hann_weight_has_no_zero_or_nan_edges() -> None:
    weight = build_window_weight(224, merge="hann", min_weight=0.05)
    assert torch.isfinite(weight).all()
    assert float(weight.min()) >= 0.05


def test_global_local_image_score_max_merge() -> None:
    global_scores = torch.tensor([1.0, 4.0])
    local_scores = torch.tensor([2.0, 3.0])
    assert torch.equal(
        combine_image_scores(global_scores, local_scores, method="max"),
        torch.tensor([2.0, 4.0]),
    )


def test_tiled_dataset_returns_global_and_nine_local_inputs(tmp_path) -> None:
    image_path = tmp_path / "sample.png"
    image = np.zeros((320, 480, 3), dtype=np.uint8)
    image[..., 1] = 127
    Image.fromarray(image).save(image_path)
    dataset = TiledMVTecImageDataset(
        records=[ImageRecord(image_path, 0, "good")],
        image_size=224,
        canvas_size=448,
        window_size=224,
        stride=112,
        resize_mode="resize",
        include_mask=False,
    )
    item = dataset[0]
    assert item["image"].shape == (3, 224, 224)
    assert item["windows"].shape == (9, 3, 224, 224)
    assert item["canvas_display_image"].shape == (448, 448, 3)
