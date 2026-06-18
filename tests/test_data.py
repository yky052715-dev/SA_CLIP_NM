from __future__ import annotations

from pathlib import Path

from PIL import Image

from sa_clip_nm.data import (
    ImageRecord,
    MVTecImageDataset,
    split_calibration_records,
    split_normal_records,
)


def test_normal_split_is_disjoint_and_reproducible() -> None:
    records = [
        ImageRecord(path=Path(f"{index}.png"), label=0, defect_type="good")
        for index in range(20)
    ]
    memory_a, calibration_a = split_normal_records(records, 0.2, seed=42)
    memory_b, calibration_b = split_normal_records(records, 0.2, seed=42)
    assert memory_a == memory_b
    assert calibration_a == calibration_b
    assert set(memory_a).isdisjoint(calibration_a)
    assert len(memory_a) + len(calibration_a) == len(records)


def test_calibration_split_is_disjoint_reproducible_and_nonempty() -> None:
    records = [
        ImageRecord(path=Path(f"{index}.png"), label=0, defect_type="good")
        for index in range(5)
    ]
    fit_a, validation_a = split_calibration_records(records, 0.5, seed=1042)
    fit_b, validation_b = split_calibration_records(records, 0.5, seed=1042)
    assert fit_a == fit_b
    assert validation_a == validation_b
    assert fit_a
    assert validation_a
    assert set(fit_a).isdisjoint(validation_a)
    assert len(fit_a) + len(validation_a) == len(records)


def test_dataset_applies_same_mask_geometry(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
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
    assert int(item["mask"].sum()) == 16 * 16
