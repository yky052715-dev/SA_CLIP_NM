from __future__ import annotations

from pathlib import Path

from PIL import Image

from sa_clip_nm.data import build_records, build_visa_records


def _write_image(path: Path, mode: str = "RGB") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new(mode, (4, 4), color=255)
    image.save(path)


def test_build_visa_records_splits_normals_and_keeps_masks(tmp_path: Path) -> None:
    root = tmp_path
    category = "candle"
    for index in range(4):
        _write_image(root / category / "Data" / "Images" / "Normal" / f"{index:04d}.JPG")
    for index in range(2):
        _write_image(root / category / "Data" / "Images" / "Anomaly" / f"{index:03d}.JPG")
        _write_image(root / category / "Data" / "Masks" / "Anomaly" / f"{index:03d}.png", mode="L")

    csv_path = root / category / "image_anno.csv"
    csv_path.write_text(
        "image,label,mask\n"
        "candle/Data/Images/Normal/0000.JPG,normal,\n"
        "candle/Data/Images/Normal/0001.JPG,normal,\n"
        "candle/Data/Images/Normal/0002.JPG,normal,\n"
        "candle/Data/Images/Normal/0003.JPG,normal,\n"
        "candle/Data/Images/Anomaly/000.JPG,anomaly,candle/Data/Masks/Anomaly/000.png\n"
        "candle/Data/Images/Anomaly/001.JPG,anomaly,candle/Data/Masks/Anomaly/001.png\n",
        encoding="utf-8",
    )

    train, test = build_visa_records(
        root,
        category,
        normal_train_fraction=0.5,
        split_seed=42,
    )

    assert len(train) == 2
    assert all(record.label == 0 for record in train)
    assert len(test) == 4
    assert sum(record.label == 0 for record in test) == 2
    assert sum(record.label == 1 for record in test) == 2
    assert all(record.mask_path is not None for record in test if record.label == 1)
    assert {record.path for record in train}.isdisjoint({record.path for record in test})


def test_build_records_dispatches_to_visa(tmp_path: Path) -> None:
    root = tmp_path
    category = "pcb1"
    for index in range(2):
        _write_image(root / category / "Data" / "Images" / "Normal" / f"{index:04d}.JPG")
    _write_image(root / category / "Data" / "Images" / "Anomaly" / "000.JPG")
    _write_image(root / category / "Data" / "Masks" / "Anomaly" / "000.png", mode="L")
    (root / category / "image_anno.csv").write_text(
        "image,label,mask\n"
        "pcb1/Data/Images/Normal/0000.JPG,normal,\n"
        "pcb1/Data/Images/Normal/0001.JPG,normal,\n"
        "pcb1/Data/Images/Anomaly/000.JPG,anomaly,\n",
        encoding="utf-8",
    )

    train, test = build_records(
        root,
        category,
        {"dataset": "visa", "visa_normal_train_fraction": 0.5, "visa_split_seed": 1},
    )

    assert len(train) == 1
    assert len(test) == 2
