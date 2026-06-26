from __future__ import annotations

import numpy as np
import pytest

from fuse_clip_dino_maps import _align_payload, _fuse_maps


def test_fuse_maps_weighted_average() -> None:
    clip = np.ones((2, 4, 4), dtype=np.float32) * 2
    dino = np.ones((2, 4, 4), dtype=np.float32) * 10
    fused = _fuse_maps(clip, dino, alpha=0.75)
    assert fused.shape == clip.shape
    assert np.allclose(fused, 4.0)


def test_align_payload_reorders_by_path() -> None:
    reference = {
        "paths": np.asarray(["a.png", "b.png", "c.png"], dtype=object),
        "maps": np.zeros((3, 2, 2), dtype=np.float32),
    }
    other = {
        "paths": np.asarray(["c.png", "a.png", "b.png"], dtype=object),
        "maps": np.asarray([3, 1, 2], dtype=np.float32).reshape(3, 1, 1),
        "labels": np.asarray([30, 10, 20], dtype=np.uint8),
    }
    aligned = _align_payload(reference, other)
    assert aligned["paths"].tolist() == ["a.png", "b.png", "c.png"]
    assert aligned["maps"].reshape(-1).tolist() == [1, 2, 3]
    assert aligned["labels"].tolist() == [10, 20, 30]


def test_align_payload_rejects_missing_path() -> None:
    reference = {"paths": np.asarray(["a.png", "missing.png"], dtype=object)}
    other = {"paths": np.asarray(["a.png"], dtype=object)}
    with pytest.raises(ValueError):
        _align_payload(reference, other)
