from __future__ import annotations

import numpy as np

from sa_clip_nm.visualization import colorize_anomaly_map, make_anomaly_overlay


def test_overlay_alpha_endpoints() -> None:
    image = np.full((8, 8, 3), 128, dtype=np.uint8)
    anomaly_map = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    colored = colorize_anomaly_map(anomaly_map, color_max=1.0)
    assert np.array_equal(
        make_anomaly_overlay(image, anomaly_map, color_max=1.0, alpha=0.0),
        image,
    )
    assert np.array_equal(
        make_anomaly_overlay(image, anomaly_map, color_max=1.0, alpha=1.0),
        colored,
    )


def test_overlay_preserves_shape_and_uint8_type() -> None:
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    anomaly_map = np.zeros((4, 6), dtype=np.float32)
    overlay = make_anomaly_overlay(image, anomaly_map, color_max=0.0)
    assert overlay.shape == image.shape
    assert overlay.dtype == np.uint8