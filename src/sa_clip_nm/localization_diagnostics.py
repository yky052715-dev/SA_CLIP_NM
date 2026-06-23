from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .visualization import colorize_anomaly_map, make_anomaly_overlay


def save_localization_diagnostics(
    output_dir: str | Path,
    prefix: str,
    image: np.ndarray,
    ground_truth: np.ndarray,
    anomaly_map: np.ndarray,
    nearest_map: np.ndarray,
    prediction: np.ndarray,
    layer_patch_maps: Mapping[int, np.ndarray],
    fused_patch_map: np.ndarray,
    threshold: float,
    color_max: float,
) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    safe_prefix = prefix.replace("/", "_").replace("\\", "_")

    for layer, values in sorted(layer_patch_maps.items()):
        np.save(
            directory / f"{safe_prefix}_layer{layer}_patch.npy",
            np.asarray(values, dtype=np.float32),
        )
    np.save(
        directory / f"{safe_prefix}_fused_patch.npy",
        np.asarray(fused_patch_map, dtype=np.float32),
    )

    Image.fromarray(
        colorize_anomaly_map(anomaly_map, color_max=color_max)
    ).save(directory / f"{safe_prefix}_bilinear.png")
    Image.fromarray(
        colorize_anomaly_map(nearest_map, color_max=color_max)
    ).save(directory / f"{safe_prefix}_nearest.png")
    Image.fromarray(
        (np.asarray(prediction).astype(np.uint8) * 255)
    ).save(directory / f"{safe_prefix}_prediction.png")
    Image.fromarray(
        (np.asarray(ground_truth).astype(np.uint8) * 255)
    ).save(directory / f"{safe_prefix}_ground_truth.png")

    overlay = make_anomaly_overlay(
        image,
        anomaly_map,
        color_max=color_max,
    )
    figure, axis = plt.subplots(1, 1, figsize=(4, 4))
    axis.imshow(overlay)
    values = np.asarray(anomaly_map, dtype=np.float32)
    if float(values.min()) <= threshold <= float(values.max()):
        axis.contour(
            values,
            levels=[threshold],
            colors="white",
            linewidths=0.8,
        )
    axis.axis("off")
    figure.tight_layout(pad=0)
    figure.savefig(
        directory / f"{safe_prefix}_overlay_contour.png",
        dpi=160,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(figure)
