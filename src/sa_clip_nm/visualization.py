from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def colorize_anomaly_map(
    anomaly_map: np.ndarray,
    color_max: float,
    cmap: str = "jet",
) -> np.ndarray:
    values = np.asarray(anomaly_map, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("anomaly_map must be a two-dimensional array")
    scale = max(float(color_max), 1.0e-6)
    normalized = np.clip(values / scale, 0.0, 1.0)
    colored = plt.get_cmap(cmap)(normalized)[..., :3]
    return np.rint(colored * 255.0).astype(np.uint8)


def make_anomaly_overlay(
    image: np.ndarray,
    anomaly_map: np.ndarray,
    color_max: float,
    alpha: float = 0.50,
    cmap: str = "jet",
) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    base = np.asarray(image)
    if base.ndim != 3 or base.shape[2] != 3:
        raise ValueError("image must have shape [height, width, 3]")
    if base.shape[:2] != np.asarray(anomaly_map).shape:
        raise ValueError("image and anomaly_map must have the same spatial size")
    if np.issubdtype(base.dtype, np.integer):
        base_float = np.clip(base, 0, 255).astype(np.float32) / 255.0
    else:
        base_float = np.clip(base.astype(np.float32), 0.0, 1.0)
    colored = colorize_anomaly_map(anomaly_map, color_max, cmap).astype(np.float32) / 255.0
    overlay = (1.0 - alpha) * base_float + alpha * colored
    return np.rint(np.clip(overlay, 0.0, 1.0) * 255.0).astype(np.uint8)


def save_result_figure(
    image: np.ndarray,
    ground_truth: np.ndarray,
    anomaly_map: np.ndarray,
    prediction: np.ndarray,
    output_path: str | Path,
    title: str,
    color_max: float,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    colored_map = colorize_anomaly_map(anomaly_map, color_max)
    overlay = make_anomaly_overlay(image, anomaly_map, color_max)
    Image.fromarray(colored_map).save(path.with_name(f"{path.stem}_heatmap.png"))
    Image.fromarray(overlay).save(path.with_name(f"{path.stem}_overlay.png"))

    figure, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[1].imshow(ground_truth, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("GT")
    axes[2].imshow(overlay)
    axes[2].set_title("Heatmap overlay")
    axes[3].imshow(prediction, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Prediction")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)