from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
    figure, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[1].imshow(ground_truth, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("GT")
    heatmap = axes[2].imshow(
        anomaly_map,
        cmap="jet",
        vmin=0.0,
        vmax=max(color_max, 1e-6),
    )
    axes[2].set_title("Anomaly map")
    axes[3].imshow(prediction, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Prediction")
    for axis in axes:
        axis.axis("off")
    figure.colorbar(heatmap, ax=axes[2], fraction=0.046, pad=0.04)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)

