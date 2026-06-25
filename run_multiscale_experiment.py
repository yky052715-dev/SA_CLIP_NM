from __future__ import annotations

import argparse

import torch

from sa_clip_nm.config import load_config, set_seed
from sa_clip_nm.multiscale_pipeline import run_multiscale_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run feature-space multiscale neighborhood aggregation"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--output-dir")
    parser.add_argument("--data-root")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--memory-ratio", type=float)
    parser.add_argument("--scales", nargs="+", type=int)
    parser.add_argument("--scale-weights", nargs="+", type=float)
    parser.add_argument("--pixel-image-quantile", type=float)
    parser.add_argument("--threshold-split-seed", type=int)
    parser.add_argument("--no-visualizations", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.categories:
        config["data"]["categories"] = args.categories
    if args.output_dir:
        config["experiment"]["output_dir"] = args.output_dir
    if args.data_root:
        config["data"]["root"] = args.data_root
    if args.batch_size is not None:
        config["model"]["batch_size"] = args.batch_size
    if args.memory_ratio is not None:
        config["memory"]["ratio"] = args.memory_ratio
    if args.scales is not None:
        config.setdefault("multiscale", {})["scales"] = args.scales
    if args.scale_weights is not None:
        config.setdefault("multiscale", {})["scale_weights"] = args.scale_weights
    if args.pixel_image_quantile is not None:
        config["calibration"]["pixel_image_quantile"] = args.pixel_image_quantile
    if args.threshold_split_seed is not None:
        config["calibration"]["threshold_split_seed"] = args.threshold_split_seed
    if args.no_visualizations:
        config["inference"]["save_visualizations"] = False
        config.setdefault("diagnostics", {})["enabled"] = False
    set_seed(int(config["experiment"]["seed"]))
    run_multiscale_experiment(config, args.device)


if __name__ == "__main__":
    main()
