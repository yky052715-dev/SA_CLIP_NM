from __future__ import annotations

import argparse

import torch

from sa_clip_nm.config import load_config, set_seed
from sa_clip_nm.highres_pipeline import run_highres_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one paired-seed high-resolution CLIP experiment"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, choices=[224, 336], required=True)
    parser.add_argument("--evaluation-size", type=int, default=448)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--memory-ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--threshold-split-seed", type=int, required=True)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config["data"]["root"] = args.data_root
    config["data"]["image_size"] = args.image_size
    config["experiment"]["output_dir"] = args.output_dir
    config["experiment"]["seed"] = args.seed
    config["calibration"]["threshold_split_seed"] = (
        args.threshold_split_seed
    )
    config["high_resolution"]["evaluation_size"] = args.evaluation_size
    config["model"]["batch_size"] = args.batch_size
    config["memory"]["ratio"] = args.memory_ratio
    set_seed(args.seed)
    run_highres_experiment(config, args.device)


if __name__ == "__main__":
    main()
