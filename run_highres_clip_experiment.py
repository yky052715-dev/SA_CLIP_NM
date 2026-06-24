from __future__ import annotations

import argparse

import torch

from sa_clip_nm.config import load_config, set_seed
from sa_clip_nm.highres_pipeline import run_highres_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run global high-resolution CLIP localization"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--output-dir")
    parser.add_argument("--data-root")
    parser.add_argument("--image-size", type=int, choices=[224, 336, 448])
    parser.add_argument("--evaluation-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--memory-ratio", type=float)
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
    if args.image_size is not None:
        config["data"]["image_size"] = args.image_size
    if args.evaluation_size is not None:
        config.setdefault("high_resolution", {})["evaluation_size"] = (
            args.evaluation_size
        )
    if args.batch_size is not None:
        config["model"]["batch_size"] = args.batch_size
    if args.memory_ratio is not None:
        config["memory"]["ratio"] = args.memory_ratio
    if args.no_visualizations:
        config["inference"]["save_visualizations"] = False
    set_seed(int(config["experiment"]["seed"]))
    run_highres_experiment(config, args.device)


if __name__ == "__main__":
    main()
