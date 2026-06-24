from __future__ import annotations

import argparse

import torch

from sa_clip_nm.config import load_config, set_seed
from sa_clip_nm.position_tiled_pipeline import run_position_tiled_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run position-calibrated tiled SA-CLIP-NM"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--output-dir")
    parser.add_argument("--data-root")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--position-rho", type=float)
    parser.add_argument("--position-quantile", type=float)
    parser.add_argument(
        "--layer-fusion",
        choices=["mean", "geometric", "minimum", "weighted_mean"],
    )
    parser.add_argument("--local-memory-ratio", type=float)
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
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed
    if args.position_rho is not None:
        config["position_calibration"]["rho"] = args.position_rho
    if args.position_quantile is not None:
        config["position_calibration"]["quantile"] = args.position_quantile
    if args.layer_fusion:
        config["inference"]["layer_fusion"] = args.layer_fusion
    if args.local_memory_ratio is not None:
        config["localization"]["local_memory_ratio"] = (
            args.local_memory_ratio
        )
    if args.no_visualizations:
        config["inference"]["save_visualizations"] = False
    set_seed(int(config["experiment"]["seed"]))
    run_position_tiled_experiment(config, args.device)


if __name__ == "__main__":
    main()
