from __future__ import annotations

import argparse

import torch

from sa_clip_nm.config import load_config, set_seed
from sa_clip_nm.pipeline import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SA-CLIP-NM on MVTec AD")
    parser.add_argument("--config", required=True, help="Path to a YAML config")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, for example cuda or cpu",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        help="Override active CLIP blocks, e.g. --layers 6",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        help="Override dataset categories",
    )
    parser.add_argument("--output-dir", help="Override experiment.output_dir")
    parser.add_argument("--data-root", help="Override data.root")
    parser.add_argument("--seed", type=int, help="Override experiment.seed")
    parser.add_argument(
        "--spatial-mode",
        choices=["none", "fixed", "adaptive"],
        help="Override retrieval.spatial_mode",
    )
    parser.add_argument("--fixed-lambda", type=float)
    parser.add_argument("--lambda-max", type=float)
    parser.add_argument(
        "--pixel-threshold-method",
        choices=[
            "global_quantile",
            "image_max_quantile",
            "image_topk_quantile",
        ],
        help="Override calibration.pixel_threshold_method",
    )
    parser.add_argument("--pixel-quantile", type=float)
    parser.add_argument("--pixel-image-quantile", type=float)
    parser.add_argument("--pixel-topk-fraction", type=float)
    parser.add_argument(
        "--threshold-split-seed",
        type=int,
        help="Override calibration.threshold_split_seed",
    )
    parser.add_argument(
        "--layer-fusion",
        choices=["mean", "geometric", "minimum", "weighted_mean"],
        help="Override inference.layer_fusion",
    )
    parser.add_argument(
        "--layer-fusion-epsilon",
        type=float,
        help="Override inference.layer_fusion_epsilon",
    )
    parser.add_argument(
        "--upsample-mode",
        choices=["bilinear", "nearest"],
        help="Override inference.upsample_mode",
    )
    parser.add_argument(
        "--diagnostics", action="store_true", help="Enable P0 diagnostic outputs"
    )
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.layers:
        config["model"]["active_layers"] = args.layers
    if args.categories:
        config["data"]["categories"] = args.categories
    if args.output_dir:
        config["experiment"]["output_dir"] = args.output_dir
    if args.data_root:
        config["data"]["root"] = args.data_root
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed
    if args.spatial_mode:
        config["retrieval"]["spatial_mode"] = args.spatial_mode
    if args.fixed_lambda is not None:
        config["retrieval"]["fixed_lambda"] = args.fixed_lambda
    if args.lambda_max is not None:
        config["retrieval"]["lambda_max"] = args.lambda_max
    if args.pixel_threshold_method:
        config["calibration"]["pixel_threshold_method"] = (
            args.pixel_threshold_method
        )
    if args.pixel_quantile is not None:
        config["calibration"]["pixel_quantile"] = args.pixel_quantile
    if args.pixel_image_quantile is not None:
        config["calibration"]["pixel_image_quantile"] = args.pixel_image_quantile
    if args.pixel_topk_fraction is not None:
        config["calibration"]["pixel_topk_fraction"] = args.pixel_topk_fraction
    if args.threshold_split_seed is not None:
        config["calibration"]["threshold_split_seed"] = args.threshold_split_seed
    if args.layer_fusion:
        config["inference"]["layer_fusion"] = args.layer_fusion
    if args.layer_fusion_epsilon is not None:
        config["inference"]["layer_fusion_epsilon"] = args.layer_fusion_epsilon
    if args.upsample_mode:
        config["inference"]["upsample_mode"] = args.upsample_mode
    if args.diagnostics:
        config.setdefault("diagnostics", {})["enabled"] = True
    set_seed(int(config["experiment"]["seed"]))
    run_experiment(config, args.device)


if __name__ == "__main__":
    main()
