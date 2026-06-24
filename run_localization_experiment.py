from __future__ import annotations

import argparse

import torch

from sa_clip_nm.config import load_config, set_seed
from sa_clip_nm.localization_protocol import localization_mode
from sa_clip_nm.pipeline import run_experiment
from sa_clip_nm.tiled_pipeline import run_tiled_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run global or tiled SA-CLIP-NM localization experiments"
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
    parser.add_argument(
        "--localization-mode", choices=["global", "tiled"]
    )
    parser.add_argument(
        "--layer-fusion",
        choices=["mean", "geometric", "minimum", "weighted_mean"],
    )
    parser.add_argument("--local-canvas-size", type=int)
    parser.add_argument("--local-window-size", type=int)
    parser.add_argument("--local-stride", type=int)
    parser.add_argument("--local-memory-ratio", type=float)
    parser.add_argument("--local-memory-max-entries", type=int)
    parser.add_argument("--window-batch-size", type=int)
    parser.add_argument("--diagnostics", action="store_true")
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
    if args.localization_mode:
        config.setdefault("localization", {})["mode"] = args.localization_mode
        config["localization"]["enabled"] = args.localization_mode == "tiled"
    if args.layer_fusion:
        config["inference"]["layer_fusion"] = args.layer_fusion
    overrides = {
        "canvas_size": args.local_canvas_size,
        "window_size": args.local_window_size,
        "stride": args.local_stride,
        "local_memory_ratio": args.local_memory_ratio,
        "local_memory_max_entries": args.local_memory_max_entries,
        "window_batch_size": args.window_batch_size,
    }
    for key, value in overrides.items():
        if value is not None:
            config.setdefault("localization", {})[key] = value
    if args.diagnostics:
        config.setdefault("diagnostics", {})["enabled"] = True
    if args.no_visualizations:
        config["inference"]["save_visualizations"] = False

    set_seed(int(config["experiment"]["seed"]))
    if localization_mode(config) == "tiled":
        run_tiled_experiment(config, args.device)
    else:
        run_experiment(config, args.device)


if __name__ == "__main__":
    main()
