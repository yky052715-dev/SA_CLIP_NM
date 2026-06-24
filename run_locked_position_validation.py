from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sa_clip_nm.config import config_fingerprint, load_config, save_json, set_seed
from sa_clip_nm.locked_localization import (
    apply_locked_method,
    locked_method_fingerprint,
)
from sa_clip_nm.position_tiled_pipeline import run_position_tiled_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the locked SA-CLIP-NM localization method on Validation10"
    )
    parser.add_argument(
        "--config",
        default="configs/mvtec_validation10_position_localization.yaml",
    )
    parser.add_argument(
        "--selection",
        default="configs/selected_localization_method.json",
    )
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    selection_path = Path(args.selection)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    config = apply_locked_method(config, selection)

    # Paths may change between machines; method parameters may not.
    if args.data_root:
        config["data"]["root"] = args.data_root
    if args.output_dir:
        config["experiment"]["output_dir"] = args.output_dir

    output_dir = Path(config["experiment"]["output_dir"])
    save_json(selection, output_dir / "selected_localization_method.json")
    save_json(
        {
            "status": "locked_before_validation",
            "method_id": selection["method_id"],
            "selected_on": selection["selected_on"],
            "selection_path": str(selection_path.resolve()),
            "selection_fingerprint": locked_method_fingerprint(selection),
            "resolved_config_fingerprint": config_fingerprint(config),
            "validation_categories": selection["validation_categories"],
        },
        output_dir / "locked_localization_protocol.json",
    )
    set_seed(int(config["experiment"]["seed"]))
    run_position_tiled_experiment(config, args.device)


if __name__ == "__main__":
    main()
