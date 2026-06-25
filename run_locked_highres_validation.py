from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sa_clip_nm.config import config_fingerprint, load_config, save_json, set_seed
from sa_clip_nm.highres_pipeline import run_highres_experiment
from sa_clip_nm.locked_highres_localization import (
    apply_locked_highres_method,
    locked_highres_method_fingerprint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the locked high-resolution SA-CLIP-NM localization method"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--selection",
        default="configs/selected_highres_localization_method.json",
    )
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--no-visualizations", action="store_true")
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
    config = apply_locked_highres_method(config, selection)

    # Runtime/data fields may change between datasets and machines; method fields may not.
    if args.data_root:
        config["data"]["root"] = args.data_root
    if args.output_dir:
        config["experiment"]["output_dir"] = args.output_dir
    if args.categories:
        config["data"]["categories"] = [str(value) for value in args.categories]
    if args.batch_size is not None:
        config["model"]["batch_size"] = args.batch_size
    if args.no_visualizations:
        config["inference"]["save_visualizations"] = False

    output_dir = Path(config["experiment"]["output_dir"])
    save_json(selection, output_dir / "selected_highres_localization_method.json")
    save_json(
        {
            "status": "locked_before_external_validation",
            "method_id": selection["method_id"],
            "selected_on": selection["selected_on"],
            "selection_path": str(selection_path.resolve()),
            "selection_fingerprint": locked_highres_method_fingerprint(selection),
            "resolved_config_fingerprint": config_fingerprint(config),
            "categories": [str(value) for value in config["data"]["categories"]],
            "warning": (
                "Method parameters are locked. External validation results "
                "must not be used to tune this selection."
            ),
        },
        output_dir / "locked_highres_localization_protocol.json",
    )
    set_seed(int(config["experiment"]["seed"]))
    run_highres_experiment(config, args.device)


if __name__ == "__main__":
    main()
