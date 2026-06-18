#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/iad_project/SA_CLIP_NM}"
cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
python run_experiment.py \
  --config configs/mvtec15.yaml \
  --device cuda

