#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CONFIG="${CONFIG:-configs/mvtec_dev5_threshold_ablation.yaml}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-outputs/dev5_threshold_selection}"
export RUN_MODE="development"

exec bash "${PROJECT_DIR}/scripts/run_threshold_ablation_set.sh"
