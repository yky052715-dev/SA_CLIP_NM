#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CONFIG="${CONFIG:-configs/mvtec_validation10_threshold_ablation.yaml}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-outputs/validation10_locked_threshold}"
export RUN_MODE="validation"
export SELECTION_MANIFEST="${SELECTION_MANIFEST:-outputs/dev5_threshold_selection/selected_threshold_method.json}"

exec bash "${PROJECT_DIR}/scripts/run_threshold_ablation_set.sh"
