#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/mvtec_validation10_position_localization.yaml}"
SELECTION="${SELECTION:-${PROJECT_ROOT}/configs/selected_localization_method.json}"
DATA_ROOT="${DATA_ROOT:-/home/ubuntu/yyk/datasets/mvtec}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/localization_refinement/validation10/locked_rho010_mem01}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/run_locked_position_validation.py" \
    --config "${CONFIG}" \
    --selection "${SELECTION}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --device cuda

echo "[validation10-position-locked] complete: ${OUTPUT_DIR}"
