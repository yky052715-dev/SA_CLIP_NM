#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/mvtec_dev5_localization.yaml}"
DATA_ROOT="${DATA_ROOT:-/home/ubuntu/yyk/datasets/mvtec}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/localization_refinement/dev5}"

run_case() {
    local case_name="$1"
    local fusion="$2"
    local upsample="$3"
    local output_dir="${OUTPUT_ROOT}/${case_name}"

    echo "[localization-ablation] ${case_name}"
    "${PYTHON_BIN}" "${PROJECT_ROOT}/run_experiment.py" \
        --config "${CONFIG}" \
        --data-root "${DATA_ROOT}" \
        --output-dir "${output_dir}" \
        --layer-fusion "${fusion}" \
        --upsample-mode "${upsample}" \
        --diagnostics
}

run_case "F0_mean_bilinear" "mean" "bilinear"
run_case "F1_geometric_bilinear" "geometric" "bilinear"
run_case "F2_minimum_bilinear" "minimum" "bilinear"
run_case "F3_mean_nearest_diagnostic" "mean" "nearest"

echo "[localization-ablation] complete: ${OUTPUT_ROOT}"
