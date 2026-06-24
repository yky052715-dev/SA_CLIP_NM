#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/mvtec_dev5_tiled_localization.yaml}"
DATA_ROOT="${DATA_ROOT:-/home/ubuntu/yyk/datasets/mvtec}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/localization_refinement/dev5_p2}"

run_case() {
    local case_name="$1"
    local mode="$2"
    local fusion="$3"
    local output_dir="${OUTPUT_ROOT}/${case_name}"

    echo "[tiled-localization-ablation] ${case_name}"
    "${PYTHON_BIN}" "${PROJECT_ROOT}/run_localization_experiment.py" \
        --config "${CONFIG}" \
        --data-root "${DATA_ROOT}" \
        --output-dir "${output_dir}" \
        --localization-mode "${mode}" \
        --layer-fusion "${fusion}"
}

run_case "L0_global_mean" "global" "mean"
run_case "L1_global_geometric" "global" "geometric"
run_case "L2_tiled_mean" "tiled" "mean"
run_case "L3_tiled_geometric" "tiled" "geometric"

"${PYTHON_BIN}" "${PROJECT_ROOT}/summarize_tiled_localization_ablation.py" \
    --root "${OUTPUT_ROOT}"

echo "[tiled-localization-ablation] complete: ${OUTPUT_ROOT}"
