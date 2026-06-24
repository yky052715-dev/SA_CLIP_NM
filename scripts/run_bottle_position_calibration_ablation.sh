#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/mvtec_dev5_position_localization.yaml}"
DATA_ROOT="${DATA_ROOT:-/home/ubuntu/yyk/datasets/mvtec}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/localization_refinement/p3_bottle}"

run_case() {
    local case_name="$1"
    local rho="$2"
    echo "[position-calibration] ${case_name}"
    "${PYTHON_BIN}" "${PROJECT_ROOT}/run_position_localization_experiment.py" \
        --config "${CONFIG}" \
        --data-root "${DATA_ROOT}" \
        --categories bottle \
        --output-dir "${OUTPUT_ROOT}/${case_name}" \
        --position-rho "${rho}" \
        --layer-fusion mean
}

run_case "P31_position_rho025" "0.25"
run_case "P32_position_rho050" "0.50"

"${PYTHON_BIN}" "${PROJECT_ROOT}/summarize_position_calibration.py" \
    --root "${OUTPUT_ROOT}"

echo "[position-calibration] complete: ${OUTPUT_ROOT}"
