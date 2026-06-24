#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/mvtec_dev5_highres_clip.yaml}"
DATA_ROOT="${DATA_ROOT:-/home/ubuntu/yyk/datasets/mvtec}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/localization_refinement/highres_clip_dev5}"

run_case() {
    local case_name="$1"
    local image_size="$2"
    echo "[highres-clip] ${case_name}"
    "${PYTHON_BIN}" "${PROJECT_ROOT}/run_highres_clip_experiment.py" \
        --config "${CONFIG}" \
        --data-root "${DATA_ROOT}" \
        --output-dir "${OUTPUT_ROOT}/${case_name}" \
        --image-size "${image_size}" \
        --evaluation-size 448 \
        --batch-size 8 \
        --memory-ratio 0.10 \
        --device cuda
}

run_case "H0_clip224" "224"
run_case "H1_clip336" "336"
run_case "H2_clip448" "448"

"${PYTHON_BIN}" "${PROJECT_ROOT}/summarize_highres_clip_ablation.py" \
    --root "${OUTPUT_ROOT}"

echo "[highres-clip] complete: ${OUTPUT_ROOT}"
