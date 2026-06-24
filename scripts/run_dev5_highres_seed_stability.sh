#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/mvtec_dev5_highres_clip.yaml}"
DATA_ROOT="${DATA_ROOT:-/home/ubuntu/yyk/datasets/mvtec}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/localization_refinement/highres_clip_stability}"
SEEDS=(42 3407 2024)

is_complete() {
    local output_dir="$1"
    "${PYTHON_BIN}" - "${output_dir}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "experiment_complete.json"
if not path.is_file():
    raise SystemExit(1)
payload = json.loads(path.read_text(encoding="utf-8"))
raise SystemExit(0 if payload.get("status") == "complete" else 1)
PY
}

run_case() {
    local seed="$1"
    local case_name="$2"
    local image_size="$3"
    local memory_ratio="$4"
    local output_dir="${OUTPUT_ROOT}/seed_${seed}/${case_name}"

    if is_complete "${output_dir}"; then
        echo "[highres-seed] skip complete: seed=${seed} ${case_name}"
        return
    fi
    echo "[highres-seed] seed=${seed} ${case_name}"
    "${PYTHON_BIN}" "${PROJECT_ROOT}/run_highres_clip_stability_experiment.py" \
        --config "${CONFIG}" \
        --data-root "${DATA_ROOT}" \
        --output-dir "${output_dir}" \
        --image-size "${image_size}" \
        --evaluation-size 448 \
        --batch-size 8 \
        --memory-ratio "${memory_ratio}" \
        --seed "${seed}" \
        --threshold-split-seed "${seed}" \
        --device cuda
}

for seed in "${SEEDS[@]}"; do
    run_case "${seed}" "H0_clip224" "224" "0.10"
    run_case "${seed}" "H1_clip336_matched" "336" "0.0444444444"
done

"${PYTHON_BIN}" "${PROJECT_ROOT}/summarize_highres_seed_stability.py" \
    --root "${OUTPUT_ROOT}" \
    --seeds 42 3407 2024

echo "[highres-seed] complete: ${OUTPUT_ROOT}"
