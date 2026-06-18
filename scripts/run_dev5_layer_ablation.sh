#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-sa_clip_nm}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
DATA_ROOT="${DATA_ROOT:-/home/ubuntu/yyk/datasets}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-outputs/dev5_layer_ablation}"
CONFIG="${CONFIG:-configs/mvtec_dev5_layer_ablation.yaml}"
RESUME="${RESUME:-0}"

CATEGORIES=(bottle metal_nut grid leather screw)

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export HF_ENDPOINT

if [[ "${CONDA_DEFAULT_ENV:-}" != "${ENV_NAME}" ]]; then
  echo "ERROR: Current Conda environment is '${CONDA_DEFAULT_ENV:-none}'."
  echo "Run: conda activate ${ENV_NAME}"
  exit 1
fi

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is not available.")
print("GPU:", torch.cuda.get_device_name(0))
PY

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "ERROR: MVTec AD root does not exist: ${DATA_ROOT}"
  exit 1
fi

for category in "${CATEGORIES[@]}"; do
  if [[ ! -d "${DATA_ROOT}/${category}/train/good" ]]; then
    echo "ERROR: Missing ${DATA_ROOT}/${category}/train/good"
    exit 1
  fi
  if [[ ! -d "${DATA_ROOT}/${category}/test" ]]; then
    echo "ERROR: Missing ${DATA_ROOT}/${category}/test"
    exit 1
  fi
done

if [[ -e "${EXPERIMENT_DIR}" && "${RESUME}" != "1" ]]; then
  echo "ERROR: Experiment directory already exists: ${EXPERIMENT_DIR}"
  echo "Use another name, for example:"
  echo "EXPERIMENT_DIR=outputs/dev5_layer_ablation_v2 bash scripts/run_dev5_layer_ablation.sh"
  echo "Or resume completed cases:"
  echo "RESUME=1 bash scripts/run_dev5_layer_ablation.sh"
  exit 1
fi

mkdir -p "${EXPERIMENT_DIR}"

run_case() {
  local name="$1"
  shift
  local output_dir="${EXPERIMENT_DIR}/${name}"
  local log_file="${EXPERIMENT_DIR}/${name}.log"

  if [[ -f "${output_dir}/metrics_summary.json" && "${RESUME}" == "1" ]]; then
    echo "Skipping completed case: ${name}"
    return
  fi
  if [[ -e "${output_dir}" || -e "${log_file}" ]]; then
    echo "ERROR: Incomplete or conflicting output exists for ${name}:"
    echo "  ${output_dir}"
    echo "  ${log_file}"
    echo "Move that case aside before resuming."
    exit 1
  fi

  echo "=================================================="
  echo "Starting layer case: ${name}"
  echo "Output: ${output_dir}"
  echo "Start time: $(date)"
  echo "=================================================="

  python run_experiment.py \
    --config "${CONFIG}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${output_dir}" \
    --spatial-mode adaptive \
    --lambda-max 0.10 \
    --device cuda \
    "$@" 2>&1 | tee "${log_file}"

  echo "Finished: ${name}"
  echo "End time: $(date)"
  echo
}

run_case block3 --layers 3
run_case block6 --layers 6
run_case block9 --layers 9
run_case block12 --layers 12
run_case fusion_3_6_9_12 --layers 3 6 9 12

python summarize_layer_ablation.py \
  --experiment-dir "${EXPERIMENT_DIR}" \
  2>&1 | tee "${EXPERIMENT_DIR}/summary.log"

echo
echo "=================================================="
echo "Layer ablation finished."
echo "Results: ${EXPERIMENT_DIR}"
echo "Main summary: ${EXPERIMENT_DIR}/layer_ablation_means.md"
echo "Per-category summary: ${EXPERIMENT_DIR}/layer_ablation_per_category.md"
echo "=================================================="
