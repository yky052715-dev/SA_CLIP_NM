#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-sa_clip_nm}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
DATA_ROOT="${DATA_ROOT:-/home/ubuntu/yyk/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mvtec15_adaptive}"
LOG_FILE="${LOG_FILE:-outputs/mvtec15_adaptive.log}"

CATEGORIES=(
  bottle
  cable
  capsule
  carpet
  grid
  hazelnut
  leather
  metal_nut
  pill
  screw
  tile
  toothbrush
  transistor
  wood
  zipper
)

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

missing=()
for category in "${CATEGORIES[@]}"; do
  if [[ ! -d "${DATA_ROOT}/${category}/train/good" ]]; then
    missing+=("${category}: train/good")
  fi
  if [[ ! -d "${DATA_ROOT}/${category}/test" ]]; then
    missing+=("${category}: test")
  fi
done

if (( ${#missing[@]} > 0 )); then
  echo "ERROR: Incomplete MVTec AD categories:"
  printf '  %s\n' "${missing[@]}"
  exit 1
fi

if [[ -e "${OUTPUT_DIR}" || -e "${LOG_FILE}" ]]; then
  echo "ERROR: Output or log already exists:"
  echo "  ${OUTPUT_DIR}"
  echo "  ${LOG_FILE}"
  echo "Move the existing files or override OUTPUT_DIR and LOG_FILE."
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_DIR}")"
mkdir -p "$(dirname "${LOG_FILE}")"

echo "=================================================="
echo "SA-CLIP-NM Full MVTec AD Experiment"
echo "Start time: $(date)"
echo "Project: ${PROJECT_DIR}"
echo "Dataset: ${DATA_ROOT}"
echo "Output: ${OUTPUT_DIR}"
echo "Log: ${LOG_FILE}"
echo "HF endpoint: ${HF_ENDPOINT}"
echo "Spatial mode: adaptive"
echo "Lambda max: 0.10"
echo "Memory sampling: k-center"
echo "Categories: ${#CATEGORIES[@]}"
echo "=================================================="

python run_experiment.py \
  --config configs/mvtec15.yaml \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --spatial-mode adaptive \
  --lambda-max 0.10 \
  --device cuda \
  2>&1 | tee "${LOG_FILE}"

echo
echo "=================================================="
echo "Full MVTec AD experiment finished"
echo "End time: $(date)"
echo "=================================================="

SUMMARY="${OUTPUT_DIR}/metrics_summary.md"
if [[ ! -f "${SUMMARY}" ]]; then
  echo "ERROR: Summary was not generated: ${SUMMARY}"
  exit 1
fi

echo
cat "${SUMMARY}"
