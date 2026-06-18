#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-sa_clip_nm}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export HF_ENDPOINT

if [[ "${CONDA_DEFAULT_ENV:-}" != "${ENV_NAME}" ]]; then
  echo "ERROR: Current Conda environment is '${CONDA_DEFAULT_ENV:-none}'."
  echo "Activate the expected environment first:"
  echo "conda activate ${ENV_NAME}"
  exit 1
fi

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is not available in the current environment.")
print("GPU:", torch.cuda.get_device_name(0))
PY

run_case() {
  local name="$1"
  shift
  local output_dir="outputs/${name}"
  local log_file="outputs/${name}.log"

  if [[ -e "${output_dir}" || -e "${log_file}" ]]; then
    echo "ERROR: Output already exists for ${name}:"
    echo "  ${output_dir}"
    echo "  ${log_file}"
    echo "Rename or move the existing output before rerunning."
    exit 1
  fi

  mkdir -p outputs
  echo "=================================================="
  echo "Starting: ${name}"
  echo "Output: ${output_dir}"
  echo "Log: ${log_file}"
  echo "HF_ENDPOINT: ${HF_ENDPOINT}"
  echo "Start time: $(date)"
  echo "=================================================="

  python run_experiment.py \
    --config configs/mvtec_dev5.yaml \
    --output-dir "${output_dir}" \
    --device cuda \
    "$@" 2>&1 | tee "${log_file}"

  echo "Finished: ${name}"
  echo "End time: $(date)"
  echo
}

run_case "dev5_adaptive" \
  --spatial-mode adaptive \
  --lambda-max 0.10

run_case "dev5_no_spatial" \
  --spatial-mode none

run_case "dev5_fixed005" \
  --spatial-mode fixed \
  --fixed-lambda 0.05

echo "=================================================="
echo "All dev5 spatial ablations finished."
echo "=================================================="

for summary in \
  outputs/dev5_adaptive/metrics_summary.md \
  outputs/dev5_no_spatial/metrics_summary.md \
  outputs/dev5_fixed005/metrics_summary.md
do
  echo
  echo "----- ${summary} -----"
  cat "${summary}"
done

