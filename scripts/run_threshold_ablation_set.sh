#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-sa_clip_nm}"
DATA_ROOT="${DATA_ROOT:-/home/ubuntu/yyk/datasets}"
CONFIG="${CONFIG:?Set CONFIG to a threshold YAML file}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:?Set EXPERIMENT_DIR}"
RUN_MODE="${RUN_MODE:?Set RUN_MODE to development or validation}"
MODEL_SEED="${MODEL_SEED:-42}"
THRESHOLD_SPLIT_SEEDS_TEXT="${THRESHOLD_SPLIT_SEEDS:-42 43 44}"
SELECTION_MANIFEST="${SELECTION_MANIFEST:-}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
RESUME="${RESUME:-0}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export HF_ENDPOINT HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

if [[ "${RUN_MODE}" != "development" && "${RUN_MODE}" != "validation" ]]; then
  echo "ERROR: RUN_MODE must be development or validation."
  exit 1
fi
CONFIG_STAGE="$(
  python - "${CONFIG}" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
print(config["experiment"].get("protocol_stage", ""))
PY
)"
if [[ "${CONFIG_STAGE}" != "${RUN_MODE}" ]]; then
  echo "ERROR: Config protocol_stage is '${CONFIG_STAGE}', not '${RUN_MODE}'."
  exit 1
fi
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

mapfile -t CATEGORIES < <(
  python - "${CONFIG}" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
for category in config["data"]["categories"]:
    print(category)
PY
)

for category in "${CATEGORIES[@]}"; do
  if [[ ! -d "${DATA_ROOT}/${category}/train/good" ]]; then
    echo "ERROR: Missing ${DATA_ROOT}/${category}/train/good"
    exit 1
  fi
done

if [[ -e "${EXPERIMENT_DIR}" && "${RESUME}" != "1" ]]; then
  echo "ERROR: Experiment directory already exists: ${EXPERIMENT_DIR}"
  echo "Use a new directory or run with RESUME=1."
  exit 1
fi
mkdir -p "${EXPERIMENT_DIR}"

METHOD_SPECS=()
if [[ "${RUN_MODE}" == "development" ]]; then
  METHOD_SPECS=(
    "global_quantile|0.995|0.95|0.01"
    "image_max_quantile|0.995|0.95|0.01"
    "image_topk_quantile|0.995|0.95|0.01"
  )
else
  if [[ -z "${SELECTION_MANIFEST}" || ! -f "${SELECTION_MANIFEST}" ]]; then
    echo "ERROR: validation requires SELECTION_MANIFEST from dev5."
    exit 1
  fi
  mapfile -t METHOD_SPECS < <(
    python - "${SELECTION_MANIFEST}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    manifest = json.load(handle)
parameters = manifest["selected_parameters"]
print(
    "|".join(
        [
            manifest["selected_method"],
            str(parameters["pixel_quantile"]),
            str(parameters["pixel_image_quantile"]),
            str(parameters["pixel_topk_fraction"]),
        ]
    )
)
PY
  )
  python - "${SELECTION_MANIFEST}" \
    "${EXPERIMENT_DIR}/selection_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
payload = json.loads(source.read_text(encoding="utf-8"))
target.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
PY
fi

desired_fingerprint() {
  local output_dir="$1"
  local method="$2"
  local split_seed="$3"
  local pixel_quantile="$4"
  local image_quantile="$5"
  local topk_fraction="$6"
  python - \
    "${CONFIG}" "${DATA_ROOT}" "${output_dir}" "${MODEL_SEED}" \
    "${method}" "${split_seed}" "${pixel_quantile}" \
    "${image_quantile}" "${topk_fraction}" <<'PY'
import sys
from sa_clip_nm.config import config_fingerprint, load_config

config = load_config(sys.argv[1])
config["data"]["root"] = sys.argv[2]
config["experiment"]["output_dir"] = sys.argv[3]
config["experiment"]["seed"] = int(sys.argv[4])
config["model"]["active_layers"] = [3, 6]
config["retrieval"]["spatial_mode"] = "none"
config["calibration"]["pixel_threshold_method"] = sys.argv[5]
config["calibration"]["threshold_split_seed"] = int(sys.argv[6])
config["calibration"]["pixel_quantile"] = float(sys.argv[7])
config["calibration"]["pixel_image_quantile"] = float(sys.argv[8])
config["calibration"]["pixel_topk_fraction"] = float(sys.argv[9])
print(config_fingerprint(config))
PY
}

is_complete() {
  local output_dir="$1"
  local expected_fingerprint="$2"
  python - \
    "${output_dir}/experiment_complete.json" \
    "${expected_fingerprint}" \
    "${CATEGORIES[@]}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
fingerprint = sys.argv[2]
categories = sys.argv[3:]
if not path.is_file():
    raise SystemExit(1)
with path.open("r", encoding="utf-8") as handle:
    marker = json.load(handle)
complete = (
    marker.get("status") == "complete"
    and marker.get("expected_categories") == categories
    and marker.get("completed_categories") == categories
    and marker.get("expected_count") == len(categories)
    and marker.get("completed_count") == len(categories)
    and marker.get("config_fingerprint") == fingerprint
)
raise SystemExit(0 if complete else 1)
PY
}

run_case() {
  local method="$1"
  local split_seed="$2"
  local pixel_quantile="$3"
  local image_quantile="$4"
  local topk_fraction="$5"
  local case_name="${method}_split${split_seed}"
  local output_dir="${EXPERIMENT_DIR}/${case_name}"
  local log_file="${EXPERIMENT_DIR}/${case_name}.log"
  local fingerprint
  fingerprint="$(
    desired_fingerprint \
      "${output_dir}" "${method}" "${split_seed}" \
      "${pixel_quantile}" "${image_quantile}" "${topk_fraction}"
  )"

  if [[ "${RESUME}" == "1" ]] && is_complete "${output_dir}" "${fingerprint}"; then
    echo "Skipping completed case: ${case_name}"
    return
  fi
  if [[ "${RESUME}" == "1" && ( -e "${output_dir}" || -e "${log_file}" ) ]]; then
    echo "Rerunning incomplete or mismatched case: ${case_name}"
  elif [[ -e "${output_dir}" || -e "${log_file}" ]]; then
    echo "ERROR: Output exists for ${case_name}."
    echo "Rerun with RESUME=1 to verify and restart mismatched cases."
    exit 1
  fi

  echo "=================================================="
  echo "Starting: ${case_name}"
  echo "Model seed: ${MODEL_SEED}"
  echo "Threshold split seed: ${split_seed}"
  echo "Output: ${output_dir}"
  echo "=================================================="

  python run_experiment.py \
    --config "${CONFIG}" \
    --data-root "${DATA_ROOT}" \
    --layers 3 6 \
    --output-dir "${output_dir}" \
    --spatial-mode none \
    --seed "${MODEL_SEED}" \
    --threshold-split-seed "${split_seed}" \
    --pixel-threshold-method "${method}" \
    --pixel-quantile "${pixel_quantile}" \
    --pixel-image-quantile "${image_quantile}" \
    --pixel-topk-fraction "${topk_fraction}" \
    --device cuda \
    2>&1 | tee "${log_file}"
}

read -r -a THRESHOLD_SPLIT_SEEDS_ARRAY <<< "${THRESHOLD_SPLIT_SEEDS_TEXT}"
for spec in "${METHOD_SPECS[@]}"; do
  IFS="|" read -r method pixel_quantile image_quantile topk_fraction <<< "${spec}"
  for split_seed in "${THRESHOLD_SPLIT_SEEDS_ARRAY[@]}"; do
    run_case \
      "${method}" "${split_seed}" "${pixel_quantile}" \
      "${image_quantile}" "${topk_fraction}"
  done
done

SUMMARY_ARGS=(
  --experiment-dir "${EXPERIMENT_DIR}"
  --mode "${RUN_MODE}"
)
if [[ "${RUN_MODE}" == "validation" ]]; then
  SUMMARY_ARGS+=(--selection-manifest "${SELECTION_MANIFEST}")
fi
python summarize_threshold_ablation.py "${SUMMARY_ARGS[@]}" \
  2>&1 | tee "${EXPERIMENT_DIR}/summary.log"

echo
echo "Threshold experiment finished: ${EXPERIMENT_DIR}"
