#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/mvtec_dev5_multiscale.yaml}
DATA_ROOT=${DATA_ROOT:-/home/ubuntu/yyk/datasets/mvtec}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/localization_refinement/multiscale_dev5}
DEVICE=${DEVICE:-cuda}
BATCH_SIZE=${BATCH_SIZE:-8}

export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

run_case() {
  local name=$1
  shift
  echo "[multiscale] ${name}"
  python run_multiscale_experiment.py \
    --config "${CONFIG}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_ROOT}/${name}" \
    --batch-size "${BATCH_SIZE}" \
    --no-visualizations \
    --device "${DEVICE}" \
    "$@"
}

run_case M0_r1 --scales 1 --scale-weights 1.0
run_case M1_r3 --scales 3 --scale-weights 1.0
run_case M2_r5 --scales 5 --scale-weights 1.0
run_case M3_r1_r3_070_030 --scales 1 3 --scale-weights 0.7 0.3
run_case M4_r1_r3_r5_060_030_010 --scales 1 3 5 --scale-weights 0.6 0.3 0.1

echo "[multiscale] summary"
python summarize_multiscale_ablation.py --root "${OUTPUT_ROOT}"

echo "[multiscale] optional M2.5:"
echo "If M2_r5 AUROC drops < 1pp and localization is not catastrophic, run:"
echo "python run_multiscale_experiment.py --config ${CONFIG} --data-root ${DATA_ROOT} --output-dir ${OUTPUT_ROOT}/M2_5_r1_r5_050_050 --batch-size ${BATCH_SIZE} --no-visualizations --device ${DEVICE} --scales 1 5 --scale-weights 0.5 0.5"
