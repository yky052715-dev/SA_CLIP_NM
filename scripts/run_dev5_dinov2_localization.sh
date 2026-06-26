#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/home/ubuntu/yyk/datasets/mvtec}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/dual_backbone/dinov2_dev5}
DEVICE=${DEVICE:-cuda}
BATCH_SIZE=${BATCH_SIZE:-4}
HUB_DIR=${HUB_DIR:-}

CMD=(
  python run_dinov2_mvtec_localization.py
  --data-root "${DATA_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --categories bottle metal_nut grid leather screw
  --input-size 518
  --evaluation-size 448
  --batch-size "${BATCH_SIZE}"
  --device "${DEVICE}"
)

if [[ -n "${HUB_DIR}" ]]; then
  CMD+=(--hub-dir "${HUB_DIR}")
fi

"${CMD[@]}"
