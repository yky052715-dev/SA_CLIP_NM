#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/mvtec_dev5_highres_clip_adaptive_threshold.yaml}
DATA_ROOT=${DATA_ROOT:-/home/ubuntu/yyk/datasets/mvtec}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/dual_backbone/clip336_dev5_maps}
DEVICE=${DEVICE:-cuda}
BATCH_SIZE=${BATCH_SIZE:-8}

python export_highres_clip_maps.py \
  --config "${CONFIG}" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --categories bottle metal_nut grid leather screw \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}"
