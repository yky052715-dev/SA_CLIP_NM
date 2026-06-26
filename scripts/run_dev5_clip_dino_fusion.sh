#!/usr/bin/env bash
set -euo pipefail

CLIP_ROOT=${CLIP_ROOT:-outputs/dual_backbone/clip336_dev5_maps}
DINO_ROOT=${DINO_ROOT:-outputs/dual_backbone/dinov2_dev5}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/dual_backbone/fusion_dev5}

python fuse_clip_dino_maps.py \
  --clip-root "${CLIP_ROOT}" \
  --dino-root "${DINO_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --categories bottle metal_nut grid leather screw \
  --alphas 0.5 0.7 0.3
