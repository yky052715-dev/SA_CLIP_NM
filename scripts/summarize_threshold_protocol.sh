#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEV5_DIR="${DEV5_DIR:-outputs/dev5_threshold_selection}"
VALIDATION10_DIR="${VALIDATION10_DIR:-outputs/validation10_locked_threshold}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/threshold_protocol_summary}"

cd "${PROJECT_DIR}"
python summarize_threshold_protocol.py \
  --dev5-dir "${DEV5_DIR}" \
  --validation10-dir "${VALIDATION10_DIR}" \
  --output-dir "${OUTPUT_DIR}"
