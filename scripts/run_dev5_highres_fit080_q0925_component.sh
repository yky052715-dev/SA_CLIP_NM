#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/home/ubuntu/yyk/datasets/mvtec}"
BASE_CONFIG="${BASE_CONFIG:-${PROJECT_ROOT}/configs/mvtec_dev5_highres_clip.yaml}"
CANDIDATE_CONFIG="${CANDIDATE_CONFIG:-${PROJECT_ROOT}/configs/mvtec_dev5_highres_clip_fit080_q0925_component.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/localization_refinement/highres_fit080_q0925_component_seed42}"
SEED="${SEED:-42}"
THRESHOLD_SEED="${THRESHOLD_SEED:-${SEED}}"
DEVICE="${DEVICE:-cuda}"

run_case() {
    local case_name="$1"
    local config_path="$2"
    local image_size="$3"
    local memory_ratio="$4"

    echo "[highres-fit080-q0925-component] ${case_name}"
    "${PYTHON_BIN}" "${PROJECT_ROOT}/run_highres_clip_stability_experiment.py" \
        --config "${config_path}" \
        --data-root "${DATA_ROOT}" \
        --output-dir "${OUTPUT_ROOT}/${case_name}" \
        --image-size "${image_size}" \
        --evaluation-size 448 \
        --batch-size 8 \
        --memory-ratio "${memory_ratio}" \
        --seed "${SEED}" \
        --threshold-split-seed "${THRESHOLD_SEED}" \
        --device "${DEVICE}"
}

run_case "H0_clip224" "${BASE_CONFIG}" 224 0.10
run_case "H1_clip336_fit080_q0925_component" "${CANDIDATE_CONFIG}" 336 0.0444444444

"${PYTHON_BIN}" - <<PY
import json
import math
from pathlib import Path

root = Path(r"${OUTPUT_ROOT}")
cases = ["H0_clip224", "H1_clip336_fit080_q0925_component"]
metrics = [
    "pixel_AUROC",
    "pixel_F1_calibrated",
    "pixel_F1_oracle",
    "localization_overseg_anomaly_macro",
    "localization_recall_anomaly_macro",
    "localization_small_defect_f1_macro",
    "localization_small_defect_recall_macro",
    "localization_test_normal_image_positive_rate",
    "inference_ms_per_image",
]

def finite(value):
    return value is not None and not (isinstance(value, float) and math.isnan(value))

def load(case):
    return json.loads((root / case / "metrics_summary.json").read_text())

def mean_from_categories(summary, key):
    values = [row.get(key) for row in summary["categories"]]
    values = [float(value) for value in values if finite(value)]
    return sum(values) / len(values) if values else float("nan")

summaries = {case: load(case) for case in cases}
print("\n# H1 fit080 q=0.925 + component postprocess seed42")
print("\n## Mean metrics")
print("| metric | H0 | H1 | delta |")
print("|---|---:|---:|---:|")
for key in metrics:
    h0 = summaries[cases[0]].get(key)
    h1 = summaries[cases[1]].get(key)
    if not finite(h0):
        h0 = mean_from_categories(summaries[cases[0]], key)
    if not finite(h1):
        h1 = mean_from_categories(summaries[cases[1]], key)
    print(f"| {key} | {h0:.6f} | {h1:.6f} | {h1 - h0:+.6f} |")

print("\n## H1 per-category")
print("| category | F1 | oracle | overseg | recall | small? | normal_fp | q | min_area_px |")
print("|---|---:|---:|---:|---:|---|---:|---:|---:|")
for row in summaries[cases[1]]["categories"]:
    print(
        f"| {row['category']} "
        f"| {row.get('pixel_F1_calibrated'):.3f} "
        f"| {row.get('pixel_F1_oracle'):.3f} "
        f"| {row.get('localization_overseg_anomaly_macro'):.3f} "
        f"| {row.get('localization_recall_anomaly_macro'):.3f} "
        f"| - "
        f"| {row.get('localization_test_normal_image_positive_rate'):.3f} "
        f"| {row.get('pixel_image_quantile_calibrated')} "
        f"| {row.get('postprocess_min_component_area_pixels')} |"
    )
PY

echo "[highres-fit080-q0925-component] complete: ${OUTPUT_ROOT}"
