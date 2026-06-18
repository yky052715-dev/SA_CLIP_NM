#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-sa_clip_nm}"

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "ERROR: Conda environment already exists: ${ENV_NAME}"
  echo "Activate it directly or choose another name:"
  echo "ENV_NAME=sa_clip_nm_v2 bash scripts/create_conda_env.sh"
  exit 1
fi

conda create -y -n "${ENV_NAME}" python=3.10.8 pip setuptools wheel

conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${ENV_NAME}" python -m pip install \
  torch==2.1.2 \
  torchvision==0.16.2 \
  --index-url https://download.pytorch.org/whl/cu121

conda run -n "${ENV_NAME}" python -m pip install -r requirements-base.txt
conda run -n "${ENV_NAME}" python -m pip install -e . --no-deps

echo ""
echo "Environment created: ${ENV_NAME}"
echo "Activate it with:"
echo "conda activate ${ENV_NAME}"
echo ""
echo "Then verify:"
echo "python scripts/check_environment.py"
echo "python smoke_test.py"

