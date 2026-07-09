#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH="$PWD/src:$PWD/LLaVA${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1

BACKEND="${BACKEND:-native}"

python scripts/infer.py --backend "$BACKEND" "$@"
