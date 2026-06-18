#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/eval_pope.sh [random|popular|adversarial|all] [--dry-run] [extra eval_pope.py args...]

Examples:
  bash scripts/eval_pope.sh
  bash scripts/eval_pope.sh random --limit 20
  bash scripts/eval_pope.sh all --load-in-4bit

Environment overrides:
  MODEL_PATH    default: models/llava-hf/llava-1.5-7b-hf
  IMAGE_ROOT    default: data/benchmarks/pope/coco/images
  POPE_ROOT     default: data/benchmarks/pope/coco
  OUTPUT_ROOT   default: experiments/outputs
  PYTHON        default: python
EOF
}

MODEL_PATH="${MODEL_PATH:-models/llava-hf/llava-1.5-7b-hf}"
POPE_ROOT="${POPE_ROOT:-data/benchmarks/pope/coco}"
IMAGE_ROOT="${IMAGE_ROOT:-${POPE_ROOT}/images}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/outputs}"
PYTHON_BIN="${PYTHON:-python}"

split="random"
dry_run=0
extra_args=()

if [[ $# -gt 0 && "$1" =~ ^(random|popular|adversarial|all)$ ]]; then
  split="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

if [[ "$split" == "all" ]]; then
  splits=(random popular adversarial)
else
  splits=("$split")
fi

quote_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

run_split() {
  local current_split="$1"
  local label_file="${POPE_ROOT}/coco_pope_${current_split}.json"
  local output_file="${OUTPUT_ROOT}/pope_${current_split}_answers.jsonl"
  local metrics_file="${OUTPUT_ROOT}/pope_${current_split}_metrics.json"
  local cmd=(
    "$PYTHON_BIN" scripts/eval_pope.py
    --model-path "$MODEL_PATH"
    --label-file "$label_file"
    --image-root "$IMAGE_ROOT"
    --output "$output_file"
    --metrics-output "$metrics_file"
    --load-in-4bit
  )

  if [[ ${#extra_args[@]} -gt 0 ]]; then
    cmd+=("${extra_args[@]}")
  fi

  if [[ "$dry_run" -eq 1 ]]; then
    quote_cmd "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
}

for current_split in "${splits[@]}"; do
  run_split "$current_split"
done
