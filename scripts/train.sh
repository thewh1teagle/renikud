#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/train.sh                                      # train from scratch
#   ./scripts/train.sh --resume outputs/.../checkpoint-N   # resume training
#   ./scripts/train.sh --resume outputs/.../checkpoint-N --reset-steps  # finetune (load weights, reset steps)

RESUME=""
RESET_STEPS=""
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume) RESUME="--resume $2"; shift 2 ;;
        --reset-steps) RESET_STEPS="--reset-steps"; shift ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done

uv run accelerate launch src/train.py \
  --train-dataset dataset/train.jsonl \
  --eval-dataset dataset/val.jsonl \
  --output-dir outputs/g2p-classifier \
  $RESUME \
  $RESET_STEPS \
  "${EXTRA[@]+"${EXTRA[@]}"}"
