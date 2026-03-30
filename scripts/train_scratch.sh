#!/usr/bin/env bash
set -euo pipefail

uv run accelerate launch src/train.py \
  --train-dataset dataset/.cache/train \
  --eval-dataset dataset/.cache/val \
  --output-dir outputs/g2p-classifier \
  --train-batch-size 16 \
  --lang hebrew \
  --no-fp16 \
  "$@"
