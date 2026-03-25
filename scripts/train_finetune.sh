#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT=${1:?"Usage: $0 <checkpoint-dir>"}

uv run src/train.py \
  --train-dataset dataset/.cache/train \
  --eval-dataset dataset/.cache/val \
  --output-dir outputs/g2p-classifier-ft \
  --init-from-checkpoint "$CHECKPOINT" \
  --encoder-lr 2e-6 \
  --head-lr 1e-5 \
  --warmup-steps 200
