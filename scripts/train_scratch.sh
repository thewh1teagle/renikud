#!/usr/bin/env bash
set -euo pipefail

uv run src/train.py \
  --train-dataset dataset/.cache/train \
  --eval-dataset dataset/.cache/val \
  --output-dir outputs/g2p-classifier \
  --train-batch-size 16
