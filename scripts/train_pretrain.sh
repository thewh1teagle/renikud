#!/usr/bin/env bash
set -euo pipefail

TRAIN_FILE=${1:?"Usage: $0 <train.txt> <eval.txt> [output_dir]"}
EVAL_FILE=${2:?"Usage: $0 <train.txt> <eval.txt> [output_dir]"}
OUTPUT_DIR=${3:-outputs/mlm-pretrain}

uv run accelerate launch --multi_gpu --mixed_precision fp16 scripts/pretrain_mlm.py \
    --train-file "$TRAIN_FILE" \
    --eval-file "$EVAL_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --train-batch-size 64 \
    --eval-batch-size 64 \
    --epochs 3 \
    --lr 1e-4 \
    --warmup-steps 2000 \
    --gradient-accumulation-steps 4 \
    --save-steps 2000 \
    --logging-steps 100 \
    --save-total-limit 5 \
    --dataloader-workers 4 \
    --fp16
