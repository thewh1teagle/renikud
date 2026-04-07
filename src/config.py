"""CLI argument parsing for training."""

from __future__ import annotations

import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Hebrew G2P classifier model")
    parser.add_argument("--train-dataset", type=str, required=True, help="Path to train.jsonl")
    parser.add_argument("--eval-dataset", type=str, required=True, help="Path to val.jsonl")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=20)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--freeze-encoder-steps", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--reset-steps", action="store_true", default=False, help="Load weights from checkpoint but reset step counter (for finetuning)")
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=torch.cuda.is_available(),
    )
    parser.add_argument("--flash-attention", action="store_true", default=False)
    parser.add_argument("--dataloader-workers", type=int, default=0)
    return parser.parse_args()
