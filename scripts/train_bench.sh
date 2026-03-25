#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT=${1:?"Usage: $0 <checkpoint-dir>"}

if [ ! -f gt.tsv ]; then
  wget https://raw.githubusercontent.com/thewh1teagle/heb-g2p-benchmark/refs/heads/main/gt.tsv
fi

uv run scripts/benchmark.py --checkpoint "$CHECKPOINT" --gt gt.tsv
