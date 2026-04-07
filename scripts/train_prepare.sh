#!/usr/bin/env bash
set -euo pipefail

INPUT=${1:?"Usage: $0 <input.tsv>"}

uv run scripts/split_dataset.py "$INPUT" dataset/train.tsv dataset/val.tsv

go build -o aligner/align ./aligner/
./aligner/align dataset/train.tsv dataset/train.jsonl
./aligner/align dataset/val.tsv dataset/val.jsonl

