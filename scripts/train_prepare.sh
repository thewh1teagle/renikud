#!/usr/bin/env bash
set -euo pipefail

INPUT=${1:?"Usage: $0 <input.tsv>"}

uv run python -c "import sys; sys.path.insert(0, 'src'); from tokenization import save_tokenizer; save_tokenizer('src/tokenizer.json')"

uv run scripts/split_dataset.py "$INPUT" dataset/train.tsv dataset/val.tsv

uv run src/data_align.py dataset/train.tsv dataset/train_alignment.jsonl
uv run src/data_align.py dataset/val.tsv dataset/val_alignment.jsonl

uv run src/data_tokenize.py dataset/train_alignment.jsonl dataset/.cache/train
uv run src/data_tokenize.py dataset/val_alignment.jsonl dataset/.cache/val
