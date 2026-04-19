#!/usr/bin/env bash
set -euo pipefail

if [ ! -f knesset_phonemes_v1.txt ]; then
  wget -c https://huggingface.co/datasets/thewh1teagle/phonikud-phonemes-data/resolve/main/knesset_phonemes_v1.txt.7z
  7z x knesset_phonemes_v1.txt.7z
fi

uv run scripts/prepare_dataset.py knesset_phonemes_v1.txt knesset.txt

uv run scripts/split_dataset.py knesset.txt dataset/train.txt dataset/val.txt

uv run scripts/prepare_tokens.py dataset/train.txt dataset/.cache/train
uv run scripts/prepare_tokens.py dataset/val.txt dataset/.cache/val
