"""
Split a TSV dataset into train and val sets.

Usage:
    uv run scripts/split_dataset.py dataset/data.tsv dataset/train.tsv dataset/val.tsv
"""

import argparse
import random
from pathlib import Path

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("train", type=Path)
    parser.add_argument("val", type=Path)
    parser.add_argument("--val-max", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print('Splitting dataset...')
    lines = args.input.read_text(encoding="utf-8").splitlines()
    random.seed(args.seed)
    random.shuffle(lines)

    val_size = min(args.val_max, len(lines))
    train_lines = lines[val_size:]
    val_lines = lines[:val_size]

    for path, data in tqdm([(args.train, train_lines), (args.val, val_lines)], desc="Writing"):
        path.write_text("\n".join(data) + "\n", encoding="utf-8")

    print(f"Total: {len(lines)} | Train: {len(train_lines)} | Val: {len(val_lines)}")


if __name__ == "__main__":
    main()
