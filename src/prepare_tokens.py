"""
Pretokenize dataset files and save as Arrow datasets for fast loading.

Usage:
    uv run src/prepare_tokens.py --input dataset/train.txt --output dataset/.cache/train
    uv run src/prepare_tokens.py --input dataset/val.txt --output dataset/.cache/val
"""

import argparse
import os
from pathlib import Path

from datasets import Dataset
from tqdm import tqdm
from constants import MAX_LEN
from tokenization import encode_ipa, load_encoder_tokenizer


def process_file(input_path: str, output_path: str, max_len: int):
    input_p = Path(input_path)
    output_p = Path(output_path)

    # Skip if cache is newer than source
    if output_p.exists():
        cache_info = output_p / "dataset_info.json"
        if cache_info.exists():
            if os.path.getmtime(cache_info) > os.path.getmtime(input_p):
                print(f"Skipping {input_p.name} (cache is up to date)")
                return

    tokenizer = load_encoder_tokenizer()

    lines = input_p.read_text(encoding="utf-8").strip().split("\n")

    rows = {"encoder_ids": [], "encoder_mask": [], "decoder_ids": []}
    skipped = 0

    for line in tqdm(lines, desc=f"Processing {input_p.name}"):
        parts = line.split("\t")
        if len(parts) != 2:
            skipped += 1
            continue

        hebrew, ipa = parts[0].strip(), parts[1].strip()
        if not hebrew or not ipa:
            skipped += 1
            continue

        enc = tokenizer(hebrew, truncation=True, max_length=max_len, return_tensors="np")
        try:
            dec_ids = encode_ipa(ipa)
        except ValueError:
            skipped += 1
            continue

        if not dec_ids:
            skipped += 1
            continue

        rows["encoder_ids"].append(enc["input_ids"][0].tolist())
        rows["encoder_mask"].append(enc["attention_mask"][0].tolist())
        rows["decoder_ids"].append(dec_ids)

    ds = Dataset.from_dict(rows)
    output_p.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(output_p))
    print(f"Saved {len(ds)} samples to {output_p} (skipped {skipped})")


def main():
    parser = argparse.ArgumentParser(description="Pretokenize G2P dataset")
    parser.add_argument("--input", type=str, required=True, help="Input .txt file (TSV)")
    parser.add_argument("--output", type=str, required=True, help="Output Arrow dataset dir")
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    args = parser.parse_args()

    process_file(args.input, args.output, args.max_len)


if __name__ == "__main__":
    main()
