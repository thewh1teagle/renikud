"""
Tokenized Arrow dataset preparation for nikud diacritization.
Reads vocalized Hebrew text, strips nikud for input_ids, extracts nikud/shin labels per Hebrew character.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import datasets

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from constants import IGNORE_INDEX
from nikud import is_hebrew_letter, remove_nikud, sort_diacritics, extract_labels, _NIKUD_PATTERN
from tokenization import load_tokenizer

_worker_tokenizer = None


def iter_char_diacritics(text: str):
    i = 0
    while i < len(text):
        char = text[i]
        i += 1
        diacritics = ""
        while i < len(text) and _NIKUD_PATTERN.match(text[i]):
            diacritics += text[i]
            i += 1
        yield char, diacritics


def tokenize_batch(batch):
    global _worker_tokenizer
    if _worker_tokenizer is None:
        _worker_tokenizer = load_tokenizer()

    vocalized_sentences = batch["vocalized"]
    batch_input_ids = []
    batch_attention_mask = []
    batch_nikud_labels = []
    batch_shin_labels = []

    for vocalized in vocalized_sentences:
        normalized = sort_diacritics(vocalized)
        stripped = remove_nikud(normalized)

        # Extract per-Hebrew-char labels from the normalized vocalized text
        char_nikud: dict[int, int] = {}
        char_shin: dict[int, int] = {}
        stripped_pos = 0
        for char, diacritics in iter_char_diacritics(normalized):
            if is_hebrew_letter(char):
                n_id, s_id = extract_labels(char, diacritics)
                char_nikud[stripped_pos] = n_id
                char_shin[stripped_pos] = s_id
            stripped_pos += len(char)  # diacritics don't appear in stripped

        enc = _worker_tokenizer(
            stripped,
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )
        offsets = enc["offset_mapping"]
        seq_len = len(enc["input_ids"])

        nikud_labels = [IGNORE_INDEX] * seq_len
        shin_labels = [IGNORE_INDEX] * seq_len

        for tok_idx, (start, end) in enumerate(offsets):
            if end - start != 1:
                continue
            if start in char_nikud:
                nikud_labels[tok_idx] = char_nikud[start]
                shin_labels[tok_idx] = char_shin[start]

        batch_input_ids.append(enc["input_ids"])
        batch_attention_mask.append(enc["attention_mask"])
        batch_nikud_labels.append(nikud_labels)
        batch_shin_labels.append(shin_labels)

    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "nikud_labels": batch_nikud_labels,
        "shin_labels": batch_shin_labels,
        "vocalized": vocalized_sentences,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input .txt file (one vocalized Hebrew sentence per line)")
    parser.add_argument("output", help="Output Arrow directory")
    parser.add_argument("--workers", type=int, default=min(64, os.cpu_count()))
    parser.add_argument("--batch_size", type=int, default=1000)
    args = parser.parse_args()

    print(f"Reading {args.input}...")
    lines = Path(args.input).read_text(encoding="utf-8").splitlines()
    lines = [l for l in lines if l.strip()]
    ds = datasets.Dataset.from_dict({"vocalized": lines})

    tokenized_ds = ds.map(
        tokenize_batch,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.workers,
        desc="Tokenizing",
    )

    print(f"Saving to {args.output}...")
    tokenized_ds.save_to_disk(args.output)
    print("Done.")


if __name__ == "__main__":
    main()
