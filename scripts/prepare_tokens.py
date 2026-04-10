"""
Tokenized Arrow dataset preparation.
Uses HF Datasets .map() with multi-processing to handle 5M+ sentences.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import datasets

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from constants import IGNORE_INDEX
from phonology import CONSONANT_TO_ID, VOWEL_TO_ID, STRESS_NONE, is_hebrew_letter, chunk_to_labels, ORTHOGRAPHIC_MARKERS
from tokenization import load_tokenizer

# Global tokenizer for worker processes
_worker_tokenizer = None

def tokenize_batch(batch):
    global _worker_tokenizer
    if _worker_tokenizer is None:
        _worker_tokenizer = load_tokenizer()

    hebrew_sentences = batch["hebrew"]
    alignments = batch["alignment"]

    encodings = _worker_tokenizer(
        hebrew_sentences,
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
    )

    batch_consonant_labels = []
    batch_vowel_labels = []
    batch_stress_labels = []

    for i in range(len(hebrew_sentences)):
        hebrew = hebrew_sentences[i]
        alignment = alignments[i]
        offsets = encodings["offset_mapping"][i]
        seq_len = len(encodings["input_ids"][i])

        c_labels = [IGNORE_INDEX] * seq_len
        v_labels = [IGNORE_INDEX] * seq_len
        s_labels = [IGNORE_INDEX] * seq_len

        # Map alignment to char positions
        char_labels = {}
        align_iter = iter(alignment)
        for char_pos, orig_char in enumerate(hebrew):
            if not is_hebrew_letter(orig_char) and orig_char not in ORTHOGRAPHIC_MARKERS and orig_char != " ":
                continue
            try:
                _, chunk = next(align_iter)
            except StopIteration:
                break
            if is_hebrew_letter(orig_char) or orig_char in ORTHOGRAPHIC_MARKERS:
                char_labels[char_pos] = chunk_to_labels(chunk)

        # Align tokens to labels
        for tok_idx, (start, end) in enumerate(offsets):
            if end - start != 1:
                continue

            char_idx = start
            if char_idx in char_labels:
                c, v, s = char_labels[char_idx]
                c_labels[tok_idx] = CONSONANT_TO_ID.get(c, IGNORE_INDEX)
                v_labels[tok_idx] = VOWEL_TO_ID.get(v, IGNORE_INDEX)
                s_labels[tok_idx] = s
            elif char_idx < len(hebrew) and not is_hebrew_letter(hebrew[char_idx]) and hebrew[char_idx] != " ":
                c_labels[tok_idx] = CONSONANT_TO_ID["∅"]
                v_labels[tok_idx] = VOWEL_TO_ID["∅"]
                s_labels[tok_idx] = STRESS_NONE

        batch_consonant_labels.append(c_labels)
        batch_vowel_labels.append(v_labels)
        batch_stress_labels.append(s_labels)

    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "consonant_labels": batch_consonant_labels,
        "vowel_labels": batch_vowel_labels,
        "stress_labels": batch_stress_labels,
        "phonemes": batch["phonemes"],
        "tags": batch.get("tags", [[] for _ in hebrew_sentences]),
    }

def main():
    parser = argparse.ArgumentParser(description="Fast Arrow dataset preparation")
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("output", help="Output Arrow directory")
    parser.add_argument("--workers", type=int, default=min(64, os.cpu_count()))
    parser.add_argument("--batch_size", type=int, default=1000)
    args = parser.parse_args()

    print(f"Reading {args.input}...")
    ds = datasets.load_dataset("json", data_files=args.input, split="train")

    tokenized_ds = ds.map(
        tokenize_batch,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.workers,
        remove_columns=["alignment"],
        desc="Tokenizing"
    )

    print(f"Saving to {args.output}...")
    tokenized_ds.save_to_disk(args.output)
    print("Done.")

if __name__ == "__main__":
    main()
