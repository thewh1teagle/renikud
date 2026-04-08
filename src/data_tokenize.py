"""
Optimized Tokenized Arrow dataset preparation.
Uses HF Datasets .map() with multi-processing to handle 5M+ sentences.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import datasets
from tqdm import tqdm

from constants import (
    CONSONANT_TO_ID,
    VOWEL_TO_ID,
    STRESS_YES,
    STRESS_NONE,
    IGNORE_INDEX,
    is_hebrew_letter,
    TOKENIZER_PATH,
)
from tokenization import load_tokenizer

STRESS_MARK = "ˈ"
VOWELS_SET = set("aeiou")

# Global tokenizer for worker processes
_worker_tokenizer = None

def parse_ipa_chunk(chunk: str) -> tuple[str, str, int]:
    if not chunk or chunk == " ":
        return ("∅", "∅", STRESS_NONE)

    pos = 0
    stress = STRESS_NONE
    if STRESS_MARK in chunk:
        stress = STRESS_YES
        chunk = chunk.replace(STRESS_MARK, "")

    consonant = "∅"
    for multi in ("tʃ", "dʒ", "ts"):
        if chunk.startswith(multi):
            consonant = multi
            pos = len(multi)
            break
    else:
        if pos < len(chunk) and chunk[pos] not in VOWELS_SET:
            consonant = chunk[pos]
            pos += 1

    vowel = chunk[pos:] if pos < len(chunk) else "∅"
    if not vowel:
        vowel = "∅"

    if vowel.endswith("χ"):
        consonant = "χ"
        vowel = vowel[:-1] or "∅"

    return (
        consonant if consonant in CONSONANT_TO_ID else "∅",
        vowel if vowel in VOWEL_TO_ID else "∅",
        stress
    )

def parse_json_line(example):
    """
    Step 1: Convert raw text line to structured Hebrew and Alignment columns.
    Bypasses schema mismatch issues with dynamic JSON keys.
    """
    try:
        obj = json.loads(example["text"])
        hebrew, alignment = next(iter(obj.items()))
        return {"hebrew": hebrew, "alignment": alignment, "valid": True}
    except Exception:
        return {"hebrew": "", "alignment": [], "valid": False}

def process_batch(batch):
    """
    Step 2: Tokenize and align labels in batches.
    """
    global _worker_tokenizer
    if _worker_tokenizer is None:
        # Crucial: Ensure your load_tokenizer uses use_fast=True
        _worker_tokenizer = load_tokenizer(TOKENIZER_PATH)

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
            if not is_hebrew_letter(orig_char) and orig_char != " ":
                continue
            try:
                _, chunk = next(align_iter)
            except StopIteration:
                break
            if is_hebrew_letter(orig_char):
                char_labels[char_pos] = parse_ipa_chunk(chunk)

        # Align tokens to labels
        for tok_idx, (start, end) in enumerate(offsets):
            # Skip special tokens or multi-char subwords
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

    # Tokenize and align
    tokenized_ds = ds.map(
        process_batch,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.workers,
        remove_columns=["alignment"],
        desc="Tokenizing"
    )

    # 4. Save to Disk
    print(f"Saving to {args.output}...")
    tokenized_ds.save_to_disk(args.output)
    print("Done.")

if __name__ == "__main__":
    main()