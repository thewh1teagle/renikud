"""Prepare Arrow dataset from raw TSV for CTC-based G2P training.

Input TSV:  text<TAB>ipa   (one sentence per line)
Output:     Arrow dataset with:
  - input_ids, attention_mask  — tokenized text
  - active_mask                — which token positions are input_chars
  - target_ids                 — IPA token ids (1-indexed, 0 reserved for CTC blank)

No alignment needed — CTC handles it during training.

Usage:
    uv run src/data_prepare.py dataset/train.tsv dataset/.cache/train --lang hebrew
    uv run src/data_prepare.py dataset/val.tsv   dataset/.cache/val   --lang hebrew
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import unicodedata
import regex as re

import datasets
from tqdm import tqdm

from constants import TOKENIZER_PATH, MAX_LEN
from lang_pack import get_lang_pack, LangPack
from tokenization import load_tokenizer

_IPA_PUNCT = re.compile(r"[!,\.?\s\"'()\[\]]")


def tokenize_ipa(ipa: str, token_to_id: dict[str, int], sorted_tokens: list[str]) -> list[int] | None:
    """Greedy longest-match IPA tokenization. Returns 1-indexed ids (0 = CTC blank)."""
    ipa = _IPA_PUNCT.sub("", ipa)
    if not ipa:
        return []
    ids = []
    i = 0
    while i < len(ipa):
        matched = False
        for tok in sorted_tokens:
            if ipa[i:].startswith(tok):
                ids.append(token_to_id[tok] + 1)  # +1: shift so 0 = blank
                i += len(tok)
                matched = True
                break
        if not matched:
            return None
    return ids


def strip_nikud(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    return re.sub(r"[\p{M}|]", "", text)


_worker_state: dict = {}


def _init_worker(lang_name: str):
    lang_pack = get_lang_pack(lang_name)
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    token_to_id = lang_pack.token_to_id()
    null_tok = lang_pack.output_tokens[0]
    sorted_tokens = sorted(
        [t for t in token_to_id if t != null_tok],
        key=len, reverse=True
    )
    _worker_state.update({
        "lang_pack": lang_pack,
        "tokenizer": tokenizer,
        "token_to_id": token_to_id,
        "sorted_tokens": sorted_tokens,
    })


def process_chunk(lines: list[str]) -> tuple[list[dict], int]:
    lang_pack: LangPack = _worker_state["lang_pack"]
    tokenizer = _worker_state["tokenizer"]
    token_to_id = _worker_state["token_to_id"]
    sorted_tokens = _worker_state["sorted_tokens"]

    records = []
    skipped = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            skipped += 1
            continue

        raw_text, ipa = parts
        text = strip_nikud(raw_text)

        # Tokenize full IPA string (strip punctuation/spaces)
        target_ids = tokenize_ipa(ipa, token_to_id, sorted_tokens)
        if target_ids is None or len(target_ids) == 0:
            skipped += 1
            continue

        # Tokenize text
        enc = tokenizer(
            text,
            truncation=True,
            max_length=MAX_LEN,
            return_offsets_mapping=True,
            return_tensors=None,
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        offset_mapping = enc["offset_mapping"]

        # active_mask: 1 for input_char positions, 0 for everything else
        active_mask = []
        for start, end in offset_mapping:
            if end - start == 1:
                active_mask.append(1 if text[start] in lang_pack else 0)
            else:
                active_mask.append(0)

        n_active = sum(active_mask)
        if n_active == 0:
            skipped += 1
            continue

        # CTC requires input_length >= target_length
        # input_length = n_active * K (K=3 slots per char)
        if n_active * 3 < len(target_ids):
            skipped += 1
            continue

        records.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "active_mask": active_mask,
            "target_ids": target_ids,
        })

    return records, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input TSV (text<TAB>ipa)")
    parser.add_argument("output", help="Output Arrow dataset directory")
    parser.add_argument("--lang", default="hebrew")
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        lines = f.readlines()

    chunk_size = max(1, len(lines) // (args.workers * 4))
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]

    all_records = []
    total_skipped = 0

    with mp.Pool(args.workers, initializer=_init_worker, initargs=(args.lang,)) as pool:
        for records, skipped in tqdm(
            pool.imap(process_chunk, chunks), total=len(chunks), desc="Preparing"
        ):
            all_records.extend(records)
            total_skipped += skipped

    print(f"\nProcessed: {len(all_records):,}")
    print(f"Skipped:   {total_skipped:,}")

    dataset = datasets.Dataset.from_list(all_records)
    dataset.save_to_disk(args.output)
    print(f"Saved to:  {args.output}")


if __name__ == "__main__":
    main()
