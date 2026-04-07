"""Map Hebrew+IPA alignment pairs to per-token label IDs for classifier training."""

from __future__ import annotations

from constants import (
    CONSONANT_TO_ID,
    VOWEL_TO_ID,
    STRESS_YES,
    STRESS_NONE,
    IGNORE_INDEX,
    is_hebrew_letter,
)

STRESS_MARK = "ˈ"
VOWELS_SET = set("aeiou")


def parse_chunk(chunk: str) -> tuple[str, str, int]:
    """Parse a single IPA chunk (e.g. 'ʃa', 'lˈo') into (consonant, vowel, stress)."""
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

    if consonant not in CONSONANT_TO_ID:
        consonant = "∅"
    if vowel not in VOWEL_TO_ID:
        vowel = "∅"

    return (consonant, vowel, stress)


def alignment_to_phonemes(alignment: list) -> str:
    """Reconstruct the GT IPA string from alignment pairs."""
    return "".join(chunk for _, chunk in alignment)


def label_sentence(example: dict, tokenizer) -> dict:
    """Tokenize a Hebrew sentence and assign consonant/vowel/stress label IDs per token."""
    hebrew = example["hebrew"]
    alignment = example["alignment"]
    phonemes = alignment_to_phonemes(alignment)

    encoding = tokenizer(
        hebrew,
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
        return_tensors=None,
    )

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    offset_mapping = encoding["offset_mapping"]

    seq_len = len(input_ids)
    consonant_labels = [IGNORE_INDEX] * seq_len
    vowel_labels = [IGNORE_INDEX] * seq_len
    stress_labels = [IGNORE_INDEX] * seq_len

    char_labels: dict[int, tuple[str, str, int]] = {}
    align_iter = iter(alignment)
    for char_pos, orig_char in enumerate(hebrew):
        if not is_hebrew_letter(orig_char) and orig_char != " ":
            continue
        try:
            align_char, chunk = next(align_iter)
        except StopIteration:
            break
        if is_hebrew_letter(orig_char):
            char_labels[char_pos] = parse_chunk(chunk)

    for tok_idx, (start, end) in enumerate(offset_mapping):
        if end - start != 1:
            continue
        char_idx = start
        if char_idx in char_labels:
            consonant, vowel, stress = char_labels[char_idx]
            consonant_labels[tok_idx] = CONSONANT_TO_ID.get(consonant, IGNORE_INDEX)
            vowel_labels[tok_idx] = VOWEL_TO_ID.get(vowel, IGNORE_INDEX)
            stress_labels[tok_idx] = stress
        elif not is_hebrew_letter(hebrew[char_idx]) and hebrew[char_idx] != " ":
            consonant_labels[tok_idx] = CONSONANT_TO_ID["∅"]
            vowel_labels[tok_idx] = VOWEL_TO_ID["∅"]
            stress_labels[tok_idx] = STRESS_NONE

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "consonant_labels": consonant_labels,
        "vowel_labels": vowel_labels,
        "stress_labels": stress_labels,
        "hebrew": hebrew,
        "phonemes": phonemes,
    }
