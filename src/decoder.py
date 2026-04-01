"""Decode per-token logits into an IPA string."""

from __future__ import annotations

import re

import torch

from constants import (
    ID_TO_CONSONANT,
    ID_TO_VOWEL,
    CONSONANT_TO_ID,
    CONSONANT_NONE,
    VOWEL_NONE,
    STRESS_YES,
    STRESS_MARK,
    is_hebrew_letter,
)
from phonology import HEBREW_LETTER_CONSONANT_IDS, FURTIVE_PATAH_LETTER, FURTIVE_PATAH_IPA, LETTERS_WITH_GERESH


def build_tokenizer_vocab(tokenizer) -> dict[int, str]:
    """Map token_id -> single character string for Hebrew letter lookup."""
    vocab = tokenizer.get_vocab()
    return {v: k for k, v in vocab.items()}


def _best_stress_per_word(offset_mapping: list[tuple[int, int]], text: str, stress_logits: torch.Tensor) -> set[int]:
    """
    For each whitespace-delimited word, pick at most one token index to carry stress —
    the one with the highest stress logit score among those that predicted stress.
    Returns a set of token indices that are allowed to emit stress.
    """
    word_spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    words: dict[int, list[int]] = {i: [] for i in range(len(word_spans))}

    for tok_idx, (start, end) in enumerate(offset_mapping):
        if end - start != 1:
            continue
        for word_idx, (ws, we) in enumerate(word_spans):
            if ws <= start < we:
                words[word_idx].append(tok_idx)
                break

    stressed: set[int] = set()
    for toks in words.values():
        if toks:
            stressed.add(max(toks, key=lambda t: stress_logits[t, STRESS_YES].item()))
    return stressed


def decode(
    text: str,
    offset_mapping: list[tuple[int, int]],
    consonant_logits: torch.Tensor,
    vowel_logits: torch.Tensor,
    stress_logits: torch.Tensor,
) -> str:
    """Decode per-token logits into an IPA string."""
    consonant_preds = consonant_logits.argmax(dim=-1)  # [S]
    vowel_preds = vowel_logits.argmax(dim=-1)           # [S]
    stressed_positions = _best_stress_per_word(offset_mapping, text, stress_logits)

    result = []
    prev_char_end = 0

    for tok_idx, (start, end) in enumerate(offset_mapping):
        if start > prev_char_end:
            result.append(text[prev_char_end:start])

        if end - start != 1:
            if end > start:
                prev_char_end = end
            continue

        char = text[start:end]
        prev_char_end = end

        if not is_hebrew_letter(char):
            # Skip geresh apostrophe after letters that use it as a digraph marker
            if char == "'" and start > 0 and text[start - 1] in LETTERS_WITH_GERESH:
                pass
            else:
                result.append(char)
            continue

        consonant = ID_TO_CONSONANT.get(int(consonant_preds[tok_idx]), CONSONANT_NONE)
        vowel = ID_TO_VOWEL.get(int(vowel_preds[tok_idx]), VOWEL_NONE)
        stress = tok_idx in stressed_positions

        # Apply per-letter consonant constraint at inference
        allowed = HEBREW_LETTER_CONSONANT_IDS.get(char, (CONSONANT_TO_ID[CONSONANT_NONE],))
        if CONSONANT_TO_ID.get(consonant, 0) not in allowed:
            for cid in sorted(allowed, key=lambda x: -consonant_logits[tok_idx, x].item()):
                consonant = ID_TO_CONSONANT[cid]
                break

        # Geresh rule: if next char is apostrophe, force the geresh consonant variant
        if char in LETTERS_WITH_GERESH and end < len(text) and text[end] == "'":
            from phonology import HEBREW_LETTER_CONSONANTS
            variants = HEBREW_LETTER_CONSONANTS.get(char, ())
            if len(variants) >= 2:
                consonant = variants[1]

        # Assemble IPA chunk: [consonant][ˈ][vowel]
        # Exception: word-final ח with vowel a — furtive patah flips to [ˈ]aχ
        word_final = end >= len(text) or not text[end].isalpha()
        chunk = ""
        if char == FURTIVE_PATAH_LETTER and word_final and vowel == "a":
            if stress:
                chunk += STRESS_MARK
            chunk += FURTIVE_PATAH_IPA
        else:
            if consonant != CONSONANT_NONE:
                chunk += consonant
            if stress:
                chunk += STRESS_MARK
            if vowel != VOWEL_NONE:
                chunk += vowel

        result.append(chunk)

    if prev_char_end < len(text):
        result.append(text[prev_char_end:])

    return "".join(result)
