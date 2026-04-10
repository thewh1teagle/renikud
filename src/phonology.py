"""
Phonological constants and constraints for Hebrew G2P — single source of truth.

Provides the full Hebrew phonological inventory (vowels, consonants, stress),
letter → consonant mappings, masking utilities, and label parsing.

Notes:
  - "" in HEBREW_LETTER_CONSONANTS means the letter can be silent (→ ∅)
  - ח has no silent option: word-final ח always produces χ (furtive patah
    is handled as a reversed [vowel]χ chunk in the aligner and infer.py)
"""

from __future__ import annotations

from typing import Final

import torch
import regex as re

from aligner.align import HEBREW_LETTER_CONSONANTS

# ---------------------------------------------------------------------------
# Orthographic formatting markers that produce no phonemes
# ---------------------------------------------------------------------------
ORTHOGRAPHIC_MARKERS: Final[tuple[str, ...]] = ("'", '"')

def normalize_graphemes(text: str) -> str:
    """Consistently normalizes Geresh and Gershayim variants to strict ASCII."""
    text = re.sub(r"[׳'`´]", "'", text)
    text = re.sub(r"[״”“]", '"', text)
    return text


# ---------------------------------------------------------------------------
# Hebrew Unicode range
# ---------------------------------------------------------------------------
ALEF_ORD: Final[int] = ord("א")
TAF_ORD: Final[int] = ord("ת")


def is_hebrew_letter(char: str) -> bool:
    return ALEF_ORD <= ord(char) <= TAF_ORD


# ---------------------------------------------------------------------------
# Vowel vocabulary
# ∅ (index 0) means no vowel — letter is vowel-less or silent
# ---------------------------------------------------------------------------
VOWEL_NONE: Final[str] = "∅"
VOWELS: Final[tuple[str, ...]] = (VOWEL_NONE, "a", "e", "i", "o", "u")
VOWEL_TO_ID: Final[dict[str, int]] = {v: i for i, v in enumerate(VOWELS)}
ID_TO_VOWEL: Final[dict[int, str]] = {i: v for i, v in enumerate(VOWELS)}
NUM_VOWEL_CLASSES: Final[int] = len(VOWELS)

# ---------------------------------------------------------------------------
# Consonant vocabulary
# ∅ (index 0) means silent — letter produces no consonant
# ---------------------------------------------------------------------------
CONSONANT_NONE: Final[str] = "∅"
CONSONANTS: Final[tuple[str, ...]] = (
    CONSONANT_NONE,
    "b", "v", "d", "h", "z", "χ", "t", "j", "k", "l", "m", "n", "s", "f", "p",
    "ts", "tʃ", "w", "ʔ", "ɡ", "ʁ", "ʃ", "ʒ", "dʒ",
)
CONSONANT_TO_ID: Final[dict[str, int]] = {c: i for i, c in enumerate(CONSONANTS)}
ID_TO_CONSONANT: Final[dict[int, str]] = {i: c for i, c in enumerate(CONSONANTS)}
NUM_CONSONANT_CLASSES: Final[int] = len(CONSONANTS)

# ---------------------------------------------------------------------------
# Stress vocabulary
# 0 = no stress, 1 = stress (ˈ before vowel)
# ---------------------------------------------------------------------------
NUM_STRESS_CLASSES: Final[int] = 2
STRESS_NONE: Final[int] = 0
STRESS_YES: Final[int] = 1
STRESS_MARK: Final[str] = "ˈ"

# ---------------------------------------------------------------------------
# Letter-level phonological rules
# ---------------------------------------------------------------------------

# Letter whose word-final + vowel-a chunk reverses to [vowel]χ (furtive patah)
FURTIVE_PATAH_LETTER: Final[str] = "ח"
FURTIVE_PATAH_IPA: Final[str] = "aχ"  # the reversed IPA output: vowel precedes consonant

# Letters where a following apostrophe is a digraph marker, not punctuation
LETTERS_WITH_GERESH: Final[frozenset[str]] = frozenset("גזצץ")

# ID-keyed view derived from above — used for masking and inference
HEBREW_LETTER_CONSONANT_IDS: dict[str, tuple[int, ...]] = {
    char: tuple(CONSONANT_TO_ID["∅"] if sym == "" else CONSONANT_TO_ID[sym] for sym in syms)
    for char, syms in HEBREW_LETTER_CONSONANTS.items()
}

# ---------------------------------------------------------------------------
# Consonant masking
# ---------------------------------------------------------------------------

def build_consonant_mask() -> torch.Tensor:
    """
    Return a boolean mask [num_hebrew_letters, NUM_CONSONANT_CLASSES].
    True = this consonant class is forbidden for this letter.
    Indexed by (ord(char) - ord('א')).
    """
    n_letters = TAF_ORD - ALEF_ORD + 1
    mask = torch.ones(n_letters, NUM_CONSONANT_CLASSES, dtype=torch.bool)
    for char, allowed_ids in HEBREW_LETTER_CONSONANT_IDS.items():
        idx = ord(char) - ALEF_ORD
        for cid in allowed_ids:
            mask[idx, cid] = False
    return mask


def apply_consonant_mask(
    consonant_logits: torch.Tensor,
    input_ids: torch.Tensor,
    tokenizer_vocab: dict[int, str],
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Zero out forbidden consonant classes for each position.

    consonant_logits: [B, S, NUM_CONSONANT_CLASSES]
    input_ids:        [B, S]
    tokenizer_vocab:  maps token_id -> character string
    mask:             [num_hebrew_letters, NUM_CONSONANT_CLASSES] from build_consonant_mask()
    """
    mask = mask.to(consonant_logits.device)
    B, S, _ = consonant_logits.shape
    masked = consonant_logits.clone()

    for b in range(B):
        for s in range(S):
            token_id = input_ids[b, s].item()
            char = tokenizer_vocab.get(token_id, "")
            if len(char) == 1 and is_hebrew_letter(char):
                letter_idx = ord(char) - ALEF_ORD
                masked[b, s][mask[letter_idx]] = -1e9

    return masked


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

_VOWELS_SET = set(VOWELS) - {VOWEL_NONE}


def chunk_to_labels(chunk: str) -> tuple[str, str, int]:
    """Parse an aligned IPA chunk into (consonant, vowel, stress) labels."""

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
        if pos < len(chunk) and chunk[pos] not in _VOWELS_SET:
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
