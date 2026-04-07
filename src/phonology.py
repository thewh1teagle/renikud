"""
Phonological constraints for Hebrew G2P — single source of truth.

Provides the letter → consonant mapping in two forms:
  - HEBREW_LETTER_CONSONANTS: symbol-based (for the DP aligner)
  - HEBREW_LETTER_CONSONANT_IDS: ID-based (for model masking / inference)

Also provides build_consonant_mask() and apply_consonant_mask() which
previously lived as private methods on G2PModel.

Notes:
  - "" in HEBREW_LETTER_CONSONANTS means the letter can be silent (→ ∅)
  - ח has no silent option: word-final ח always produces χ (furtive patah
    is handled as a reversed [vowel]χ chunk in the aligner and infer.py)
"""

from __future__ import annotations

import torch

from constants import CONSONANT_TO_ID, ALEF_ORD, TAF_ORD, NUM_CONSONANT_CLASSES, is_hebrew_letter
from aligner.align import HEBREW_LETTER_CONSONANTS

# ---------------------------------------------------------------------------
# Letter whose word-final + vowel-a chunk reverses to [vowel]χ (furtive patah)
FURTIVE_PATAH_LETTER: str = "ח"
FURTIVE_PATAH_IPA: str = "aχ"  # the reversed IPA output: vowel precedes consonant

# Letters where a following apostrophe is a digraph marker, not punctuation
LETTERS_WITH_GERESH: frozenset[str] = frozenset("גזצץ")

# ID-keyed view derived from above — used for masking and inference
HEBREW_LETTER_CONSONANT_IDS: dict[str, tuple[int, ...]] = {
    char: tuple(CONSONANT_TO_ID["∅"] if sym == "" else CONSONANT_TO_ID[sym] for sym in syms)
    for char, syms in HEBREW_LETTER_CONSONANTS.items()
}


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
