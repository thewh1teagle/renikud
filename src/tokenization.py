"""Tokenization helpers for Hebrew G2P."""

from __future__ import annotations
import unicodedata
import itertools
from functools import lru_cache

from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerFast

from constants import (
    CTC_BLANK_ID,
    ENCODER_MODEL,
    ID_TO_TOKEN,
    PAD_ID,
    TOKEN_TO_ID,
    UNK_ID,
    VALID_OUTPUT_TOKENS,
)


def normalize_ipa(text: str) -> str:
    """Normalize an IPA label without changing its visible formatting."""
    return unicodedata.normalize("NFD", text).strip()


def encode_ipa(text: str, allow_unk: bool = False) -> list[int]:
    """Encode an IPA string into decoder token ids."""
    normalized = normalize_ipa(text)

    # 1. Fast strict validation using Set Difference
    if not allow_unk:
        unknown_chars = set(normalized) - VALID_OUTPUT_TOKENS
        if unknown_chars:
            # Preserve appearance order for the error message, remove duplicates
            bad_chars = list(dict.fromkeys(c for c in normalized if c in unknown_chars))
            joined = ", ".join(repr(c) for c in bad_chars)
            raise ValueError(f"Unknown IPA tokens: {joined}")

    # 2. Fast mapping using List Comprehension
    return [TOKEN_TO_ID.get(char, UNK_ID) for char in normalized]


def decode_ipa(token_ids: list[int], skip_special: bool = True) -> str:
    """Decode token ids back into an IPA string."""
    chars = []
    for token_id in token_ids:
        token_id = int(token_id)
        if skip_special and token_id in (CTC_BLANK_ID, PAD_ID):
            continue
        
        if token_id not in ID_TO_TOKEN:
            raise ValueError(f"Unknown token id: {token_id}")
            
        chars.append(ID_TO_TOKEN[token_id])

    return "".join(chars)


def decode_ctc(token_ids: list[int]) -> str:
    """Collapse CTC repeats and blanks, then decode."""
    # itertools.groupby cleanly collapses adjacent duplicates: [1,1,0,2] -> [1,0,2]
    collapsed = [k for k, _g in itertools.groupby(token_ids)]

    # Remove CTC blanks BEFORE decoding
    cleaned = [t for t in collapsed if t != CTC_BLANK_ID]

    return decode_ipa(cleaned, skip_special=True)


def beam_search_ctc(log_probs: list[list[float]], beam_size: int) -> str:
    """
    Character-level CTC beam search.
    log_probs: [T, vocab_size] as nested lists or numpy array.
    Returns the decoded IPA string.
    """
    import math
    neg_inf = -math.inf

    # Each beam: (prefix_tuple, score)
    beams: dict[tuple, float] = {(): 0.0}

    for t_probs in log_probs:
        next_beams: dict[tuple, float] = {}
        for prefix, score in beams.items():
            last = prefix[-1] if prefix else None
            for token_id, lp in enumerate(t_probs):
                if lp <= neg_inf:
                    continue
                if token_id == CTC_BLANK_ID:
                    # Blank: extend with blank (prefix unchanged)
                    key = prefix
                elif token_id == last:
                    # Same as last non-blank: only allowed after a blank
                    # so we don't extend (stays collapsed)
                    key = prefix
                else:
                    key = prefix + (token_id,)
                new_score = score + lp
                if key not in next_beams or next_beams[key] < new_score:
                    next_beams[key] = new_score

        # Keep top beam_size beams
        beams = dict(sorted(next_beams.items(), key=lambda x: -x[1])[:beam_size])

    best_prefix = max(beams, key=lambda k: beams[k])
    return decode_ipa(list(best_prefix), skip_special=True)


def unwrap_encoder_model(encoder):
    """Unwrap Dicta's diacritization model wrapper when present."""
    return encoder.bert if hasattr(encoder, "bert") else encoder


@lru_cache(maxsize=1)
def load_encoder_tokenizer(model_name: str = ENCODER_MODEL) -> PreTrainedTokenizerFast:
    """
    Load the encoder tokenizer securely. 
    Bypasses the broken AutoTokenizer logic for character-level models.
    """
    tokenizer_file = hf_hub_download(repo_id=model_name, filename="tokenizer.json")
    
    return PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_file,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
    )