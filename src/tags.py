"""Input tags — special tokens that condition the model on sentence-level properties.

Tags are prepended to the input sequence after [CLS] in a fixed order.
Missing categories default to their UNKNOWN variant so the sequence length
is always constant regardless of which tags are specified.

Current categories (in fixed order):
  1. Gender — disambiguates gender-sensitive words like לך (lexa vs. laχ)

Adding a new category:
  1. Add its tokens and UNKNOWN default below
  2. Append it to TAG_CATEGORIES
  3. Add the tokens to SPECIAL_TOKENS in tokenization.py
"""

from __future__ import annotations

from tokenization import build_vocab

# ---------------------------------------------------------------------------
# Gender
# ---------------------------------------------------------------------------
GENDER_UNKNOWN = "[GENDER_UNKNOWN]"
GENDER_MALE    = "[GENDER_MALE]"
GENDER_FEMALE  = "[GENDER_FEMALE]"

# ---------------------------------------------------------------------------
# Fixed order of categories: (set of valid tokens, default unknown token)
# ---------------------------------------------------------------------------
TAG_CATEGORIES: list[tuple[frozenset[str], str]] = [
    (frozenset({GENDER_UNKNOWN, GENDER_MALE, GENDER_FEMALE}), GENDER_UNKNOWN),
]

_vocab = build_vocab()


def build_tag_prefix(tags: list[str]) -> list[str]:
    """Return ordered tag tokens for the given tags, filling unknowns for missing categories."""
    tag_set = set(tags)
    result = []
    for valid_tokens, default in TAG_CATEGORIES:
        match = tag_set & valid_tokens
        result.append(match.pop() if match else default)
    return result


def tag_prefix_ids(tags: list[str]) -> list[int]:
    """Return token IDs for the ordered tag prefix."""
    return [_vocab[t] for t in build_tag_prefix(tags)]
