"""Unicode codepoint tokenizer for G2P.

Each character maps directly to its Unicode codepoint as the token id.
Special tokens use private-use codepoints (U+E000–U+E004).

Normalizer:
  - NFKC
  - Lowercase
  - StripAccents

Adding a new language requires no changes here — the vocab is infinite
by definition (any codepoint is a valid id).
"""

from __future__ import annotations

import unicodedata


PAD_ID  = 0
CLS_ID  = 1
SEP_ID  = 2
UNK_ID  = 3
MASK_ID = 4

# Embedding table size: covers Latin, Hebrew, Arabic, and most Middle Eastern/European scripts.
# Increase if adding languages with higher codepoints (e.g. 0x10000 for CJK).
VOCAB_SIZE = 0x0700  # 1792


def normalize(text: str, strip_accents: bool = True) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    if strip_accents:
        text = "".join(
            c for c in unicodedata.normalize("NFD", text)
            if unicodedata.category(c) != "Mn"
        )
    return text


class UnicodeTokenizer:
    """Character-level tokenizer where token_id = ord(char).

    Returns dicts compatible with the rest of the pipeline:
    input_ids, attention_mask, offset_mapping.
    """

    def __init__(self, strip_accents: bool = True):
        self.strip_accents = strip_accents

    def __call__(
        self,
        text: str,
        truncation: bool = True,
        max_length: int = 512,
        return_offsets_mapping: bool = False,
        return_tensors: str | None = None,
    ) -> dict:
        norm = normalize(text, strip_accents=self.strip_accents)

        # Truncate to max_length - 2 to leave room for CLS and SEP
        chars = list(norm)
        if truncation and len(chars) > max_length - 2:
            chars = chars[:max_length - 2]

        input_ids = [CLS_ID] + [ord(c) for c in chars] + [SEP_ID]
        attention_mask = [1] * len(input_ids)

        result: dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if return_offsets_mapping:
            # CLS and SEP get (0, 0) offsets
            offsets = [(0, 0)]
            pos = 0
            for c in chars:
                offsets.append((pos, pos + 1))
                pos += 1
            offsets.append((0, 0))
            result["offset_mapping"] = offsets

        if return_tensors == "pt":
            import torch
            result["input_ids"] = torch.tensor([result["input_ids"]], dtype=torch.long)
            result["attention_mask"] = torch.tensor([result["attention_mask"]], dtype=torch.long)
            if return_offsets_mapping:
                result["offset_mapping"] = torch.tensor([result["offset_mapping"]], dtype=torch.long)

        return result


def load_tokenizer(_path=None, lang_pack=None) -> UnicodeTokenizer:
    """Drop-in replacement for the old PreTrainedTokenizerFast loader."""
    strip_accents = lang_pack.strip_accents if lang_pack is not None else True
    return UnicodeTokenizer(strip_accents=strip_accents)
