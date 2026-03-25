"""Character-level tokenizer for Hebrew G2P.

Vocab:
  - Special tokens: [PAD], [CLS], [SEP], [UNK], [MASK]
  - Hebrew letters: alef-tav (including final forms)
  - ASCII lowercase + digits + punctuation + space
  - Hebrew punctuation: maqaf, geresh, gershayim

Normalizer:
  - NFKC
  - Lowercase
  - StripAccents
"""

from __future__ import annotations

import string
import unicodedata
from pathlib import Path

from tokenizers import Tokenizer, AddedToken
from tokenizers.models import WordPiece
from tokenizers.normalizers import Sequence, NFKC, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Split
from tokenizers.processors import TemplateProcessing
from tokenizers import Regex
from transformers import PreTrainedTokenizerFast


SPECIAL_TOKENS = ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"]


def build_vocab() -> dict[str, int]:
    hebrew = [
        chr(cp)
        for cp in range(0x05D0, 0x05EB)
        if unicodedata.category(chr(cp)) == "Lo"
    ]
    # Hebrew punctuation — see https://en.wikipedia.org/wiki/Unicode_and_HTML_for_the_Hebrew_alphabet
    hebrew_punct = [
        "\u05BE",  # maqaf (Hebrew hyphen)
        "\u05F3",  # geresh
        "\u05F4",  # gershayim
    ]
    chars = (
        list(string.ascii_lowercase)
        + list(string.digits)
        + list(string.punctuation)
        + [" "]
        + hebrew
        + hebrew_punct
    )

    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for c in chars:
        if c not in vocab:
            vocab[c] = len(vocab)
    return vocab


def build_tokenizer() -> Tokenizer:
    vocab = build_vocab()

    tokenizer = Tokenizer(WordPiece(vocab, unk_token="[UNK]", continuing_subword_prefix="##"))
    tokenizer.normalizer = Sequence([NFKC(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Split(pattern=Regex("[\\s\\S]"), behavior="isolated")

    cls_id = vocab["[CLS]"]
    sep_id = vocab["[SEP]"]
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)],
    )
    tokenizer.add_special_tokens([AddedToken(t, special=True) for t in SPECIAL_TOKENS])

    return tokenizer


def save_tokenizer(path: str | Path) -> None:
    build_tokenizer().save(str(path))


def load_tokenizer(path: str | Path) -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast(
        tokenizer_file=str(path),
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
    )
