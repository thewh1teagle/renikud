"""Language pack dataclass — the only language-specific configuration needed.

To add a new language, create src/languages/<name>.py with a LangPack instance
and register it in src/languages/__init__.py.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class LangPack:
    name: str
    input_chars: frozenset[str]
    output_tokens: tuple[str, ...]  # index 0 = null/silent token
    max_slot_len: int = 3           # max IPA tokens one char can emit
    extra_chars: frozenset[str] = frozenset()  # script punctuation to include in vocab but not phonemize
    strip_accents: bool = True  # False for languages where accents are phonemically meaningful

    def token_to_id(self) -> dict[str, int]:
        return {t: i for i, t in enumerate(self.output_tokens)}

    def id_to_token(self) -> dict[int, str]:
        return {i: t for i, t in enumerate(self.output_tokens)}

    def __contains__(self, char: str) -> bool:
        return char in self.input_chars
