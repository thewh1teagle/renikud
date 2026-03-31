"""Language pack — the only language-specific configuration needed.

A language pack defines:
  - input_chars:   which characters the model handles (predicts phonemes for)
  - output_tokens: the atomic IPA units the model can emit per character slot
  - max_slot_len:  maximum number of output tokens one input char can produce

Everything not in input_chars passes through unchanged at inference.
Add a new language by adding a new LangPack instance — no other code changes needed.
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
    strip_accents: bool = True  # False for languages where accents are phonemically meaningful (French, Spanish, etc.)

    def token_to_id(self) -> dict[str, int]:
        return {t: i for i, t in enumerate(self.output_tokens)}

    def id_to_token(self) -> dict[int, str]:
        return {i: t for i, t in enumerate(self.output_tokens)}

    def __contains__(self, char: str) -> bool:
        return char in self.input_chars


# ---------------------------------------------------------------------------
# Hebrew
# ---------------------------------------------------------------------------
# output_tokens: null + all atomic IPA units Hebrew letters can produce.
# Stressed vowels are single tokens (ˈa, ˈe, ...) so one char can emit
# e.g. ["b", "ˈa"] as a length-2 slot — within max_slot_len=3.
# ---------------------------------------------------------------------------
HEBREW = LangPack(
    name="hebrew",
    input_chars=frozenset(
        "אבגדהוזחטיכךלמםנןסעפףצץקרשת"
    ),
    extra_chars=frozenset("\u05BE\u05F3\u05F4"),  # maqaf, geresh, gershayim
    output_tokens=(
        # 0 = null (silent, no output)
        "∅",
        # plain consonants
        "b", "v", "d", "h", "z", "χ", "t", "j", "k", "l",
        "m", "n", "s", "f", "p", "ts", "tʃ", "w", "ʔ", "ɡ", "ʁ", "ʃ", "ʒ", "dʒ",
        # plain vowels (unstressed)
        "a", "e", "i", "o", "u",
        # stressed vowels (stress mark fused into token)
        "ˈa", "ˈe", "ˈi", "ˈo", "ˈu",
    ),
    max_slot_len=3,
)

LANG_PACKS: dict[str, LangPack] = {
    "hebrew": HEBREW,
}


def get_lang_pack(name: str) -> LangPack:
    if name not in LANG_PACKS:
        raise ValueError(f"Unknown language pack: {name!r}. Available: {list(LANG_PACKS)}")
    return LANG_PACKS[name]
