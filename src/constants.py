"""Constants for the Hebrew G2P classifier model."""

from typing import Final

MAX_LEN: Final[int] = 256

# ---------------------------------------------------------------------------
# 1. Vowel vocabulary
# ∅ (index 0) means no vowel — letter is vowel-less or silent
# ---------------------------------------------------------------------------
VOWEL_NONE: Final[str] = "∅"
VOWELS: Final[tuple[str, ...]] = (VOWEL_NONE, "a", "e", "i", "o", "u")
VOWEL_TO_ID: Final[dict[str, int]] = {v: i for i, v in enumerate(VOWELS)}
ID_TO_VOWEL: Final[dict[int, str]] = {i: v for i, v in enumerate(VOWELS)}
NUM_VOWEL_CLASSES: Final[int] = len(VOWELS)

# ---------------------------------------------------------------------------
# 2. Consonant vocabulary
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
# 3. Stress vocabulary
# 0 = no stress, 1 = stress (ˈ before vowel)
# ---------------------------------------------------------------------------
NUM_STRESS_CLASSES: Final[int] = 2
STRESS_NONE: Final[int] = 0
STRESS_YES: Final[int] = 1
STRESS_MARK: Final[str] = "ˈ"

# Hebrew Unicode range א-ת
ALEF_ORD: Final[int] = ord("א")
TAF_ORD: Final[int] = ord("ת")


def is_hebrew_letter(char: str) -> bool:
    return ALEF_ORD <= ord(char) <= TAF_ORD


# Label ignore index — used for non-Hebrew positions (CLS, SEP, spaces, punct)
IGNORE_INDEX: Final[int] = -100
