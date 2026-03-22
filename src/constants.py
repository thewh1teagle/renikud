"""Constants for the Hebrew G2P classifier model."""

from typing import Final

ENCODER_MODEL: Final[str] = "dicta-il/dictabert-large-char-menaked"
MAX_LEN: Final[int] = 256

# ---------------------------------------------------------------------------
# 1. Vowel vocabulary
# ∅ (index 0) means no vowel — letter is vowel-less or silent
# ---------------------------------------------------------------------------
VOWEL_NONE: Final[str] = "∅"
VOWELS: Final[tuple[str, ...]] = (VOWEL_NONE, "a", "e", "i", "o", "u", "aχ")
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

# ---------------------------------------------------------------------------
# 4. Per-letter allowed consonant IDs (for logit masking)
# Only these consonant classes are valid for each Hebrew letter.
# ∅ (index 0) is included only for letters that can genuinely be silent:
#   א, ע (quiescent), ה (word-final mater lectionis), ו, י (matres lectionis).
# All other letters always produce a consonant in Modern Hebrew.
# ---------------------------------------------------------------------------
HEBREW_LETTER_TO_ALLOWED_CONSONANTS: Final[dict[str, tuple[int, ...]]] = {
    "א": (CONSONANT_TO_ID["∅"], CONSONANT_TO_ID["ʔ"]),
    "ב": (CONSONANT_TO_ID["b"], CONSONANT_TO_ID["v"]),
    "ג": (CONSONANT_TO_ID["ɡ"], CONSONANT_TO_ID["dʒ"]),
    "ד": (CONSONANT_TO_ID["d"],),
    "ה": (CONSONANT_TO_ID["∅"], CONSONANT_TO_ID["h"]),
    "ו": (CONSONANT_TO_ID["∅"], CONSONANT_TO_ID["v"], CONSONANT_TO_ID["w"]),
    "ז": (CONSONANT_TO_ID["z"], CONSONANT_TO_ID["ʒ"]),
    "ח": (CONSONANT_TO_ID["∅"], CONSONANT_TO_ID["χ"]),
    "ט": (CONSONANT_TO_ID["t"],),
    "י": (CONSONANT_TO_ID["∅"], CONSONANT_TO_ID["j"]),
    "כ": (CONSONANT_TO_ID["k"], CONSONANT_TO_ID["χ"]),
    "ך": (CONSONANT_TO_ID["k"], CONSONANT_TO_ID["χ"]),
    "ל": (CONSONANT_TO_ID["l"],),
    "מ": (CONSONANT_TO_ID["m"],),
    "ם": (CONSONANT_TO_ID["m"],),
    "נ": (CONSONANT_TO_ID["n"],),
    "ן": (CONSONANT_TO_ID["n"],),
    "ס": (CONSONANT_TO_ID["s"],),
    "ע": (CONSONANT_TO_ID["∅"], CONSONANT_TO_ID["ʔ"]),
    "פ": (CONSONANT_TO_ID["p"], CONSONANT_TO_ID["f"]),
    "ף": (CONSONANT_TO_ID["p"], CONSONANT_TO_ID["f"]),
    "צ": (CONSONANT_TO_ID["ts"], CONSONANT_TO_ID["tʃ"]),
    "ץ": (CONSONANT_TO_ID["ts"], CONSONANT_TO_ID["tʃ"]),
    "ק": (CONSONANT_TO_ID["k"],),
    "ר": (CONSONANT_TO_ID["ʁ"],),
    "ש": (CONSONANT_TO_ID["ʃ"], CONSONANT_TO_ID["s"]),
    "ת": (CONSONANT_TO_ID["t"],),
}

# Hebrew Unicode range א-ת
ALEF_ORD: Final[int] = ord("א")
TAF_ORD: Final[int] = ord("ת")


def is_hebrew_letter(char: str) -> bool:
    return ALEF_ORD <= ord(char) <= TAF_ORD


# Label ignore index — used for non-Hebrew positions (CLS, SEP, spaces, punct)
IGNORE_INDEX: Final[int] = -100
