"""Core constants for Hebrew G2P preprocessing and decoding."""

from typing import Final
import string

VOCAB_VERSION: Final[str] = "hebrew-ipa-char-v1"
ENCODER_MODEL: Final[str] = "dicta-il/dictabert-large-char-menaked"
MAX_LEN: Final[int] = 256
PROJECTION_DIM: Final[int] = 256
UPSAMPLE_FACTOR: Final[int] = 2

# 1. Special Tokens (Must remain at specific indices for CTC)
CTC_BLANK_TOKEN: Final[str] = "<blank>"
PAD_TOKEN: Final[str] = "<pad>"
UNK_TOKEN: Final[str] = "<unk>"
SPECIAL_TOKENS: Final[tuple[str, ...]] = (CTC_BLANK_TOKEN, PAD_TOKEN, UNK_TOKEN)

# 2. Linguistic Phoneme Units (Single Source of Truth)
# Based exactly on the Hebrew Phonemes spec.
STRESS_MARKS: Final[tuple[str, ...]] = ("ˈ",)
VOWEL_UNITS: Final[tuple[str, ...]] = ("a", "e", "i", "o", "u")
CONSONANT_UNITS: Final[tuple[str, ...]] = (
    "b", "v", "d", "h", "z", "χ", "t", "j", "k", "l", "m", "n", "s", "f", "p", 
    "ts", "tʃ", "w", "ʔ", "ɡ", "ʁ", "ʃ", "ʒ", "dʒ"
)

# 3. Punctuation & Spacing
SPACE_TOKEN: Final[str] = " "

# Copied from string.punctuation
PUNCTUATION_TOKENS: str = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""

# 4. Dynamic Character-Level Extraction
# Breaks down composite phonemes (ts, tʃ, dʒ) into unique individual characters.
# Using sorted() guarantees the IDs will be exactly the same on every run.
_unique_ipa_chars = sorted(list(set("".join(STRESS_MARKS + VOWEL_UNITS + CONSONANT_UNITS))))

# 4.5 ASCII Safety Net (Pass-through for English and Numbers)
# We add standard ASCII letters and digits, strictly ensuring no duplicates with IPA
_ascii_fallback = sorted(list(set(string.ascii_letters + string.digits) - set(_unique_ipa_chars)))

# The actual modeling tokens used by the CTC head
IPA_CHAR_TOKENS: Final[tuple[str, ...]] = (
    SPACE_TOKEN,
    *PUNCTUATION_TOKENS,
    *_unique_ipa_chars,
    *_ascii_fallback
)

# 5. Final Vocabulary & Mappings
DECODER_VOCAB: Final[tuple[str, ...]] = SPECIAL_TOKENS + IPA_CHAR_TOKENS

CTC_BLANK_ID: Final[int] = 0
PAD_ID: Final[int] = 1
UNK_ID: Final[int] = 2

TOKEN_TO_ID: Final[dict[str, int]] = {token: idx for idx, token in enumerate(DECODER_VOCAB)}
ID_TO_TOKEN: Final[dict[int, str]] = {idx: token for idx, token in enumerate(DECODER_VOCAB)}
VALID_OUTPUT_TOKENS: Final[frozenset[str]] = frozenset(IPA_CHAR_TOKENS)
