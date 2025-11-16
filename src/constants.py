"""
https://en.wikipedia.org/wiki/Unicode_and_HTML_for_the_Hebrew_alphabet#Compact_table
"""

# To keep
SHVA = '\u05b0'
SEGOL = '\u05b6'
HIRIK = '\u05b4'
PATAH = '\u05b7'
HOLAM = '\u05b9'
QUBUTS = '\u05bb'
DAGESH = '\u05bc'
SIN_DOT = '\u05c2'
QAMATS_QATAN = '\u05c7'
HATAMA = '\u05ab'

# To remove
HATAF_SEGOL = '\u05b1'
HATAF_PATAH = '\u05b2'
HATAF_QAMATS = '\u05b3'
TSERE = '\u05b5'
QAMATS = '\u05b8'
HOLAM_HASER_FOR_VAV = '\u05ba'

HEBREW_PATTERN = r'[\u0590-\u05ff|]' # Phonikud Hebrew pattern

HEBREW_PUNCTUATION = r"""?!,.:"'-"""

HEBREW_ALPHABET = 'אבגדהוזחטיכךלמםנןסעפףצץקרשת'
RELEVANT_DIACRITICS = [
    # To keep
    SHVA,
    SEGOL,
    HIRIK,
    PATAH,
    HOLAM,
    QUBUTS,
    DAGESH,
    SIN_DOT,
    QAMATS_QATAN,
    HATAMA,

    # To remove
    HATAF_SEGOL,
    HATAF_PATAH,
    HATAF_QAMATS,
    TSERE,
    QAMATS,
    HOLAM_HASER_FOR_VAV,
]

RELEVANT_CHARS = set(''.join(RELEVANT_DIACRITICS) + HEBREW_ALPHABET + HEBREW_PUNCTUATION + ' ')

DEDUPLICATE_MAP = {
    HATAF_SEGOL: SEGOL,
    HATAF_PATAH: PATAH, 
    HATAF_QAMATS: HOLAM, 
    TSERE: SEGOL,
    QAMATS: PATAH,
    HOLAM_HASER_FOR_VAV: HOLAM
}