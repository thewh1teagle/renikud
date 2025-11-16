"""
wget "https://huggingface.co/datasets/thewh1teagle/phonikud-phonemes-data/resolve/main/knesset_phonemes_v1.txt.7z"
7z x knesset_phonemes_v1.txt.7z
"""

import string
import csv
from tqdm import tqdm
import regex as re

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

def deduplicate_diacritics(text: str):
    new_text = ''
    for c in text:
        if c in DEDUPLICATE_MAP:
            new_text += DEDUPLICATE_MAP[c]
        else:
            new_text += c
    return new_text

def clean_dagesh(text: str):
    # Keep DAGESH for בכפת letters, remove from others
    KEEP_DAGESH = 'בכךפףו'
    # Match letter + diacritics, remove DAGESH if letter not in KEEP_DAGESH
    def replace_dagesh(match):
        letter = match.group(1)
        diacritics = match.group(2)
        if letter not in KEEP_DAGESH:
            diacritics = diacritics.replace(DAGESH, '')
        return letter + diacritics
    return re.sub(r'(\p{L})(\p{Mn}+)', replace_dagesh, text)

def clean_text(text: str):
    cleaned = ''
    for c in text:
        if c in RELEVANT_CHARS:
            cleaned += c
    return cleaned

if __name__ == "__main__":
    file = 'knesset_phonemes_v1.txt'
    out_file = 'renikud_data_v1.txt'
    total = sum(1 for _ in open(file, encoding='utf-8'))
    with open(file, encoding='utf-8') as f, open(out_file, 'w', encoding='utf-8') as out:
        reader = csv.reader(f, delimiter='\t')
        for row in tqdm(reader, total=total):
            text, phonemes = row
            # text = "הַ|כְּנֶ֫סֶת"
            # breakpoint()
            text = clean_text(text)
            text = clean_dagesh(text)
            text = deduplicate_diacritics(text)
            out.write(f'{text}\n')

