"""
wget "https://huggingface.co/datasets/thewh1teagle/phonikud-phonemes-data/resolve/main/knesset_phonemes_v1.txt.7z"
7z x knesset_phonemes_v1.txt.7z
"""

import csv
from tqdm import tqdm
import regex as re
from constants import DEDUPLICATE_MAP, DAGESH, RELEVANT_CHARS

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

