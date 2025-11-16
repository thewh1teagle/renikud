"""
Prepare training data for Renikud model

Usage:
    wget "https://huggingface.co/datasets/thewh1teagle/phonikud-phonemes-data/resolve/main/knesset_phonemes_v1.txt.7z"
    7z x knesset_phonemes_v1.txt.7z
    python -m src.prepare_data
"""

import csv
from tqdm import tqdm
import regex as re
from constants import DEDUPLICATE_MAP, DAGESH, RELEVANT_CHARS, VOWEL_CLASSES, CAN_HAVE_DAGESH

def deduplicate_diacritics(text: str):
    """Map various diacritics to their canonical forms"""
    new_text = ''
    for c in text:
        if c in DEDUPLICATE_MAP:
            new_text += DEDUPLICATE_MAP[c]
        else:
            new_text += c
    return new_text

def clean_dagesh(text: str):
    """Keep DAGESH only for letters that can have it: בכךפףו"""
    # Match letter + diacritics, remove DAGESH if letter not in CAN_HAVE_DAGESH
    def replace_dagesh(match):
        letter = match.group(1)
        diacritics = match.group(2)
        if letter not in CAN_HAVE_DAGESH:
            diacritics = diacritics.replace(DAGESH, '')
        return letter + diacritics
    return re.sub(r'(\p{L})(\p{Mn}+)', replace_dagesh, text)

def clean_text(text: str):
    """Keep only relevant characters"""
    cleaned = ''
    for c in text:
        if c in RELEVANT_CHARS:
            cleaned += c
    return cleaned

def validate_text(text: str):
    """Check if text contains only valid vowels and diacritics"""
    # Extract all vowels from text
    nikud_pattern = re.compile(r'[\u05b0-\u05bc\u05c1\u05c2\u05c7]')
    diacritics = nikud_pattern.findall(text)
    
    valid_diacritics = set(VOWEL_CLASSES[1:]) | {DAGESH, '\u05c1', '\u05c2'}  # vowels + dagesh + shin/sin dots
    
    for diacritic in diacritics:
        if diacritic not in valid_diacritics:
            return False
    return True

if __name__ == "__main__":
    file = 'knesset_phonemes_v1.txt'
    out_file = 'renikud_data_v1.txt'
    
    print(f"📖 Reading from {file}")
    print(f"📝 Writing to {out_file}")
    print("🔧 Processing with Renikud simplifications:")
    print("   - Deduplicate diacritics (e.g., QAMATS -> PATAH)")
    print("   - Keep DAGESH only for בכךפףו")
    print("   - Keep only relevant characters")
    print(f"   - Keep only 6 main vowels: {VOWEL_CLASSES[1:]}")
    
    total = sum(1 for _ in open(file, encoding='utf-8'))
    skipped = 0
    
    with open(file, encoding='utf-8') as f, open(out_file, 'w', encoding='utf-8') as out:
        reader = csv.reader(f, delimiter='\t')
        for row in tqdm(reader, total=total):
            if len(row) < 2:
                skipped += 1
                continue
                
            text, phonemes = row[0], row[1]
            
            # Clean and simplify
            text = clean_text(text)
            text = clean_dagesh(text)
            text = deduplicate_diacritics(text)
            
            # Validate
            if not validate_text(text):
                skipped += 1
                continue
            
            # Skip if too short
            if len(text.strip()) < 2:
                skipped += 1
                continue
            
            out.write(f'{text}\n')
    
    print(f"✅ Done! Skipped {skipped} lines")
    print(f"📊 Output: {out_file}")
