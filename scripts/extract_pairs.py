"""
Extract Hebrew-IPA pairs from train_decoded_ipa.tsv by validating word-by-word
using the aligner. Collects consecutive valid word pairs into sequences.

Usage:
    uv run scripts/extract_pairs.py train_decoded_ipa.tsv pairs.tsv
"""

import sys
import re
sys.path.insert(0, 'src')
from tqdm import tqdm

from aligner.align import align_word

HEBREW_RE = re.compile(r'[\u05d0-\u05ea]')


def split_words(text: str) -> list[str]:
    return text.split()


def is_hebrew_word(w: str) -> bool:
    return bool(HEBREW_RE.search(w))


def try_align(heb_word: str, ipa_word: str) -> bool:
    clean_heb = re.sub(r'[^\u05d0-\u05eaא-ת\'\-]', '', heb_word)
    if not clean_heb:
        return False
    return align_word(clean_heb, ipa_word) is not None


def extract_sequences(heb: str, pho: str) -> list[tuple[str, str]]:
    heb_words = split_words(heb)
    pho_words = split_words(pho)

    if not heb_words or not pho_words:
        return []

    sequences = []
    current_heb = []
    current_pho = []

    hi, pi = 0, 0
    while hi < len(heb_words) and pi < len(pho_words):
        hw = heb_words[hi]
        pw = pho_words[pi]

        if not is_hebrew_word(hw):
            if current_heb:
                sequences.append((' '.join(current_heb), ' '.join(current_pho)))
                current_heb, current_pho = [], []
            hi += 1
            continue

        if try_align(hw, pw):
            current_heb.append(hw)
            current_pho.append(pw)
            hi += 1
            pi += 1
        else:
            # Lookahead: try next N Hebrew words against same IPA word
            LOOKAHEAD = 3
            resynced = False
            for skip in range(1, LOOKAHEAD + 1):
                if hi + skip < len(heb_words) and try_align(heb_words[hi + skip], pw):
                    if current_heb:
                        sequences.append((' '.join(current_heb), ' '.join(current_pho)))
                        current_heb, current_pho = [], []
                    hi += skip  # skip unaligned heb words, retry same pi
                    resynced = True
                    break
            if not resynced:
                # Also try next N IPA words against current Hebrew word
                for skip in range(1, LOOKAHEAD + 1):
                    if pi + skip < len(pho_words) and try_align(hw, pho_words[pi + skip]):
                        if current_heb:
                            sequences.append((' '.join(current_heb), ' '.join(current_pho)))
                            current_heb, current_pho = [], []
                        pi += skip
                        resynced = True
                        break
            if not resynced:
                if current_heb:
                    sequences.append((' '.join(current_heb), ' '.join(current_pho)))
                    current_heb, current_pho = [], []
                hi += 1
                pi += 1

    if current_heb:
        sequences.append((' '.join(current_heb), ' '.join(current_pho)))

    return sequences


def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    pairs = []
    total_lines = 0

    with open(input_path, encoding='utf-8') as f:
        lines = f.readlines()
    for line in tqdm(lines, desc="Extracting"):
        parts = line.rstrip('\n').split('\t')
        if len(parts) < 3:
            continue
        total_lines += 1
        transcript = parts[1] + ' = ' + parts[2]
        segments = transcript.split(' = ')

        i = 0
        while i + 1 < len(segments):
            heb = segments[i].strip()
            pho = segments[i + 1].strip()
            i += 2
            if not heb or not pho:
                continue
            if not HEBREW_RE.search(heb):
                continue
            for seq in extract_sequences(heb, pho):
                if len(seq[0].split()) >= 2:
                    pairs.append(seq)

    with open(output_path, 'w', encoding='utf-8') as f:
        for heb, pho in pairs:
            f.write(f'{heb}\t{pho}\n')

    print(f'Done: {len(pairs)} sequences from {total_lines} lines')


if __name__ == '__main__':
    main()
