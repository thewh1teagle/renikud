# /// script
# requires-python = ">=3.12"
# dependencies = ["tokenizers", "transformers"]
# ///
"""Round-trip test: vocalized Hebrew → strip nikud → tokenize → extract labels → reconstruct → assert == original."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from nikud import is_hebrew_letter, remove_nikud, sort_diacritics, extract_labels, NIKUD_CLASSES, SHIN_CLASSES, SHIN_LETTER, _NIKUD_PATTERN
from tokenization import load_tokenizer

KNESSET = Path(__file__).parent.parent.parent / "knesset.txt"
N = 100


def iter_char_diacritics(text: str):
    """Yield (pos, letter, diacritics) for each character in text.
    diacritics is the string of nikud codepoints immediately following the letter.
    """
    i = 0
    while i < len(text):
        char = text[i]
        i += 1
        diacritics = ""
        while i < len(text) and _NIKUD_PATTERN.match(text[i]):
            diacritics += text[i]
            i += 1
        yield char, diacritics


def reconstruct(stripped: str, offset_mapping, nikud_ids: list[int], shin_ids: list[int]) -> str:
    """Reinsert nikud onto stripped text using per-token predictions."""
    result = []
    prev_end = 0
    tok_idx = 0

    for start, end in offset_mapping:
        if start > prev_end:
            result.append(stripped[prev_end:start])
        if end - start != 1:
            if end > start:
                prev_end = end
            tok_idx += 1
            continue

        char = stripped[start:end]
        prev_end = end

        if not is_hebrew_letter(char):
            result.append(char)
            tok_idx += 1
            continue

        nikud = NIKUD_CLASSES[nikud_ids[tok_idx]]
        shin = ""
        if char == SHIN_LETTER:
            shin = SHIN_CLASSES[shin_ids[tok_idx]]

        result.append(char + "".join(sorted(shin + nikud)))
        tok_idx += 1

    if prev_end < len(stripped):
        result.append(stripped[prev_end:])

    return "".join(result)


def main():
    tokenizer = load_tokenizer()
    lines = KNESSET.read_text(encoding="utf-8").splitlines()[:N]

    passed = 0
    failed = 0
    for i, original in enumerate(lines):
        original = sort_diacritics(original)
        stripped = remove_nikud(original)

        # Extract labels from original by walking char+diacritics
        nikud_ids = []
        shin_ids = []
        for char, diacritics in iter_char_diacritics(original):
            if is_hebrew_letter(char):
                n_id, s_id = extract_labels(char, diacritics)
                nikud_ids.append(n_id)
                shin_ids.append(s_id)

        # Tokenize stripped text
        enc = tokenizer(stripped, return_offsets_mapping=True)
        offset_mapping = enc["offset_mapping"]

        # Align nikud_ids to token positions (Hebrew letters only, in order)
        heb_tok_indices = [
            tok_idx for tok_idx, (s, e) in enumerate(offset_mapping)
            if e - s == 1 and is_hebrew_letter(stripped[s:e])
        ]

        assert len(heb_tok_indices) == len(nikud_ids), (
            f"line {i}: {len(heb_tok_indices)} Hebrew tokens vs {len(nikud_ids)} labels"
        )

        # Build per-token label arrays (0 for non-Hebrew positions)
        full_nikud = [0] * len(offset_mapping)
        full_shin = [0] * len(offset_mapping)
        for label_idx, tok_idx in enumerate(heb_tok_indices):
            full_nikud[tok_idx] = nikud_ids[label_idx]
            full_shin[tok_idx] = shin_ids[label_idx]

        reconstructed = reconstruct(stripped, offset_mapping, full_nikud, full_shin)

        if reconstructed == original:
            passed += 1
        else:
            failed += 1
            if failed <= 5:
                print(f"FAIL line {i}:")
                print(f"  orig: {original}")
                print(f"  reco: {reconstructed}")
                # find first diff
                for j, (a, b) in enumerate(zip(original, reconstructed)):
                    if a != b:
                        print(f"  first diff at pos {j}: orig={repr(a)} reco={repr(b)}")
                        break

    print(f"\n{passed}/{passed+failed} passed")


if __name__ == "__main__":
    main()
