"""
Align Hebrew characters to IPA chunks using a DP aligner.

Each Hebrew letter is assigned exactly one IPA chunk (consonant + optional vowel + optional stress).
The alignment is constrained by the known possible phonemes per Hebrew letter.

Usage:
    uv run src/data_align.py dataset/train.tsv ./dataset/train_alignment.jsonl

Input TSV:   hebrew_text<TAB>ipa_text  (one sentence per line, Hebrew may have nikud)
Output JSONL: one JSON object per line, key=hebrew sentence, value=[[char, ipa_chunk], ...]
              failures saved to <output>_failures.txt
"""

import argparse
import json
import unicodedata
import multiprocessing as mp
import regex as re
from tqdm import tqdm

from phonology import HEBREW_LETTER_CONSONANTS as HEBREW_CONSONANTS

VOWELS = ("a", "e", "i", "o", "u")
STRESS = "ˈ"
SPACE = " "


def strip_nikud(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    return re.sub(r"[\p{M}|]", "", text)


def align_word(heb_word: str, ipa_word: str) -> list[tuple[str, str]] | None:
    """
    Align a Hebrew word to IPA using cost-based DP.
    Each letter consumes one IPA chunk: [consonant][stress?][vowel?]
    Silent letters (ה/א/ע/etc acting as matres) consume nothing (empty chunk).
    Cost = number of empty chunks; minimizing this prefers consonants over silence.
    """
    n, m = len(heb_word), len(ipa_word)
    INF = float("inf")
    cost = [[INF] * (m + 1) for _ in range(n + 1)]
    back = [[-1] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0

    for i in range(1, n + 1):
        char = heb_word[i - 1]
        is_last = i == n

        for j_prev in range(m + 1):
            if cost[i - 1][j_prev] == INF:
                continue
            c = cost[i - 1][j_prev]
            rest = ipa_word[j_prev:]

            def update(length: int, penalty: int = 0) -> None:
                k = j_prev + length
                if k <= m and c + penalty < cost[i][k]:
                    cost[i][k] = c + penalty
                    back[i][k] = j_prev

            for consonant in HEBREW_CONSONANTS.get(char, ("",)):
                if not consonant:
                    update(0, penalty=1)
                    continue
                if not rest.startswith(consonant):
                    continue
                pos = len(consonant)
                s = pos + 1 if pos < len(rest) and rest[pos] == STRESS else pos
                for stress_end in ({s, pos} if s != pos else {pos}):
                    for vowel in VOWELS + ("",):
                        if vowel:
                            if rest[stress_end:].startswith(vowel):
                                update(stress_end + len(vowel))
                        else:
                            update(stress_end)

            # ו/י as pure vowel (with optional preceding stress)
            if char in ("ו", "י"):
                vowel_map = {"ו": ("u", "o"), "י": ("i",)}
                for vowel in vowel_map[char]:
                    for s in (1, 0):
                        if s and (not rest or rest[0] != STRESS):
                            continue
                        if rest[s:].startswith(vowel):
                            update(s + len(vowel))

            # Word-final ח furtive patah: [stress?][vowel]χ
            if char == "ח" and is_last:
                s = 1 if rest.startswith(STRESS) else 0
                for vowel in VOWELS + ("",):
                    p = s + len(vowel) if vowel and rest[s:].startswith(vowel) else (s if not vowel else None)
                    if p is not None and rest[p:].startswith("χ"):
                        update(p + 1, penalty=1)

    if cost[n][m] == INF:
        return None
    chunks, j = [], m
    for i in range(n, 0, -1):
        j_prev = back[i][j]
        chunks.append((heb_word[i - 1], ipa_word[j_prev:j]))
        j = j_prev
    return chunks[::-1]


def align_sentence(heb: str, ipa: str) -> list[tuple[str, str]] | None:
    """
    Align a full sentence by splitting on spaces and aligning word by word.
    Non-Hebrew characters (punctuation, digits) are stripped before alignment.
    Spaces are passed through as (' ', ' ').
    """
    heb_words = heb.split(" ")
    ipa_words = ipa.split(" ")

    if len(heb_words) != len(ipa_words):
        return None

    result = []
    for i, (hw, iw) in enumerate(zip(heb_words, ipa_words)):
        if not hw:
            continue

        # Keep only Hebrew letters for alignment
        heb_core = re.sub(r"[^\u05d0-\u05ea]", "", hw)
        # Keep only IPA phoneme characters (strip punctuation like . , ? !)
        ipa_core = re.sub(r"[^abdefghijklmnoprstuvwzɡʁʃʒʔˈχ]", "", iw)

        if not heb_core:
            continue

        aligned = align_word(heb_core, ipa_core)
        if aligned is None:
            return None
        result.extend(aligned)

        if i < len(heb_words) - 1:
            result.append((" ", " "))

    return result


def process_chunk(lines: list[str]) -> list[tuple[str, list | None, str] | None]:
    """Align a batch of TSV lines. Returns list of (heb, result, ipa) or None to skip."""
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            out.append(None)
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            out.append(None)
            continue
        heb_raw, ipa = parts
        heb = strip_nikud(heb_raw)
        result = align_sentence(heb, ipa)
        out.append((heb, result, ipa))
    return out


def main():
    parser = argparse.ArgumentParser(description="Align Hebrew chars to IPA chunks via DP")
    parser.add_argument("input", help="Input TSV file (hebrew<TAB>ipa)")
    parser.add_argument("output", help="Output JSONL file (one sentence per line)")
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    args = parser.parse_args()

    total = 0
    aligned_count = 0
    failed_count = 0

    failures_path = args.output.replace(".jsonl", "_failures.txt")
    with open(args.input, encoding="utf-8") as fin:
        lines = fin.readlines()

    chunk_size = max(1, len(lines) // (args.workers * 4))
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]

    with open(args.output, "w", encoding="utf-8") as fout, \
         open(failures_path, "w", encoding="utf-8") as ffail, \
         mp.Pool(args.workers) as pool:

        for batch in tqdm(pool.imap(process_chunk, chunks), total=len(chunks), desc="Aligning"):
            for item in batch:
                if item is None:
                    continue
                heb, result, ipa = item
                total += 1
                if result is None:
                    failed_count += 1
                    ffail.write(f"{heb}\t{ipa}\n")
                else:
                    aligned_count += 1
                    fout.write(json.dumps({heb: result}, ensure_ascii=False) + "\n")

    print(f"\nTotal:    {total:,}")
    print(f"Aligned:  {aligned_count:,} ({aligned_count/total:.1%})")
    print(f"Failed:   {failed_count:,} ({failed_count/total:.1%})")


if __name__ == "__main__":
    main()
