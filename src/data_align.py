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


def _candidates(char: str, rest: str, is_last: bool) -> list[int]:
    """
    Return valid IPA-consumption lengths for *char* at *rest*,
    in priority order.  The greedy forward pass picks the first
    candidate whose remaining suffix is reachable.

    Priority tiers (tried in order):
      T1  consonant only                       (let next letter take the vowel)
      T2  consonant + vowel   (no stress)      (take the vowel ourselves)
      T3  consonant + stress + vowel           (take stress + vowel)
      T4  ו/י pure-vowel carriers              (mater lectionis)
      T5  furtive patah (word-final ח)
      T6  consonant + stress  (no vowel)       (rare — stress without vowel)
      T7  silent  (empty chunk)
    """
    allowed = HEBREW_CONSONANTS.get(char, ("",))
    t1: list[int] = []  # consonant only
    t2: list[int] = []  # consonant + vowel
    t3: list[int] = []  # consonant + stress + vowel
    t4: list[int] = []  # ו/י vowel carrier
    t5: list[int] = []  # furtive patah
    t6: list[int] = []  # consonant + stress (no vowel)

    for consonant in allowed:
        if not consonant:
            continue
        if not rest.startswith(consonant):
            continue
        clen = len(consonant)
        t1.append(clen)
        # consonant + stress + vowel / consonant + stress only
        if clen < len(rest) and rest[clen] == STRESS:
            for v in VOWELS:
                if rest[clen + 1:].startswith(v):
                    t3.append(clen + 1 + len(v))
            t6.append(clen + 1)
        # consonant + vowel (no stress)
        for v in VOWELS:
            if rest[clen:].startswith(v):
                t2.append(clen + len(v))

    # ו/י as pure vowel carriers
    if char in ("ו", "י"):
        vowel_map = {"ו": ("u", "o"), "י": ("i",)}
        for v in vowel_map[char]:
            if rest.startswith(STRESS + v):
                t4.append(1 + len(v))
            if rest.startswith(v):
                t4.append(len(v))

    # Furtive patah — word-final ח
    if char == "ח" and is_last:
        for v in VOWELS:
            if rest.startswith(STRESS + v + "χ"):
                t5.append(2 + len(v))
            if rest.startswith(v + "χ"):
                t5.append(len(v) + 1)
        if rest.startswith(STRESS + "χ"):
            t5.append(2)
        if rest.startswith("χ"):
            t5.append(1)

    # Assemble in priority order, deduplicate
    silent = [0] if "" in allowed else []
    all_lengths = t1 + t2 + t3 + t4 + t5 + t6 + silent
    seen: set[int] = set()
    unique: list[int] = []
    for ln in all_lengths:
        if ln not in seen:
            seen.add(ln)
            unique.append(ln)
    return unique


def align_word(heb_word: str, ipa_word: str) -> list[tuple[str, str]] | None:
    """
    Align Hebrew word to IPA: backward reachability DP + greedy forward pick.

    Phase 1: reach[i][j] = can heb[i:] consume ipa[j:] exactly?
    Phase 2: walk forward, picking the longest valid chunk per letter.
    """
    n = len(heb_word)
    m = len(ipa_word)

    # Phase 1 — backward reachability
    reach = [[False] * (m + 1) for _ in range(n + 1)]
    reach[n][m] = True

    for i in range(n - 1, -1, -1):
        char = heb_word[i]
        is_last = (i == n - 1)
        for j in range(m + 1):
            for ln in _candidates(char, ipa_word[j:], is_last):
                if j + ln <= m and reach[i + 1][j + ln]:
                    reach[i][j] = True
                    break

    if not reach[0][0]:
        return None

    # Phase 2 — greedy forward
    chunks: list[tuple[str, str]] = []
    j = 0
    for i in range(n):
        char = heb_word[i]
        is_last = (i == n - 1)
        for ln in _candidates(char, ipa_word[j:], is_last):
            if j + ln <= m and reach[i + 1][j + ln]:
                chunks.append((char, ipa_word[j:j + ln]))
                j += ln
                break

    return chunks


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
