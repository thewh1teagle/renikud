"""Align Hebrew letters to IPA chunks."""

import multiprocessing as mp
import argparse
import json
import unicodedata
from functools import lru_cache

import regex as re
from tqdm import tqdm

from phonology import HEBREW_LETTER_CONSONANTS as HEBREW_CONSONANTS

VOWELS = ("a", "e", "i", "o", "u")
STRESS = "ˈ"
VOWEL_CARRIERS = {"ו": ("u", "o"), "י": ("i",)}
DEFAULT_ORDER = (
    "plain",
    "vowel",
    "stressed_vowel",
    "carrier",
    "furtive",
    "stressed",
    "silent",
)
VOWEL_ORDER = (
    "vowel",
    "stressed_vowel",
    "plain",
    "carrier",
    "furtive",
    "stressed",
    "silent",
)
GLIDE_ORDER = (
    "stressed_vowel",
    "vowel",
    "plain",
    "carrier",
    "furtive",
    "stressed",
    "silent",
)
SILENT_ORDER = (
    "silent",
    "plain",
    "vowel",
    "stressed_vowel",
    "carrier",
    "furtive",
    "stressed",
)
HEB_RE = r"[^\u05d0-\u05ea]"
IPA_RE = r"[^abdefghijklmnoprstuvwzɡʁʃʒʔˈχ]"


def strip_nikud(text: str) -> str:
    return re.sub(r"[\p{M}|]", "", unicodedata.normalize("NFD", text))


def _ordered_candidates(heb_word: str, ipa_word: str, i: int, j: int) -> list[str]:
    char = heb_word[i]
    rest = ipa_word[j:]
    same_prev = i > 0 and heb_word[i - 1] == char
    next_char = heb_word[i + 1] if i + 1 < len(heb_word) else ""
    next_next = heb_word[i + 2] if i + 2 < len(heb_word) else ""
    same_next = next_char == char
    is_last = i == len(heb_word) - 1
    allowed = HEBREW_CONSONANTS.get(char, ("",))
    allow_repeat_silent = same_prev or (same_next and "" in allowed)
    vowel_first = next_char == next_next == "י" or (
        next_char == "ח" and i + 1 == len(heb_word) - 1
    )
    glide_first = same_next and (
        (char == "ו" and rest.startswith("w")) or (char == "י" and rest.startswith("j"))
    )

    groups = {key: [] for key in DEFAULT_ORDER}
    for consonant in allowed:
        if not consonant or not rest.startswith(consonant):
            continue
        tail = rest[len(consonant) :]
        groups["plain"].append(consonant)
        if tail.startswith(STRESS):
            stressed = consonant + STRESS
            stressed_tail = tail[1:]
            groups["stressed"].append(stressed)
            groups["stressed_vowel"].extend(
                stressed + vowel for vowel in VOWELS if stressed_tail.startswith(vowel)
            )
        groups["vowel"].extend(
            consonant + vowel for vowel in VOWELS if tail.startswith(vowel)
        )

    for vowel in VOWEL_CARRIERS.get(char, ()):
        groups["carrier"].extend(
            chunk for chunk in (STRESS + vowel, vowel) if rest.startswith(chunk)
        )

    if is_last and char == "ח":
        for stress in (STRESS, ""):
            groups["furtive"].extend(
                stress + vowel + "χ"
                for vowel in (*VOWELS, "")
                if rest.startswith(stress + vowel + "χ")
            )

    if "" in allowed or allow_repeat_silent:
        groups["silent"].append("")

    order = DEFAULT_ORDER

    if same_next:
        if glide_first:
            order = GLIDE_ORDER
        elif char not in ("ו", "י") and "" in allowed:
            order = SILENT_ORDER
    elif vowel_first:
        order = VOWEL_ORDER

    return list(
        dict.fromkeys(chunk for group_name in order for chunk in groups[group_name])
    )


def align_word(heb_word: str, ipa_word: str) -> list[tuple[str, str]] | None:
    """Align one Hebrew word to one IPA word."""
    n = len(heb_word)

    @lru_cache(maxsize=None)
    def solve(i: int, j: int) -> tuple[tuple[str, str], ...] | None:
        if i == n:
            return () if j == len(ipa_word) else None

        for chunk in _ordered_candidates(heb_word, ipa_word, i, j):
            if (tail := solve(i + 1, j + len(chunk))) is not None:
                return ((heb_word[i], chunk),) + tail
        return None

    return list(result) if (result := solve(0, 0)) is not None else None


def align_sentence(heb: str, ipa: str) -> list[tuple[str, str]] | None:
    """Align a sentence."""
    heb_words, ipa_words = heb.split(), ipa.split()

    if len(heb_words) != len(ipa_words):
        return None

    result = []
    for hw, iw in zip(heb_words, ipa_words):
        heb_core = re.sub(HEB_RE, "", hw)
        ipa_core = re.sub(IPA_RE, "", iw)
        if not heb_core:
            continue
        if (aligned := align_word(heb_core, ipa_core)) is None:
            return None
        if result:
            result.append((" ", " "))
        result.extend(aligned)

    return result


def process_chunk(lines: list[str]) -> list[tuple[str, list | None, str] | None]:
    """Align TSV lines."""
    out = []
    for line in lines:
        heb_raw, sep, ipa = line.strip().partition("\t")
        if sep:
            heb = strip_nikud(heb_raw)
            out.append((heb, align_sentence(heb, ipa), ipa))
        else:
            out.append(None)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Align Hebrew chars to IPA chunks via DP"
    )
    parser.add_argument("input", help="Input TSV file (hebrew<TAB>ipa)")
    parser.add_argument("output", help="Output JSONL file (one sentence per line)")
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    args = parser.parse_args()

    total = aligned_count = failed_count = 0

    failures_path = args.output.replace(".jsonl", "_failures.txt")
    with open(args.input, encoding="utf-8") as fin:
        lines = list(fin)

    chunk_size = max(1, len(lines) // (args.workers * 4))
    chunks = [lines[i : i + chunk_size] for i in range(0, len(lines), chunk_size)]

    with (
        open(args.output, "w", encoding="utf-8") as fout,
        open(failures_path, "w", encoding="utf-8") as ffail,
        mp.Pool(args.workers) as pool,
    ):
        for batch in tqdm(
            pool.imap(process_chunk, chunks), total=len(chunks), desc="Aligning"
        ):
            for heb, result, ipa in filter(None, batch):
                total += 1
                if result is not None:
                    aligned_count += 1
                    fout.write(json.dumps({heb: result}, ensure_ascii=False) + "\n")
                else:
                    failed_count += 1
                    ffail.write(f"{heb}\t{ipa}\n")

    print(f"\nTotal:    {total:,}")
    print(f"Aligned:  {aligned_count:,} ({aligned_count / total:.1%})")
    print(f"Failed:   {failed_count:,} ({failed_count / total:.1%})")


if __name__ == "__main__":
    main()
