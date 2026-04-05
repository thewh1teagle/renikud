"""Align Hebrew letters to IPA chunks."""

import multiprocessing as mp
import argparse
import json
import re
import unicodedata

from tqdm import tqdm

from phonology import HEBREW_LETTER_CONSONANTS as HEBREW_CONSONANTS

VOWELS = ("a", "e", "i", "o", "u")
STRESS = "ˈ"
HEB_RE = r"[^\u05d0-\u05ea]"
IPA_RE = r"[^abdefghijklmnoprstuvwzɡʁʃʒʔˈχ]"


def strip_nikud(text: str) -> str:
    return re.sub(r"[\p{M}|]", "", unicodedata.normalize("NFD", text))


def _chunk_pattern(char: str) -> str:
    """Regex capture group for the IPA chunk a single Hebrew letter produces."""
    allowed = HEBREW_CONSONANTS.get(char, ("",))
    cons_alts = sorted([re.escape(c) for c in allowed if c], key=len, reverse=True)
    has_silent = "" in allowed

    vowel_re = f"ˈ?(?:{'|'.join(VOWELS)})?"

    patterns = []

    if cons_alts:
        cons_re = "|".join(cons_alts)
        cons_re = f"(?:{cons_re})" if len(cons_alts) > 1 else cons_alts[0]
        patterns.append(f"{cons_re}{vowel_re}")

    if char == "ו":
        patterns.append(r"ˈ?(?:u|o)")
    if char == "י":
        patterns.append(r"ˈ?i")

    if char == "ח":
        patterns.append(f"ˈ?(?:{'|'.join(VOWELS)})?χ")

    if has_silent:
        patterns.append(f"ˈ?(?:{'|'.join(VOWELS)})|ˈ|")

    combined = "|".join(f"(?:{p})" for p in patterns if p)
    if has_silent or char in ("ו", "י"):
        combined += "|"
    return f"({combined})"


_LETTER_PATTERNS: dict[str, str] = {
    char: _chunk_pattern(char) for char in HEBREW_CONSONANTS
}


def align_word(heb_word: str, ipa_word: str) -> list[tuple[str, str]] | None:
    """Align one Hebrew word to one IPA word."""
    pattern = "".join(_LETTER_PATTERNS.get(c, r"(\S*)") for c in heb_word)
    m = re.fullmatch(pattern, ipa_word)
    if m is None:
        return None
    return list(zip(heb_word, m.groups()))


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
        description="Align Hebrew chars to IPA chunks via regex"
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
