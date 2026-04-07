#!/usr/bin/env -S uv run
# /// script
# dependencies = []
# ///

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


VOWELS = set("aeiou")
ALLOWED = {
    "א": {"ʔ", ""},
    "ב": {"b", "v"},
    "ג": {"ɡ", "dʒ"},
    "ד": {"d"},
    "ה": {"h", ""},
    "ו": {"v", "w", ""},
    "ז": {"z", "ʒ"},
    "ח": {"χ"},
    "ט": {"t"},
    "י": {"j", ""},
    "כ": {"k", "χ"},
    "ך": {"k", "χ"},
    "ל": {"l"},
    "מ": {"m"},
    "ם": {"m"},
    "נ": {"n"},
    "ן": {"n"},
    "ס": {"s"},
    "ע": {"ʔ", ""},
    "פ": {"p", "f"},
    "ף": {"p", "f"},
    "צ": {"ts", "tʃ"},
    "ץ": {"ts", "tʃ"},
    "ק": {"k"},
    "ר": {"ʁ"},
    "ש": {"ʃ", "s", ""},
    "ת": {"t"},
}


def is_hebrew_letter(ch: str) -> bool:
    return "\u05d0" <= ch <= "\u05ea"


def parse_chunk(chunk: str) -> tuple[str, str, bool]:
    if chunk in {"", " "}:
        return "", "", False

    stress = "ˈ" in chunk
    chunk = chunk.replace("ˈ", "")

    consonant = ""
    for multi in ("tʃ", "dʒ", "ts"):
        if chunk.startswith(multi):
            consonant = multi
            chunk = chunk[len(multi) :]
            break
    else:
        if chunk and chunk[0] not in VOWELS:
            consonant = chunk[0]
            chunk = chunk[1:]

    vowel = chunk
    if vowel.endswith("χ"):
        consonant = "χ"
        vowel = vowel[:-1]

    return consonant, vowel, stress


def filtered_hebrew(text: str) -> str:
    return "".join(ch for ch in text if is_hebrew_letter(ch) or ch == " ")


def check_example(row_num: int, example: dict) -> list[tuple]:
    issues: list[tuple] = []
    hebrew = example["hebrew"]
    alignment = example["alignment"]

    aligned_chars = "".join(ch for ch, _ in alignment)
    expected_chars = filtered_hebrew(hebrew)
    if aligned_chars != expected_chars:
        issues.append(("text_mismatch", row_num, aligned_chars[:80], expected_chars[:80]))
        return issues

    for pos, (char, chunk) in enumerate(alignment):
        if char == " ":
            if chunk != " ":
                issues.append(("space_chunk_not_space", row_num, pos, repr(chunk)))
            continue

        consonant, vowel, stress = parse_chunk(chunk)

        if vowel and any(ch not in VOWELS for ch in vowel):
            issues.append(("bad_vowel_tail", row_num, pos, char, chunk))
            continue

        ok = consonant in ALLOWED.get(char, set())
        if char == "ו" and consonant == "" and vowel in {"u", "o"}:
            ok = True
        if char == "י" and consonant == "" and vowel == "i":
            ok = True
        if char in {"א", "ה", "ו", "י", "ע", "ש"} and consonant == "" and vowel in VOWELS | {""}:
            ok = True
        if char == "ח" and consonant == "χ" and vowel in VOWELS | {""}:
            ok = True

        if not ok:
            issues.append(("illegal_chunk", row_num, pos, char, chunk, consonant, vowel, stress))

    return issues


def main() -> None:
    path = Path("dataset/train.jsonl")
    limit = 1000
    stats = Counter()
    issues: list[tuple] = []
    samples: list[tuple[str, list[list[str]]]] = []

    with path.open() as f:
        for row_num, line in enumerate(f, 1):
            if row_num > limit:
                break
            example = json.loads(line)
            stats["rows"] += 1
            stats["pairs"] += len(example["alignment"])
            stats["spaces"] += sum(1 for ch, _ in example["alignment"] if ch == " ")
            stats["punctuation_omitted"] += sum(
                1 for ch in example["hebrew"] if not is_hebrew_letter(ch) and ch != " "
            )
            issues.extend(check_example(row_num, example))
            if len(samples) < 5:
                samples.append((example["hebrew"], example["alignment"][:12]))

    print("checked_rows=", stats["rows"], sep="")
    print("pairs=", stats["pairs"], sep="")
    print("spaces=", stats["spaces"], sep="")
    print("punctuation_omitted=", stats["punctuation_omitted"], sep="")
    print("issue_count=", len(issues), sep="")
    for issue in issues[:20]:
        print("issue", issue)
    print("samples:")
    for hebrew, alignment_prefix in samples:
        print(hebrew)
        print(alignment_prefix)


if __name__ == "__main__":
    main()
