"""
Align Hebrew characters to IPA chunks using a DP aligner.

Each Hebrew letter is assigned exactly one IPA chunk (consonant + optional vowel + optional stress).
The alignment is constrained by the known possible phonemes per Hebrew letter.

Usage:
    uv run src/align_data.py dataset/train.txt ./dataset/train_alignment.jsonl

Input TSV:   hebrew_text<TAB>ipa_text  (one sentence per line, Hebrew may have nikud)
Output JSONL: one JSON object per line, key=hebrew sentence, value=[[char, ipa_chunk], ...]
              failures saved to <output>_failures.txt
"""

import argparse
import json
import unicodedata
import regex as re
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Hebrew letter -> allowed leading consonants (the IPA token that starts its chunk)
# ∅ means the letter can be silent (emit nothing)
# ---------------------------------------------------------------------------
HEBREW_CONSONANTS: dict[str, tuple[str, ...]] = {
    "א": ("ʔ", ""),
    "ב": ("b", "v"),
    "ג": ("ɡ", "dʒ"),
    "ד": ("d",),
    "ה": ("h", ""),
    "ו": ("v", "w", ""),      # can also be the vowel u/o — handled via vowel-only path
    "ז": ("z", "ʒ"),
    "ח": ("χ",),
    "ט": ("t",),
    "י": ("j", ""),            # can also be the vowel i — handled via vowel-only path
    "כ": ("k", "χ"),
    "ך": ("k", "χ"),
    "ל": ("l",),
    "מ": ("m",),
    "ם": ("m",),
    "נ": ("n",),
    "ן": ("n",),
    "ס": ("s",),
    "ע": ("ʔ", ""),
    "פ": ("p", "f"),
    "ף": ("p", "f"),
    "צ": ("ts", "tʃ"),
    "ץ": ("ts", "tʃ"),
    "ק": ("k",),
    "ר": ("ʁ",),
    "ש": ("ʃ", "s"),
    "ת": ("t",),
}

VOWELS = ("a", "e", "i", "o", "u")
STRESS = "ˈ"
SPACE = " "


def strip_nikud(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    return re.sub(r"[\p{M}|]", "", text)


def align_word(heb_word: str, ipa_word: str) -> list[tuple[str, str]] | None:
    """
    Align a single Hebrew word to its IPA using DP.
    Returns list of (hebrew_char, ipa_chunk) or None if no valid alignment found.

    Each IPA chunk has the form: [consonant] [stress?] [vowel?]
    where consonant comes from the letter's allowed set.
    """
    n = len(heb_word)
    m = len(ipa_word)

    # dp[i][j] = True if we can align heb_word[:i] to ipa_word[:j]
    # back[i][j] = j_prev to reconstruct the path
    dp = [[False] * (m + 1) for _ in range(n + 1)]
    back = [[-1] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = True

    for i in range(1, n + 1):
        char = heb_word[i - 1]
        allowed = HEBREW_CONSONANTS.get(char, ("",))

        for j_prev in range(m + 1):
            if not dp[i - 1][j_prev]:
                continue

            rest = ipa_word[j_prev:]

            # Try each allowed consonant for this letter
            for consonant in allowed:
                pos = 0

                # Match consonant prefix
                if consonant and not rest[pos:].startswith(consonant):
                    continue
                pos += len(consonant)

                # Optionally consume stress mark
                stress_pos = pos
                if pos < len(rest) and rest[pos] == STRESS:
                    stress_pos = pos + 1

                # Try with and without stress, with and without vowel
                for s in (stress_pos, pos):  # with stress or without
                    # Try each vowel or no vowel
                    for vowel in VOWELS + ("",):
                        v_pos = s
                        if vowel:
                            if not rest[s:].startswith(vowel):
                                continue
                            v_pos = s + len(vowel)
                        j_new = j_prev + v_pos
                        if j_new <= m and not dp[i][j_new]:
                            dp[i][j_new] = True
                            back[i][j_new] = j_prev

            # Special case: word-final ח consumes [stress?]aχ.
            # The preceding vowel (o/u/e) is handled by the ו/י special case above.
            # For words without a preceding vowel letter (e.g. שמח -> samˈeaχ),
            # also allow consuming [vowel]aχ as a fallback.
            # Only applies when ח is the last letter of the word (i == n).
            if char == "ח" and i == n:
                for prefix in ("aχ", "ˈaχ"):
                    if rest.startswith(prefix):
                        j_new = j_prev + len(prefix)
                        if j_new <= m:
                            dp[i][j_new] = True
                            back[i][j_new] = j_prev
                for vowel in VOWELS:
                    for prefix in (f"{vowel}aχ", f"ˈ{vowel}aχ"):
                        if rest.startswith(prefix):
                            j_new = j_prev + len(prefix)
                            if j_new <= m and not dp[i][j_new]:
                                dp[i][j_new] = True
                                back[i][j_new] = j_prev

            # Special case: ו/י as pure vowel (u, o, i) with optional stress
            if char in ("ו", "י"):
                vowel_map = {"ו": ("u", "o"), "י": ("i",)}
                for vowel in vowel_map[char]:
                    for s_offset in (0, 1):  # without/with preceding stress
                        pos = 0
                        if s_offset and (not rest or rest[0] != STRESS):
                            continue
                        pos += s_offset
                        if rest[pos:].startswith(vowel):
                            j_new = j_prev + pos + len(vowel)
                            if j_new <= m and not dp[i][j_new]:
                                dp[i][j_new] = True
                                back[i][j_new] = j_prev

    if not dp[n][m]:
        return None

    # Reconstruct path
    chunks = []
    j = m
    for i in range(n, 0, -1):
        j_prev = back[i][j]
        chunks.append((heb_word[i - 1], ipa_word[j_prev:j]))
        j = j_prev
    chunks.reverse()
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
        ipa_core = re.sub(r"[^\w\u02c8\u0294\u0261\u0281\u0283\u0292χʁʃʒʔɡˈaeiou]", "", iw)

        if not heb_core:
            continue

        aligned = align_word(heb_core, ipa_core)
        if aligned is None:
            return None
        result.extend(aligned)

        if i < len(heb_words) - 1:
            result.append((" ", " "))

    return result


def main():
    parser = argparse.ArgumentParser(description="Align Hebrew chars to IPA chunks via DP")
    parser.add_argument("input", help="Input TSV file (hebrew<TAB>ipa)")
    parser.add_argument("output", help="Output JSONL file (one sentence per line)")
    args = parser.parse_args()

    total = 0
    aligned_count = 0
    failed_count = 0

    failures_path = args.output.replace(".jsonl", "_failures.txt")
    with open(args.input, encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout, \
         open(failures_path, "w", encoding="utf-8") as ffail:

        lines = fin.readlines()
        for line in tqdm(lines, desc="Aligning"):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue

            heb_raw, ipa = parts
            heb = strip_nikud(heb_raw)
            total += 1

            result = align_sentence(heb, ipa)
            if result is None:
                failed_count += 1
                ffail.write(f"{heb}\t{ipa}\n")
                continue

            aligned_count += 1
            fout.write(json.dumps({heb: result}, ensure_ascii=False) + "\n")

    print(f"\nTotal:    {total:,}")
    print(f"Aligned:  {aligned_count:,} ({aligned_count/total:.1%})")
    print(f"Failed:   {failed_count:,} ({failed_count/total:.1%})")


if __name__ == "__main__":
    main()
