"""Nikud (diacritics) constants and label extraction for Hebrew diacritization."""

from __future__ import annotations

import regex
import re

NIKUD_CLASSES = [
    "",          # 0 — no nikud
    "\u05B0",    # shva
    "\u05B1",    # hataf segol
    "\u05B2",    # hataf patah
    "\u05B3",    # hataf qamats
    "\u05B4",    # hiriq
    "\u05B5",    # tsere
    "\u05B6",    # segol
    "\u05B7",    # patah
    "\u05B8",    # qamats
    "\u05B9",    # holam
    "\u05BA",    # holam haser
    "\u05BB",    # qubuts
    "\u05BC",    # dagesh
    "\u05B0\u05BC",  # shva + dagesh
    "\u05B1\u05BC",  # hataf segol + dagesh
    "\u05B2\u05BC",  # hataf patah + dagesh
    "\u05B3\u05BC",  # hataf qamats + dagesh
    "\u05B4\u05BC",  # hiriq + dagesh
    "\u05B5\u05BC",  # tsere + dagesh
    "\u05B6\u05BC",  # segol + dagesh
    "\u05B7\u05BC",  # patah + dagesh
    "\u05B8\u05BC",  # qamats + dagesh
    "\u05B9\u05BC",  # holam + dagesh
    "\u05BA\u05BC",  # holam haser + dagesh
    "\u05BB\u05BC",  # qubuts + dagesh
    "\u05C7",        # qamats qatan
    "\u05BC\u05C7",  # dagesh + qamats qatan
]

SHIN_CLASSES = [
    "\u05C1",  # shin dot
    "\u05C2",  # sin dot
]

NIKUD_TO_ID = {n: i for i, n in enumerate(NIKUD_CLASSES)}
SHIN_TO_ID = {s: i for i, s in enumerate(SHIN_CLASSES)}

NUM_NIKUD_CLASSES = len(NIKUD_CLASSES)
NUM_SHIN_CLASSES = len(SHIN_CLASSES)

_NIKUD_PATTERN = re.compile(r"[\u05B0-\u05BD\u05C1\u05C2\u05C7]")

ALEF_ORD = ord("\u05D0")  # א
TAF_ORD = ord("\u05EA")   # ת
SHIN_LETTER = "\u05E9"    # ש


def is_hebrew_letter(char: str) -> bool:
    return ALEF_ORD <= ord(char) <= TAF_ORD


def sort_diacritics(text: str) -> str:
    def _cb(match):
        return match.group(1) + "".join(sorted(match.group(2)))
    return regex.sub(r"(\p{L})(\p{M}+)", _cb, text)


def remove_nikud(text: str) -> str:
    return _NIKUD_PATTERN.sub("", text)


def extract_labels(letter: str, diacritics: str) -> tuple[int, int]:
    """Given a Hebrew letter and its following diacritic string, return (nikud_id, shin_id).

    shin_id is 0 (shin dot) by default; only meaningful when letter == ש.
    """
    shin_id = 0
    shin = ""
    remaining = diacritics

    # Extract shin/sin dot first (U+05C1 / U+05C2)
    if "\u05C1" in remaining:
        shin = "\u05C1"
        shin_id = SHIN_TO_ID["\u05C1"]
        remaining = remaining.replace("\u05C1", "")
    elif "\u05C2" in remaining:
        shin = "\u05C2"
        shin_id = SHIN_TO_ID["\u05C2"]
        remaining = remaining.replace("\u05C2", "")

    nikud_id = NIKUD_TO_ID.get(remaining, 0)
    return nikud_id, shin_id
