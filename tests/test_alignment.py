import csv
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_align import align_word

DATA_DIR = Path(__file__).parent / "data"


HEBREW_RANGE = ("\u05d0", "\u05ea")
SILENT = "∅"
IPA_NORMALIZE = {"sh": "ʃ", "x": "χ", "g": "ɡ", "r": "ʁ"}


def hebrew_letters(text: str) -> int:
    return sum(1 for c in text if HEBREW_RANGE[0] <= c <= HEBREW_RANGE[1])


def normalize_ipa(s: str) -> str:
    for ascii_form, ipa in IPA_NORMALIZE.items():
        s = s.replace(ascii_form, ipa)
    return s


def load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def csv_params(filename: str) -> list:
    return [
        pytest.param(row["hebrew"], row["alignment"], id=f"{filename}:{row['hebrew']}")
        for row in load_csv(DATA_DIR / filename)
    ]


ALL_ROWS = csv_params("basic.csv") + csv_params("advanced.csv")


INVALID_ROWS = [
    pytest.param(row["hebrew"], row["ipa"], id=f"invalid:{row['hebrew']}")
    for row in load_csv(DATA_DIR / "invalid.csv")
]


@pytest.mark.parametrize("heb,ipa", INVALID_ROWS)
def test_align_word_returns_none_for_invalid(heb, ipa):
    heb_core = "".join(c for c in heb if HEBREW_RANGE[0] <= c <= HEBREW_RANGE[1])
    assert align_word(heb_core, ipa) is None


@pytest.mark.parametrize("heb,alignment", ALL_ROWS)
def test_letter_count_equals_span_count(heb, alignment):
    spans = alignment.split("|")
    assert hebrew_letters(heb) == len(spans), (
        f"{heb!r}: {hebrew_letters(heb)} letters but {len(spans)} spans"
    )


@pytest.mark.parametrize("heb,alignment", ALL_ROWS)
def test_align_word_matches_csv(heb, alignment):
    heb_core = "".join(c for c in heb if HEBREW_RANGE[0] <= c <= HEBREW_RANGE[1])
    spans = [normalize_ipa(s) for s in alignment.split("|")]
    ipa = "".join(s.replace(SILENT, "") for s in spans)
    expected = [s.replace(SILENT, "") for s in spans]

    result = align_word(heb_core, ipa)
    assert result is not None, f"align_word returned None for IPA {ipa!r}"
    assert [chunk for _, chunk in result] == expected
