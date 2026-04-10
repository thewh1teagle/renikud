import csv
import re
from pathlib import Path
import pytest
from align import align_word

def normalize_graphemes(text: str) -> str:
    text = re.sub(r"[׳'`´]", "'", text)
    text = re.sub(r'[״""]', '"', text)
    return text

GOLDEN_DIR = Path(__file__).parent / "testdata"
SILENT_MARKER = "_"

def normalize_ipa(text: str) -> str:
    """
    Normalizes 'dirty' IPA characters to their standard versions 
    to match the cleaner alignment map.
    """
    replacements = {
        "g": "ɡ",  # Standard IPA g
        "x": "χ",  # Standard IPA chi
        "r": "ʁ",  # Standard IPA resh
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def read_golden_files() -> list[tuple[str, str, str]]:
    if not GOLDEN_DIR.exists():
        return []
        
    data = []
    for filepath in sorted(GOLDEN_DIR.glob("*.tsv")):
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                heb = normalize_graphemes(row["hebrew"].strip())
                aligned_ipa = row["aligned_ipa"].strip()
                if heb and aligned_ipa:
                    # Normalize both the full IPA and the chunks in the TSV
                    normalized_aligned = normalize_ipa(aligned_ipa)
                    full_ipa = normalized_aligned.replace("|", "").replace(SILENT_MARKER, "")
                    data.append((heb, full_ipa, normalized_aligned))
    return data

GOLDEN_DATA = read_golden_files()

@pytest.mark.parametrize("hebrew, full_ipa, expected_aligned_str", GOLDEN_DATA)
def test_strict_golden_alignment(hebrew, full_ipa, expected_aligned_str):
    expected_chunks = [c if c != SILENT_MARKER else "" for c in expected_aligned_str.split("|")]
    expected_alignment = list(zip(hebrew, expected_chunks))
    
    actual_alignment = align_word(hebrew, full_ipa)
    
    assert actual_alignment is not None, f"Failed to align '{hebrew}'"
    assert actual_alignment == expected_alignment, (
        f"\nMismatch for '{hebrew}':\n"
        f"Expected: {expected_alignment}\n"
        f"Actual:   {actual_alignment}"
    )