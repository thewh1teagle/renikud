"""Shared constants."""

from pathlib import Path
from typing import Final

TOKENIZER_PATH: Final[Path] = Path(__file__).parent / "tokenizer.json"
MAX_LEN: Final[int] = 256
IGNORE_INDEX: Final[int] = -100
