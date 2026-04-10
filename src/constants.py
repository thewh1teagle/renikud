"""Constants for the Hebrew G2P classifier model."""

from typing import Final

MAX_LEN: Final[int] = 256

# Label ignore index — used for non-Hebrew positions (CLS, SEP, spaces, punct)
IGNORE_INDEX: Final[int] = -100
