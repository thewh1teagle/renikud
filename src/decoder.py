"""Decode per-token nikud logits back into vocalized Hebrew text."""

from __future__ import annotations

import torch

from nikud import NIKUD_CLASSES, SHIN_CLASSES, SHIN_LETTER, MAT_LECT_TOKEN, is_hebrew_letter


def decode(
    text: str,
    offset_mapping: list[tuple[int, int]],
    nikud_logits: torch.Tensor,
    shin_logits: torch.Tensor,
) -> str:
    """Reinsert predicted nikud onto stripped Hebrew text."""
    nikud_preds = nikud_logits.argmax(dim=-1)   # [S]
    shin_preds = shin_logits.argmax(dim=-1)     # [S]

    result = []
    prev_end = 0

    for tok_idx, (start, end) in enumerate(offset_mapping):
        if start > prev_end:
            result.append(text[prev_end:start])

        if end - start != 1:
            if end > start:
                prev_end = end
            continue

        char = text[start:end]
        prev_end = end

        if not is_hebrew_letter(char):
            result.append(char)
            continue

        nikud = NIKUD_CLASSES[int(nikud_preds[tok_idx])]
        shin = SHIN_CLASSES[int(shin_preds[tok_idx])] if char == SHIN_LETTER else ""

        if nikud == MAT_LECT_TOKEN:
            result.append(char + shin + "\u05AF")
            continue

        result.append(char + "".join(sorted(shin + nikud)))

    if prev_end < len(text):
        result.append(text[prev_end:])

    return "".join(result)
