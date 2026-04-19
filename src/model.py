"""Hebrew diacritization model — per-character prediction of nikud and shin dot."""

from __future__ import annotations

import torch
import torch.nn as nn
from constants import IGNORE_INDEX
from encoder import build_encoder
from nikud import NUM_NIKUD_CLASSES, NUM_SHIN_CLASSES


class NikudModel(nn.Module):
    """
    Per-character Hebrew diacritization model.

    For each Hebrew letter in the input, predicts:
      - nikud class  (shva / patah / qamats / dagesh+vowel combos / ∅ / ...)
      - shin class   (shin dot / sin dot) — only meaningful for ש

    Non-Hebrew characters are passed through unchanged at inference.
    """

    def __init__(self, dropout_rate: float = 0.1, flash_attention: bool = False) -> None:
        super().__init__()

        self.encoder = build_encoder(flash_attention=flash_attention)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout_rate)

        self.nikud_head = nn.Linear(hidden_size, NUM_NIKUD_CLASSES)
        self.shin_head = nn.Linear(hidden_size, NUM_SHIN_CLASSES)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        nikud_labels: torch.Tensor | None = None,
        shin_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = self.dropout(encoder_outputs.last_hidden_state)  # [B, S, H]

        nikud_logits = self.nikud_head(hidden)   # [B, S, NUM_NIKUD_CLASSES]
        shin_logits = self.shin_head(hidden)     # [B, S, NUM_SHIN_CLASSES]

        output: dict[str, torch.Tensor] = {
            "nikud_logits": nikud_logits,
            "shin_logits": shin_logits,
        }

        if nikud_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            loss = (
                loss_fct(nikud_logits.view(-1, NUM_NIKUD_CLASSES), nikud_labels.view(-1))
                + loss_fct(shin_logits.view(-1, NUM_SHIN_CLASSES), shin_labels.view(-1))
            )
            output["loss"] = loss

        return output
