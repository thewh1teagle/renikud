"""Hebrew G2P classifier model — per-character prediction of consonant, vowel, and stress."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

from constants import (
    ENCODER_MODEL,
    NUM_CONSONANT_CLASSES,
    NUM_VOWEL_CLASSES,
    NUM_STRESS_CLASSES,
    HEBREW_LETTER_TO_ALLOWED_CONSONANTS,
    IGNORE_INDEX,
    is_hebrew_letter,
)
from tokenization import unwrap_encoder_model


class HebrewG2PClassifier(nn.Module):
    """
    Per-character Hebrew G2P model.

    For each Hebrew letter in the input, predicts:
      - consonant class (from a per-letter constrained set)
      - vowel class     (a / e / i / o / u / ∅)
      - stress          (yes / no)

    Non-Hebrew characters (spaces, punctuation, digits, Latin) are passed
    through unchanged at inference — the heads are never called for them.
    """

    def __init__(self, encoder_model: str = ENCODER_MODEL, dropout_rate: float = 0.1) -> None:
        super().__init__()

        encoder = AutoModel.from_pretrained(encoder_model, trust_remote_code=True)
        self.encoder = unwrap_encoder_model(encoder)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout_rate)

        # Three independent classification heads
        self.consonant_head = nn.Linear(hidden_size, NUM_CONSONANT_CLASSES)
        self.vowel_head = nn.Linear(hidden_size, NUM_VOWEL_CLASSES)
        self.stress_head = nn.Linear(hidden_size, NUM_STRESS_CLASSES)

        # Precompute consonant mask: [vocab_size, NUM_CONSONANT_CLASSES]
        # mask[i, j] = True means consonant class j is FORBIDDEN for Hebrew letter i
        # Built once at init, moved to device on first forward pass
        self._consonant_mask: torch.Tensor | None = None
        self._build_consonant_mask()

    def _build_consonant_mask(self) -> None:
        """
        Build a boolean mask [num_hebrew_letters, NUM_CONSONANT_CLASSES].
        True = this consonant class is forbidden for this letter.
        Hebrew letters are indexed by (ord(char) - ord('א')).
        """
        from constants import ALEF_ORD, TAF_ORD
        n_letters = TAF_ORD - ALEF_ORD + 1
        # Start with all forbidden, then allow the valid ones
        mask = torch.ones(n_letters, NUM_CONSONANT_CLASSES, dtype=torch.bool)
        for char, allowed_ids in HEBREW_LETTER_TO_ALLOWED_CONSONANTS.items():
            idx = ord(char) - ALEF_ORD
            for cid in allowed_ids:
                mask[idx, cid] = False
        self._consonant_mask = mask

    def _apply_consonant_mask(
        self,
        consonant_logits: torch.Tensor,
        input_ids: torch.Tensor,
        tokenizer_vocab: dict[int, str],
    ) -> torch.Tensor:
        """
        Zero out forbidden consonant classes for each position based on the
        input Hebrew character at that position.

        consonant_logits: [B, S, NUM_CONSONANT_CLASSES]
        input_ids:        [B, S]
        tokenizer_vocab:  maps token_id -> character string
        """
        from constants import ALEF_ORD
        mask = self._consonant_mask.to(consonant_logits.device)
        B, S, _ = consonant_logits.shape
        masked = consonant_logits.clone()

        for b in range(B):
            for s in range(S):
                token_id = input_ids[b, s].item()
                char = tokenizer_vocab.get(token_id, "")
                if len(char) == 1 and is_hebrew_letter(char):
                    letter_idx = ord(char) - ALEF_ORD
                    # Set forbidden logits to -inf
                    masked[b, s][mask[letter_idx]] = -1e9

        return masked

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        consonant_labels: torch.Tensor | None = None,
        vowel_labels: torch.Tensor | None = None,
        stress_labels: torch.Tensor | None = None,
        tokenizer_vocab: dict[int, str] | None = None,
    ) -> dict[str, torch.Tensor]:
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = self.dropout(encoder_outputs.last_hidden_state)  # [B, S, H]

        consonant_logits = self.consonant_head(hidden)  # [B, S, NUM_CONSONANT_CLASSES]
        vowel_logits = self.vowel_head(hidden)           # [B, S, NUM_VOWEL_CLASSES]
        stress_logits = self.stress_head(hidden)         # [B, S, NUM_STRESS_CLASSES]

        # Apply per-letter consonant mask if vocab provided
        if tokenizer_vocab is not None:
            consonant_logits = self._apply_consonant_mask(consonant_logits, input_ids, tokenizer_vocab)

        output: dict[str, torch.Tensor] = {
            "consonant_logits": consonant_logits,
            "vowel_logits": vowel_logits,
            "stress_logits": stress_logits,
        }

        if consonant_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            loss = (
                loss_fct(consonant_logits.view(-1, NUM_CONSONANT_CLASSES), consonant_labels.view(-1))
                + loss_fct(vowel_logits.view(-1, NUM_VOWEL_CLASSES), vowel_labels.view(-1))
                + loss_fct(stress_logits.view(-1, NUM_STRESS_CLASSES), stress_labels.view(-1))
            )
            output["loss"] = loss

        return output

    def parameter_groups(self, encoder_lr: float, head_lr: float, weight_decay: float) -> list[dict]:
        """Discriminative LRs: lower for encoder, higher for classification heads."""
        no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}

        def is_no_decay(name: str) -> bool:
            return any(term in name for term in no_decay)

        return [
            {
                "params": [p for n, p in self.encoder.named_parameters() if not is_no_decay(n)],
                "lr": encoder_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.encoder.named_parameters() if is_no_decay(n)],
                "lr": encoder_lr,
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if not n.startswith("encoder.") and not is_no_decay(n)
                ],
                "lr": head_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if not n.startswith("encoder.") and is_no_decay(n)
                ],
                "lr": head_lr,
                "weight_decay": 0.0,
            },
        ]
