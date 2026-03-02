"""Model definition for Hebrew G2P with an upsampled CTC head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from constants import (
    CTC_BLANK_ID,
    DECODER_VOCAB,
    ENCODER_MODEL,
    PROJECTION_DIM,
    UPSAMPLE_FACTOR,
)
from tokenization import unwrap_encoder_model


class HebrewG2PCTC(nn.Module):
    """Minimal encoder + upsample + CTC head model."""

    def __init__(
        self,
        encoder_model: str = ENCODER_MODEL,
        projection_dim: int = PROJECTION_DIM,
        upsample_factor: int = UPSAMPLE_FACTOR,
        vocab_size: int = len(DECODER_VOCAB),
    ) -> None:
        super().__init__()
        if upsample_factor < 1:
            raise ValueError("upsample_factor must be >= 1")

        encoder = AutoModel.from_pretrained(
            encoder_model,
            trust_remote_code=True,
        )
        self.encoder = unwrap_encoder_model(encoder)
        self.upsample_factor = upsample_factor

        hidden_size = self.encoder.config.hidden_size
        self.projection = nn.Linear(hidden_size, projection_dim)
        self.slot_embedding = nn.Embedding(upsample_factor, projection_dim)
        self.classifier = nn.Linear(projection_dim, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden_states = self.projection(encoder_outputs.last_hidden_state)

        if self.upsample_factor > 1:
            hidden_states = hidden_states.repeat_interleave(self.upsample_factor, dim=1)
            expanded_mask = attention_mask.repeat_interleave(self.upsample_factor, dim=1)
            # Break symmetry: add slot position embedding [0,1,0,1,...] to each pair
            T = hidden_states.shape[1]
            slot_ids = torch.arange(T, device=hidden_states.device) % self.upsample_factor
            hidden_states = hidden_states + self.slot_embedding(slot_ids)
        else:
            expanded_mask = attention_mask

        logits = self.classifier(hidden_states)
        output = {"logits": logits, "input_lengths": expanded_mask.sum(dim=1).to(dtype=torch.long)}

        if labels is not None:
            output["loss"] = self._compute_ctc_loss(logits, expanded_mask, labels)

        return output

    def parameter_groups(self, encoder_lr: float, head_lr: float, weight_decay: float) -> list[dict]:
        """Return AdamW parameter groups with discriminative LRs and correct weight decay."""
        no_decay = {"bias", "LayerNorm.weight"}
        return [
            {
                "params": [p for n, p in self.encoder.named_parameters() if not any(nd in n for nd in no_decay)],
                "lr": encoder_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.encoder.named_parameters() if any(nd in n for nd in no_decay)],
                "lr": encoder_lr,
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in self.projection.named_parameters() if not any(nd in n for nd in no_decay)]
                        + [p for n, p in self.slot_embedding.named_parameters() if not any(nd in n for nd in no_decay)]
                        + [p for n, p in self.classifier.named_parameters() if not any(nd in n for nd in no_decay)],
                "lr": head_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.projection.named_parameters() if any(nd in n for nd in no_decay)]
                        + [p for n, p in self.slot_embedding.named_parameters() if any(nd in n for nd in no_decay)]
                        + [p for n, p in self.classifier.named_parameters() if any(nd in n for nd in no_decay)],
                "lr": head_lr,
                "weight_decay": 0.0,
            },
        ]

    def _compute_ctc_loss(
        self,
        logits: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
        input_lengths = attention_mask.sum(dim=1).to(dtype=torch.long)

        target_mask = labels.ne(-100)
        target_lengths = target_mask.sum(dim=1).to(dtype=torch.long)
        flat_targets = labels.masked_select(target_mask).to(dtype=torch.long)

        return F.ctc_loss(
            log_probs=log_probs,
            targets=flat_targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            blank=CTC_BLANK_ID,
            zero_infinity=True,
        )
