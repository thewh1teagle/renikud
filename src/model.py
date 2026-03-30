"""G2P model with structured upsample + CTC loss.

Architecture:
  - Upsample: each input char → K slots via learned conv (K=3)
  - Transformer encoder over all slots (full sentence attention)
  - Linear head → IPA vocab + blank per slot
  - CTC loss vs raw IPA token sequence (no alignment needed)

At inference:
  - Slots [i*K : i*K+K] always belong to char i (structural, not learned)
  - Greedy decode per char: skip blanks and consecutive repeats
  - Passthrough for non-input_chars
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import build_encoder
from lang_pack import LangPack

UPSAMPLE_FACTOR = 3  # slots per input char


class G2PModel(nn.Module):
    def __init__(self, lang_pack: LangPack, dropout_rate: float = 0.1, flash_attention: bool = False):
        super().__init__()
        self.lang_pack = lang_pack
        self.K = UPSAMPLE_FACTOR

        self.encoder = build_encoder(flash_attention=flash_attention)
        H = self.encoder.config.hidden_size

        # IPA vocab + blank (blank = index 0 by convention for CTCLoss)
        self.vocab_size = len(lang_pack.output_tokens) + 1  # +1 for blank
        self.blank_id = 0
        # shift all real token ids by 1: null/∅ → 1, b → 2, etc.

        # Upsample: embed each char position into K slots
        # Simple: learned linear that produces K*H from H, then reshape
        self.upsample = nn.Linear(H, self.K * H)

        # Slot positional bias: distinguish slot 0,1,2 within a char
        self.slot_pos_emb = nn.Embedding(self.K, H)

        self.dropout = nn.Dropout(dropout_rate)
        self.head = nn.Linear(H, self.vocab_size)

        self.ctc_loss = nn.CTCLoss(blank=self.blank_id, reduction="mean", zero_infinity=True)

    def _upsample(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        hidden: [B, L, H]
        returns: [B, L*K, H]  — K slots per input position
        """
        B, L, H = hidden.shape
        # Project to K*H then reshape to [B, L, K, H]
        expanded = self.upsample(hidden).view(B, L, self.K, H)  # [B, L, K, H]
        # Add slot positional embeddings
        slot_idx = torch.arange(self.K, device=hidden.device)
        expanded = expanded + self.slot_pos_emb(slot_idx)       # broadcast over B, L
        # Reshape to [B, L*K, H] — slots are interleaved: char0_slot0, char0_slot1, char0_slot2, char1_slot0...
        return expanded.reshape(B, L * self.K, H)

    def forward(
        self,
        input_ids: torch.Tensor,          # [B, S]
        attention_mask: torch.Tensor,     # [B, S]
        active_mask: torch.Tensor | None = None,      # [B, S] bool — which positions are input_chars
        target_ids: list[torch.Tensor] | None = None, # B x [T_i] — IPA token ids (1-indexed, 0=blank)
    ) -> dict[str, torch.Tensor]:

        # Encode at char level first
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = enc_out.last_hidden_state   # [B, S, H]

        # Upsample to slot level
        slotted = self._upsample(hidden)     # [B, S*K, H]
        slotted = self.dropout(slotted)

        # Project to vocab logits
        logits = self.head(slotted)          # [B, S*K, vocab_size]

        output: dict[str, torch.Tensor] = {"logits": logits}

        if active_mask is not None and target_ids is not None:
            B, S = input_ids.shape
            loss_total = torch.tensor(0.0, device=hidden.device)
            valid = 0

            for b in range(B):
                # Get active char positions for this sample
                active_pos = active_mask[b].nonzero(as_tuple=True)[0]  # [L_active]
                if active_pos.numel() == 0 or target_ids[b].numel() == 0:
                    continue

                # Gather the slots corresponding to active chars only
                # active_pos[i] → slots [active_pos[i]*K : active_pos[i]*K + K]
                slot_indices = (active_pos.unsqueeze(1) * self.K +
                                torch.arange(self.K, device=hidden.device)).reshape(-1)  # [L_active * K]

                active_logits = logits[b, slot_indices, :]   # [L_active*K, vocab_size]
                log_probs = F.log_softmax(active_logits, dim=-1)  # [L_active*K, vocab_size]

                T_in = log_probs.shape[0]
                T_out = target_ids[b].shape[0]

                if T_in < T_out:
                    continue  # shouldn't happen with K=3 but be safe

                # CTCLoss expects [T, B, C] input
                loss_b = self.ctc_loss(
                    log_probs.unsqueeze(1),              # [T_in, 1, vocab_size]
                    target_ids[b].unsqueeze(0),          # [1, T_out]
                    torch.tensor([T_in], device=hidden.device),
                    torch.tensor([T_out], device=hidden.device),
                )
                loss_total = loss_total + loss_b
                valid += 1

            output["loss"] = loss_total / max(valid, 1)

        return output
