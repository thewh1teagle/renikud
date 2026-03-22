"""Hebrew G2P classifier model — per-character prediction of consonant, vowel, and stress.
Upgraded with NeoBERT/ModernBERT architecture (RoPE, SwiGLU, RMSNorm).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

# Import the NeoBERT architecture from your local file
from encoder.neobert import NeoBERT, NeoBERTConfig

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

        # 1. Load the original DictaBERT blueprint and pre-trained weights
        dicta_config = AutoConfig.from_pretrained(encoder_model, trust_remote_code=True)
        dicta_pretrained = AutoModel.from_pretrained(encoder_model, trust_remote_code=True)
        dicta_base = unwrap_encoder_model(dicta_pretrained)

        # 2. Map DictaBERT's exact dimensions to the NeoBERT config
        neo_config = NeoBERTConfig(
            vocab_size=dicta_config.vocab_size,
            pad_token_id=dicta_config.pad_token_id,
            hidden_size=dicta_config.hidden_size,                 
            num_hidden_layers=dicta_config.num_hidden_layers,     
            num_attention_heads=dicta_config.num_attention_heads, 
            intermediate_size=dicta_config.intermediate_size,     
            max_length=dicta_config.max_position_embeddings       
        )

        # 3. Initialize the modern architecture (NeoBERT)
        self.encoder = NeoBERT(neo_config)
        hidden_size = self.encoder.config.hidden_size

        # 4. TRANSPLANT: Copy DictaBERT's pre-trained character embeddings
        # NeoBERT's embedding layer is self.encoder.encoder
        # DictaBERT's embedding layer is dicta_base.embeddings.word_embeddings
        with torch.no_grad():
            self.encoder.encoder.weight.copy_(dicta_base.embeddings.word_embeddings.weight)
            
        # Free up memory (we don't need the old DictaBERT layers taking up VRAM)
        del dicta_pretrained
        del dicta_base

        self.dropout = nn.Dropout(dropout_rate)

        # Three independent classification heads
        self.consonant_head = nn.Linear(hidden_size, NUM_CONSONANT_CLASSES)
        self.vowel_head = nn.Linear(hidden_size, NUM_VOWEL_CLASSES)
        self.stress_head = nn.Linear(hidden_size, NUM_STRESS_CLASSES)

        # Precompute consonant mask: [vocab_size, NUM_CONSONANT_CLASSES]
        self._consonant_mask: torch.Tensor | None = None
        self._build_consonant_mask()

    def _build_consonant_mask(self) -> None:
        """
        Build a boolean mask [num_hebrew_letters, NUM_CONSONANT_CLASSES].
        True = this consonant class is forbidden for this letter.
        """
        from constants import ALEF_ORD, TAF_ORD
        n_letters = TAF_ORD - ALEF_ORD + 1
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
        Zero out forbidden consonant classes for each position based on the input character.
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
        attention_mask: torch.Tensor | None = None,
        consonant_labels: torch.Tensor | None = None,
        vowel_labels: torch.Tensor | None = None,
        stress_labels: torch.Tensor | None = None,
        tokenizer_vocab: dict[int, str] | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        
        # NeoBERT forward pass
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        
        # NeoBERT natively returns a BaseModelOutput object with last_hidden_state
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
        """Discriminative LRs updated to target NeoBERT's RMSNorm layers."""
        
        # NeoBERT uses RMSNorm (attention_norm, ffn_norm, layer_norm) instead of LayerNorm
        no_decay = {"bias", "attention_norm.weight", "ffn_norm.weight", "layer_norm.weight"}

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