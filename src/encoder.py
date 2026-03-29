"""ModernBERT encoder for Hebrew G2P, initialized from scratch.

Architecture follows ModernBERT-base (answerdotai/ModernBERT-base):
  - RoPE positional embeddings
  - Pre-LN with RMSNorm
  - Flash Attention
  - 8192 token context

Vocab size matches the tokenizer built in tokenization.py.
"""

from __future__ import annotations

import torch
from transformers import ModernBertConfig, ModernBertModel

from tokenization import build_vocab


def _vocab_size() -> int:
    return len(build_vocab())


MODERNBERT_CONFIG = ModernBertConfig(
    vocab_size=_vocab_size(),
    hidden_size=768,
    num_hidden_layers=22,
    num_attention_heads=12,
    intermediate_size=1152,
    hidden_activation="gelu",
    mlp_dropout=0.0,
    attention_dropout=0.0,
    max_position_embeddings=8192,
    initializer_range=0.02,
    norm_eps=1e-5,
    pad_token_id=0,
)


def build_encoder(flash_attention: bool = False) -> ModernBertModel:
    config = MODERNBERT_CONFIG
    if flash_attention:
        config = ModernBertConfig(**MODERNBERT_CONFIG.to_dict())
        config._attn_implementation = "flash_attention_2"
        config.dtype = torch.bfloat16
    return ModernBertModel(config)
