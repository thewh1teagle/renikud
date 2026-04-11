"""Tiny BERT encoder for Hebrew G2P — ESP32-friendly (~238K params).

Uses transformers.BertModel with a minimal config:
  vocab_size=104 (character-level Hebrew vocab)
  hidden_size=96, num_hidden_layers=3, num_attention_heads=4
  intermediate_size=192 → ~238K total params, ~232 KB int8

This replaces the custom NeoBERT encoder while keeping the same
.config.hidden_size interface expected by G2PModel.
"""

from __future__ import annotations

from transformers import BertConfig, BertModel

from tokenization import build_vocab


def _vocab_size() -> int:
    return len(build_vocab())


def build_encoder(flash_attention: bool = False) -> BertModel:
    config = BertConfig(
        vocab_size=_vocab_size(),       # 104 character-level Hebrew tokens
        hidden_size=96,
        num_hidden_layers=3,
        num_attention_heads=4,          # head_dim = 24
        intermediate_size=192,          # 2× hidden (tight for ESP32)
        max_position_embeddings=512,
        type_vocab_size=1,              # no segment embeddings needed
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        pad_token_id=0,
    )
    return BertModel(config, add_pooling_layer=False)
