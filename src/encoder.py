"""BERT-large encoder for Hebrew G2P.

Uses transformers BertModel with default bert-large-uncased architecture,
initialized from scratch (no pretrained weights).
Set ONNX_EXPORT=1 before importing to use ONNX-compatible ops.
"""

from __future__ import annotations

from transformers import BertConfig, BertModel

from tokenization import build_vocab


def _vocab_size() -> int:
    return len(build_vocab())


def build_encoder(flash_attention: bool = False) -> BertModel:
    config = BertConfig(
        vocab_size=_vocab_size(),
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        **{"attn_implementation": "flash_attention_2"} if flash_attention else {},
    )
    return BertModel(config)
