"""ModernBERT encoder for Hebrew G2P, initialized from scratch.

Same architecture as answerdotai/ModernBERT-base (22 layers, hidden=768,
intermediate=1152, 12 heads, RoPE, SwiGLU, Pre-RMSNorm) but with a
104-token Hebrew character vocabulary instead of the original 50368-token vocab.
"""

from __future__ import annotations

from transformers import ModernBertConfig
from transformers.models.modernbert.modeling_modernbert import ModernBertModel

from tokenization import build_vocab


def _vocab_size() -> int:
    return len(build_vocab())


def build_config() -> ModernBertConfig:
    return ModernBertConfig(vocab_size=_vocab_size(), pad_token_id=0)


def build_encoder(flash_attention: bool = False) -> ModernBertModel:
    return ModernBertModel(build_config())
