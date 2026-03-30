"""NeoBERT encoder for Hebrew G2P, initialized from scratch.

Architecture follows chandar-lab/NeoBERT shrunk to ~113M params:
  - 16 layers, 768 hidden, 12 heads, 3072 FFN
  - SwiGLU activation
  - RoPE positional embeddings
  - Pre-RMSNorm
  - Full attention every layer (xformers)
  - 4096 token context

Vocab size matches the tokenizer built in tokenization.py.
Set NEOBERT_ONNX_EXPORT=1 before importing to use ONNX-compatible ops.
"""

from __future__ import annotations

from neobert.model import NeoBERT, NeoBERTConfig

from tokenization import build_vocab


def _vocab_size() -> int:
    return len(build_vocab())


def build_encoder(flash_attention: bool = False) -> NeoBERT:
    config = NeoBERTConfig(
        vocab_size=_vocab_size(),      # 104 instead of 30522 (character-level Hebrew vocab)
        num_hidden_layers=6,           # 28 in full NeoBERT; 6 gives ~19M with our tiny vocab
        hidden_size=512,               # reduced from 768
        intermediate_size=2048,        # 4x hidden
        num_attention_heads=8,         # reduced from 12
        max_length=4096,
    )
    return NeoBERT(config)