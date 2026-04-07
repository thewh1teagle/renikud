"""NeoBERT encoder loaded from HuggingFace."""

from __future__ import annotations

from transformers import AutoModel


HF_MODEL = "thewh1teagle/bert-char-he"
TRUST_REMOTE_CODE = True


def build_encoder(flash_attention: bool = False):
    return AutoModel.from_pretrained(HF_MODEL, trust_remote_code=TRUST_REMOTE_CODE)
