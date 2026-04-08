"""Run inference with the Hebrew G2P classifier model.

Usage:
    uv run src/infer.py --checkpoint outputs/g2p-classifier/step-5000 --text "שלום עולם"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from constants import MAX_LEN, TOKENIZER_PATH
from decoder import build_tokenizer_vocab, decode
from model import G2PModel
from tokenization import load_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Infer IPA from Hebrew text using classifier model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    return parser.parse_args()


def load_checkpoint(model: G2PModel, checkpoint_dir: str) -> None:
    from safetensors.torch import load_file
    import torch
    base = Path(checkpoint_dir)
    safetensors_path = base / "model.safetensors"
    bin_path = base / "pytorch_model.bin"
    if safetensors_path.exists():
        state = load_file(str(safetensors_path), device="cpu")
    elif bin_path.exists():
        state = torch.load(bin_path, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"No checkpoint weights found in {checkpoint_dir}")
    model.load_state_dict(state, strict=False)


def phonemize(text: str, model: G2PModel, tokenizer, device: torch.device, max_len: int) -> str:
    """Convert unvocalized Hebrew text to IPA using the classifier model."""
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offset_mapping = encoding.pop("offset_mapping")[0].tolist()  # [S, 2]
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer_vocab=build_tokenizer_vocab(tokenizer),
        )

    return decode(
        text=text,
        offset_mapping=offset_mapping,
        consonant_logits=out["consonant_logits"][0],
        vowel_logits=out["vowel_logits"][0],
        stress_logits=out["stress_logits"][0],
    )


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer(TOKENIZER_PATH)
    model = G2PModel()
    load_checkpoint(model, args.checkpoint)
    model.to(device).eval()

    print(phonemize(args.text, model, tokenizer, device, args.max_len))


if __name__ == "__main__":
    main()
