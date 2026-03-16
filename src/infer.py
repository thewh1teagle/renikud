"""Run inference with the Hebrew G2P classifier model.

Usage:
    uv run src/infer.py --checkpoint outputs/g2p-classifier/checkpoint-5000 --text "שלום עולם"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from constants import (
    ID_TO_CONSONANT,
    ID_TO_VOWEL,
    HEBREW_LETTER_TO_ALLOWED_CONSONANTS,
    CONSONANT_TO_ID,
    STRESS_YES,
    MAX_LEN,
    is_hebrew_letter,
)
from model import HebrewG2PClassifier
from tokenization import load_encoder_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Infer IPA from Hebrew text using classifier model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    return parser.parse_args()


def load_checkpoint(model: HebrewG2PClassifier, checkpoint_dir: str) -> None:
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


def build_tokenizer_vocab(tokenizer) -> dict[int, str]:
    """Map token_id -> single character string for Hebrew letter lookup."""
    vocab = tokenizer.get_vocab()
    return {v: k for k, v in vocab.items()}


def phonemize(text: str, model: HebrewG2PClassifier, tokenizer, device: torch.device, max_len: int) -> str:
    """Convert unvocalized Hebrew text to IPA using the classifier model."""
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offset_mapping = encoding.pop("offset_mapping")[0]  # [S, 2]
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    tokenizer_vocab = build_tokenizer_vocab(tokenizer)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer_vocab=tokenizer_vocab,
        )

    consonant_preds = out["consonant_logits"][0].argmax(dim=-1)  # [S]
    vowel_preds = out["vowel_logits"][0].argmax(dim=-1)           # [S]
    stress_preds = out["stress_logits"][0].argmax(dim=-1)         # [S]

    result = []
    prev_char_end = 0

    for tok_idx, (start, end) in enumerate(offset_mapping.tolist()):
        # Pass through any characters skipped by the tokenizer
        if start > prev_char_end:
            result.append(text[prev_char_end:start])

        if end - start != 1:
            # CLS, SEP, or multi-char token — skip
            if end > start:
                prev_char_end = end
            continue

        char = text[start:end]
        prev_char_end = end

        if not is_hebrew_letter(char):
            # Non-Hebrew: pass through as-is
            result.append(char)
            continue

        # Get predictions
        consonant = ID_TO_CONSONANT.get(int(consonant_preds[tok_idx]), "∅")
        vowel = ID_TO_VOWEL.get(int(vowel_preds[tok_idx]), "∅")
        stress = int(stress_preds[tok_idx]) == STRESS_YES

        # Apply per-letter consonant constraint at inference
        allowed = HEBREW_LETTER_TO_ALLOWED_CONSONANTS.get(char, (CONSONANT_TO_ID["∅"],))
        consonant_id = CONSONANT_TO_ID.get(consonant, 0)
        if consonant_id not in allowed:
            # Fall back to most probable allowed consonant
            logits = out["consonant_logits"][0, tok_idx]
            for cid in sorted(allowed, key=lambda x: -logits[x].item()):
                consonant = ID_TO_CONSONANT[cid]
                break

        # Assemble IPA chunk in benchmark-compatible order: [consonant][ˈ][vowel]
        # (dataset/GT encodes stress before the vowel, e.g. "ʁˈa")
        chunk = ""
        if consonant != "∅":
            chunk += consonant
        if stress:
            chunk += "ˈ"
        if vowel != "∅":
            chunk += vowel

        result.append(chunk)

    # Append any remaining characters
    if prev_char_end < len(text):
        result.append(text[prev_char_end:])

    return "".join(result)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_encoder_tokenizer()
    model = HebrewG2PClassifier()
    load_checkpoint(model, args.checkpoint)
    model.to(device).eval()

    print(phonemize(args.text, model, tokenizer, device, args.max_len))


if __name__ == "__main__":
    main()
