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
    TOKENIZER_PATH,
    is_hebrew_letter,
)
from model import HebrewG2PClassifier
from tokenization import load_tokenizer


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


def _best_stress_per_word(offset_mapping: list[tuple[int, int]], text: str, stress_logits: torch.Tensor) -> set[int]:
    """
    For each whitespace-delimited word, pick at most one token index to carry stress —
    the one with the highest stress logit score among those that predicted stress.
    Returns a set of token indices that are allowed to emit stress.
    """
    import re
    # Group single-char token indices by word span
    word_spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    words: dict[int, list[int]] = {i: [] for i in range(len(word_spans))}

    for tok_idx, (start, end) in enumerate(offset_mapping):
        if end - start != 1:
            continue
        for word_idx, (ws, we) in enumerate(word_spans):
            if ws <= start < we:
                words[word_idx].append(tok_idx)
                break

    # Per word: pick the token with the highest stress score (every word must have exactly one stress)
    stressed: set[int] = set()
    for toks in words.values():
        if toks:
            stressed.add(max(toks, key=lambda t: stress_logits[t, STRESS_YES].item()))
    return stressed


def _decode(
    text: str,
    offset_mapping: list[tuple[int, int]],
    consonant_logits: torch.Tensor,
    vowel_logits: torch.Tensor,
    stress_logits: torch.Tensor,
) -> str:
    """Decode per-token logits into an IPA string."""
    consonant_preds = consonant_logits.argmax(dim=-1)  # [S]
    vowel_preds = vowel_logits.argmax(dim=-1)           # [S]
    stressed_positions = _best_stress_per_word(offset_mapping, text, stress_logits)

    result = []
    prev_char_end = 0

    for tok_idx, (start, end) in enumerate(offset_mapping):
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
            # Skip geresh apostrophe after letters that use it as a digraph marker
            if char == "'" and start > 0 and text[start - 1] in "גזצץ":
                pass
            else:
                result.append(char)
            continue

        # Get predictions
        consonant = ID_TO_CONSONANT.get(int(consonant_preds[tok_idx]), "∅")
        vowel = ID_TO_VOWEL.get(int(vowel_preds[tok_idx]), "∅")
        stress = tok_idx in stressed_positions

        # Apply per-letter consonant constraint at inference
        allowed = HEBREW_LETTER_TO_ALLOWED_CONSONANTS.get(char, (CONSONANT_TO_ID["∅"],))
        if CONSONANT_TO_ID.get(consonant, 0) not in allowed:
            # Fall back to most probable allowed consonant
            for cid in sorted(allowed, key=lambda x: -consonant_logits[tok_idx, x].item()):
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


def phonemize(text: str, model: HebrewG2PClassifier, tokenizer, device: torch.device, max_len: int) -> str:
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

    return _decode(
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
    model = HebrewG2PClassifier()
    load_checkpoint(model, args.checkpoint)
    model.to(device).eval()

    print(phonemize(args.text, model, tokenizer, device, args.max_len))


if __name__ == "__main__":
    main()
