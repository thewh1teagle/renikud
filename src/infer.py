"""Run inference with the Hebrew G2P classifier model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from constants import (
    ID_TO_CONSONANT,
    ID_TO_VOWEL,
    STRESS_YES,
    VOWEL_TO_ID,
    MAX_LEN,
    is_hebrew_letter,
)

# Index for ∅ (no vowel) — cannot place primary stress on these tokens
VOWEL_EMPTY_ID = VOWEL_TO_ID["∅"]

# ג׳ / ז׳ / צ׳ — geresh signals affricate/fricative; IPA is on the letter, not a literal apostrophe
GIMEL_ZAYIN_TSADI = frozenset({"ג", "ז", "צ"})


def _is_hebrew_geresh(ch: str) -> bool:
    """U+05F3 ׳, ASCII ', or typographic ' (U+2019) used as Hebrew geresh."""
    return len(ch) == 1 and ch in ("\u05f3", "'", "\u2019")


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


def _best_stress_per_word(
    offset_mapping: list[tuple[int, int]],
    text: str,
    stress_logits: torch.Tensor,
    vowel_preds: torch.Tensor,
) -> set[int]:
    """
    Ensures each word has exactly one primary stress when it has at least one vowel token.

    Prefer tokens where STRESS_YES > STRESS_NO; if the model is unconfident on all vowels,
    fall back to the vowel with the largest (YES - NO) margin so Hebrew words never lack stress.
    """
    import re

    STRESS_NO = 0 if STRESS_YES == 1 else 1

    word_spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    words: dict[int, list[int]] = {i: [] for i in range(len(word_spans))}

    for tok_idx, (start, end) in enumerate(offset_mapping):
        if end - start != 1:
            continue
        for word_idx, (ws, we) in enumerate(word_spans):
            if ws <= start < we:
                words[word_idx].append(tok_idx)
                break

    stressed: set[int] = set()
    for toks in words.values():
        if not toks:
            continue

        vowel_toks = [t for t in toks if int(vowel_preds[t].item()) != VOWEL_EMPTY_ID]

        if not vowel_toks:
            continue

        confident_toks = [
            t for t in vowel_toks if stress_logits[t, STRESS_YES] > stress_logits[t, STRESS_NO]
        ]

        if confident_toks:
            best_tok = max(
                confident_toks,
                key=lambda t: (stress_logits[t, STRESS_YES] - stress_logits[t, STRESS_NO]).item(),
            )
            stressed.add(best_tok)
        else:
            best_tok = max(
                vowel_toks,
                key=lambda t: (stress_logits[t, STRESS_YES] - stress_logits[t, STRESS_NO]).item(),
            )
            stressed.add(best_tok)

    return stressed


def _decode(
    text: str,
    offset_mapping: list[tuple[int, int]],
    consonant_logits: torch.Tensor,
    vowel_logits: torch.Tensor,
    stress_logits: torch.Tensor,
) -> str:
    """Decode per-token logits into an IPA string."""
    consonant_preds = consonant_logits.argmax(dim=-1)
    vowel_preds = vowel_logits.argmax(dim=-1)
    stressed_positions = _best_stress_per_word(offset_mapping, text, stress_logits, vowel_preds)

    result: list[str] = []
    prev_char_end = 0
    last_src_char: str | None = None

    def emit_raw_gap(gap: str) -> None:
        nonlocal last_src_char
        for c in gap:
            if _is_hebrew_geresh(c) and last_src_char in GIMEL_ZAYIN_TSADI:
                continue
            result.append(c)
            last_src_char = c

    for tok_idx, (start, end) in enumerate(offset_mapping):
        if start > prev_char_end:
            emit_raw_gap(text[prev_char_end:start])

        if end - start != 1:
            if end > start:
                prev_char_end = end
            continue

        char = text[start:end]
        prev_char_end = end

        if not is_hebrew_letter(char):
            if _is_hebrew_geresh(char) and last_src_char in GIMEL_ZAYIN_TSADI:
                continue
            result.append(char)
            last_src_char = char
            continue

        consonant = ID_TO_CONSONANT.get(int(consonant_preds[tok_idx]), "∅")
        vowel = ID_TO_VOWEL.get(int(vowel_preds[tok_idx]), "∅")
        stress = tok_idx in stressed_positions

        # The model's forward pass already masked out impossible consonants via -1e9.
        # No fallback logic needed here!

        chunk = ""
        if consonant != "∅":
            chunk += consonant
        if stress:
            chunk += "ˈ"
        if vowel != "∅":
            chunk += vowel

        result.append(chunk)
        last_src_char = char

    if prev_char_end < len(text):
        emit_raw_gap(text[prev_char_end:])

    return "".join(result)


def phonemize(
    text: str, 
    model: HebrewG2PClassifier, 
    tokenizer, 
    vocab_cache: dict[int, str], 
    device: torch.device, 
    max_len: int
) -> str:
    """Convert unvocalized Hebrew text to IPA using the classifier model."""
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offset_mapping = encoding.pop("offset_mapping")[0].tolist()  
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Use hardware acceleration if available
    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=device.type=="cuda"):
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer_vocab=vocab_cache, # Passed statically instead of rebuilding
        )

    return _decode(
        text=text,
        offset_mapping=offset_mapping,
        consonant_logits=out["consonant_logits"][0].float(), # Cast back to fp32 for argmax/math
        vowel_logits=out["vowel_logits"][0].float(),
        stress_logits=out["stress_logits"][0].float(),
    )


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_encoder_tokenizer()
    vocab_cache = build_tokenizer_vocab(tokenizer)
    
    model = HebrewG2PClassifier()
    load_checkpoint(model, args.checkpoint)
    model.to(device).eval()

    print(phonemize(args.text, model, tokenizer, vocab_cache, device, args.max_len))


if __name__ == "__main__":
    main()