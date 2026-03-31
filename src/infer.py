"""Inference for CTC-based G2P model.

Each input_char owns exactly K=3 slots in the encoder output.
Greedy decode per char: read slots left→right, skip blanks and consecutive repeats.
Non-input_chars pass through unchanged.
One stress per word is enforced: highest-confidence stressed vowel wins.

Usage:
    uv run src/infer.py --checkpoint outputs/g2p/checkpoint-5000 --text "שלום עולם"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from constants import MAX_LEN
from lang_pack import LangPack
from languages import get_lang_pack
from model import G2PModel, UPSAMPLE_FACTOR
from tokenization import load_tokenizer

UNSTRESS = {"ˈa": "a", "ˈe": "e", "ˈi": "i", "ˈo": "o", "ˈu": "u"}
STRESSED = set(UNSTRESS.keys())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--lang", type=str, default="hebrew")
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    return parser.parse_args()


def load_checkpoint(model: G2PModel, checkpoint_dir: str) -> None:
    base = Path(checkpoint_dir)
    safetensors_path = base / "model.safetensors"
    bin_path = base / "pytorch_model.bin"
    if safetensors_path.exists():
        from safetensors.torch import load_file
        state = load_file(str(safetensors_path), device="cpu")
    elif bin_path.exists():
        state = torch.load(bin_path, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    model.load_state_dict(state, strict=False)


def _enforce_one_stress(word_chunks: list[tuple[list[str], float]]) -> list[str]:
    """Given a list of (tokens, max_stress_logit) per char in a word,
    keep only the highest-confidence stressed vowel and downgrade the rest.

    word_chunks: list of (tokens, stress_logit) per char
    Returns: flat list of tokens with at most one stressed vowel
    """
    # Find all stressed positions and their scores
    stressed = [
        (i, score)
        for i, (tokens, score) in enumerate(word_chunks)
        if any(t in STRESSED for t in tokens)
    ]

    if len(stressed) <= 1:
        return [t for tokens, _ in word_chunks for t in tokens]

    # Keep the winner (highest logit), downgrade the rest
    winner_idx = max(stressed, key=lambda x: x[1])[0]

    result = []
    for i, (tokens, _) in enumerate(word_chunks):
        for t in tokens:
            if t in STRESSED and i != winner_idx:
                result.append(UNSTRESS[t])
            else:
                result.append(t)
    return result


def phonemize(text: str, model: G2PModel, tokenizer, lang_pack: LangPack, device: torch.device, max_len: int) -> str:
    K = UPSAMPLE_FACTOR
    id_to_token = lang_pack.id_to_token()
    token_to_id = {v: k for k, v in id_to_token.items()}
    blank_id = 0

    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offset_mapping = enc.pop("offset_mapping")[0].tolist()
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = out["logits"][0]      # [S*K, vocab_size]
    preds = logits.argmax(dim=-1)  # [S*K]

    # Build per-char chunks: (tokens, stress_logit)
    # Group chars into words separated by passthrough chars
    result = []
    prev_end = 0
    current_word: list[tuple[list[str], float]] = []

    def flush_word():
        if current_word:
            result.extend(_enforce_one_stress(current_word))
            current_word.clear()

    for tok_idx, (start, end) in enumerate(offset_mapping):
        if start > prev_end:
            flush_word()
            result.append(text[prev_end:start])

        if end - start != 1:
            if end > start:
                prev_end = end
            continue

        char = text[start:end]
        prev_end = end

        if char not in lang_pack:
            # Passthrough char — flush current word first
            flush_word()
            result.append(char)
            continue

        # Decode K slots for this char
        slot_start = tok_idx * K
        slot_preds = preds[slot_start:slot_start + K].tolist()

        tokens = []
        prev = blank_id
        for p in slot_preds:
            if p != blank_id and p != prev:
                tok = id_to_token.get(p - 1, "")
                if tok and tok != "∅":
                    tokens.append(tok)
            prev = p

        # Stress logit: max logit of any stressed vowel emitted in this char's slots
        stress_logit = max(
            (logits[slot_start + s, token_to_id[t] + 1].item()
             for s, p in enumerate(slot_preds)
             for t in [id_to_token.get(p - 1, "")]
             if t in STRESSED),
            default=float("-inf"),
        )

        current_word.append((tokens, stress_logit))

    flush_word()

    if prev_end < len(text):
        result.append(text[prev_end:])

    return "".join(result)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lang_pack = get_lang_pack(args.lang)
    tokenizer = load_tokenizer(lang_pack=lang_pack)
    model = G2PModel(lang_pack=lang_pack)
    load_checkpoint(model, args.checkpoint)
    model.to(device).eval()
    print(phonemize(args.text, model, tokenizer, lang_pack, device, args.max_len))


if __name__ == "__main__":
    main()
