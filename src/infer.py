"""Inference for CTC-based G2P model.

Each input_char owns exactly K=3 slots in the encoder output.
Greedy decode per char: read slots left→right, skip blanks and consecutive repeats.
Non-input_chars pass through unchanged.

Usage:
    uv run src/infer.py --checkpoint outputs/g2p/checkpoint-5000 --text "שלום עולם"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from constants import MAX_LEN, TOKENIZER_PATH
from lang_pack import get_lang_pack, LangPack
from model import G2PModel, UPSAMPLE_FACTOR
from tokenization import load_tokenizer


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


def phonemize(text: str, model: G2PModel, tokenizer, lang_pack: LangPack, device: torch.device, max_len: int) -> str:
    K = UPSAMPLE_FACTOR
    id_to_token = lang_pack.id_to_token()
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
    active_mask = torch.tensor([
        [1 if (e - s == 1 and text[s] in lang_pack) else 0
         for s, e in offset_mapping]
    ], dtype=torch.bool, device=device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = out["logits"][0]        # [S*K, vocab_size]
    preds = logits.argmax(dim=-1)    # [S*K]

    result = []
    prev_end = 0

    for tok_idx, (start, end) in enumerate(offset_mapping):
        if start > prev_end:
            result.append(text[prev_end:start])

        if end - start != 1:
            if end > start:
                prev_end = end
            continue

        char = text[start:end]
        prev_end = end

        if char not in lang_pack:
            result.append(char)
            continue

        # Read K slots for this char
        slot_start = tok_idx * K
        slot_preds = preds[slot_start:slot_start + K].tolist()

        # Greedy CTC decode: skip blanks and consecutive repeats
        tokens = []
        prev = blank_id
        for p in slot_preds:
            if p != blank_id and p != prev:
                # token ids are 1-indexed: real_id = p - 1
                tok = id_to_token.get(p - 1, "")
                if tok and tok != "∅":
                    tokens.append(tok)
            prev = p

        result.extend(tokens)

    if prev_end < len(text):
        result.append(text[prev_end:])

    return "".join(result)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lang_pack = get_lang_pack(args.lang)
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    model = G2PModel(lang_pack=lang_pack)
    load_checkpoint(model, args.checkpoint)
    model.to(device).eval()
    print(phonemize(args.text, model, tokenizer, lang_pack, device, args.max_len))


if __name__ == "__main__":
    main()
