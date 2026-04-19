"""Run inference with the Hebrew diacritization model.

Usage:
    uv run src/infer.py --checkpoint outputs/nikud/step-5000 --text "שלום עולם"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file

from constants import MAX_LEN
from decoder import decode
from model import NikudModel
from nikud import remove_nikud, sort_diacritics
from tokenization import load_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Add nikud to Hebrew text using the diacritization model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--input", type=str, default=None, help="Input text file (one sentence per line)")
    parser.add_argument("--output", type=str, default=None, help="Output text file")
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    return parser.parse_args()


def load_checkpoint(model: NikudModel, checkpoint_dir: str) -> None:
    state = load_file(str(Path(checkpoint_dir) / "model.safetensors"), device="cpu")
    model.load_state_dict(state)


def diacritize(text: str, model: NikudModel, tokenizer, device: torch.device, max_len: int) -> str:
    """Add nikud to unvocalized Hebrew text."""
    stripped = remove_nikud(sort_diacritics(text))

    encoding = tokenizer(
        stripped,
        truncation=True,
        max_length=max_len,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offset_mapping = encoding.pop("offset_mapping")[0].tolist()
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)

    return decode(
        text=stripped,
        offset_mapping=offset_mapping,
        nikud_logits=out["nikud_logits"][0],
        shin_logits=out["shin_logits"][0],
    )


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer()
    model = NikudModel()
    load_checkpoint(model, args.checkpoint)
    model.to(device).eval()

    if args.input:
        lines = Path(args.input).read_text(encoding="utf-8").splitlines()
        results = [diacritize(l, model, tokenizer, device, args.max_len) for l in lines if l.strip()]
        output = "\n".join(results)
        if args.output:
            Path(args.output).write_text(output + "\n", encoding="utf-8")
        else:
            print(output)
    else:
        result = diacritize(args.text, model, tokenizer, device, args.max_len)
        if args.output:
            Path(args.output).write_text(result + "\n", encoding="utf-8")
        else:
            print(result)


if __name__ == "__main__":
    main()
