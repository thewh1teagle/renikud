"""Evaluation helpers for the Hebrew diacritization model."""

from __future__ import annotations

import torch
import jiwer
import regex

from constants import IGNORE_INDEX
from decoder import decode
from nikud import remove_nikud, sort_diacritics

_MAT_LECT_RE = regex.compile(r"\p{L}\u05AF")


def normalize_nikud(text: str, keep_matres: bool = False) -> str:
    text = sort_diacritics(text)
    if not keep_matres:
        text = _MAT_LECT_RE.sub("", text)
    return text


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    mask = labels != IGNORE_INDEX
    if mask.sum() == 0:
        return 0.0
    preds = logits.argmax(dim=-1)
    return (preds[mask] == labels[mask]).float().mean().item()


def decode_batch(texts: list[str], out: dict, tokenizer) -> list[str]:
    preds = []
    for i, text in enumerate(texts):
        stripped = remove_nikud(sort_diacritics(text))
        encoding = tokenizer(stripped, truncation=True, max_length=512, return_offsets_mapping=True)
        preds.append(decode(
            text=stripped,
            offset_mapping=encoding["offset_mapping"],
            nikud_logits=out["nikud_logits"][i],
            shin_logits=out["shin_logits"][i],
        ))
    return preds


def evaluate(model, eval_loader, device, fp16: bool, tokenizer) -> dict:
    model.eval()
    total_loss = 0.0
    nikud_acc_sum = shin_acc_sum = 0.0
    n = 0
    refs, hyps = [], []

    with torch.no_grad():
        for batch in eval_loader:
            texts = batch.pop("texts")
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.autocast("cuda", enabled=fp16):
                out = model(**batch)

            total_loss += out["loss"].item()
            nikud_acc_sum += compute_accuracy(out["nikud_logits"], batch["nikud_labels"])
            shin_acc_sum += compute_accuracy(out["shin_logits"], batch["shin_labels"])
            n += 1

            refs.extend([normalize_nikud(t) for t in texts])
            hyps.extend([normalize_nikud(t) for t in decode_batch(texts, out, tokenizer)])

    model.train()
    return {
        "eval_loss": total_loss / n,
        "nikud_acc": nikud_acc_sum / n,
        "shin_acc": shin_acc_sum / n,
        "mean_acc": (nikud_acc_sum + shin_acc_sum) / (2 * n),
        "cer": jiwer.cer(refs, hyps),
        "wer": jiwer.wer(refs, hyps),
    }
