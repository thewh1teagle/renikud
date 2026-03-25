"""Evaluation helpers for the Hebrew G2P classifier."""

from __future__ import annotations

import torch

from constants import IGNORE_INDEX


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Per-token accuracy ignoring IGNORE_INDEX positions."""
    mask = labels != IGNORE_INDEX
    if mask.sum() == 0:
        return 0.0
    preds = logits.argmax(dim=-1)
    return (preds[mask] == labels[mask]).float().mean().item()


def evaluate(model, eval_loader, device, fp16: bool) -> dict:
    model.eval()
    total_loss = 0.0
    consonant_acc_sum = vowel_acc_sum = stress_acc_sum = 0.0
    n = 0

    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", enabled=fp16):
                out = model(**batch)
            total_loss += out["loss"].item()
            consonant_acc_sum += compute_accuracy(out["consonant_logits"], batch["consonant_labels"])
            vowel_acc_sum += compute_accuracy(out["vowel_logits"], batch["vowel_labels"])
            stress_acc_sum += compute_accuracy(out["stress_logits"], batch["stress_labels"])
            n += 1

    model.train()
    return {
        "eval_loss": total_loss / n,
        "consonant_acc": consonant_acc_sum / n,
        "vowel_acc": vowel_acc_sum / n,
        "stress_acc": stress_acc_sum / n,
        "mean_acc": (consonant_acc_sum + vowel_acc_sum + stress_acc_sum) / (3 * n),
    }
