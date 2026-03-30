"""Evaluation for CTC-based G2P model."""

from __future__ import annotations

import torch


def evaluate(model, eval_loader, device, fp16: bool) -> dict:
    model.eval()
    total_loss = 0.0
    n = 0

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            active_mask = batch["active_mask"].to(device)
            target_ids = batch["target_ids"]

            with torch.autocast("cuda", enabled=fp16):
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    active_mask=active_mask,
                    target_ids=target_ids,
                )
            total_loss += out["loss"].item()
            n += 1

    model.train()
    return {
        "eval_loss": total_loss / max(n, 1),
        "mean_acc": 0.0,
    }
