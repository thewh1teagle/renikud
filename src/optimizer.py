"""Optimizer configuration and LR schedule for Hebrew G2P training."""

from __future__ import annotations

import math

from model import G2PModel


def cosine_lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def parameter_groups(
    model: G2PModel,
    encoder_lr: float,
    head_lr: float,
    weight_decay: float,
) -> list[dict]:
    """Discriminative LRs: lower for encoder, higher for classification heads."""
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight"}

    def is_no_decay(name: str) -> bool:
        return any(term in name for term in no_decay)

    return [
        {
            "params": [p for n, p in model.encoder.named_parameters() if not is_no_decay(n)],
            "lr": encoder_lr,
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.encoder.named_parameters() if is_no_decay(n)],
            "lr": encoder_lr,
            "weight_decay": 0.0,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if not n.startswith("encoder.") and not is_no_decay(n)
            ],
            "lr": head_lr,
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if not n.startswith("encoder.") and is_no_decay(n)
            ],
            "lr": head_lr,
            "weight_decay": 0.0,
        },
    ]
