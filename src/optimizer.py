"""Optimizer configuration for Hebrew G2P training."""

from __future__ import annotations

from model import G2PModel


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
