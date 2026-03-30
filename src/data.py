"""Dataset loading and collation for CTC-based G2P training."""

from __future__ import annotations

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader


class G2PDataCollator:
    pad_id: int = 0

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids, attention_mask, active_mask = [], [], []
        target_ids = []

        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(list(f["input_ids"]) + [self.pad_id] * pad)
            attention_mask.append(list(f["attention_mask"]) + [0] * pad)
            active_mask.append(list(f["active_mask"]) + [0] * pad)
            target_ids.append(torch.tensor(f["target_ids"], dtype=torch.long))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "active_mask": torch.tensor(active_mask, dtype=torch.bool),
            "target_ids": target_ids,  # list of variable-length tensors
        }


def make_dataloaders(args) -> tuple[DataLoader, DataLoader]:
    collator = G2PDataCollator()
    train_loader = DataLoader(
        load_from_disk(args.train_dataset),
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.dataloader_workers,
    )
    eval_loader = DataLoader(
        load_from_disk(args.eval_dataset),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.dataloader_workers,
    )
    return train_loader, eval_loader
