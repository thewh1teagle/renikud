"""Dataset loading and collation for nikud diacritization training."""

from __future__ import annotations

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader

from constants import IGNORE_INDEX


class NikudDataCollator:
    """Pad dataset features to the same length within a batch."""

    pad_id: int = 0
    ignore_id: int = IGNORE_INDEX

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids, attention_mask = [], []
        nikud_labels, shin_labels = [], []
        texts = []

        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(list(f["input_ids"]) + [self.pad_id] * pad)
            attention_mask.append(list(f["attention_mask"]) + [0] * pad)
            nikud_labels.append(list(f["nikud_labels"]) + [self.ignore_id] * pad)
            shin_labels.append(list(f["shin_labels"]) + [self.ignore_id] * pad)
            texts.append(f["vocalized"])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "nikud_labels": torch.tensor(nikud_labels, dtype=torch.long),
            "shin_labels": torch.tensor(shin_labels, dtype=torch.long),
            "texts": texts,
        }


def make_dataloaders(args) -> tuple[DataLoader, DataLoader]:
    collator = NikudDataCollator()
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
