"""Dataset loading and collation for classifier training."""

from __future__ import annotations

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader

from constants import IGNORE_INDEX
from tags import TAG_CATEGORIES, tag_prefix_ids


class ClassifierDataCollator:
    """Pad classifier dataset features to the same length within a batch."""

    pad_id: int = 0
    ignore_id: int = IGNORE_INDEX

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(len(f["input_ids"]) for f in features) + len(TAG_CATEGORIES)

        input_ids, attention_mask = [], []
        consonant_labels, vowel_labels, stress_labels = [], [], []
        texts, phonemes = [], []

        for f in features:
            prefix = tag_prefix_ids(f.get("tags", []))
            ids = [f["input_ids"][0]] + prefix + list(f["input_ids"][1:])
            pad = max_len - len(ids)
            ignore_prefix = [self.ignore_id] * len(prefix)
            input_ids.append(ids + [self.pad_id] * pad)
            attention_mask.append([1] * len(ids) + [0] * pad)
            consonant_labels.append([f["consonant_labels"][0]] + ignore_prefix + list(f["consonant_labels"][1:]) + [self.ignore_id] * pad)
            vowel_labels.append([f["vowel_labels"][0]] + ignore_prefix + list(f["vowel_labels"][1:]) + [self.ignore_id] * pad)
            stress_labels.append([f["stress_labels"][0]] + ignore_prefix + list(f["stress_labels"][1:]) + [self.ignore_id] * pad)
            texts.append(f["hebrew"])
            phonemes.append(f["phonemes"])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "consonant_labels": torch.tensor(consonant_labels, dtype=torch.long),
            "vowel_labels": torch.tensor(vowel_labels, dtype=torch.long),
            "stress_labels": torch.tensor(stress_labels, dtype=torch.long),
            "texts": texts,
            "phonemes": phonemes,
        }


def make_dataloaders(args) -> tuple[DataLoader, DataLoader]:
    collator = ClassifierDataCollator()
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
