"""Dataset loading and collation utilities for Hebrew G2P."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from datasets import load_from_disk


def load_tokenized_dataset(path: str):
    """Load a pretokenized Arrow dataset from disk."""
    return load_from_disk(path)


def load_dataset_splits(train_path: str, eval_path: str):
    """Load train/eval datasets from disk."""
    train_dataset = load_tokenized_dataset(train_path)
    eval_dataset = load_tokenized_dataset(eval_path)
    return train_dataset, eval_dataset


@dataclass
class G2PDataCollator:
    """Pad tokenized encoder inputs and decoder labels for CTC training."""

    encoder_pad_id: int = 0
    label_pad_id: int = -100

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        max_input_len = max(len(feature["encoder_ids"]) for feature in features)
        max_label_len = max(len(feature["decoder_ids"]) for feature in features)

        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []

        for feature in features:
            encoder_ids = list(feature["encoder_ids"])
            encoder_mask = list(feature["encoder_mask"])
            decoder_ids = list(feature["decoder_ids"])

            input_pad_len = max_input_len - len(encoder_ids)
            label_pad_len = max_label_len - len(decoder_ids)

            input_ids.append(encoder_ids + [self.encoder_pad_id] * input_pad_len)
            attention_mask.append(encoder_mask + [0] * input_pad_len)
            labels.append(decoder_ids + [self.label_pad_id] * label_pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
