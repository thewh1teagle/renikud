"""Dataset loading and collation for classifier training."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import torch
from datasets import Dataset, load_dataset, load_from_disk
from torch.utils.data import DataLoader

from constants import IGNORE_INDEX
from encoder import HF_MODEL, TRUST_REMOTE_CODE
from labeling import label_sentence


def prepare_dataset(jsonl_path: str, name: str) -> Dataset:
    cache_dir = Path(jsonl_path).parent / ".cache" / name
    if cache_dir.exists():
        return load_from_disk(str(cache_dir))

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, trust_remote_code=TRUST_REMOTE_CODE)
    dataset = load_dataset("json", data_files=jsonl_path, split="train")
    dataset = dataset.map(
        label_sentence,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=mp.cpu_count(),
        remove_columns=["alignment"],
        desc=f"Tokenizing {name}",
    )
    dataset.save_to_disk(str(cache_dir))
    return dataset


class ClassifierDataCollator:
    pad_id: int = 0
    ignore_id: int = IGNORE_INDEX

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids, attention_mask = [], []
        consonant_labels, vowel_labels, stress_labels = [], [], []
        texts, phonemes = [], []

        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(list(f["input_ids"]) + [self.pad_id] * pad)
            attention_mask.append(list(f["attention_mask"]) + [0] * pad)
            consonant_labels.append(list(f["consonant_labels"]) + [self.ignore_id] * pad)
            vowel_labels.append(list(f["vowel_labels"]) + [self.ignore_id] * pad)
            stress_labels.append(list(f["stress_labels"]) + [self.ignore_id] * pad)
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
    train_ds = prepare_dataset(args.train_dataset, "train")
    eval_ds = prepare_dataset(args.eval_dataset, "val")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.dataloader_workers,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.dataloader_workers,
    )
    return train_loader, eval_loader
