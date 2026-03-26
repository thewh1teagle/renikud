# Training

## Commands

### 1. Prepare dataset

```console
./scripts/train_prepare.sh dataset/data.tsv
```

### 2. Train from scratch

```console
./scripts/train_scratch.sh
```

### 3. Fine-tune from a checkpoint

```console
./scripts/train_finetune.sh outputs/g2p-classifier/checkpoint-5000
```

## Learning Rates

- `--encoder-lr 2e-5` — default for training from scratch
- `--head-lr 1e-4` — higher LR for the three classification heads
- For fine-tuning use lower rates: `--encoder-lr 2e-6 --head-lr 1e-5`

## Data Format

Input TSV: `hebrew_text<TAB>ipa_text` — one sentence per line, no header. Hebrew side may have nikud (diacritics are stripped automatically by the aligner).

The aligner outputs JSONL where each line is `{"hebrew sentence": [["char", "ipa_chunk"], ...]}`. Failed alignments are saved to `<output>_failures.txt`.
