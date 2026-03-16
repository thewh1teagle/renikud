# Training

## Commands

### 1. Align data

```console
uv run src/align_data.py dataset/train.tsv dataset/train_alignment.jsonl
uv run src/align_data.py dataset/val.tsv dataset/val_alignment.jsonl
```

### 2. Prepare tokenized dataset

```console
uv run src/prepare_tokens.py dataset/train_alignment.jsonl dataset/.cache/train
uv run src/prepare_tokens.py dataset/val_alignment.jsonl dataset/.cache/val
```

### 3. Train

```console
uv run src/train.py \
  --train-dataset dataset/.cache/train \
  --eval-dataset dataset/.cache/val \
  --output-dir outputs/my-run \
  --epochs 1 \
  --encoder-lr 2e-6 \
  --head-lr 1e-5 \
  --train-batch-size 32 \
  --gradient-accumulation-steps 1 \
  --warmup-steps 200 \
  --logging-steps 50 \
  --save-steps 200
```

### Fine-tuning from a checkpoint

Use `--init-from-checkpoint` to load weights only (resets optimizer state):

```console
uv run src/train.py \
  --train-dataset dataset/.cache/train \
  --eval-dataset dataset/.cache/val \
  --output-dir outputs/my-run \
  --init-from-checkpoint outputs/previous-run/checkpoint-1200 \
  --epochs 1
```

## Learning Rates

- `--encoder-lr 2e-6` — safe for fine-tuning the 300M param DictaBERT encoder
- `--head-lr 1e-5` — higher LR for the three classification heads

## Data Format

Input TSV: `hebrew_text<TAB>ipa_text` — one sentence per line, no header. Hebrew side may have nikud (diacritics are stripped automatically by the aligner).

The aligner outputs JSONL where each line is `{"hebrew sentence": [["char", "ipa_chunk"], ...]}`. Failed alignments are saved to `<output>_failures.txt`.
