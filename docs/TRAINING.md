# Training

## Commands

```console
uv run src/prepare_data.py --input data.tsv --output-dir dataset/mydata --lines 907200 --max-val 500
uv run src/prepare_tokens.py --input dataset/mydata/train.txt --output dataset/.cache/mydata-train
uv run src/prepare_tokens.py --input dataset/mydata/val.txt --output dataset/.cache/mydata-val

uv run src/train.py \
  --train-dataset dataset/.cache/mydata-train \
  --eval-dataset dataset/.cache/mydata-val \
  --output-dir outputs/my-run \
  --init-from-checkpoint outputs/g2p-mixed3-highlr/checkpoint-18000 \
  --epochs 1 \
  --encoder-lr 2e-5 \
  --head-lr 1e-4 \
  --train-batch-size 16 \
  --gradient-accumulation-steps 16 \
  --warmup-steps 200 \
  --logging-steps 50 \
  --save-steps 500
```

## Batch Size is Critical for CTC

**Always use an effective batch size of at least 256** (`--train-batch-size 16 --gradient-accumulation-steps 16`).

CTC loss works by summing over all possible alignments between input and output sequences. With a small batch, the gradient is computed from very few alignment paths — the signal is noisy, and the model oscillates instead of converging. With a large effective batch, gradients average over many more paths and point consistently in the right direction.

Observed effect in this project:
- Effective batch 256 → loss reached **0.07**
- Effective batch 16 → loss stuck at **0.25**

Gradient accumulation (`--gradient-accumulation-steps N`) is mathematically equivalent to a large batch but uses no extra VRAM — it accumulates gradients over N steps before each optimizer update.

## Learning Rates

- `--encoder-lr 2e-5` — safe for fine-tuning the 300M param DictaBERT encoder
- `--head-lr 1e-4` — higher LR for the projection + CTC classifier head

Do not raise encoder LR above `2e-5` — it causes NaN gradients and training instability.

## Fine-tuning from a Checkpoint

Use `--init-from-checkpoint` to load weights only (resets optimizer state). This is preferred over `--resume-from-checkpoint` when training on a new dataset.

## Data Format

Input TSV: `hebrew_text<TAB>ipa_text` — one sentence per line, no header. Hebrew side should have no nikud (diacritics are stripped automatically by `prepare_data.py`).
