# Training

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) — Python package manager
- [Go](https://go.dev/) — required to build the aligner

## Commands

### 1. Prepare dataset

```console
./scripts/train_prepare.sh dataset/data.tsv
```

### 2. Train

```console
./scripts/train.sh                                                    # from scratch
./scripts/train.sh --resume outputs/g2p-classifier/checkpoint-5000   # resume
./scripts/train.sh --resume outputs/g2p-classifier/checkpoint-5000 --reset-steps  # finetune (load weights, reset steps)
```

## Upload Checkpoint to HuggingFace

```console
./scripts/upload_checkpoint.sh outputs/g2p-classifier/checkpoint-5000
```

## Download Checkpoint

```console
./scripts/download_checkpoint.sh                  # downloads to ./checkpoint
./scripts/train.sh --resume checkpoint --reset-steps  # finetune from downloaded
```

## CUDA Version

```console
uv sync --extra cu130  # CUDA 13.0
uv sync --extra cu128  # CUDA 12.8
```

## Flash Attention

Enable with `--flash-attention`. Install a prebuilt wheel first:

- **x86_64**: https://github.com/mjun0812/flash-attention-prebuild-wheels
- **aarch64 (ARM)**: https://pypi.jetson-ai-lab.io/sbsa/cu130

```console
./scripts/train.sh --flash-attention
```

## Learning Rates

- `--encoder-lr 2e-5` — default for training from scratch
- `--head-lr 1e-4` — higher LR for the three classification heads
- For fine-tuning use lower rates: `--encoder-lr 2e-6 --head-lr 1e-5`

## Data Format

Input TSV: `hebrew_text<TAB>ipa_text` — one sentence per line, no header. Hebrew side may have nikud (diacritics are stripped automatically by the aligner).

The aligner outputs JSONL where each line is `{"hebrew": "...", "alignment": [["char", "ipa_chunk"], ...]}`. Failed alignments are saved to `<output>_failures.txt`.
