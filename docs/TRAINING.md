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

## Upload Checkpoint to HuggingFace

```console
./scripts/upload_checkpoint.sh outputs/g2p-classifier/checkpoint-5000
```

## Download Checkpoint

```console
./scripts/download_checkpoint.sh                  # downloads to ./checkpoint
```

To fine-tune from a downloaded checkpoint:

```console
./scripts/download_checkpoint.sh checkpoint
./scripts/train_finetune.sh checkpoint
```

## xFormers

xFormers must match your PyTorch CUDA version. If you see a version mismatch warning (e.g. built for cu128 but you have cu130), build from source:

```console
pip install xformers --no-binary xformers
```

Note: `pyproject.toml` pins PyTorch to the `pytorch-cu130` index. If your system uses a different CUDA version, remove or update that index entry before installing.

## Flash Attention

ModernBERT supports Flash Attention 2 for faster training and lower VRAM usage. Enable with `--flash-attention`:

Install a prebuilt wheel first:

- **x86_64**: https://github.com/mjun0812/flash-attention-prebuild-wheels
- **aarch64 (ARM)**: https://pypi.jetson-ai-lab.io/sbsa/cu130

```console
./scripts/train_scratch.sh --flash-attention
```

Install a prebuilt wheel first:

- **x86_64**: https://github.com/mjun0812/flash-attention-prebuild-wheels
- **aarch64 (ARM)**: https://pypi.jetson-ai-lab.io/sbsa/cu130

Validate:

```console
uv run python -c "import flash_attn; print(flash_attn.__version__)"
```

## Learning Rates

- `--encoder-lr 2e-5` — default for training from scratch
- `--head-lr 1e-4` — higher LR for the three classification heads
- For fine-tuning use lower rates: `--encoder-lr 2e-6 --head-lr 1e-5`

## Data Format

Input TSV: `hebrew_text<TAB>ipa_text` — one sentence per line, no header. Hebrew side may have nikud (diacritics are stripped automatically by the aligner).

The aligner outputs JSONL where each line is `{"hebrew sentence": [["char", "ipa_chunk"], ...]}`. Failed alignments are saved to `<output>_failures.txt`.
