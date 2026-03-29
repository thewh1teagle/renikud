# renikud

Hebrew grapheme-to-phoneme (G2P) training project for converting unvocalized Hebrew text into IPA.

Model: [thewh1teagle/renikud](https://huggingface.co/thewh1teagle/renikud)

## Architecture

ModernBERT-base encoder trained from scratch (22 layers, 768 hidden, RoPE, Flash Attention) → three linear heads (consonant, vowel, stress) per Hebrew letter.

Each Hebrew letter gets exactly one output slot predicting a (consonant, vowel, stress) triple. Uses a custom character-level tokenizer built for Hebrew.

See `docs/ARCHITECTURE.md` for full design details.

## Training

See `docs/TRAINING.md` for data preparation, training commands, upload/download, and hyperparameters.

## Benchmark

```console
./scripts/train_bench.sh outputs/g2p-classifier/checkpoint-5000
```

## Inference (ONNX)

Export a checkpoint to ONNX:

```console
./scripts/export_onnx.sh outputs/g2p-classifier/checkpoint-5000
```

Then use the Python package:

```python
from renikud_onnx import G2P

g2p = G2P("model.onnx")
print(g2p.phonemize("שלום עולם"))
# → ʃalˈom ʔolˈam
```

Or the Rust crate — see `renikud-rs/`.
