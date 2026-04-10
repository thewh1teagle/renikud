# renikud

Hebrew grapheme-to-phoneme (G2P) training project for converting unvocalized Hebrew text into IPA.

Model: [thewh1teagle/renikud](https://huggingface.co/thewh1teagle/renikud)

## Architecture

NeoBERT encoder (~19M params) trained from scratch → three coupled classification heads (consonant, vowel, stress) per Hebrew letter.

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
./scripts/ckpt_export.sh outputs/g2p-classifier/checkpoint-5000
```

Then use the Python package:

```python
from renikud_onnx import G2P

g2p = G2P("model.onnx")
print(g2p.phonemize("שלום עולם"))
# → ʃalˈom ʔolˈam
```

Or the Rust crate — see `renikud-rs/`.
