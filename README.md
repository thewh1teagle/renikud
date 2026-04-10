# renikud

Hebrew grapheme-to-phoneme (G2P) training project for converting unvocalized Hebrew text into IPA.

🤗 Model: [thewh1teagle/renikud](https://huggingface.co/thewh1teagle/renikud)

## Features

- Context-aware Hebrew G2P (text → IPA)
- Letter-level constrained decoding
- Passthrough for non-Hebrew text
- Runs on ONNX Runtime (no PyTorch)
- ~20 MB, real-time inference

## Usage

Inference is published as **`renikud-onnx`** on PyPI. Install and download the ONNX weights from Hugging Face (they are not bundled with the wheel):

```console
pip install renikud-onnx
wget https://huggingface.co/thewh1teagle/renikud/resolve/main/model.onnx -O model.onnx
```

```python
from renikud_onnx import G2P

g2p = G2P("model.onnx")
print(g2p.phonemize("שלום עולם"))
# → ʃalˈom ʔolˈam
```

See `renikud-onnx/README.md` for the same install / download / usage flow. For Rust inference, see `renikud-rs/`.

## Architecture

See `docs/ARCHITECTURE.md` for model design and implementation details.

## Training

See `docs/TRAINING.md` for data preparation, training commands, upload/download, ONNX export, benchmark, and hyperparameters.
