# renikud-onnx

Hebrew grapheme-to-phoneme (G2P) inference via ONNX. Converts unvocalized Hebrew text to IPA phonemes.

## Features

- Context-aware Hebrew G2P (text → IPA)
- Letter-level constrained decoding
- Passthrough for non-Hebrew text
- Runs on ONNX Runtime (no PyTorch)
- ~20 MB, real-time inference

## Install

```console
pip install renikud-onnx
```

Download the model file from Hugging Face:

```console
wget https://huggingface.co/thewh1teagle/renikud/resolve/main/model.onnx -O model.onnx
```

## Usage

```python
from renikud_onnx import G2P

g2p = G2P("model.onnx")
print(g2p.phonemize("שלום עולם"))
# → ʃalˈom ʔolˈam
```
