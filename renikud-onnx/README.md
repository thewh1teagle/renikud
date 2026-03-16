# renikud-onnx

Hebrew grapheme-to-phoneme (G2P) inference via ONNX. Converts unvocalized Hebrew text to IPA phonemes.

## Install

```console
pip install renikud-onnx
```

## Usage

```python
from renikud_onnx import G2P

g2p = G2P("model.onnx")
print(g2p.phonemize("שלום עולם"))
# → ʃalˈom ʔolˈam
```

## Export

To export a trained checkpoint to ONNX (requires the training repo):

```console
uv run scripts/export.py --checkpoint ../outputs/g2p-augmented/checkpoint-1500 --output model.onnx
```

The vocab is embedded as metadata inside the `.onnx` file — no extra files needed at runtime.
