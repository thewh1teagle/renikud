# NeoBERT

Source: https://huggingface.co/chandar-lab/NeoBERT

Fetched and modified for this project:

- Reduced from 28 to 16 layers to get ~113M parameters (vs 250M original), keeping hidden size at 768. Smaller model means smaller ONNX export size.
- Added `NEOBERT_ONNX_EXPORT=1` env var support for ONNX-compatible ops (real-valued RoPE, pure PyTorch SwiGLU).
