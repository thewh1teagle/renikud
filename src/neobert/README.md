# NeoBERT

Source: https://huggingface.co/chandar-lab/NeoBERT

Fetched and modified for this project:

- Shrunk from the original 250M parameter configuration to a smaller model for faster training and smaller ONNX export size.
- Added `ONNX_EXPORT=1` env var support for ONNX-compatible ops (real-valued RoPE, pure PyTorch SwiGLU).
