"""
wget https://huggingface.co/thewh1teagle/renikud/resolve/main/model.onnx
uv run examples/basic.py
"""
from renikud_onnx import G2P

g2p = G2P("model.onnx")
print(g2p.phonemize("הוא רצה את זה גם, אבל היא רצה מהר והקדימה אותו"))
