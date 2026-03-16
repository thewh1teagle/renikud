"""
Hebrew G2P demo using the classifier model.

Usage:
    uv run src/app.py outputs/g2p-classifier-v3/checkpoint-1800
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
import torch

from constants import MAX_LEN
from infer import load_checkpoint, phonemize
from model import HebrewG2PClassifier
from tokenization import load_encoder_tokenizer

checkpoint = sys.argv[1] if len(sys.argv) > 1 else None
if not checkpoint:
    print("Usage: uv run src/app.py <checkpoint_dir>")
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = load_encoder_tokenizer()
model = HebrewG2PClassifier()
load_checkpoint(model, checkpoint)
model.to(device).eval()


def predict(text: str) -> str:
    if not text.strip():
        return ""
    return phonemize(text, model, tokenizer, device, MAX_LEN)


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Hebrew text", placeholder="הקלד טקסט בעברית...", lines=6, rtl=True),
    outputs=gr.Textbox(label="IPA", lines=6),
    title="Hebrew G2P",
    examples=[
        ["שלום עולם"],
        ["הוא צפה בסרט וראה חיה שצפה במים"],
        ["ראיתי את זה בוואטסאפ של חבר שלי"],
    ],
)

if __name__ == "__main__":
    demo.launch()
