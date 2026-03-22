"""
Export HebrewG2PClassifier to a self-contained ONNX file with vocab metadata embedded.

Usage:
    uv run scripts/export.py --checkpoint ../outputs/g2p-classifier-v3/checkpoint-1200 --output model.onnx
    uv run scripts/export.py --checkpoint ../outputs/g2p-classifier-v3/checkpoint-1200 --output model.onnx --int8
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from constants import CONSONANTS, VOWELS, HEBREW_LETTER_TO_ALLOWED_CONSONANTS
from infer import load_checkpoint
from model import HebrewG2PClassifier
from tokenization import load_encoder_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="model.onnx")
    parser.add_argument("--int8", action="store_true", help="Quantize weights to INT8 (dynamic quantization, no calibration needed)")
    args = parser.parse_args()

    tokenizer = load_encoder_tokenizer()
    vocab = tokenizer.get_vocab()  # {token: id}
    tokenizer_vocab = {v: k for k, v in vocab.items()}  # {id: token}

    model = HebrewG2PClassifier()
    load_checkpoint(model, args.checkpoint)
    if args.int8:
        model.float().eval()
    else:
        model.half().eval()

    dummy_ids = torch.zeros(1, 16, dtype=torch.long)
    dummy_mask = torch.ones(1, 16, dtype=torch.long)

    export_path = args.output
    if args.int8:
        tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        export_path = tmp.name
        tmp.close()

    torch.onnx.export(
        model,
        (dummy_ids, dummy_mask, tokenizer_vocab),
        export_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["consonant_logits", "vowel_logits", "stress_logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "consonant_logits": {0: "batch", 1: "seq_len"},
            "vowel_logits": {0: "batch", 1: "seq_len"},
            "stress_logits": {0: "batch", 1: "seq_len"},
        },
        opset_version=17,
    )

    if args.int8:
        quantize_dynamic(export_path, args.output, weight_type=QuantType.QInt8)
        Path(export_path).unlink(missing_ok=True)
        Path(export_path + ".data").unlink(missing_ok=True)

    onnx_model = onnx.load(args.output, load_external_data=True)
    meta = onnx_model.metadata_props

    entry = meta.add()
    entry.key = "vocab"
    entry.value = json.dumps(vocab)

    entry = meta.add()
    entry.key = "consonant_vocab"
    entry.value = json.dumps({str(i): c for i, c in enumerate(CONSONANTS)})

    entry = meta.add()
    entry.key = "vowel_vocab"
    entry.value = json.dumps({str(i): v for i, v in enumerate(VOWELS)})

    entry = meta.add()
    entry.key = "letter_consonant_constraints"
    entry.value = json.dumps({letter: list(ids) for letter, ids in HEBREW_LETTER_TO_ALLOWED_CONSONANTS.items()})

    entry = meta.add()
    entry.key = "cls_token_id"
    entry.value = str(tokenizer.cls_token_id)

    entry = meta.add()
    entry.key = "sep_token_id"
    entry.value = str(tokenizer.sep_token_id)

    onnx.save_model(onnx_model, args.output, save_as_external_data=False)

    data_file = Path(args.output + ".data")
    if data_file.exists():
        data_file.unlink()

    quant_label = " (int8)" if args.int8 else " (fp16)"
    print(f"Exported to {args.output}{quant_label} ({Path(args.output).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
