"""Export G2P CTC model to ONNX.

Usage:
    uv run scripts/export.py --checkpoint ../outputs/g2p-classifier/checkpoint-180000-unicode
    uv run scripts/export.py --checkpoint ../outputs/g2p-classifier/checkpoint-180000-unicode --no-int8
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["NEOBERT_ONNX_EXPORT"] = "1"

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from infer import load_checkpoint
from lang_pack import get_lang_pack
from model import G2PModel
from tokenization import CLS_ID, SEP_ID


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="model.onnx")
    parser.add_argument("--lang", default="hebrew")
    parser.add_argument("--int8", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    lang_pack = get_lang_pack(args.lang)
    model = G2PModel(lang_pack=lang_pack)
    load_checkpoint(model, args.checkpoint)
    model.float().eval()

    dummy_ids = torch.zeros(1, 16, dtype=torch.long)
    dummy_mask = torch.ones(1, 16, dtype=torch.long)

    export_path = args.output
    if args.int8:
        tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        export_path = tmp.name
        tmp.close()

    torch.onnx.export(
        model,
        (dummy_ids, dummy_mask),
        export_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids":      {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "logits":         {0: "batch", 1: "seq_len"},
        },
        opset_version=18,
    )

    if args.int8:
        quantize_dynamic(export_path, args.output, weight_type=QuantType.QInt8)
        Path(export_path).unlink(missing_ok=True)
        Path(export_path + ".data").unlink(missing_ok=True)

    # Embed metadata for inference without extra files
    onnx_model = onnx.load(args.output, load_external_data=True)
    meta = onnx_model.metadata_props

    def add_meta(key, value):
        e = meta.add()
        e.key = key
        e.value = value

    add_meta("lang", args.lang)
    add_meta("cls_token_id", str(CLS_ID))
    add_meta("sep_token_id", str(SEP_ID))
    add_meta("output_tokens", json.dumps(list(lang_pack.output_tokens)))
    add_meta("input_chars", json.dumps(sorted(lang_pack.input_chars)))
    add_meta("upsample_factor", "3")
    add_meta("strip_accents", str(lang_pack.strip_accents).lower())

    onnx.save_model(onnx_model, args.output, save_as_external_data=False)
    Path(args.output + ".data").unlink(missing_ok=True)

    quant_label = " (int8)" if args.int8 else " (fp32)"
    print(f"Exported to {args.output}{quant_label} ({Path(args.output).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
