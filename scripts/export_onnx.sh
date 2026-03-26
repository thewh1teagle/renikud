#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT=${1:?"Usage: $0 <checkpoint-dir> [output.onnx]"}
OUTPUT=${2:-"model.onnx"}

cd "$(dirname "$0")/../renikud-onnx"
uv run scripts/export.py --checkpoint "../$CHECKPOINT" --output "$OUTPUT"
