#!/usr/bin/env bash
set -euo pipefail

OUTPUT=${1:-"checkpoint"}

uv run hf download thewh1teagle/renikud model.safetensors --local-dir "$OUTPUT"
echo "Downloaded to $OUTPUT/"
