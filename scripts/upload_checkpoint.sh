#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT=${1:?"Usage: $0 <checkpoint-dir> [commit-message]"}
MESSAGE=${2:-"add weights"}

uv run hf upload thewh1teagle/renikud "$CHECKPOINT" --include "model.safetensors" --commit-message "$MESSAGE"
