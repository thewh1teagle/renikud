"""Checkpoint saving for classifier training."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def save_checkpoint(model, output_dir: Path, step: int, acc: float, save_total_limit: int):
    from safetensors.torch import save_file
    ckpt_dir = output_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), str(ckpt_dir / "model.safetensors"))
    (ckpt_dir / "train_state.json").write_text(json.dumps({"step": step, "acc": acc}))
    checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    while len(checkpoints) > save_total_limit:
        shutil.rmtree(checkpoints.pop(0))
