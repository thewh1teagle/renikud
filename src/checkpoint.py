"""Checkpoint saving and resuming for classifier training."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def save_checkpoint(model, tokenizer, output_dir: Path, step: int, acc: float, save_total_limit: int) -> None:
    from safetensors.torch import save_file
    ckpt_dir = output_dir / f"step-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), str(ckpt_dir / "model.safetensors"))
    tokenizer.save_pretrained(str(ckpt_dir))
    (ckpt_dir / "train_state.json").write_text(json.dumps({"step": step, "acc": acc}))
    checkpoints = sorted(output_dir.glob("step-*"), key=lambda p: int(p.name.split("-")[1]))
    while len(checkpoints) > save_total_limit:
        shutil.rmtree(checkpoints.pop(0))


def save_epoch_checkpoint(model, tokenizer, output_dir: Path, epoch: int, step: int, acc: float) -> None:
    from safetensors.torch import save_file
    ckpt_dir = output_dir / f"epoch-{epoch}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), str(ckpt_dir / "model.safetensors"))
    tokenizer.save_pretrained(str(ckpt_dir))
    (ckpt_dir / "train_state.json").write_text(json.dumps({"step": step, "epoch": epoch, "acc": acc}))


def resume_step(checkpoint: str, scheduler) -> int:
    """Load step counter from a checkpoint and fast-forward the scheduler. Returns the step."""
    state_path = Path(checkpoint) / "train_state.json"
    if not state_path.exists():
        return 0
    saved = json.loads(state_path.read_text())
    step = saved["step"]
    for _ in range(step):
        scheduler.step()
    return step
