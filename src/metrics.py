"""Metric logging helpers for training."""

from __future__ import annotations


def log_train_metrics(loss, enc_lr, head_lr, writer, opt_step):
    print(f"[step {opt_step}] train_loss={loss:.4f} lr_encoder={enc_lr:.2e} lr_head={head_lr:.2e}")
    writer.add_scalar("train/loss", loss, opt_step)
    writer.add_scalar("train/lr_encoder", enc_lr, opt_step)
    writer.add_scalar("train/lr_head", head_lr, opt_step)


def log_eval_metrics(metrics, writer, opt_step, label):
    print(f"[{label}] loss={metrics['eval_loss']:.4f} nikud={metrics['nikud_acc']:.1%} shin={metrics['shin_acc']:.1%} char_acc={1-metrics['cer']:.1%} word_acc={1-metrics['wer']:.1%}")
    if writer is not None:
        writer.add_scalar("eval/loss", metrics["eval_loss"], opt_step)
        writer.add_scalar("eval/nikud_acc", metrics["nikud_acc"], opt_step)
        writer.add_scalar("eval/shin_acc", metrics["shin_acc"], opt_step)
        writer.add_scalar("eval/mean_acc", metrics["mean_acc"], opt_step)
        writer.add_scalar("eval/char_acc", 1 - metrics["cer"], opt_step)
        writer.add_scalar("eval/word_acc", 1 - metrics["wer"], opt_step)
