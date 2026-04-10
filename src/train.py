"""Train the Hebrew G2P classifier model.

Example:
    uv run src/train.py \
        --train-dataset dataset/.cache/classifier-train \
        --eval-dataset dataset/.cache/classifier-val \
        --output-dir outputs/g2p-classifier

Multi-GPU:
    accelerate launch src/train.py \
        --train-dataset dataset/.cache/train \
        --eval-dataset dataset/.cache/val \
        --output-dir outputs/g2p-classifier
"""

from __future__ import annotations

import math
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from tqdm import tqdm

from safetensors.torch import load_file

from checkpoint import resume_step, save_checkpoint, save_epoch_checkpoint
from config import parse_args
from data import make_dataloaders
from eval import evaluate
from model import G2PModel
from optimizer import build_optimizer, build_scheduler
from tokenization import load_tokenizer


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(mixed_precision="fp16" if args.fp16 else "no")
    device = accelerator.device

    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard")) if accelerator.is_main_process else None

    tokenizer = load_tokenizer()

    train_loader, eval_loader = make_dataloaders(args)

    model = G2PModel(flash_attention=args.flash_attention)

    if args.resume:
        state = load_file(str(Path(args.resume) / "model.safetensors"), device="cpu")
        model.load_state_dict(state, strict=False)
        if accelerator.is_main_process:
            print(f"Loaded weights from {args.resume}")

    if args.freeze_encoder_steps > 0:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        if accelerator.is_main_process:
            print("Encoder frozen.")

    total_opt_steps = math.ceil(len(train_loader) * args.epochs / args.gradient_accumulation_steps)
    optimizer = build_optimizer(model, args.encoder_lr, args.head_lr, args.weight_decay)
    scheduler = build_scheduler(optimizer, args.warmup_steps, total_opt_steps)

    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )

    opt_step = 0
    if args.resume and not args.reset_steps:
        opt_step = resume_step(args.resume, scheduler)
        if accelerator.is_main_process:
            print(f"Resumed from step {opt_step}")

    global_step = opt_step * args.gradient_accumulation_steps
    optimizer.zero_grad()

    for epoch in range(math.ceil(args.epochs)):
        epoch_loss_sum = 0.0
        epoch_steps = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}", dynamic_ncols=True, disable=not accelerator.is_main_process)

        for batch in pbar:
            if opt_step >= total_opt_steps:
                break

            if args.freeze_encoder_steps > 0 and global_step == args.freeze_encoder_steps:
                for p in accelerator.unwrap_model(model).encoder.parameters():
                    p.requires_grad_(True)
                if accelerator.is_main_process:
                    print(f"\n[step {opt_step}] Encoder unfrozen.")

            batch.pop("texts")
            batch.pop("phonemes")
            with accelerator.autocast():
                out = model(**batch)

            scaled_loss = out["loss"] / args.gradient_accumulation_steps
            accelerator.backward(scaled_loss)
            epoch_loss_sum += out["loss"].item()
            epoch_steps += 1
            global_step += 1

            if global_step % args.gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                opt_step += 1

                train_loss = epoch_loss_sum / epoch_steps
                pbar.set_postfix(
                    step=opt_step,
                    loss=f"{train_loss:.4f}",
                    enc_lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    head_lr=f"{optimizer.param_groups[2]['lr']:.2e}",
                )

                if accelerator.is_main_process:
                    if opt_step % args.logging_steps == 0:
                        print(f"[step {opt_step}] train_loss={train_loss:.4f} lr_encoder={optimizer.param_groups[0]['lr']:.2e} lr_head={optimizer.param_groups[2]['lr']:.2e}")
                        writer.add_scalar("train/loss", train_loss, opt_step)
                        writer.add_scalar("train/lr_encoder", optimizer.param_groups[0]["lr"], opt_step)
                        writer.add_scalar("train/lr_head", optimizer.param_groups[2]["lr"], opt_step)

                    if opt_step % args.save_steps == 0:
                        metrics = evaluate(accelerator.unwrap_model(model), eval_loader, device, args.fp16, tokenizer)
                        print(f"[step {opt_step}] loss={metrics['eval_loss']:.4f} consonant={metrics['consonant_acc']:.1%} vowel={metrics['vowel_acc']:.1%} stress={metrics['stress_acc']:.1%} char_acc={1-metrics['cer']:.1%} word_acc={1-metrics['wer']:.1%}")
                        writer.add_scalar("eval/loss", metrics["eval_loss"], opt_step)
                        writer.add_scalar("eval/consonant_acc", metrics["consonant_acc"], opt_step)
                        writer.add_scalar("eval/vowel_acc", metrics["vowel_acc"], opt_step)
                        writer.add_scalar("eval/stress_acc", metrics["stress_acc"], opt_step)
                        writer.add_scalar("eval/mean_acc", metrics["mean_acc"], opt_step)
                        writer.add_scalar("eval/char_acc", 1 - metrics["cer"], opt_step)
                        writer.add_scalar("eval/word_acc", 1 - metrics["wer"], opt_step)
                        save_checkpoint(accelerator.unwrap_model(model), tokenizer, output_dir, opt_step, metrics["mean_acc"], args.save_total_limit)

        if args.save_epochs and accelerator.is_main_process:
            metrics = evaluate(accelerator.unwrap_model(model), eval_loader, device, args.fp16, tokenizer)
            print(f"[epoch {epoch + 1}] loss={metrics['eval_loss']:.4f} consonant={metrics['consonant_acc']:.1%} vowel={metrics['vowel_acc']:.1%} stress={metrics['stress_acc']:.1%} char_acc={1-metrics['cer']:.1%} word_acc={1-metrics['wer']:.1%}")
            save_epoch_checkpoint(accelerator.unwrap_model(model), tokenizer, output_dir, epoch + 1, opt_step, metrics["mean_acc"])

    if accelerator.is_main_process:
        metrics = evaluate(accelerator.unwrap_model(model), eval_loader, device, args.fp16, tokenizer)
        print(f"Final: loss={metrics['eval_loss']:.4f} consonant={metrics['consonant_acc']:.1%} vowel={metrics['vowel_acc']:.1%} stress={metrics['stress_acc']:.1%} char_acc={1-metrics['cer']:.1%} word_acc={1-metrics['wer']:.1%}")
        writer.add_scalar("eval/loss", metrics["eval_loss"], opt_step)
        writer.add_scalar("eval/consonant_acc", metrics["consonant_acc"], opt_step)
        writer.add_scalar("eval/vowel_acc", metrics["vowel_acc"], opt_step)
        writer.add_scalar("eval/stress_acc", metrics["stress_acc"], opt_step)
        writer.add_scalar("eval/mean_acc", metrics["mean_acc"], opt_step)
        writer.add_scalar("eval/cer", metrics["cer"], opt_step)
        writer.add_scalar("eval/wer", metrics["wer"], opt_step)
        save_checkpoint(accelerator.unwrap_model(model), tokenizer, output_dir, opt_step, metrics["mean_acc"], args.save_total_limit)
        writer.close()


if __name__ == "__main__":
    main()
