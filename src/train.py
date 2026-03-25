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

import torch
import wandb
from accelerate import Accelerator
from tqdm import tqdm

from checkpoint import cosine_lr_lambda, save_checkpoint
from config import parse_args
from data import make_dataloaders
from eval import evaluate
from model import HebrewG2PClassifier


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(mixed_precision="fp16" if args.fp16 else "no")
    device = accelerator.device

    if accelerator.is_main_process:
        wandb.init(project="hebrew-g2p-classifier", config=vars(args), mode=args.wandb_mode)

    train_loader, eval_loader = make_dataloaders(args)

    model = HebrewG2PClassifier()

    if args.init_from_checkpoint:
        from safetensors.torch import load_file
        state = load_file(str(Path(args.init_from_checkpoint) / "model.safetensors"), device="cpu")
        model.load_state_dict(state, strict=False)
        if accelerator.is_main_process:
            print(f"Loaded weights from {args.init_from_checkpoint}")

    if args.freeze_encoder_steps > 0:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        if accelerator.is_main_process:
            print("Encoder frozen.")

    optimizer = torch.optim.AdamW(
        model.parameter_groups(args.encoder_lr, args.head_lr, args.weight_decay)
    )

    total_opt_steps = math.ceil(len(train_loader) * args.epochs / args.gradient_accumulation_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_lr_lambda(step, args.warmup_steps, total_opt_steps),
    )

    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )

    # Restore step counter when resuming
    opt_step = 0
    if args.init_from_checkpoint:
        import json
        state_path = Path(args.init_from_checkpoint) / "train_state.json"
        if state_path.exists():
            saved = json.loads(state_path.read_text())
            opt_step = saved["step"]
            # Fast-forward scheduler to correct LR
            for _ in range(opt_step):
                scheduler.step()
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
                        wandb.log({
                            "train_loss": train_loss,
                            "lr_encoder": optimizer.param_groups[0]["lr"],
                            "lr_head": optimizer.param_groups[2]["lr"],
                            "epoch": epoch,
                        }, step=opt_step)

                    if opt_step % args.save_steps == 0:
                        metrics = evaluate(accelerator.unwrap_model(model), eval_loader, device, args.fp16)
                        wandb.log(metrics, step=opt_step)
                        print(f"[step {opt_step}] consonant_acc={metrics['consonant_acc']:.4f} vowel_acc={metrics['vowel_acc']:.4f} stress_acc={metrics['stress_acc']:.4f} eval_loss={metrics['eval_loss']:.4f}")
                        save_checkpoint(accelerator.unwrap_model(model), output_dir, opt_step, metrics["mean_acc"], args.save_total_limit)

    if accelerator.is_main_process:
        metrics = evaluate(accelerator.unwrap_model(model), eval_loader, device, args.fp16)
        wandb.log(metrics)
        print(f"Final: consonant_acc={metrics['consonant_acc']:.4f} vowel_acc={metrics['vowel_acc']:.4f} stress_acc={metrics['stress_acc']:.4f}")
        save_checkpoint(accelerator.unwrap_model(model), output_dir, opt_step, metrics["mean_acc"], args.save_total_limit)
        wandb.finish()


if __name__ == "__main__":
    main()
