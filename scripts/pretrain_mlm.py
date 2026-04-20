"""MLM pre-training for the Hebrew G2P encoder.

Trains ModernBertForMaskedLM from scratch on plain Hebrew text before
fine-tuning on labeled G2P pairs. The saved checkpoint can be loaded
into G2PModel via --resume (strict=False loads encoder weights only).

Example:
    uv run scripts/pretrain_mlm.py \\
        --train-file dataset/hebrew_10m_train.txt \\
        --eval-file dataset/hebrew_10m_val.txt \\
        --output-dir outputs/mlm-pretrain

Multi-GPU:
    accelerate launch scripts/pretrain_mlm.py \\
        --train-file dataset/hebrew_10m_train.txt \\
        --eval-file dataset/hebrew_10m_val.txt \\
        --output-dir outputs/mlm-pretrain
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import torch
from accelerate import Accelerator
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from transformers.models.modernbert.modeling_modernbert import ModernBertForMaskedLM

from encoder import build_config
from tokenization import load_tokenizer

SPACE_TOKEN_ID = 73   # ' ' in our char-level vocab (verified: tok.convert_tokens_to_ids(' ') == 73)
MASK_TOKEN_ID  = 4    # '[MASK]' is index 4 in SPECIAL_TOKENS


class WholeWordMaskingCollator:
    """Masks whole Hebrew words (character sequences between spaces) instead of random chars.

    Selects ~mlm_probability of words per sentence, then masks every character in
    those words. 80% replaced with [MASK], 10% random token, 10% unchanged — same
    schedule as standard MLM. Returns input_ids, attention_mask, labels.
    """

    def __init__(self, tokenizer, mlm_probability: float = 0.15, vocab_size: int = 104):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.vocab_size = vocab_size
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        # Pad batch
        max_len = max(len(e["input_ids"]) for e in examples)
        input_ids = torch.full((len(examples), max_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros(len(examples), max_len, dtype=torch.long)
        for i, e in enumerate(examples):
            seq = e["input_ids"]
            input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            attention_mask[i, :len(seq)] = 1

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # ignore padding

        for i in range(len(examples)):
            seq = input_ids[i]
            seq_len = attention_mask[i].sum().item()

            # Find word spans: contiguous non-space, non-special tokens
            words: list[tuple[int, int]] = []  # (start, end) inclusive
            j = 0
            while j < seq_len:
                tok = seq[j].item()
                if tok in (self.pad_id, 1, 2, SPACE_TOKEN_ID):  # PAD/CLS/SEP/space
                    j += 1
                    continue
                start = j
                while j < seq_len and seq[j].item() not in (self.pad_id, 1, 2, SPACE_TOKEN_ID):
                    j += 1
                words.append((start, j - 1))

            if not words:
                continue

            n_mask = max(1, round(len(words) * self.mlm_probability))
            selected = torch.randperm(len(words))[:n_mask].tolist()

            for wi in selected:
                start, end = words[wi]
                for pos in range(start, end + 1):
                    r = torch.rand(1).item()
                    if r < 0.8:
                        input_ids[i, pos] = MASK_TOKEN_ID
                    elif r < 0.9:
                        input_ids[i, pos] = torch.randint(5, self.vocab_size, (1,)).item()
                    # else: keep original (label still set)

            # Positions not selected → ignore in loss
            mask_positions = torch.zeros(seq_len, dtype=torch.bool)
            for wi in selected:
                start, end = words[wi]
                mask_positions[start:end + 1] = True
            labels[i, :seq_len][~mask_positions] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def parse_args():
    parser = argparse.ArgumentParser(description="MLM pre-training for Hebrew G2P encoder")
    parser.add_argument("--train-file", type=str, required=True, help="Plain text file, one sentence per line")
    parser.add_argument("--eval-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--save-total-limit", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=torch.cuda.is_available())
    parser.add_argument("--dataloader-workers", type=int, default=4)
    return parser.parse_args()


class TextDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lines = [l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.lines[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


def evaluate(model, eval_loader, device, fp16: bool) -> dict:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_masked = 0
    n = 0

    with torch.no_grad():
        for batch in eval_loader:
            with torch.autocast("cuda", enabled=fp16):
                out = model(**batch)

            total_loss += out.loss.item()
            n += 1

            # Masked token accuracy: only positions where labels != -100
            labels = batch["labels"]
            mask = labels != -100
            if mask.sum() > 0:
                preds = out.logits.argmax(dim=-1)
                total_correct += (preds[mask] == labels[mask]).sum().item()
                total_masked += mask.sum().item()

    model.train()
    avg_loss = total_loss / n
    return {
        "eval_loss": avg_loss,
        "perplexity": math.exp(avg_loss),
        "masked_token_accuracy": total_correct / total_masked if total_masked > 0 else 0.0,
    }


def save_checkpoint(model, output_dir: Path, step: int, loss: float, save_total_limit: int) -> None:
    from safetensors.torch import save_model
    ckpt_dir = output_dir / f"step-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_model(model, str(ckpt_dir / "model.safetensors"))
    (ckpt_dir / "train_state.json").write_text(json.dumps({"step": step, "eval_loss": loss}))
    checkpoints = sorted(output_dir.glob("step-*"), key=lambda p: int(p.name.split("-")[1]))
    while len(checkpoints) > save_total_limit:
        shutil.rmtree(checkpoints.pop(0))


def resume_step(checkpoint: str, scheduler) -> int:
    state_path = Path(checkpoint) / "train_state.json"
    if not state_path.exists():
        return 0
    step = json.loads(state_path.read_text())["step"]
    for _ in range(step):
        scheduler.step()
    return step


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(mixed_precision="fp16" if args.fp16 else "no")

    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard")) if accelerator.is_main_process else None

    tokenizer = load_tokenizer()

    train_dataset = TextDataset(args.train_file, tokenizer, args.max_length)
    eval_dataset = TextDataset(args.eval_file, tokenizer, args.max_length)

    collator = WholeWordMaskingCollator(tokenizer=tokenizer, mlm_probability=args.mlm_probability, vocab_size=len(tokenizer))

    train_loader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True,
        collate_fn=collator, num_workers=args.dataloader_workers, pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.eval_batch_size, shuffle=False,
        collate_fn=collator, num_workers=args.dataloader_workers, pin_memory=True,
    )

    model = ModernBertForMaskedLM(build_config())

    if args.resume:
        state = load_file(str(Path(args.resume) / "model.safetensors"), device="cpu")
        model.load_state_dict(state, strict=False)
        if accelerator.is_main_process:
            print(f"Loaded weights from {args.resume}")

    no_decay = {"bias", "norm.weight"}
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if not any(t in n for t in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(t in n for t in no_decay)], "weight_decay": 0.0},
    ], lr=args.lr)

    total_opt_steps = math.ceil(len(train_loader) * args.epochs / args.gradient_accumulation_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_opt_steps)

    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )

    opt_step = 0
    if args.resume:
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

            with accelerator.autocast():
                out = model(**batch)

            scaled_loss = out.loss / args.gradient_accumulation_steps
            accelerator.backward(scaled_loss)
            epoch_loss_sum += out.loss.item()
            epoch_steps += 1
            global_step += 1

            if global_step % args.gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                opt_step += 1

                train_loss = epoch_loss_sum / epoch_steps
                pbar.set_postfix(step=opt_step, loss=f"{train_loss:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

                if accelerator.is_main_process and opt_step % args.logging_steps == 0:
                    writer.add_scalar("train/loss", train_loss, opt_step)
                    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], opt_step)

                if accelerator.is_main_process and opt_step % args.save_steps == 0:
                    metrics = evaluate(accelerator.unwrap_model(model), eval_loader, accelerator.device, args.fp16)
                    writer.add_scalar("eval/loss", metrics["eval_loss"], opt_step)
                    writer.add_scalar("eval/perplexity", metrics["perplexity"], opt_step)
                    writer.add_scalar("eval/masked_token_accuracy", metrics["masked_token_accuracy"], opt_step)
                    print(f"[step {opt_step}] eval_loss={metrics['eval_loss']:.4f} ppl={metrics['perplexity']:.2f} acc={metrics['masked_token_accuracy']:.2%}")
                    save_checkpoint(accelerator.unwrap_model(model), output_dir, opt_step, metrics["eval_loss"], args.save_total_limit)

    if accelerator.is_main_process:
        metrics = evaluate(accelerator.unwrap_model(model), eval_loader, accelerator.device, args.fp16)
        writer.add_scalar("eval/loss", metrics["eval_loss"], opt_step)
        writer.add_scalar("eval/perplexity", metrics["perplexity"], opt_step)
        writer.add_scalar("eval/masked_token_accuracy", metrics["masked_token_accuracy"], opt_step)
        print(f"[final] eval_loss={metrics['eval_loss']:.4f} ppl={metrics['perplexity']:.2f} acc={metrics['masked_token_accuracy']:.2%}")
        save_checkpoint(accelerator.unwrap_model(model), output_dir, opt_step, metrics["eval_loss"], args.save_total_limit)
        writer.close()


if __name__ == "__main__":
    main()
