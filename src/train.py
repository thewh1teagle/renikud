"""
Training script for Renikud model

Usage:
    python -m src.train --device mps --epochs 10 --batch_size 16
"""

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm, trange
from pathlib import Path

from src.config import get_args
from src.model.renikud_model import RenikudModel, RenikudLogitsOutput
from src.data import Collator, get_dataloader


def train_step(model, batch, optimizer, criterion_vowel, criterion_dagesh, criterion_sin, device, scaler):
    """Single training step"""
    optimizer.zero_grad()
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in batch.input.items()}
    vowel_targets = batch.vowel_targets.to(device)
    dagesh_targets = batch.dagesh_targets.to(device)
    sin_targets = batch.sin_targets.to(device)
    
    # Forward pass
    with torch.amp.autocast(device_type='cuda' if 'cuda' in str(device) else 'cpu', enabled='cuda' in str(device)):
        output: RenikudLogitsOutput = model(**inputs)
        
        # Calculate losses for each head
        # Vowel: CrossEntropyLoss (multi-class)
        vowel_loss = criterion_vowel(
            output.vowel_logits.view(-1, 7),  # 7 classes
            vowel_targets.view(-1)
        )
        
        # Dagesh: CrossEntropyLoss (binary as 2-class)
        dagesh_loss = criterion_dagesh(
            output.dagesh_logits.view(-1, 2),
            dagesh_targets.view(-1)
        )
        
        # Sin: CrossEntropyLoss (binary as 2-class)
        sin_loss = criterion_sin(
            output.sin_logits.view(-1, 2),
            sin_targets.view(-1)
        )
        
        # Combine losses (equal weighting)
        loss = vowel_loss + dagesh_loss + sin_loss
    
    # Backward pass
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    
    return loss.item(), vowel_loss.item(), dagesh_loss.item(), sin_loss.item()


def validate(model, val_dataloader, device):
    """Validation loop"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    criterion_vowel = nn.CrossEntropyLoss()
    criterion_dagesh = nn.CrossEntropyLoss()
    criterion_sin = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation"):
            inputs = {k: v.to(device) for k, v in batch.input.items()}
            vowel_targets = batch.vowel_targets.to(device)
            dagesh_targets = batch.dagesh_targets.to(device)
            sin_targets = batch.sin_targets.to(device)
            
            output: RenikudLogitsOutput = model(**inputs)
            
            vowel_loss = criterion_vowel(
                output.vowel_logits.view(-1, 7),
                vowel_targets.view(-1)
            )
            dagesh_loss = criterion_dagesh(
                output.dagesh_logits.view(-1, 2),
                dagesh_targets.view(-1)
            )
            sin_loss = criterion_sin(
                output.sin_logits.view(-1, 2),
                sin_targets.view(-1)
            )
            
            loss = vowel_loss + dagesh_loss + sin_loss
            total_loss += loss.item()
            num_batches += 1
    
    model.train()
    return total_loss / num_batches if num_batches > 0 else 0.0


def save_checkpoint(model, tokenizer, output_dir, step=None, loss=None):
    """Save model checkpoint"""
    output_dir = Path(output_dir)
    if step is not None and loss is not None:
        output_dir = output_dir / f"step_{step}_loss_{loss:.4f}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"💾 Saved checkpoint to {output_dir}")


def load_data(data_file: Path, val_split: int):
    """Load and split data"""
    with open(data_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Split into train and validation
    val_lines = lines[:val_split]
    train_lines = lines[val_split:]
    
    print(f"📊 Loaded {len(train_lines)} training lines, {len(val_lines)} validation lines")
    
    # For now, use the same lines for unvocalized and vocalized
    # In practice, you'd have separate unvocalized versions
    return train_lines, train_lines, val_lines, val_lines


def main():
    args = get_args()
    
    print(f"🚀 Starting Renikud training")
    print(f"   Device: {args.device}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Learning rate: {args.learning_rate}")
    
    # Create output directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer and model
    print(f"🧠 Loading model from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = RenikudModel.from_pretrained(args.model, trust_remote_code=True)
    
    model.to(args.device)
    
    # Optionally freeze BERT backbone
    if args.freeze_bert:
        model.freeze_base_model()
        print(f"✅ Model loaded and BERT backbone frozen")
    else:
        print(f"✅ Model loaded - training entire model (BERT + heads)")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    # Load data
    data_file = args.data_dir / "renikud_data_v1.txt"
    if not data_file.exists():
        # Try parent directory
        data_file = args.data_dir.parent / "renikud_data_v1.txt"
    
    train_unvocalized, train_vocalized, val_unvocalized, val_vocalized = load_data(
        data_file, int(args.val_split_num)
    )
    
    # Create dataloaders
    collator = Collator(tokenizer)
    train_dataloader = get_dataloader(
        train_unvocalized,
        train_vocalized,
        args.batch_size,
        collator,
        num_workers=args.num_workers,
        shuffle=True,
    )
    val_dataloader = get_dataloader(
        val_unvocalized,
        val_vocalized,
        args.batch_size,
        collator,
        num_workers=args.num_workers,
        shuffle=False,
    )
    
    # Setup training
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=9000, gamma=0.1)
    scaler = torch.amp.GradScaler('cuda' if 'cuda' in args.device else 'cpu', enabled='cuda' in args.device)
    
    criterion_vowel = nn.CrossEntropyLoss()
    criterion_dagesh = nn.CrossEntropyLoss()
    criterion_sin = nn.CrossEntropyLoss()
    
    # Training loop
    step = 0
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    for epoch in trange(args.epochs, desc="Epochs"):
        model.train()
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in pbar:
            loss, vowel_loss, dagesh_loss, sin_loss = train_step(
                model, batch, optimizer, criterion_vowel, criterion_dagesh, 
                criterion_sin, args.device, scaler
            )
            scheduler.step()
            step += 1
            
            pbar.set_description(
                f"Epoch {epoch+1} | L={loss:.4f} (V:{vowel_loss:.3f} D:{dagesh_loss:.3f} S:{sin_loss:.3f})"
            )
            
            # Checkpoint and validation
            if args.checkpoint_interval and step % args.checkpoint_interval == 0:
                # Validate
                val_loss = validate(model, val_dataloader, args.device)
                print(f"\n📊 Step {step} | Val loss: {val_loss:.4f}")
                
                # Save checkpoint
                save_checkpoint(model, tokenizer, args.output_dir / "last")
                
                # Save best
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stop_counter = 0
                    save_checkpoint(model, tokenizer, args.output_dir / "best")
                    print(f"🏆 New best model! Val loss: {val_loss:.4f}")
                else:
                    early_stop_counter += 1
                    print(f"📉 No improvement ({early_stop_counter}/{args.early_stopping_patience})")
                
                # Early stopping
                if early_stop_counter >= args.early_stopping_patience:
                    print(f"⏹️  Early stopping at step {step}")
                    break
        
        if early_stop_counter >= args.early_stopping_patience:
            break
    
    # Save final model
    save_checkpoint(model, tokenizer, args.output_dir / f"final_epoch_{epoch+1}")
    print(f"✅ Training complete! Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
