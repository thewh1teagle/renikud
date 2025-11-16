# Renikud Model

A simplified Hebrew diacritization model with 3 separate prediction heads:

1. **Vowel head**: 7-class classifier (6 vowels: SHVA, SEGOL, HIRIK, PATAH, HOLAM, QUBUTS + empty)
2. **Dagesh head**: Binary classifier - applied only to בכךפףו
3. **Sin head**: Binary classifier - applied only to ש

## Architecture

The model is based on BERT (dicta-il/dictabert-large-char-menaked) with 3 new classification heads:
- The BERT backbone is frozen during training
- Only the 3 new heads are trained
- Uses CrossEntropyLoss for all heads

## Data Preparation

```bash
# Download data
wget "https://huggingface.co/datasets/thewh1teagle/phonikud-phonemes-data/resolve/main/knesset_phonemes_v1.txt.7z"
7z x knesset_phonemes_v1.txt.7z

# Process data
python -m src.prepare_data
```

This will:
- Deduplicate diacritics (e.g., QAMATS -> PATAH, TSERE -> SEGOL)
- Keep DAGESH only for בכךפףו
- Keep only 6 main vowels
- Output: `renikud_data_v1.txt`

## Training

```bash
# Basic training
python -m src.train --device cuda --epochs 10 --batch_size 32

# With custom parameters
python -m src.train \
  --device cuda \
  --epochs 20 \
  --batch_size 64 \
  --learning_rate 5e-3 \
  --early_stopping_patience 3 \
  --checkpoint_interval 9000
```

## Project Structure

```
src/
├── model/
│   ├── dicta_model.py       # Base BERT model (from phonikud)
│   └── renikud_model.py     # New RenikudModel with 3 heads
├── constants.py             # Vowel classes and diacritic definitions
├── data.py                  # Dataset and collator
├── train.py                 # Training script
├── prepare_data.py          # Data preprocessing
└── config.py                # Training configuration
```

## Key Simplifications vs Original Phonikud

- **No matres lectionis** handling
- **No hatama/vocal_shva/prefix** features
- **3 separate heads** instead of combined nikud classifier (28 classes)
- **Minimal diacritics**: 6 vowels + dagesh + sin dot only
- **Conditional predictions**: dagesh only for בכךפףו, sin only for ש

## Model Usage

```python
from transformers import AutoTokenizer
from src.model.renikud_model import RenikudModel

# Load model and tokenizer
model = RenikudModel.from_pretrained("path/to/checkpoint")
tokenizer = AutoTokenizer.from_pretrained("path/to/checkpoint")

# Predict
sentences = ["שלום עולם"]
results = model.predict(sentences, tokenizer)
print(results)  # ['שָׁלוֹם עוֹלָם']
```

## Training Configuration

See `src/config.py` for all available options:
- `model`: Base model to load (default: dicta-il/dictabert-large-char-menaked)
- `device`: cuda, cuda:1, cpu, mps
- `batch_size`: Batch size (default: 32)
- `epochs`: Number of epochs (default: 20)
- `learning_rate`: Learning rate (default: 5e-3)
- `early_stopping_patience`: Early stopping patience (default: 3)
- `checkpoint_interval`: Save checkpoint every N steps (default: 9000)
- `val_split_num`: Number of validation lines (default: 250)

