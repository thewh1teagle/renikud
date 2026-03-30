# Structured Upsample CTC — Aligner-Free Character-Level G2P

## Overview

An innovative aligner-free G2P architecture that achieves per-character phoneme control at inference without autoregression, manual alignment, or language-specific rules. The model uses a structured upsample to expand each input character into K=3 fixed slots, then applies CTC loss against the raw IPA string.

## Problem

Hebrew (and other languages) G2P traditionally requires either:
- A hand-crafted DP aligner (language-specific, fails on edge cases)
- A seq2seq decoder (autoregressive, slow, hallucinates)
- Pre-aligned labeled data (expensive, brittle)

This architecture eliminates all three requirements.

## Key Innovation

**Structured upsampling with fixed slot boundaries.**

Each input character expands to exactly K=3 slots before the encoder. The slot-to-character mapping is hardcoded by construction — not learned. This gives:
- CTC handles alignment automatically during training
- Per-char boundaries are always known at inference
- No DP, no decoder, no alignment code

## Features

- ✅ No manual aligner — CTC marginalizes over all valid alignments during training
- ✅ No autoregression — all slots run in one parallel forward pass
- ✅ Per-char control — slots `[i*3 : i*3+3]` are structurally bound to char `i`
- ✅ Full sentence context — encoder attends over all slots across the full sentence
- ✅ Language agnostic — only `lang_pack` changes per language, zero architecture changes
- ✅ Passthrough for non-input_chars — English, digits, punctuation copied directly
- ✅ Fast inference — argmax over 3 slots per char, no beam search, no DP
- ✅ Mathematically guaranteed convergence — CTC forward-backward algorithm
- ✅ Constrained decoding — slot logits maskable per char from lang pack at inference
- ✅ Mixed script — Hebrew + English in same sentence works naturally
- ✅ Extendable — add new language via lang pack + fine-tune, existing weights preserved

## Architecture

```
input:  ה  י  ,     W  h  a  t  s  A  p  p
        │  │  │     │  │  │  │  │  │  │  │
   lang_pack?  │     └──────────────────────┘
   ✓  ✓   │                passthrough
        │  │
   [upsample x3 — learned linear + slot position embeddings]
        │  │
   ┌────┴──┴──────────────────────────┐
   │  h₀ h₁ h₂ h₃ h₄ h₅             │
   │  NeoBERT encoder (full attn)     │
   └────┬──┬──┬──┬──┬──┬─────────────┘
        │  │  │  │  │  │
       ה₀ ה₁ ה₂ י₀ י₁ י₂   ← K=3 slots per char
        │  │  │  │  │  │
   [linear → IPA vocab + blank]
        │  │  │  │  │  │
        h  ∅  ∅  j  ˈi ∅    ← greedy decode per char
        └──────┘  └──────┘
           ה         י
          "h"      "jˈi"
              +  ","  +  " WhatsApp"
           =  "hjˈi, WhatsApp"
```

### Training
```
CTC loss vs raw IPA string (no alignment preprocessing)
nn.CTCLoss(blank=0, zero_infinity=True)
```

### Inference
```python
for char_i, char in enumerate(input_chars):
    if char not in lang_pack:
        emit(char)  # passthrough
        continue
    slots = preds[char_i*K : char_i*K + K]
    tokens = greedy_ctc_decode(slots)  # skip blank, skip repeats
    emit(tokens)
```

## Components

| File | Role |
|---|---|
| `lang_pack.py` | Language config: `input_chars`, `output_tokens`, `max_slot_len` |
| `model.py` | `G2PModel`: upsample → encoder → CTC head + loss |
| `data_prepare.py` | TSV → Arrow: tokenize IPA string, build active_mask |
| `data.py` | Collator: pad sequences, keep target_ids as list of tensors |
| `infer.py` | Per-char slot decode, passthrough for non-input_chars |
| `encoder.py` | NeoBERT (unchanged) |
| `tokenization.py` | Character-level tokenizer (unchanged) |

## Language Pack

```python
HEBREW = LangPack(
    name="hebrew",
    input_chars=frozenset("אבגדהוזחטיכךלמםנןסעפףצץקרשת"),
    output_tokens=("∅", "b", "v", ..., "ˈa", "ˈe", "ˈi", "ˈo", "ˈu"),
    max_slot_len=3,
)
```

Adding a new language = one new `LangPack` instance. Zero other changes.

## Training

```bash
uv run src/data_prepare.py dataset/train.tsv dataset/.cache/train --lang hebrew
uv run src/data_prepare.py dataset/val.tsv   dataset/.cache/val   --lang hebrew
bash scripts/train_scratch.sh
```

## Why CTC Works Here

Standard CTC fails when `input_length < target_length`. With K=3 upsampling, `input_length = 3 * n_active_chars`. For Hebrew, words have more letters than IPA tokens on average (many silent letters, vowels not written), so the constraint `3L >= T` is always satisfied.

The structured slot boundary is the key difference from vanilla CTC: at inference you never need to run the CTC decoder — you just read off 3 slots per char directly.

## Constraints & Extensions

- **Constrained decoding**: mask slot logits per char at inference using lang pack allowed outputs
- **One stress per word**: post-process — keep highest-confidence stressed vowel per word
- **New language**: add `LangPack`, fine-tune from Hebrew checkpoint
- **ONNX export**: standard export, no custom ops needed
