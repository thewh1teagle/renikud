# Architecture

## Problem

Add nikud (vowel diacritics) to unvocalized Hebrew text. Hebrew is written without vowels — the same consonant skeleton can have multiple valid vocalizations depending on context. The model must recover the diacritics purely from the consonant skeleton and surrounding context.

## Core Idea

The model frames diacritization as **per-character classification**. Every Hebrew letter independently predicts a `(nikud, shin)` pair. Non-Hebrew characters (spaces, punctuation, digits, Latin) are passed through unchanged.

This works because Hebrew diacritics attach to individual letters — each letter carries at most one nikud combo and optionally a shin/sin dot. The model learns context-dependent choices (e.g. shva vs. no nikud) from the encoder.

## Model

`NikudModel` in `src/model.py`:

1. **Encoder** — ModernBERT-style encoder initialized from scratch with a custom 104-token Hebrew character vocabulary. See `src/encoder.py` for the exact configuration (~19M params).
2. **Two classification heads** — each head sees the encoder hidden state:
   - **Nikud head** → `hidden` → 28 classes (no nikud, shva, hataf variants, vowels, dagesh+vowel combos, qamats qatan variants)
   - **Shin head** → `hidden` → 2 classes (shin dot / sin dot) — only meaningful for ש

## Tokenizer

Custom character-level tokenizer (`src/tokenization.py`) with a 104-token vocab:
- 5 special tokens: `[PAD] [CLS] [SEP] [UNK] [MASK]`
- Hebrew letters א–ת (including final forms) + maqaf, geresh, gershayim
- ASCII lowercase, digits, punctuation, space

Each character is its own token. The tokenizer strips diacritics via `StripAccents` — nikud is handled separately before tokenization.

## Label Vocabulary

**Nikud** (28 classes):

| Class | Unicode | Description |
|---|---|---|
| `` | — | no nikud |
| `ְ` | 05B0 | shva |
| `ֱ` | 05B1 | hataf segol |
| `ֲ` | 05B2 | hataf patah |
| `ֳ` | 05B3 | hataf qamats |
| `ִ` | 05B4 | hiriq |
| `ֵ` | 05B5 | tsere |
| `ֶ` | 05B6 | segol |
| `ַ` | 05B7 | patah |
| `ָ` | 05B8 | qamats |
| `ֹ` | 05B9 | holam |
| `ֺ` | 05BA | holam haser |
| `ּ` | 05BB | qubuts |
| `ּ` | 05BC | dagesh |
| `ְּ` | 05B0+05BC | shva + dagesh |
| `ֱּ` | 05B1+05BC | hataf segol + dagesh |
| `ֲּ` | 05B2+05BC | hataf patah + dagesh |
| `ֳּ` | 05B3+05BC | hataf qamats + dagesh |
| `ִּ` | 05B4+05BC | hiriq + dagesh |
| `ֵּ` | 05B5+05BC | tsere + dagesh |
| `ֶּ` | 05B6+05BC | segol + dagesh |
| `ַּ` | 05B7+05BC | patah + dagesh |
| `ָּ` | 05B8+05BC | qamats + dagesh |
| `ֹּ` | 05B9+05BC | holam + dagesh |
| `ֺּ` | 05BA+05BC | holam haser + dagesh |
| `ּּ` | 05BB+05BC | qubuts + dagesh |
| `ׇ` | 05C7 | qamats qatan |
| `ׇּ` | 05BC+05C7 | dagesh + qamats qatan |

**Shin** (2 classes): shin dot (`׃05C1`) / sin dot (`05C2`)

## Diacritic Normalization

Raw Hebrew text may encode multi-diacritic sequences in different codepoint orders. We normalize once at the entry point using `sort_diacritics()` (`src/nikud.py`), which sorts combining marks after each base letter by codepoint value. `NIKUD_CLASSES` multi-char entries are defined in the same codepoint-sorted order. This means:

- `extract_labels` does a plain dict lookup — no re-sorting needed
- `decoder.py` sorts `shin + nikud` on output to match the normalized form

## Data Pipeline

```
knesset_phonemes_v1.txt.7z  (downloaded from HuggingFace)
  → scripts/prepare_dataset.py   extract left (Hebrew) column, strip | markers → knesset.txt
  → scripts/split_dataset.py     shuffle + split → train.txt / val.txt
  → scripts/prepare_tokens.py    sort_diacritics → strip nikud → tokenize → extract labels → Arrow dataset
  → src/train.py                 training loop
```

Label extraction (`scripts/prepare_tokens.py`): for each Hebrew letter, collect the diacritic codepoints immediately following it in the vocalized text. Strip shin/sin dot separately, look up the remainder in `NIKUD_TO_ID`. The tokenizer receives the stripped (unvocalized) text; labels are aligned to token positions via `offset_mapping`.
