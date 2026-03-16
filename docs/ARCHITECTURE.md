# Architecture

## Goal

This project trains a Hebrew grapheme-to-phoneme (G2P) model that converts unvocalized Hebrew sentences into IPA strings.

The model is a **per-character classifier**: each Hebrew letter independently predicts a (consonant, vowel, stress) triple — every letter gets exactly one output slot.

## Design Principles

- One output slot per Hebrew letter — no alignment ambiguity at inference time.
- Per-letter consonant masking — impossible consonants are zeroed out before argmax.
- Keep the code path short and explicit.
- Strict preprocessing so invalid labels are caught early.

## Data Flow

1. Raw source data is stored as TSV: `hebrew_text<TAB>ipa_text`
2. `src/align_data.py` runs a DP aligner to produce per-character alignments, saved as JSONL: `{"hebrew": [["char", "ipa_chunk"], ...]}`
3. `src/prepare_tokens.py` tokenizes the Hebrew sentence with DictaBERT and maps per-character IPA labels to token positions, saved as an Arrow dataset.
4. `src/train.py` trains the model with a plain PyTorch loop.
5. `src/infer.py` runs per-character prediction from a saved checkpoint.

## Project Layout

- `src/` — application code: alignment, tokenization, modeling, training, and inference
- `dataset/` — alignment JSONL files and tokenized Arrow caches
- `docs/` — design and operational documentation
- `plans/` — research notes and experiments
- `scripts/` — standalone evaluation and benchmarking scripts

## Vocabulary

Defined in `src/constants.py`.

**Consonants** (25 + ∅): `∅ b v d h z χ t j k l m n s f p ts tʃ w ʔ ɡ ʁ ʃ ʒ dʒ`

**Vowels** (6 + ∅ + aχ): `∅ a e i o u aχ`
- `∅` means no vowel (consonant-only syllable or silent letter)
- `aχ` is a special token for word-final ח coda (e.g. `שמח` → `samˈeaχ`)

**Stress**: binary — 0 (none) or 1 (ˈ before vowel)

## Model

Defined in `src/model.py` as `HebrewG2PClassifier`.

Pipeline:

1. Encode with `dicta-il/dictabert-large-char-menaked` (300M param character-level BERT)
2. Three linear classification heads on top of encoder hidden states:
   - **Consonant head**: projects to 25 consonant classes + ∅
   - **Vowel head**: projects to 6 vowel classes + ∅ + aχ
   - **Stress head**: projects to 2 classes (none / stressed)
3. Per-letter consonant masking: before argmax, logits for impossible consonants are set to `-1e9` using `HEBREW_LETTER_TO_ALLOWED_CONSONANTS`

## Label Alignment

`src/prepare_tokens.py` maps per-character IPA labels to tokenizer token positions using `offset_mapping`. Only single-character tokens (offset `end - start == 1`) get labels — CLS, SEP, and multi-char tokens receive `IGNORE_INDEX = -100`.

Punctuation, digits, and Latin characters are skipped when walking the alignment pairs, so they don't cause offset drift into Hebrew letter positions.

## Training

Defined in `src/train.py`.

Key features:

- Discriminative learning rates: separate LRs for encoder vs. heads via `parameter_groups()`
- Cosine schedule with linear warmup (`--warmup-steps`)
- Optional encoder freeze for the first N steps (`--freeze-encoder-steps`)
- Weight-only initialization via `--init-from-checkpoint` (loads weights, resets optimizer)
- Mixed precision (`fp16`) with grad scaler
- Checkpoints saved as `model.safetensors` + `train_state.json`; oldest pruned beyond `--save-total-limit`

## Inference

Defined in `src/infer.py`.

1. Load checkpoint weights (supports `model.safetensors` and `pytorch_model.bin`)
2. Tokenize Hebrew input with the encoder tokenizer
3. Run forward pass with per-letter consonant masking
4. For each single-character token that is a Hebrew letter, assemble: `[consonant][ˈ][vowel]`
5. Non-Hebrew characters (punctuation, spaces) are passed through as-is
