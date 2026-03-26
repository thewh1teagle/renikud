# Renikud: The Journey

A year+ of building a Hebrew G2P model from scratch — no off-the-shelf solution, no shortcuts.
This document traces every major experiment so a reader can understand the full arc.

---

## The Problem

Hebrew is written without vowels. The same string can have multiple valid pronunciations depending
on context. The goal: given unvocalized Hebrew text, output IPA pronunciation automatically.

This is harder than it sounds. Most existing tools either:
- Require nikud (vowel marks) as input — not available in the wild
- Are proprietary, black-box, or require heavy HuggingFace runtime dependencies
- Don't produce IPA (only nikud), so you can't use ASR data directly

---

## V1 — BERT Frozen, 3 Heads, Nikud Output
**Branch:** `renikud-experiment-v1-bert-frozen-3heads`

### Architecture
- Encoder: `dicta-il/dictabert-large-char-menaked` (300M, frozen)
- 3 classification heads on top:
  - **Vowel head**: 7-class (SHVA, SEGOL, HIRIK, PATAH, HOLAM, QUBUTS + empty)
  - **Dagesh head**: binary (ב כ ך פ ף ו only)
  - **Sin head**: binary (ש only)
- Only the heads were trained; backbone frozen

### What Happened
Worked to some extent. But the output was nikud (Hebrew diacritic marks), not IPA.
This created a fundamental data bottleneck — nikud-annotated corpora are small.
Tried to build a nikud ASR pipeline to generate more data. It didn't work out.
The frozen encoder also severely limited what the heads could learn.

---

## V2 — BERT Unfrozen, 4 Heads, Add Stress
**Branch:** `renikud-experiment-v2-bert-4heads-stress`

### Architecture
- Same base: `dicta-il/dictabert-large-char`
- Added a **Stress head** (binary)
- Unfroze the encoder for end-to-end training

### What Happened
Slight improvement. But still predicting nikud, still data-limited.
The fundamental limitation wasn't the model architecture — it was the output format.
Nikud doesn't map cleanly to IPA, and nikud-annotated data is scarce.

---

## V3 — BERT, 5 Heads, Prefix Classifier, Constrained Decoding
**Branch:** `renikud-experiment-v3-bert-5heads-prefix`

### Architecture
- Same BERT base
- 5 heads: Vowel (8-class), Dagesh, Sin, Stress, **Prefix** (binary — marks morphological prefix boundaries)
- Added constrained decoding: certain heads only fire on specific letters

### What Happened
Constrained decoding was a key insight that carried forward to all later versions.
But still predicting nikud. The prefix head was interesting for NLP but irrelevant for TTS.
Results improved but the nikud ceiling was hit. Time to switch to IPA.

---

## V4 — CTC, IPA Output, DictaBERT Encoder
**Branch:** `renikud-experiment-v4-ctc-g2p`

### Architecture
- Encoder: `dicta-il/dictabert-large-char-menaked` (300M, character-level BERT)
- Linear projection → 2× upsample + slot embedding → **CTC head**
- Output: IPA strings directly

### What Happened
Big shift — now predicting IPA, not nikud. This unlocked:
- ASR corpora (IPA from audio transcription pipelines)
- Phonikud silver labels
- 500K Hebrew sentences from Knesset corpus

Reached **85.1% word accuracy** on heb-g2p-benchmark.

But there were real frustrations:
- CTC gives poor fine-grained control: multiple stress marks per word, pass-through tokens
  for punctuation were unreliable
- Data augmentation was needed constantly to fix CTC artifacts
- The upsample trick was a hack — CTC requires output length ≤ input length, so you
  had to upscale the encoder output to accommodate longer IPA sequences
- Not fun to work with

---

## V4.5 — DictaBERT Encoder, Per-Character Classification (IPA)

### Architecture
- Same per-character classification idea as V5/V6 (3 heads: consonant, vowel, stress)
- Encoder: `dicta-il/dictabert-large-char-menaked` (300M, HuggingFace pretrained)
- Tokenizer: same model's character-level tokenizer (841-token vocab including nikud chars)
- Consonant masking already in place

### Training Data
Beyond the Knesset/teacher-distilled data, the model was also trained on transcriptions
from ~2000 hours of Hebrew audio using an IPA ASR pipeline — giving it exposure to
real spoken Hebrew pronunciation patterns at scale.

### Benchmark Expansion
During this phase the benchmark was expanded from 100 to 250 sentences. The new sentences
were hard, ambiguous cases generated with Gemini 2.5 Pro (excellent at crafting tricky
Hebrew homophones and context-dependent words). IPA labels were produced by the model
then corrected manually. This harder benchmark became the standard for all subsequent
evaluation.

### What Happened
This was the bridge version — proved that per-character classification + IPA output
works well with the DictaBERT backbone. The architecture was solid but came with baggage:
- Runtime dependency on HuggingFace + the pretrained model weights
- 300M params, heavy to deploy
- Tokenizer tied to the menaked model's 841-token vocab

Good results, but the goal was always a self-contained model. This version showed the
approach was right — the next step was to replace the encoder with something trained
from scratch on our own vocabulary.

---

## V5 — NeoBERT Encoder, Per-Character Classification
**Branch:** `renikud-experiment-v5-neobert`

### Architecture (by Max)
- Dropped CTC entirely — switched to **per-character classification**
- Custom NeoBERT encoder (28-layer, 768-dim, RoPE, SwiGLU, RMSNorm, ~100M params)
  built from scratch in pure PyTorch
- 3 heads per token: consonant (26-class), vowel (8-class), stress (binary)
- Consonant masking: per-letter allowed set, impossible consonants → -1e9 before argmax

### Why Per-Character Classification Works for Hebrew
Hebrew has near one-to-one letter→phoneme correspondence. Each letter produces exactly
one consonant (or silence) and optionally a vowel and stress. The model learns the
exceptions (context-dependent allophones) from surrounding characters.

### What Happened
The model learned very fast. Per-character classification solved all the CTC pain.

However, `transformers` has no built-in architecture for this setup — a from-scratch
encoder with a custom vocab and custom classification heads isn't something you can
just config into HuggingFace. The result was that the code started to bloat:
custom model class, custom config, custom weight init, wiring around the HF trainer
abstractions. It worked but it was getting unwieldy.
- Exactly one stress per word (enforced at inference)
- Clean pass-through for non-Hebrew characters
- No length constraints, no upsample tricks
- Full control over decoding

---

## V6 — ModernBERT Encoder, Custom Tokenizer (Current Main)
**Branch:** main (`renikud-experiment-v6-modernbert-custom`)

**Training data**: 5M lines of silver IPA labels generated by the V4.5 DictaBERT model (the teacher).
The student (V6, 100M params) was trained entirely on the teacher's outputs — knowledge distillation
without explicit distillation loss, just cross-entropy on teacher predictions at scale.
Result: within 1.3% of the teacher on the 250-sample benchmark, with 3× fewer parameters.

### Architecture
See `docs/ARCHITECTURE.md` for full detail.

- Encoder: ModernBERT-base (22-layer, 768-dim, RoPE, Flash Attention)
  — initialized **from scratch** with a custom 104-token Hebrew character vocabulary
  — no HuggingFace pretrained weights at runtime
- Custom character-level tokenizer (104 tokens, built from code)
- 3 classification heads: consonant (26-class), vowel (8-class), stress (binary)
- Consonant masking from `HEBREW_LETTER_TO_ALLOWED_CONSONANTS`
- One-stress-per-word enforced at inference via `_best_stress_per_word()`
- DP aligner (`data_align.py`) constrains label assignment to phonetically valid chunks
- ONNX export with embedded metadata (vocab, consonant mask, cls/sep IDs) → 113MB int8

### Training
- Pre-trained on ~5M teacher-distilled rows (teacher = DictaBERT 300M via Phonikud)
- Fine-tuned on ~1M mixed rows: 50k from train.tsv + 907k Vox-Knesset ASR +
  81k YouTube + 13k manual + 94 dj/j entries
- Fine-tuning started from checkpoint-456000 (85.1% acc)

### Results
| Stage | Word Acc |
|---|---|
| Pre-training (checkpoint-456000) | 85.1% |
| Fine-tune checkpoint-8500 | 85.3% |
| Fine-tune checkpoint-14000 | 85.3% |
| Fine-tune checkpoint-23000 | **85.7%** |
| ONNX int8 (100-sample subset) | 88.6% |

Teacher model (DictaBERT 300M): **87%**

### Key Properties
- 3× smaller than teacher (100M vs 300M params)
- Self-contained: no HuggingFace dependency at runtime
- 113MB int8 ONNX, ~38ms G2P on CPU
- Deployed in `hesay` binary: ~195MB self-contained Hebrew TTS (G2P + Piper TTS embedded)
- Supports mixed Hebrew/English input

### Still In Progress
- Fine-tuning continues — 85.7% vs teacher's 87%, gap is 1.3%
- With 1/3 the parameters and no pretrained weights, closing to within 1.3% is already
  a strong result; beating the teacher is the next milestone

---

## Key Lessons

1. **Output format matters more than architecture** — switching from nikud to IPA
   unlocked orders of magnitude more training data (ASR corpora)

2. **Per-character classification beats CTC for Hebrew** — the near one-to-one
   letter→phoneme structure makes it the natural fit; CTC's length constraint and
   blank collapse create artifacts that are painful to debug

3. **Constrained decoding is load-bearing** — the consonant mask isn't optional;
   without it the model predicts phonetically impossible outputs (e.g. ל → b)

4. **Training from scratch beats fine-tuning a pretrained model** — custom vocab,
   no dependency on external tokenizer/model files, full control

5. **Knowledge distillation works** — 100M student trained on 5M teacher outputs
   gets within 1.3% of the 300M teacher on a held-out benchmark
