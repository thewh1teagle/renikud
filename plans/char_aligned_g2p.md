# Character-Aligned Hebrew G2P — Problem Brief

## Background

We have a working Hebrew G2P model (`renikud-v5`) that converts unvocalized Hebrew sentences to IPA strings. It reaches **91% word accuracy** on the Phonikud benchmark (surpassing the Phonikud teacher at 86.1%), trained on ~1M sentences of IPA transcriptions derived from ASR data plus the Knesset phonemes dataset.

Despite high accuracy, the model **hallucinates** — it occasionally emits phonemes not grounded to any specific input character. This is a structural limitation of CTC: the alignment between input characters and output phonemes is implicit and uncontrolled.

## Goal

Build a new model with **per-character grounding**: each Hebrew letter predicts its own IPA output. This eliminates hallucination by construction — the model cannot emit a phoneme that isn't tied to an input character.

---

## Existing Model Architecture

- **Encoder:** `dicta-il/dictabert-large-char-menaked` (300M param character-level BERT)
- **Decoder:** CTC over a fixed IPA character vocabulary
- **Pipeline:** unvocalized Hebrew text → BERT encoder → linear projection → upsample (repeat_interleave, factor=2) → slot embedding → CTC classifier → IPA string
- **Problem:** CTC alignment is a black box. The model has freedom to drift — phonemes are not grounded to specific input positions.

---

## Dataset

### Knesset Phonemes (`knesset_phonemes_v1.txt`)

TSV format: `hebrew_text<TAB>ipa_text`, one sentence per line.

The Hebrew side has nikud (vocalization diacritics) in the raw file, but **nikud is stripped** before use — the model only ever sees plain א-ת.

Example:
```
אֲבַקֵּשׁ מֵ|חַבְרֵי הַ|כְּנֶ֫סֶת	ʔavakˈeʃ meχavʁˈej haknˈeset
```

The `|` characters in the Hebrew side are word-boundary markers (artifact of the source, stripped during preprocessing).

At inference time input is **plain unvocalized Hebrew**: `אבקש מחברי הכנסת`

---

## IPA Phoneme Inventory

### Stress (1)
- `ˈ` — stress mark (U+02C8)

### Vowels (5)
- `a`, `e`, `i`, `o`, `u`

### Consonants (24)
- `b` — Bet
- `v` — Vet, Vav
- `d` — Daled
- `h` — Hey
- `z` — Zain
- `χ` — Het, Haf
- `t` — Taf, Tet
- `j` — Yud
- `k` — Kuf, Kaf
- `l` — Lamed
- `m` — Mem
- `n` — Nun
- `s` — Sin, Samekh
- `f` — Fey
- `p` — Pey
- `ts` — Tsadik
- `tʃ` — Tsadik with Geresh
- `w` — Vav (foreign words)
- `ʔ` — Alef/Ayin (U+0294)
- `ɡ` — Gimel (U+0261)
- `ʁ` — Resh (U+0281)
- `ʃ` — Shin (U+0283)
- `ʒ` — Zain with Geresh (U+0292)
- `dʒ` — Gimel with Geresh

---

## Proposed Architecture: Per-Character Classifier

### Core Idea

Each Hebrew letter (א-ת, plain unvocalized) independently predicts three things:

| Head | Output space | Notes |
|---|---|---|
| **Consonant** | letter-specific fixed set (see below) | includes `∅` for silent |
| **Vowel** | {a, e, i, o, u, ∅} | vowel following this consonant |
| **Stress** | {yes, no} | is this syllable stressed |

Output assembly per character: `[ˈ if stressed] + consonant + vowel`
Final IPA = concatenation of all character outputs left to right.

### Vowel Placement Convention

**Vowel belongs to the preceding consonant** (standard Hebrew CV syllable structure). So each letter predicts the vowel that follows it phonetically.

Special cases:
- **ו** — can be consonant (`v`, `w`) + following vowel, OR it IS the vowel (`u`, `o`) — modeled by including `u`/`o` in the consonant head options for ו, with vowel=∅
- **י** — can be consonant (`j`) + following vowel, OR it IS the vowel (`i`) — same treatment

### Grapheme → Possible Consonant Outputs

Each letter has a **fixed, constrained set** of possible consonant outputs. The classifier only picks within this set — hallucination of out-of-vocabulary phonemes is impossible.

```
א → {ʔ, ∅}
ב → {b, v}
ג → {ɡ, dʒ}
ד → {d}
ה → {h, ∅}
ו → {v, w, u, o}       # u/o = "this letter is the vowel"
ז → {z, ʒ}
ח → {χ}
ט → {t}
י → {j, i}             # i = "this letter is the vowel"
כ/ך → {k, χ}
ל → {l}
מ/ם → {m}
נ/ן → {n}
ס → {s}
ע → {ʔ, ∅}
פ/ף → {p, f}
צ/ץ → {ts, tʃ}
ק → {k}
ר → {ʁ}
ש → {ʃ, s}
ת → {t}
```

### Why This Eliminates Hallucination

- Each output token is tied to a specific input character position
- The consonant classifier is constrained to the known possible outputs for that letter
- The vowel classifier is constrained to {a, e, i, o, u, ∅}
- No CTC, no free alignment — strict left-to-right character-by-character generation

---

## Key Design Decisions

1. **Input:** plain unvocalized Hebrew (א-ת only, 22 base letters + 5 final forms). No nikud, no dagesh, no geresh in the input.
2. **No nikud at any stage** — not for training labels, not for inference. Alignment is learned purely from (unvocalized Hebrew, IPA) pairs at sentence level.
3. **No bootstrapping from existing CTC model.**
4. **Alignment is implicit** — the model must learn which vowel belongs to which letter from sentence-level supervision only. Context from BERT helps resolve ambiguity (e.g. ספר → book vs. counted vs. barber).
5. **Multi-task per character:** three prediction heads sharing the same BERT hidden state at each character position.

---

## Open Problem: Alignment Supervision

The dataset provides sentence-level pairs only:
```
אבקש  →  ʔavakˈeʃ
```

There are no explicit per-character labels. The core research question is:

**How do you train per-character classifiers from sentence-level (Hebrew, IPA) pairs without explicit alignment?**

Approaches to consider:
- **Monotone seq2seq with hard attention** — decoder advances through input characters strictly left to right, emitting IPA per character. Alignment emerges from training.
- **Joint alignment + classification** — learn alignment and labels simultaneously (e.g. HMM-style or differentiable monotone alignment).
- **Other ideas welcome.**

The constraint that each letter's output is drawn from a small known set should make the alignment learning significantly easier than unconstrained seq2seq.

---

## Codebase Reference

- `src/constants.py` — IPA vocabulary, decoder vocab, token mappings
- `src/model.py` — existing CTC model (`HebrewG2PCTC`)
- `src/train.py` — training loop
- `src/prepare_data.py` — strips nikud, splits train/val
- `dataset/` — preprocessed Arrow datasets
- `knesset_phonemes_v1.txt` — raw TSV data (~900K sentences)
- Encoder: `dicta-il/dictabert-large-char-menaked` (character-level, already handles Hebrew script)
