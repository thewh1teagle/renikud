"""
Debug script: align -> tokenize -> infer for a specific sentence.
Run from project root: uv run plans/debug_sentence/debug_sentence.py
"""

import sys
import json
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import regex as re
import torch
from safetensors.torch import load_file

from aligner.align import align_word
from phonology import (
    ORTHOGRAPHIC_MARKERS, normalize_graphemes,
    CONSONANT_TO_ID, VOWEL_TO_ID, ID_TO_CONSONANT, ID_TO_VOWEL,
    STRESS_NONE, is_hebrew_letter, chunk_to_labels,
)
from tokenization import load_tokenizer, id_to_token
from model import G2PModel
from decoder import decode
from constants import MAX_LEN

# ── Sentence ─────────────────────────────────────────────────────────────────
SENTENCE_HEB_RAW = "צ׳יפס צ'יפס ארה״ב ארה\"ב"
# Hardcoded IPA for alignment demo (what we'd expect after G2P)
SENTENCE_IPA = "tʃˈips tʃˈips ʔaʁahˈav ʔaʁahˈav"

# Latest checkpoint
CHECKPOINT = Path(__file__).parent.parent.parent / "outputs/g2p-classifier/best"

# ── Helpers ───────────────────────────────────────────────────────────────────
MARKERS = "".join(ORTHOGRAPHIC_MARKERS)
HEB_RE = re.compile(rf"[^\u05d0-\u05ea{re.escape(MARKERS)}]")
IPA_RE = re.compile(r"[^abdefghijklmnoprstuvwzɡʁʃʒʔˈχ]")
NIKUD_RE = re.compile(r"[\p{M}|]")


def normalize(text: str) -> str:
    text = NIKUD_RE.sub("", unicodedata.normalize("NFD", text))
    text = normalize_graphemes(text)
    text = text.replace("-", " ")
    return text


def sep(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Normalize + Align
# ══════════════════════════════════════════════════════════════════════════════
sep("STEP 1: NORMALIZE & ALIGN")

heb_norm = normalize(SENTENCE_HEB_RAW)
ipa_norm = SENTENCE_IPA.replace("-", " ")

print(f"Raw Hebrew  : {SENTENCE_HEB_RAW!r}")
print(f"Norm Hebrew : {heb_norm!r}")
print(f"IPA         : {ipa_norm!r}")

heb_words = heb_norm.split()
ipa_words = ipa_norm.split()
print(f"\nWord count  : heb={len(heb_words)}  ipa={len(ipa_words)}")

full_alignment = []
for hw, iw in zip(heb_words, ipa_words):
    heb_core = HEB_RE.sub("", hw)
    ipa_core = IPA_RE.sub("", iw)
    print(f"\n  Word: {hw!r} (core={heb_core!r}) → IPA: {iw!r} (core={ipa_core!r})")
    aligned = align_word(heb_core, ipa_core)
    if aligned is None:
        print("    ❌ ALIGNMENT FAILED")
    else:
        for pair in aligned:
            print(f"    {pair[0]!r:6s} → {pair[1]!r}")
        if full_alignment:
            full_alignment.append((" ", " "))
        full_alignment.extend(aligned)

print(f"\nFull alignment ({len(full_alignment)} pairs):")
for h, p in full_alignment:
    print(f"  {h!r:6s} → {p!r}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Tokenize + label assignment
# ══════════════════════════════════════════════════════════════════════════════
sep("STEP 2: TOKENIZE & LABEL ASSIGNMENT")

tokenizer = load_tokenizer()
encoding = tokenizer(
    heb_norm,
    truncation=True,
    max_length=512,
    return_offsets_mapping=True,
)
input_ids = encoding["input_ids"]
offsets   = encoding["offset_mapping"]
tokens    = [tokenizer.convert_ids_to_tokens([tid])[0] for tid in input_ids]

print(f"Sentence: {heb_norm!r}")
print(f"Tokens ({len(tokens)}): {tokens}")
print()

from constants import IGNORE_INDEX

# Replicate prepare_tokens logic
char_labels: dict[int, tuple] = {}
align_iter = iter(full_alignment)
for char_pos, orig_char in enumerate(heb_norm):
    if not is_hebrew_letter(orig_char) and orig_char not in ORTHOGRAPHIC_MARKERS and orig_char != " ":
        continue
    try:
        _, chunk = next(align_iter)
    except StopIteration:
        break
    if is_hebrew_letter(orig_char) or orig_char in ORTHOGRAPHIC_MARKERS:
        char_labels[char_pos] = chunk_to_labels(chunk)

print("Char-level labels (char_pos → (consonant, vowel, stress)):")
for pos, lbl in sorted(char_labels.items()):
    print(f"  [{pos}] {heb_norm[pos]!r}  → {lbl}")

print()
print(f"{'Tok':>4}  {'Token':12s}  {'Offset':10s}  {'Char':6s}  {'Consonant':12s}  {'Vowel':8s}  {'Stress'}")
print("-" * 70)
for tok_idx, (start, end) in enumerate(offsets):
    tok = tokens[tok_idx]
    char = heb_norm[start] if start < len(heb_norm) else "—"
    if end - start == 1 and start in char_labels:
        c, v, s = char_labels[start]
        cid = CONSONANT_TO_ID.get(c, IGNORE_INDEX)
        vid = VOWEL_TO_ID.get(v, IGNORE_INDEX)
        print(f"  {tok_idx:>2}  {tok:12s}  {str((start,end)):10s}  {char!r:6s}  {c or '∅':12s} ({cid:2d})  {v or '∅':8s} ({vid})  {s}")
    else:
        print(f"  {tok_idx:>2}  {tok:12s}  {str((start,end)):10s}  {char!r:6s}  (skip)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Inference
# ══════════════════════════════════════════════════════════════════════════════
sep(f"STEP 3: INFERENCE (checkpoint: {CHECKPOINT.name})")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = G2PModel()
state = load_file(str(CHECKPOINT / "model.safetensors"), device="cpu")
model.load_state_dict(state)
model.to(device).eval()
print(f"Model loaded from {CHECKPOINT}")

enc = tokenizer(
    heb_norm,
    truncation=True,
    max_length=MAX_LEN,
    return_offsets_mapping=True,
    return_tensors="pt",
)
offset_mapping_raw = enc.pop("offset_mapping")[0].tolist()
input_ids_t  = enc["input_ids"].to(device)
attn_mask    = enc["attention_mask"].to(device)

with torch.no_grad():
    out = model(
        input_ids=input_ids_t,
        attention_mask=attn_mask,
        tokenizer_vocab=id_to_token(tokenizer),
    )

c_logits = out["consonant_logits"][0]   # [S, C]
v_logits = out["vowel_logits"][0]       # [S, V]
s_logits = out["stress_logits"][0]      # [S, 2]

print(f"\nLogit shapes: consonant={tuple(c_logits.shape)}  vowel={tuple(v_logits.shape)}  stress={tuple(s_logits.shape)}")

import torch.nn.functional as F

c_probs = F.softmax(c_logits, dim=-1)
v_probs = F.softmax(v_logits, dim=-1)
s_probs = F.softmax(s_logits, dim=-1)

print()
print(f"{'Tok':>4}  {'Token':12s}  {'Top-Consonant':22s}  {'Top-Vowel':18s}  {'Stress%'}")
print("-" * 85)
for tok_idx in range(len(tokens)):
    tok = tokens[tok_idx]
    top_c_id  = c_probs[tok_idx].argmax().item()
    top_v_id  = v_probs[tok_idx].argmax().item()
    stress_p  = s_probs[tok_idx, 1].item()
    top_c     = ID_TO_CONSONANT[top_c_id]
    top_v     = ID_TO_VOWEL[top_v_id]
    top_cp    = c_probs[tok_idx, top_c_id].item()
    top_vp    = v_probs[tok_idx, top_v_id].item()
    print(f"  {tok_idx:>2}  {tok:12s}  {top_c or '∅':8s} ({top_cp:.2f})    {top_v or '∅':6s} ({top_vp:.2f})    {stress_p:.2f}")

# Final decoded IPA
ipa_out = decode(
    text=heb_norm,
    offset_mapping=offset_mapping_raw,
    consonant_logits=c_logits,
    vowel_logits=v_logits,
    stress_logits=s_logits,
)
print(f"\nDecoded IPA : {ipa_out}")
print(f"Expected IPA: {SENTENCE_IPA}")
