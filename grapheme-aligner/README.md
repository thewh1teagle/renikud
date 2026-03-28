# grapheme-aligner

Language-agnostic grapheme-to-IPA aligner using EM + Viterbi DP.

Define possible phonemes per letter in a config file — the aligner learns probabilities from your corpus.

## Setup

```bash
uv sync
```

## Train

```bash
uv run src/train.py data.tsv models/hebrew.json --config configs/hebrew.yaml
```

Trains on `data.tsv` (format: `grapheme<TAB>ipa` per line) and saves the model.

## Infer

```bash
uv run src/infer.py data.tsv alignment.jsonl --model models/hebrew.json --config configs/hebrew.yaml
```

Outputs one JSON object per line: `{"sentence": [["letter", "ipa_chunk"], ...]}`

## New language

1. Copy `configs/hebrew.yaml`
2. Set `letters` with allowed starting IPA atoms per letter
3. Train on your corpus
