# Evaluation

## Metrics

All evaluation uses [jiwer](https://github.com/jitsi/jiwer) to compute:

- **CER** (Character Error Rate) — primary metric, measures phoneme-level accuracy
- **WER** (Word Error Rate) — secondary metric, measures per-word accuracy
- **Acc** (Word Accuracy) — `1 - WER`, used for high-level comparison

### During Training

`src/evaluate.py` computes CER and WER on the validation split at each eval step via the Hugging Face `Trainer`. Predictions are truncated to `input_lengths` before CTC decoding to avoid scoring padded frames.

### Standalone Benchmark

`scripts/benchmark.py` evaluates a checkpoint against an external ground-truth file:

```
uv run scripts/benchmark.py --checkpoint outputs/g2p-v1/checkpoint-18000 --gt gt.tsv
```

Optional `--ignore-punct` strips `.,?!` from both references and predictions before scoring.

## Benchmark Dataset

The benchmark is [heb-g2p-benchmark](https://github.com/thewh1teagle/heb-g2p-benchmark) — 100 Hebrew sentences specifically engineered to stress-test G2P systems. The sentences cover:

- Male vs. female verb forms
- Stress placement variation
- Homographs (words with different readings depending on context)

Download:
```
wget https://raw.githubusercontent.com/thewh1teagle/heb-g2p-benchmark/refs/heads/main/gt.tsv
```

## Target: Outperform Phonikud

Our goal is to surpass [Phonikud](https://github.com/thewh1teagle/phonikud), the teacher model we use to generate training data.

| Model | Acc | WER | CER | Stress WER |
|---|---|---|---|---|
| Phonikud (teacher) | 86.1% | 0.14 | 0.04 | 0.09 |
| **Ours (best)** | **91%** | — | — | — |

We have surpassed Phonikud — the model now exceeds its own training signal. Achieved by augmenting training data with ~1M sentences of IPA transcriptions derived from ASR.
