"""
Train the EM aligner on a TSV corpus and save the learned model.

Usage:
    python train.py vox-knesset-ipa-v1.tsv hebrew.json --config configs/hebrew.yaml
    python train.py vox-knesset-ipa-v1.tsv hebrew.json --config configs/hebrew.yaml --iterations 15
"""

import argparse
from tqdm import tqdm
from ipa import load_config, parse_sentence
from em import run_em, save_model


def main():
    parser = argparse.ArgumentParser(description="Train EM grapheme-to-IPA aligner")
    parser.add_argument("input", help="Input TSV (grapheme<TAB>ipa)")
    parser.add_argument("model", help="Output model JSON path")
    parser.add_argument("--config", default="configs/hebrew.yaml")
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config(args.config)
    atoms = cfg["ipa_atoms"]
    max_n = cfg["max_tokens_per_letter"]
    allowed = {letter: [str(a) for a in starts] for letter, starts in cfg["letters"].items()}

    print(f"Loading {args.input}...")
    with open(args.input, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    pairs = [line.split("\t") for line in lines if line.count("\t") == 1]

    print(f"Loaded {len(pairs):,} sentences. Parsing...")
    corpus = []
    for text, ipa in tqdm(pairs):
        parsed = parse_sentence(text, ipa, atoms)
        if parsed is not None:
            corpus.append(parsed)
    print(f"Parsed {len(corpus):,} sentences.")

    log_probs = run_em(corpus, allowed, max_n, args.iterations)
    save_model(log_probs, args.model)
    print(f"Model saved to {args.model}")


if __name__ == "__main__":
    main()
