"""
Align a TSV corpus using a pre-trained model. No EM — just Viterbi.

Usage:
    python infer.py data.tsv alignment.jsonl --model hebrew.json --config configs/hebrew.yaml
"""

import argparse
import json
from tqdm import tqdm
from ipa import load_config, align_sentence
from em import load_model


def main():
    parser = argparse.ArgumentParser(description="Align TSV using a pre-trained model")
    parser.add_argument("input", help="Input TSV (grapheme<TAB>ipa)")
    parser.add_argument("output", help="Output JSONL")
    parser.add_argument("--model", required=True, help="Trained model JSON (from train.py)")
    parser.add_argument("--config", default="configs/hebrew.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    atoms = cfg["ipa_atoms"]
    max_n = cfg["max_tokens_per_letter"]
    allowed = {letter: [str(a) for a in starts] for letter, starts in cfg["letters"].items()}

    print(f"Loading model from {args.model}...")
    log_probs = load_model(args.model)

    print(f"Loading {args.input}...")
    with open(args.input, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    pairs = [line.split("\t") for line in lines if line.count("\t") == 1]

    failures_path = args.output.replace(".jsonl", "_failures.txt")
    aligned_count = 0
    failed_count = 0

    with open(args.output, "w", encoding="utf-8") as fout, \
         open(failures_path, "w", encoding="utf-8") as ffail:
        for text, ipa in tqdm(pairs):
            result = align_sentence(text, ipa, atoms, log_probs, allowed, max_n)
            if result is None:
                failed_count += 1
                ffail.write(f"{text}\t{ipa}\n")
            else:
                aligned_count += 1
                fout.write(json.dumps({text: result}, ensure_ascii=False) + "\n")

    total = aligned_count + failed_count
    print(f"\nTotal:   {total:,}")
    print(f"Aligned: {aligned_count:,} ({aligned_count/total:.1%})")
    print(f"Failed:  {failed_count:,} ({failed_count/total:.1%})")


if __name__ == "__main__":
    main()
