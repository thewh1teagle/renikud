import json
import math
from collections import defaultdict
from tqdm import tqdm

from dp import forward_backward, valid_chunk


def run_em(
    corpus: list[tuple[list[list[str]], list[list[str]]]],
    allowed: dict[str, list[str]],
    max_n: int,
    iterations: int,
) -> dict:
    print("Collecting candidate chunks...")
    candidate_chunks: set[str] = {""}
    for letter_seqs, token_seqs in tqdm(corpus):
        for tokens in token_seqs:
            for start in range(len(tokens) + 1):
                for length in range(0, max_n + 1):
                    end = start + length
                    if end > len(tokens):
                        break
                    candidate_chunks.add("".join(tokens[start:end]))

    log_probs: dict[tuple[str, str], float] = {}
    for letter, allowed_starts in allowed.items():
        valid = [c for c in candidate_chunks if valid_chunk(c, allowed_starts)]
        if not valid:
            continue
        lp = -math.log(len(valid))
        for chunk in valid:
            log_probs[(letter, chunk)] = lp

    for iteration in range(iterations):
        print(f"EM iteration {iteration + 1}/{iterations}")
        counts: dict[tuple[str, str], float] = defaultdict(float)
        skipped = 0

        for letter_seqs, token_seqs in tqdm(corpus):
            for ls, ts in zip(letter_seqs, token_seqs):
                result = forward_backward(ls, ts, log_probs, allowed, max_n)
                if result is None:
                    skipped += 1
                    continue
                for key, val in result.items():
                    counts[key] += val

        if skipped:
            print(f"  Skipped {skipped} words (no valid alignment)")

        letter_totals: dict[str, float] = defaultdict(float)
        for (letter, _), count in counts.items():
            letter_totals[letter] += count

        log_probs = {}
        for (letter, chunk), count in counts.items():
            total = letter_totals[letter]
            ratio = count / total if total > 0 else 0.0
            log_probs[(letter, chunk)] = math.log(ratio) if ratio > 1e-300 else float("-inf")

    return log_probs


def save_model(log_probs: dict, path: str) -> None:
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    nested: dict[str, dict[str, float]] = {}
    for (letter, chunk), lp in log_probs.items():
        nested.setdefault(letter, {})[chunk] = lp
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nested, f, ensure_ascii=False, indent=2)


def load_model(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        nested = json.load(f)
    return {(letter, chunk): lp for letter, chunks in nested.items() for chunk, lp in chunks.items()}
