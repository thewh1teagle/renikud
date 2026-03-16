"""
Overfit validation: train on 50 samples, then check exact match on inference.

Step 1 - prepare tiny dataset (run once):
    head -50 knesset_phonemes_v1.txt > plans/overfit/tiny.txt
    uv run src/prepare_data.py --input plans/overfit/tiny.txt --output-dir plans/overfit/dataset --lines 50 --max-val 0
    uv run src/prepare_tokens.py --input plans/overfit/dataset/train.txt --output plans/overfit/dataset/.cache/train
    uv run src/prepare_tokens.py --input plans/overfit/dataset/train.txt --output plans/overfit/dataset/.cache/val

Step 2 - train and eval:
    uv run plans/overfit/overfit_001.py

Step 2b - eval only (skip training, use existing checkpoint):
    uv run plans/overfit/overfit_001.py --eval-only
"""

# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent.parent
TRAIN_CACHE = REPO / "plans/overfit/dataset/.cache/train"
VAL_CACHE = REPO / "plans/overfit/dataset/.cache/val"
TRAIN_TXT = REPO / "plans/overfit/dataset/train.txt"
OUTPUT_DIR = REPO / "plans/overfit/checkpoint"
EPOCHS = 1000
BATCH = 50


def run(cmd: list[str]) -> None:
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(1)


def run_capture(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr)
        sys.exit(1)
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    # Sanity check
    if not args.eval_only and (not TRAIN_CACHE.exists() or not VAL_CACHE.exists()):
        print("ERROR: dataset cache not found. Run the prepare commands in the docstring first.")
        sys.exit(1)

    if args.eval_only and not OUTPUT_DIR.exists():
        print("ERROR: no checkpoint found. Run without --eval-only first.")
        sys.exit(1)

    if not args.eval_only:
        print(f"Training for {EPOCHS} epochs on tiny dataset...")
        cmd = [
            "uv", "run", str(REPO / "src/train.py"),
            "--train-dataset", str(TRAIN_CACHE),
            "--eval-dataset", str(VAL_CACHE),
            "--output-dir", str(OUTPUT_DIR),
            "--epochs", str(EPOCHS),
            "--train-batch-size", str(BATCH),
            "--eval-batch-size", str(BATCH),
            "--save-steps", "999999",
            "--encoder-lr", "1e-4",
            "--head-lr", "1e-3",
            "--lr-scheduler-type", "cosine",
            "--freeze-encoder-steps", "50",
            "--upsample-factor", "2",
            "--report-to", "none",
        ]
        if (OUTPUT_DIR / "trainer_state.json").exists():
            cmd += ["--resume-from-checkpoint", str(OUTPUT_DIR)]
        run(cmd)

    # --- Infer on training samples and check exact match ---
    lines = TRAIN_TXT.read_text(encoding="utf-8").strip().splitlines()
    pairs = [line.split("\t") for line in lines if "\t" in line]

    matches = 0
    for hebrew, expected_ipa in pairs:
        predicted = run_capture([
            "uv", "run", str(REPO / "src/infer.py"),
            "--checkpoint", str(OUTPUT_DIR),
            "--text", hebrew,
            "--beam-size", "10",
        ])
        if predicted == expected_ipa:
            matches += 1
        else:
            print(f"MISMATCH")
            print(f"  input:    {hebrew}")
            print(f"  expected: {expected_ipa}")
            print(f"  got:      {predicted}")

    total = len(pairs)
    print(f"\nResult: {matches}/{total} exact matches ({100*matches//total}%)")
    if matches == total:
        print("PASS: model overfits perfectly.")
    elif matches >= total * 0.8:
        print("PARTIAL: model overfits mostly — acceptable.")
    else:
        print("FAIL: model is not overfitting. Check training setup.")


if __name__ == "__main__":
    main()
