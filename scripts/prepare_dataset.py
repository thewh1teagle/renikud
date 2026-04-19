"""Extract vocalized Hebrew text (left column) from a tab-separated file, stripping pipe characters."""

import re
import sys
from tqdm import tqdm

_STRIP = re.compile(r"[|\u05bd\u05ab]")

def main(input_path: str, output_path: str) -> None:
    with open(input_path, encoding="utf-8") as fin:
        lines = fin.readlines()

    with open(output_path, "w", encoding="utf-8") as fout:
        for line in tqdm(lines, desc="Processing"):
            line = line.rstrip("\n")
            if not line:
                continue
            left = line.split("\t")[0]
            left = _STRIP.sub("", left)
            fout.write(left + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.txt> <output.txt>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
