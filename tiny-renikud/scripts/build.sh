#!/bin/sh
set -e
SRC="$(dirname "$0")/../src"
gcc -O2 -o "$SRC/renikud" "$SRC/main.c" "$SRC/model.c" -lm
echo "Built $SRC/renikud"
"$SRC/renikud"
