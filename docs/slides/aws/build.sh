#!/bin/bash
# Build one AWS-style deck:  ./build.sh <name>   (compiles <name>.tex; e.g. ./build.sh terminalbench)
# Compiles twice so slide links resolve, fails loudly on any Overfull box, renders previews to png/<name>/.
# Theme: preamble.tex + art/ (from the aws-beamer-deck skill). Figures live in ../figures/.
set -u
cd "$(dirname "$0")"
N="${1:?usage: ./build.sh <deck name without .tex>}"
for i in 1 2; do pdflatex -interaction=nonstopmode -halt-on-error "$N.tex" >/dev/null || { grep -n -A5 "^!" "$N.log" | head -40; exit 1; }; done
echo "$N: pages: $(pdfinfo "$N.pdf" | awk '/Pages/{print $2}')"
if grep -nE "Overfull \\\\(h|v)box" "$N.log"; then
  echo "OVERFLOW WARNINGS FOUND"; grep -nB2 -A3 "Overfull" "$N.log" | grep -E "Overfull|lines [0-9]|frame" | head -40; exit 2
fi
echo "no overfull boxes"
rm -rf "png/$N" && mkdir -p "png/$N" && pdftoppm -r 50 -png "$N.pdf" "png/$N/p"
rm -f "$N".{aux,log,nav,out,snm,toc,vrb}
