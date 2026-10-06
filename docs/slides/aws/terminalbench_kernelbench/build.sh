#!/bin/bash
# Build deck twice (refs), then fail loudly on any overflow warning.
cd "$(dirname "$0")"
for i in 1 2; do pdflatex -interaction=nonstopmode -halt-on-error deck.tex >/dev/null || { grep -n -A5 "^!" deck.log | head -40; exit 1; }; done
echo "pages: $(pdfinfo deck.pdf | awk '/Pages/{print $2}')"
if grep -nE "Overfull \\\\(h|v)box" deck.log; then
  echo "OVERFLOW WARNINGS FOUND"; grep -nB2 -A3 "Overfull" deck.log | grep -E "Overfull|lines [0-9]|frame" | head -40; exit 2
fi
echo "no overfull boxes"
rm -rf png && mkdir png && pdftoppm -r 50 -png deck.pdf png/p
