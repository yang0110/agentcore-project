#!/bin/bash
# Build the first-call deck twice (refs), fail loudly on any overflow, export png_first_call/slide-NN.png.
cd "$(dirname "$0")"
N=first-call-agentic-rl-with-agentcore
for i in 1 2; do pdflatex -interaction=nonstopmode -halt-on-error "$N.tex" >/dev/null || { grep -n -A5 "^!" "$N.log" | head -40; exit 1; }; done
echo "pages: $(pdfinfo "$N.pdf" | awk '/Pages/{print $2}')"
if grep -nE "Overfull \\\\(h|v)box" "$N.log"; then
  echo "OVERFLOW WARNINGS FOUND"; grep -nB2 -A3 "Overfull" "$N.log" | grep -E "Overfull|lines [0-9]|frame" | head -40; exit 2
fi
echo "no overfull boxes"
rm -rf png_first_call && mkdir png_first_call && pdftoppm -r 150 -png "$N.pdf" png_first_call/slide
rm -f "$N".{aux,log,nav,out,snm,toc,vrb}
