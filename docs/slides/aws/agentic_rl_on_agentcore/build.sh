#!/bin/bash
# Build deck twice (refs), fail loudly on any overflow warning, then export every slide to png/slide-NN.png.
cd "$(dirname "$0")"
N=agentic_rl_on_agentcore
for i in 1 2; do pdflatex -interaction=nonstopmode -halt-on-error "$N.tex" >/dev/null || { grep -n -A5 "^!" "$N.log" | head -40; exit 1; }; done
echo "pages: $(pdfinfo "$N.pdf" | awk '/Pages/{print $2}')"
if grep -nE "Overfull \\\\(h|v)box" "$N.log"; then
  echo "OVERFLOW WARNINGS FOUND"; grep -nB2 -A3 "Overfull" "$N.log" | grep -E "Overfull|lines [0-9]|frame" | head -40; exit 2
fi
echo "no overfull boxes"
rm -rf png && mkdir png && pdftoppm -r 150 -png "$N.pdf" png/slide   # every slide as a PNG
rm -f "$N".{aux,log,nav,out,snm,toc,vrb}
