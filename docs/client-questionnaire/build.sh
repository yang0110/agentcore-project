#!/bin/bash
# Build deck twice (refs), then fail loudly on any overflow warning.
cd "$(dirname "$0")"
for i in 1 2; do pdflatex -interaction=nonstopmode -halt-on-error agentic_rl_client_questions.tex >/dev/null || { grep -n -A5 "^!" agentic_rl_client_questions.log | head -40; exit 1; }; done
echo "pages: $(pdfinfo agentic_rl_client_questions.pdf | awk '/Pages/{print $2}')"
if grep -nE "Overfull \\\\(h|v)box" agentic_rl_client_questions.log; then
  echo "OVERFLOW WARNINGS FOUND"; grep -nB2 -A3 "Overfull" agentic_rl_client_questions.log | grep -E "Overfull|lines [0-9]|frame" | head -40; exit 2
fi
echo "no overfull boxes"
rm -rf png && mkdir png && pdftoppm -r 150 -png agentic_rl_client_questions.pdf png/p
