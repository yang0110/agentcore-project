#!/usr/bin/env bash
# Rebuild agentcore_slime_deck.{tex,pdf} from agentcore_slime_deck.md
#
# The .md is the SOURCE OF TRUTH. The .tex is pandoc-generated — do not hand-edit
# it, your changes will be overwritten on the next run. Edit the .md instead.
#
# Requires: pandoc, xelatex (texlive-xetex), and the DejaVu fonts
# (fonts-dejavu-core) — DejaVu Sans covers every exotic glyph in the deck
# (✓ ◑ ⊥ ↔ ⇒ ① ② ③ × ε ≈ ≠ ≤ ≥ ∧ → ─ ▶).
set -euo pipefail

D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$D"
BASE=agentcore_slime_deck
MD="$BASE.md"
TEX="$BASE.tex"
PDF="$BASE.pdf"

TITLE="Agentic RL: AgentCore as Sandbox and Harness"
# ASCII author on purpose: avoids needing a CJK font just for the header line.
AUTHOR="Kaige Yang --- Senior Applied Scientist, AWS"
DATE="September 10, 2026"

TMP="$(mktemp _build_meta.XXXXXX.md)"
trap 'rm -f "$TMP" '"$BASE"'.{aux,log,nav,out,snm,toc,vrb}' EXIT

# 1. YAML metadata header + body from the first "## " slide onward
#    (skips the manual title block at the top of the .md, which pandoc replaces).
{
  printf -- '---\ntitle: "%s"\nauthor: "%s"\ndate: "%s"\n---\n' "$TITLE" "$AUTHOR" "$DATE"
  awk 'f{print} /^## /{if(!f){print; f=1}}' "$MD"
} > "$TMP"

# 2. pandoc -> Beamer
pandoc "$TMP" -t beamer --slide-level=2 \
  -V theme=Madrid -V aspectratio=169 \
  -V mainfont="DejaVu Sans" -V monofont="DejaVu Sans Mono" \
  -V CJKmainfont="Noto Serif CJK SC" \
  --pdf-engine=xelatex \
  -s -o "$TEX"

# 3. allowframebreaks on every titled frame so dense tables/lists don't overflow
#    (preserve [fragile] on code frames; fragile line first so the second rule
#    can't double-match it).
sed -i \
  -e 's/\\begin{frame}\[fragile\]{/\\begin{frame}[fragile,allowframebreaks]{/g' \
  -e 's/\\begin{frame}{/\\begin{frame}[allowframebreaks]{/g' \
  "$TEX"

# Only number the continuation pages of a frame that actually breaks (II, III,
# ...); a frame that fits one page keeps a clean, un-numbered title. Injected as
# raw LaTeX here because pandoc escapes the [..] if passed via header-includes.
awk '/\\begin\{document\}/{print "\\setbeamertemplate{frametitle continuation}[from second]"} {print}' \
  "$TEX" > "$TEX.tmp" && mv "$TEX.tmp" "$TEX"

# 4. Compile twice (longtable needs a second pass to settle column widths).
xelatex -interaction=nonstopmode -halt-on-error "$TEX" >/dev/null
xelatex -interaction=nonstopmode -halt-on-error "$TEX" >/dev/null

# 5. Sanity check: no dropped glyphs.
if grep -q 'Missing character' "$BASE.log" 2>/dev/null; then
  echo "WARNING: xelatex reported missing glyphs — check $BASE.log" >&2
fi

echo "Built $TEX and $PDF ($(pdfinfo "$PDF" 2>/dev/null | awk '/^Pages/{print $2}') pages)."
