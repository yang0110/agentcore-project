#!/usr/bin/env bash
# Rebuild scale_lanes_deck.{tex,pdf} from terminalbench_deck.md
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
BASE=scale_lanes_deck
MD="$BASE.md"
TEX="$BASE.tex"
PDF="$BASE.pdf"

TITLE="Scaling lanes on AgentCore: harness, inference and sandbox scaled together"
# ASCII author on purpose: avoids needing a CJK font just for the header line.
AUTHOR="Kaige Yang --- Senior Applied Scientist, AWS"
DATE="September 24, 2026"

TMP="$(mktemp _build_meta.XXXXXX.md)"
trap 'rm -f "$TMP" '"$BASE"'.{aux,log,nav,out,snm,toc,vrb}' EXIT

# 1. YAML metadata header + body from the first "## " slide onward
#    (skips the manual title block at the top of the .md, which pandoc replaces).
{
  printf -- '---\ntitle: "%s"\nauthor: "%s"\ndate: "%s"\n---\n' "$TITLE" "$AUTHOR" "$DATE"
  awk 'f{print} /^## /{if(!f){print; f=1}}' "$MD"
} > "$TMP"

# 2. pandoc -> Beamer
#    fontsize=10pt + tighter list/paragraph spacing so dense slides fit ONE
#    frame instead of spilling a 1-3 line orphan onto an allowframebreaks page.
pandoc "$TMP" -t beamer --slide-level=2 \
  -V theme=Madrid -V aspectratio=169 -V fontsize=10pt -V classoption=t \
  -V mainfont="DejaVu Sans" -V monofont="DejaVu Sans Mono" \
  -V CJKmainfont="Noto Serif CJK SC" \
  -H <(cat <<'PREAMBLE'
% Tighten vertical rhythm so dense slides fit on a single frame.
\usepackage{etoolbox}
\setlength{\parskip}{3pt plus 1pt minus 1pt}
\setbeamertemplate{itemize/enumerate body begin}{\small}
\setbeamertemplate{itemize/enumerate subbody begin}{\footnotesize}
\AtBeginEnvironment{itemize}{\setlength{\itemsep}{2pt}\setlength{\topsep}{2pt}}
\AtBeginEnvironment{enumerate}{\setlength{\itemsep}{2pt}\setlength{\topsep}{2pt}}
% Shrink code listings and tables so long lines do not force a page break.
\AtBeginEnvironment{Highlighting}{\footnotesize}
\AtBeginEnvironment{verbatim}{\footnotesize}
\AtBeginEnvironment{longtable}{\scriptsize}
PREAMBLE
) \
  --pdf-engine=xelatex \
  -s -o "$TEX"

# 3. No allowframebreaks here: inside a breakable frame longtable forces a page break after every
#    table. Every slide of this deck is sized to fit one frame instead.

# Short title/author for the footline (the full title overflows Madrid's footer boxes).
sed -i -e 's/^\\title{/\\title[Scaling lanes on AgentCore]{/' -e 's/^\\author{/\\author[Kaige Yang, AWS]{/' "$TEX"

# Centre every figure and its "Figure N:" line (pandoc leaves a height-limited image flush left).
# Tables are already centred (longtable default) and span the text width.
perl -0pi -e 's/^(\\includegraphics\[[^\]]*\]\{[^}]*\})$/\\centerline{$1}/mg; s/^\\emph\{(Figure \d+:[^}]*)\}$/\\centerline{\\emph{$1}}/mg' "$TEX"

# Only number the continuation pages of a frame that actually breaks (II, III,
# ...); a frame that fits one page keeps a clean, un-numbered title. Injected as
# raw LaTeX here because pandoc escapes the [..] if passed via header-includes.
awk '/\\begin\{document\}/{print "\\setbeamertemplate{frametitle continuation}[from second]"} {print}' \
  "$TEX" > "$TEX.tmp" && mv "$TEX.tmp" "$TEX"

# 4. Compile twice (longtable needs a second pass to settle column widths).
xelatex -interaction=nonstopmode -halt-on-error "$TEX" >/dev/null
xelatex -interaction=nonstopmode -halt-on-error "$TEX" >/dev/null

cp "$BASE.log" /tmp/lanes_deck.log
# 5. Sanity check: no dropped glyphs.
if grep -q 'Missing character' "$BASE.log" 2>/dev/null; then
  echo "WARNING: xelatex reported missing glyphs — check $BASE.log" >&2
fi

echo "Built $TEX and $PDF ($(pdfinfo "$PDF" 2>/dev/null | awk '/^Pages/{print $2}') pages)."
