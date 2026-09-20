#!/usr/bin/env bash
# Build report-based-deck.{tex,pdf} from deck.md — a SELF-CONTAINED deck that
# covers the AgentCore + slime material end-to-end (structured into numbered
# sections with an outline slide; no back-references to the report). It was
# seeded from ../agentcore_slime_report.md but now stands on its own.
#
# The .md is the SOURCE OF TRUTH. The .tex is pandoc-generated — do not hand-edit
# it. Edit deck.md instead. Same toolchain and Beamer settings as
# ../../slides/build_deck.sh so the two decks render identically.
#
# Requires: pandoc, xelatex (texlive-xetex), DejaVu fonts (fonts-dejavu-core).
set -euo pipefail

D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$D"
BASE=report-based-deck
MD="deck.md"
TEX="$BASE.tex"
PDF="$BASE.pdf"

TITLE="AgentCore as the Sandbox in Agentic RL"
SUBTITLE="AgentCore Runtime as the execution sandbox, integrated with slime"
# ASCII author on purpose: avoids needing a CJK font just for the header line.
AUTHOR="Kaige Yang --- Senior Applied Scientist, AWS"
DATE="September 10, 2026"

TMP="$(mktemp _build_meta.XXXXXX.md)"
trap 'rm -f "$TMP" '"$BASE"'.{aux,log,nav,out,snm,toc,vrb}' EXIT

# 1. YAML metadata header + body from the first "## " slide onward
#    (skips the manual title block at the top of the .md, which pandoc replaces).
{
  printf -- '---\ntitle: "%s"\nsubtitle: "%s"\nauthor: "%s"\ndate: "%s"\n---\n' \
    "$TITLE" "$SUBTITLE" "$AUTHOR" "$DATE"
  awk 'f{print} /^## /{if(!f){print; f=1}}' "$MD"
} > "$TMP"

# 2. pandoc -> Beamer (same style as the slides deck).
pandoc "$TMP" -t beamer --slide-level=2 \
  -V theme=Madrid -V aspectratio=169 -V fontsize=10pt -V classoption=t \
  -V mainfont="DejaVu Sans" -V monofont="DejaVu Sans Mono" \
  -V CJKmainfont="Noto Serif CJK SC" \
  -H <(cat <<'PREAMBLE'
% Tighten vertical rhythm so dense slides fit on a single frame.
\usepackage{etoolbox}
% TikZ so the ported reference diagrams render (same libraries as the report).
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,fit,backgrounds}
% A standalone section title slide at the start of each section (# headers),
% plus the \tableofcontents on the Outline slide, give the deck its structure.
\AtBeginSection[]{\frame{\sectionpage}}
\setbeamertemplate{section in toc}[sections numbered]
\setlength{\parskip}{3pt plus 1pt minus 1pt}
% Base font is 10pt; keep list levels within one step of each other so nested
% bullets don't shrink to an unreadable, inconsistent size (the old 9pt +
% \small + \footnotesize stack bottomed out near 7pt).
\setbeamertemplate{itemize/enumerate body begin}{\normalsize}
\setbeamertemplate{itemize/enumerate subbody begin}{\small}
\setbeamertemplate{itemize/enumerate subsubbody begin}{\small}
\AtBeginEnvironment{itemize}{\setlength{\itemsep}{2pt}\setlength{\topsep}{2pt}}
\AtBeginEnvironment{enumerate}{\setlength{\itemsep}{2pt}\setlength{\topsep}{2pt}}
\AtBeginEnvironment{Highlighting}{\footnotesize}
\AtBeginEnvironment{verbatim}{\footnotesize}
\AtBeginEnvironment{longtable}{\scriptsize}
% longtable's default pre/post glue is \bigskip; trim it so the trailing
% "-> Report" pointer stays on the table's own frame instead of spilling.
\setlength\LTpre{2pt}\setlength\LTpost{2pt}
PREAMBLE
) \
  --pdf-engine=xelatex \
  -s -o "$TEX"

# 3. allowframebreaks on every titled frame so dense tables/lists don't overflow.
sed -i \
  -e 's/\\begin{frame}\[fragile\]{/\\begin{frame}[fragile,allowframebreaks]{/g' \
  -e 's/\\begin{frame}{/\\begin{frame}[allowframebreaks]{/g' \
  "$TEX"

# Only number the continuation pages of a frame that actually breaks.
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
