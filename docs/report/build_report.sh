#!/usr/bin/env bash
# Build agentcore_slime_report.{tex,pdf} — the REPORT (not the slides).
#
# Source of truth: agentcore_slime_report.md (this dir). It was seeded from the
# deck (the since-removed ../slides/agentcore_slime_deck.md; now ../slides/aws/agentcore_slime.tex) but is now EDITED INDEPENDENTLY so
# report refinements don't disturb the slides. Each "## " becomes a numbered
# *section* of a flowing PDF report, via the Eisvogel pandoc template (article
# class, TOC, shaded code boxes). The .tex is pandoc output — edit the .md.
#
# Requires: pandoc, xelatex (texlive-xetex), the DejaVu fonts
# (fonts-dejavu-core), and eisvogel.latex (vendored next to this script).
set -euo pipefail

D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$D"

SRC="agentcore_slime_report.md"              # report-specific source of truth
BASE=agentcore_slime_report
TEX="$BASE.tex"
PDF="$BASE.pdf"
TEMPLATE=eisvogel.latex

TITLE="Using AgentCore Runtime in an Agentic RL Workflow"
SUBTITLE="Hands-on guidance, grounded in a working slime + AgentCore project"
# ASCII author on purpose: avoids needing a CJK font just for the header line.
AUTHOR="Kaige Yang --- Senior Applied Scientist, AWS"
DATE="September 10, 2026"

TMP="$(mktemp _build_meta.XXXXXX.md)"
trap 'rm -f "$TMP" '"$BASE"'.{aux,log,out,toc}' EXIT

# 1. YAML metadata header + body from the first "## " section onward (skips the
#    manual title block at the top of the .md — the template renders the title
#    page instead). Drop the standalone "---" lines: they are slide-group
#    dividers in the deck and have no meaning in a flowing report.
{
  cat <<YAML
---
title: "$TITLE"
subtitle: "$SUBTITLE"
author: "$AUTHOR"
date: "$DATE"
titlepage: true
titlepage-rule-color: "5F5F5F"
toc: true
toc-own-page: true
colorlinks: true
code-block-font-size: \footnotesize
mainfont: "DejaVu Sans"
monofont: "DejaVu Sans Mono"
CJKmainfont: "Noto Serif CJK SC"
header-includes: |
  \usepackage{tikz}
  \usetikzlibrary{arrows.meta,positioning,fit,backgrounds}
---
YAML
  awk 'f{if($0!="---")print} /^## /{if(!f){print;f=1}}' "$SRC"
} > "$TMP"

# 2. pandoc -> LaTeX (Eisvogel template). --number-sections numbers the former
#    slide titles; default skylighting highlighting fills Eisvogel's shaded box.
# --shift-heading-level-by=-1 promotes the deck's "## " (level 2) titles to
# level-1 sections so --number-sections numbers them 1, 2, 3 (not 0.0.N).
pandoc "$TMP" \
  --template="$TEMPLATE" \
  --pdf-engine=xelatex \
  --number-sections \
  --shift-heading-level-by=-1 \
  --top-level-division=section \
  -s -o "$TEX"

# 3. Compile twice so the TOC settles.
xelatex -interaction=nonstopmode -halt-on-error "$TEX" >/dev/null
xelatex -interaction=nonstopmode -halt-on-error "$TEX" >/dev/null

# 4. Sanity check: no dropped glyphs.
if grep -q 'Missing character' "$BASE.log" 2>/dev/null; then
  echo "WARNING: xelatex reported missing glyphs — check $BASE.log" >&2
fi

echo "Built $TEX and $PDF ($(pdfinfo "$PDF" 2>/dev/null | awk '/^Pages/{print $2}') pages)."
