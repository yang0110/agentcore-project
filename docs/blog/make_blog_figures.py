#!/usr/bin/env python3
"""Light-theme copies of the deck's charts for the Medium blog (white page).

    python3 make_blog_figures.py     # -> figures/fig_*.png + figures/data.json

Runs the deck's make_figures.py with three changes: a light palette, output into figures/,
and RUNS pinned to the 9 runs the deck reports (smoke 2-8 + both train48 hybrid runs), so
later logs in the demo folder do not shift the numbers.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DECK = os.path.join(HERE, "..", "slides", "aws", "agentic_rl_on_agentcore")
OUT = os.path.join(HERE, "figures")

src = open(os.path.join(DECK, "make_figures.py")).read()
subs = [
    ('HERE = os.path.dirname(os.path.abspath(__file__))',
     f'HERE = {os.path.abspath(DECK)!r}\nOUT = {OUT!r}'),
    ('BG, PANEL, INK, MUTED, GRID = "#161D26", "#232F3E", "#F2F4F7", "#A3ACB9", "#333E4C"',
     'BG, PANEL, INK, MUTED, GRID = "#FFFFFF", "#D5DAE1", "#16191F", "#5F6B7A", "#E3E6EA"'),
    ('C = {"microvm": "#3B8BEB", "docker": "#E2602C", "gpu": "#1BAF7A"}',
     'C = {"microvm": "#2F7FDE", "docker": "#E2602C", "gpu": "#12936A"}'),
    ('RUNS = sorted(p[:-len(".demo.csv")] for p in glob.glob(os.path.join(LOGS, "*.demo.csv")))',
     'RUNS = sorted(p[:-len(".demo.csv")] for p in glob.glob(os.path.join(LOGS, "*.demo.csv"))\n'
     '              if re.match(r"(smoke\\d\\w*|train48)-hybrid_", os.path.basename(p)))'),
]
for old, new in subs:
    assert src.count(old) == 1, old
    src = src.replace(old, new)
# every output (fig_*.png, data.json) goes to OUT, not the deck folder
src, n = re.subn(r'os\.path\.join\(HERE, "(fig_\w+\.png|data\.json)"\)', r'os.path.join(OUT, "\1")', src)
assert n >= 11, n
exec(compile(src, "make_figures.py[blog]", "exec"), {"__name__": "__main__"})
