#!/usr/bin/env python3
"""Reorganize agentcore_slime_deck.tex into the 6 requested sections by RELOCATING the existing
frame blocks (matched by a unique title substring) under new \\section headers, rewriting the
Outline, and applying trivial global string fixes. Content-heavy frame rewrites are done afterward
by hand-edits; this script only moves + relabels + does safe global substitutions."""
import re, sys

SRC = "agentcore_slime_deck.tex"
OUT = "agentcore_slime_deck.tex"

txt = open(SRC).read()

# --- split into preamble (through the titlepage frame) and the rest ---------------------------
# Preamble ends right after the titlepage frame's \end{frame}.
m = re.search(r"\\begin\{frame\}\s*\n\s*\\titlepage\s*\n\s*\\end\{frame\}", txt)
assert m, "titlepage frame not found"
preamble = txt[:m.end()] + "\n"
body = txt[m.end():]

# --- collect all frame blocks -----------------------------------------------------------------
frame_re = re.compile(r"\\begin\{frame\}.*?\\end\{frame\}", re.DOTALL)
frames = frame_re.findall(body)

def find(key):
    """Return the single frame whose header (first ~300 chars) contains `key` (verbatim)."""
    hits = [f for f in frames if key in f[:300]]
    assert len(hits) == 1, f"key {key!r} matched {len(hits)} frames"
    return hits[0]

# Frame keys (unique title substrings) grouped into the 6 sections + tail.
S1 = ["What is AgentCore?", "AgentCore --- features", "AgentCore as a \\emph{Harness}",
      "AgentCore as a \\emph{Sandbox} --- use cases"]
S2 = ["What is Agentic RL?", "From trajectory to training signal",
      "Agent Lightning v1.0 --- the main idea", "Agent Lightning v1.0 --- architecture",
      "The three decouplings", "The slime framework", "Sync vs.\\ async mode"]
S3 = ["Agentic RL: AgentCore as Sandbox", "Agentic RL: AgentCore as Harness + Sandbox",
      "Target topology: harness and sandbox"]  # + NEW key-advantages frame appended by hand-edit
S4 = ["Integrating AgentCore sandbox into slime", "Using AgentCore as a \\emph{sandbox} --- code",
      "Integrating the Claude Code (Docker) harness",
      "Using AgentCore Runtime as a \\emph{harness} --- code",
      "The full two-Runtime path", "Bounding episode cost"]
S5 = ["The $2\\times3$ demo matrix", "Running the demos --- step by step (1/2)",
      "Running the demos --- step by step (2/2)",
      "Demo A --- MBPP", "Demo A --- how it is built", "Demo A --- code sketch",
      "Demo B --- KernelBench", "Demo B --- how it is built", "Demo B --- code sketch",
      "Demo A2 --- MBPP", "Demo A2 --- how it is built", "Demo A2 --- code sketch",
      "Demo B2 --- KernelBench", "Demo B2 --- how it is built", "Demo B2 --- code sketch",
      "Demo A2 / B2 --- replicating", "Demo A2 / B2 --- the replica ratio",
      "Demo A3 --- MBPP", "Demo A3 --- how it is built", "Demo A3 --- code sketch",
      "Demo B3 --- KernelBench", "Demo B3 --- how it is built", "Demo B3 --- code sketch",
      "Demo A3 / B3 --- replicating", "Demo A3 / B3 --- the replica ratio"]
S6 = ["Networking: Runtimes dial back", "Security \\& VPC: who sits where", "The VPC-mode trap",
      "Why the sandbox GPUs must NOT", "Can we just expose the adapter publicly",
      "Why decouple? --- five components", "Why scale up?", "How to scale up?",
      "Scaling deep-dive (1/3)", "Scaling deep-dive (2/3)", "Scaling deep-dive (3/3)"]
TAIL = ["Summary", "References", "Thank you"]  # Summary via \section*, refs, thanks

all_keys = S1 + S2 + S3 + S4 + S5 + S6 + TAIL
# sanity: every frame is placed exactly once
used = set()
for k in all_keys:
    f = find(k)
    assert id(f) not in used, f"frame double-placed via key {k!r}"
    used.add(id(f))
placed = {id(find(k)) for k in all_keys}
missing = [f.split(chr(10),1)[0] for f in frames
           if id(f) not in placed and "{Outline}" not in f[:60]]
assert not missing, f"UNPLACED frames: {missing}"

NEW_OUTLINE = r"""\begin{frame}{Outline}
  \small
  \begin{enumerate}\setlength\itemsep{0.35em}
    \item What is \textbf{AgentCore}? --- features \& the use case of each feature
    \item What is \textbf{Agentic RL}? --- use cases, and the workflow \& components
    \item The role of \textbf{AgentCore Runtime} in agentic RL --- \textbf{harness \& sandbox},
      and the key advantages (high level)
    \item \textbf{When \& how} to use AgentCore in agentic RL
    \item \textbf{Demos}: \texttt{demo\_a}, \texttt{demo\_b}, \texttt{demo\_a2},
      \texttt{demo\_b2}, \texttt{demo\_a3}, \texttt{demo\_b3}
      (each: high-level + target use case, then how to run \& build)
    \item \textbf{Networking \& scaling up}
  \end{enumerate}
\end{frame}"""

SEC = "\n% ============================================================\n\\section{@@}\n% ============================================================\n\n"
def sec(title): return SEC.replace("@@", title)

def emit(keys):
    return "\n\n".join(find(k) for k in keys)

parts = [preamble, NEW_OUTLINE, "\n"]
parts += [sec("What is AgentCore?"), emit(S1)]
parts += [sec("What is Agentic RL?"), emit(S2)]
parts += [sec("AgentCore Runtime in Agentic RL: Harness \\& Sandbox"), emit(S3)]
parts += [sec("When \\& How to Use AgentCore in Agentic RL"), emit(S4)]
parts += [sec("Demos"), emit(S5)]
parts += [sec("Networking \\& Scaling Up"), emit(S6)]
# tail: Summary uses \section*, then references + thanks
parts += ["\n% ============================================================\n\\section*{Summary}\n% ============================================================\n\n"]
parts += [find("Summary")]
parts += ["\n\n% ============================================================\n"]
parts += [find("References")]
parts += ["\n\n" + find("Thank you")]
parts += ["\n\n\\end{document}\n"]

new = "\n".join(p for p in parts)

# --- trivial global fixes ---------------------------------------------------------------------
new = new.replace("arXiv:2608.17528", "arXiv:2508.03680")
new = new.replace("https://arxiv.org/abs/2608.17528", "https://arxiv.org/abs/2508.03680")
new = new.replace("your EC2 (e.g.\\ L40S)", "your EC2 (e.g.\\ g6 / L4)")

open(OUT, "w").write(new)
print("wrote", OUT, "frames placed:", len(all_keys))
