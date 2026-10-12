"""Agent skills: short how-to guides the agent can read on demand (docs/skills-in-agentic-rl.md).

A skill is a folder under SKILLS_DIR (default: the demo's skills/) holding a SKILL.md, whose front
matter gives `name`, `description` and `families`, plus optional files (scripts, templates).
SKILLS=listed installs the family's skills into the episode's sandbox at open and appends a
one-line-per-skill listing to the system prompt. The agent reads a skill with its ordinary bash
tool (`cat <root>/<name>/SKILL.md`), so there is no new tool schema, and the skill text reaches the
model as a tool result (loss-masked; the policy is never trained to reproduce it). `uninstall` removes
the folder before grading. SKILLS=off (default) changes nothing.

Knobs: SKILLS=off|listed, SKILLS_SET=all|a,b (names to offer), SKILLS_DIR, SKILLS_EVAL=same|off
(off = evaluate without skills, the internalization check). Every file's literal `SKILL_DIR` is
replaced by the installed folder's absolute path.
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import os
import re
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_DIR = Path(__file__).resolve().parent.parent / "skills"
# First writable root wins (a capability-less image may not allow /opt).
_ROOTS = ("/opt/acs_skills", "/tmp/acs_skills")


@dataclass
class Skill:
    name: str
    description: str
    families: tuple[str, ...]
    files: dict[str, str] = field(default_factory=dict)   # relative path -> text


@dataclass
class Installed:
    root: str
    names: list[str]
    version: str
    listing: str


def _front_matter(text: str) -> dict[str, str]:
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        return {}
    out = {}
    for line in m.group(1).splitlines():
        k, _, v = line.partition(":")
        if v:
            out[k.strip()] = v.strip()
    return out


def mode(evaluation: bool = False) -> str:
    m = os.environ.get("SKILLS", "off").strip().lower() or "off"
    if m not in ("off", "listed"):
        raise ValueError(f"SKILLS={m!r}: expected off|listed")
    if evaluation and os.environ.get("SKILLS_EVAL", "same").strip().lower() == "off":
        return "off"
    return m


_CACHE: dict[str, list[Skill]] = {}


def library() -> list[Skill]:
    root = Path(os.environ.get("SKILLS_DIR", "") or _DEFAULT_DIR)
    key = str(root)
    if key not in _CACHE:
        skills = []
        for md in sorted(root.glob("*/SKILL.md")):
            text = md.read_text()
            fm = _front_matter(text)
            fams = tuple(f.strip() for f in fm.get("families", "").strip("[]").split(",") if f.strip())
            files = {str(p.relative_to(md.parent)): p.read_text()
                     for p in sorted(md.parent.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}
            skills.append(Skill(fm.get("name", md.parent.name), fm.get("description", ""), fams, files))
            limit = int(os.environ.get("MINI_AGENT_MAX_TOOL_CHARS", "4000"))
            if len(text) > limit:   # `cat SKILL.md` comes back head+tail clipped: the middle is lost
                logger.warning("[skills] %s is %d chars > MINI_AGENT_MAX_TOOL_CHARS=%d", md, len(text), limit)
        _CACHE[key] = skills
    return _CACHE[key]


def select(family: str) -> list[Skill]:
    want = os.environ.get("SKILLS_SET", "all").strip()
    names = None if want in ("", "all") else {n.strip() for n in want.split(",")}
    return [s for s in library() if family in s.families and (names is None or s.name in names)]


def version(skills: list[Skill]) -> str:
    h = hashlib.sha256()
    for s in skills:
        for path, text in sorted(s.files.items()):
            h.update(f"{s.name}/{path}\0{text}\0".encode())
    return h.hexdigest()[:10]


def listing(skills: list[Skill], root: str) -> str:
    """List skills as FILE PATHS. A `- name: description` list reads like a tool list: Qwen3-4B then
    emits <tool_call>{"name": "tb-python-env"} (0/8 valid tool calls vs 7/8 with no listing, probed
    on the smoke's SGLang 2026-10-12); this wording gave 8/8 valid calls, all `cat <path>` reads."""
    lines = [f"  {root}/{s.name}/SKILL.md  ({s.description})" for s in skills]
    return ("\n\nHelpful notes are saved as files. To read one, call the bash tool with `cat` on its path:\n"
            + "\n".join(lines))


def _reference_block(prompt: str) -> str | None:
    """The task's reference program (the first ```python block of a KernelBench prompt)."""
    m = re.search(r"```python\n(.*?)```", prompt or "", re.S)
    return m.group(1) if m else None


async def install(sb, family: str, *, prompt: str = "", evaluation: bool = False) -> Installed | None:
    """Write the family's skills into the sandbox; return what to list (None = skills off / none apply).

    One exec: a base64 tar unpacked by python3 (always on the image; the in-session runner needs it)
    into the first writable root. KernelBench: the prompt's reference Model is saved as
    kb-self-check/reference.py, so the self-check never needs the agent to re-type it."""
    if mode(evaluation) == "off":
        return None
    skills = select(family)
    if not skills:
        return None
    ver = version(skills)
    ref = _reference_block(prompt) if family == "kernelbench" else None
    for root in _ROOTS:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for s in skills:
                files = dict(s.files)
                if s.name == "kb-self-check" and ref:
                    files["reference.py"] = ref
                for rel, text in files.items():
                    data = text.replace("SKILL_DIR", f"{root}/{s.name}").encode()
                    info = tarfile.TarInfo(f"{s.name}/{rel}")
                    info.size, info.mode = len(data), 0o755 if rel.endswith((".py", ".sh")) else 0o644
                    tar.addfile(info, io.BytesIO(data))
        b64 = base64.b64encode(buf.getvalue()).decode()
        code, _out, _err = await sb.exec(
            f"rm -rf {root} && mkdir -p {root} && python3 -c \"import base64,io,sys,tarfile; "
            f"tarfile.open(fileobj=io.BytesIO(base64.b64decode(sys.argv[1]))).extractall(sys.argv[2])\" "
            f"{b64} {root}", timeout=60)
        if code == 0:
            return Installed(root, [s.name for s in skills], ver, listing(skills, root))
    raise RuntimeError(f"cannot install skills into any of {_ROOTS} (last exit {code}: {(_err or '')[:200]})")


async def uninstall(sb, inst: Installed | None) -> None:
    """Remove the skills before anything grades the sandbox."""
    if inst is None:
        return
    try:
        await sb.exec(f"rm -rf {inst.root}", timeout=30)
    except Exception:  # noqa: BLE001 — teardown only; the grader never reads this path
        pass


def reads(transcript: list[dict], inst: Installed | None) -> list[str]:
    """Skill names the agent read (any bash command or read_file path naming <root>/<name>/)."""
    if inst is None:
        return []
    seen = []
    for msg in transcript or []:
        if msg.get("role") != "assistant" or not isinstance(msg.get("content"), list):
            continue
        for tu in msg["content"]:
            if not isinstance(tu, dict) or tu.get("type") != "tool_use":
                continue
            inp = tu.get("input") or {}
            text = " ".join(str(inp.get(k, "")) for k in ("command", "path"))
            for name in inst.names:
                if f"{inst.root}/{name}/" in text and name not in seen:
                    seen.append(name)
    return seen


def demo_fields(transcript: list[dict], inst: Installed | None) -> dict[str, str]:
    """The [demo] skill fields (empty when skills are off, so default runs log nothing new)."""
    if inst is None:
        return {}
    got = reads(transcript, inst)
    return {"skills_listed": str(len(inst.names)), "skill_reads": str(len(got)),
            "skills_read": "+".join(got) or "none", "skills_version": inst.version}
