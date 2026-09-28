#!/usr/bin/env python3
"""Fetch a small SWE-bench Verified subset for demo_cpu_swebench.

Pulls princeton-nlp/SWE-bench_Verified (HF `datasets`, fallback: the parquet via huggingface_hub),
picks ~20 instances biased to FAST, SMALL test suites across DIVERSE repos, resolves each instance's
official eval recipe (test_cmd + eval_commands from the swebench package's
MAP_REPO_VERSION_TO_SPECS_PY, pip-downloaded at fetch time — never re-typed here), and writes:

    swebench/manifest.json          (gitignored) everything the deploy + grader need per instance:
                                    image, repo, version, base_commit, test_patch (HIDDEN — stays on the
                                    trainer), test_files, test_cmd, eval_commands, parser,
                                    FAIL_TO_PASS, PASS_TO_PASS, gold patch (oracle check only)
    data/swebench_train.jsonl       slime rows, same shape as data/tb_*.jsonl but family=swebench
    data/swebench_eval.jsonl        (disjoint instances, every 5th)

Selection (all knobs env):
  * repo in the grader's supported set (agentcore_slime_demo/swebench.PARSER_BY_REPO) with a per-repo
    quota (SWE_QUOTA, default django:6,sympy:4,sphinx-doc:3,pytest-dev:2 and 1 each for astropy,
    requests, xarray, pylint, flask = 20) — matplotlib / scikit-learn / seaborn are excluded (4-8 GB
    images, minutes-long suites);
  * difficulty "<15 min fix", <= SWE_MAX_F2P (3) FAIL_TO_PASS, SWE_MIN_P2P..SWE_MAX_P2P (1..150) PASS_TO_PASS,
    <= 2 test files touched; within a repo the smallest suites (F2P+P2P) win;
  * the prebuilt Docker Hub image `swebench/sweb.eval.x86_64.<id with __ -> _1776_>` must exist with a
    compressed size <= SWE_MAX_IMAGE_MB (3000) — checked via the Docker Hub tags API.

The data files only list instances present in swebench/arn_by_image.json (once deployed) and, if
swebench/oracle.csv exists, only those whose gold patch solved and whose empty patch did not
(`usable=1`), the same filter demo_cpu_tasks' Terminal-Bench rows use. Re-run with --data-only after deploy/check.

    python3 fetch_swebench.py              # fetch + select + manifest + data
    python3 fetch_swebench.py --data-only  # rebuild data/ from the existing manifest (+ pool/oracle filters)

No AWS, no Docker. Network: huggingface.co, pypi.org, hub.docker.com.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "swebench"
DATA = HERE / "data"
sys.path.insert(0, str(HERE))

from agentcore_slime_demo.swebench import PARSER_BY_REPO, SWEBenchTask  # noqa: E402

DATASET = "princeton-nlp/SWE-bench_Verified"
SWEBENCH_PKG = os.environ.get("SWE_SPECS_PKG", "swebench==3.0.17")   # last release with the py spec table
NON_TEST_EXTS = [".json", ".png", "csv", ".txt", ".md", ".jpg", ".jpeg", ".pkl", ".yml", ".yaml", ".toml"]
DEFAULT_QUOTA = ("django/django:6,sympy/sympy:4,sphinx-doc/sphinx:3,pytest-dev/pytest:2,"
                 "astropy/astropy:1,psf/requests:1,pydata/xarray:1,pylint-dev/pylint:1,pallets/flask:1")


def load_rows() -> list[dict]:
    try:
        from datasets import load_dataset

        return [dict(r) for r in load_dataset(DATASET, split="test")]
    except Exception as e:  # noqa: BLE001 — fall back to the raw parquet
        print(f"datasets load failed ({e}); trying the parquet via huggingface_hub", file=sys.stderr)
        import pandas as pd
        from huggingface_hub import hf_hub_download, list_repo_files

        f = next(p for p in list_repo_files(DATASET, repo_type="dataset") if p.endswith(".parquet"))
        return pd.read_parquet(hf_hub_download(DATASET, f, repo_type="dataset")).to_dict("records")


def load_specs() -> dict:
    """MAP_REPO_VERSION_TO_SPECS_PY from the official swebench package (a pure-constants module)."""
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "-q", "-d", tmp, SWEBENCH_PKG],
                       check=True)
        whl = next(Path(tmp).glob("*.whl"))
        subprocess.run(["unzip", "-q", "-o", str(whl), "swebench/harness/constants/python.py", "-d", tmp],
                       check=True)
        spec = importlib.util.spec_from_file_location("swe_py_constants",
                                                      Path(tmp) / "swebench/harness/constants/python.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.MAP_REPO_VERSION_TO_SPECS_PY


def test_directives(repo: str, test_patch: str) -> list[str]:
    """swebench get_test_directives: test files from the patch; django wants dotted module names."""
    ds = [d for d in re.findall(r"diff --git a/.* b/(.*)", test_patch)
          if not any(d.endswith(ext) for ext in NON_TEST_EXTS)]
    if repo == "django/django":
        ds = [(d[:-3] if d.endswith(".py") else d) for d in ds]
        ds = [(d[len("tests/"):] if d.startswith("tests/") else d).replace("/", ".") for d in ds]
    return ds


def docker_image(instance_id: str) -> str:
    return f"swebench/sweb.eval.x86_64.{instance_id.replace('__', '_1776_')}:latest"


def hub_size_mb(image: str) -> float | None:
    repo, tag = image.split(":")
    url = f"https://hub.docker.com/v2/repositories/{repo}/tags/{tag}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            d = json.load(r)
    except Exception:  # noqa: BLE001 — missing image / API hiccup: not selectable
        return None
    sizes = [i.get("size", 0) for i in d.get("images", []) if i.get("architecture") == "amd64"]
    return round((sizes[0] if sizes else d.get("full_size", 0)) / 2**20, 1)


def select(rows: list[dict], specs: dict) -> list[dict]:
    quota = dict((k, int(v)) for k, v in (p.rsplit(":", 1) for p in os.environ.get("SWE_QUOTA", DEFAULT_QUOTA).split(",")))
    max_f2p, max_p2p = int(os.environ.get("SWE_MAX_F2P", "3")), int(os.environ.get("SWE_MAX_P2P", "150"))
    min_p2p = int(os.environ.get("SWE_MIN_P2P", "1"))   # >=1 regression test, so "still passes" is checked
    max_mb = float(os.environ.get("SWE_MAX_IMAGE_MB", "3000"))
    only = {t for t in os.environ.get("SWE_INSTANCES", "").split(",") if t}
    cands: dict[str, list[dict]] = {}
    for r in rows:
        f2p, p2p = json.loads(r["FAIL_TO_PASS"]), json.loads(r["PASS_TO_PASS"])
        files = re.findall(r"diff --git a/.* b/(.*)", r["test_patch"])
        if only:
            if r["instance_id"] not in only:
                continue
        elif (r["repo"] not in quota or r["repo"] not in PARSER_BY_REPO or r["difficulty"] != "<15 min fix"
              or not f2p or len(f2p) > max_f2p or not min_p2p <= len(p2p) <= max_p2p or len(files) > 2):
            continue
        if r["version"] not in specs.get(r["repo"], {}):
            continue
        cands.setdefault(r["repo"], []).append({**r, "_f2p": f2p, "_p2p": p2p, "_files": files})
    picked = []
    for repo, cs in sorted(cands.items()):
        cs.sort(key=lambda c: (len(c["_f2p"]) + len(c["_p2p"]), c["instance_id"]))
        n = len(cs) if only else quota[repo]
        for c in cs:
            if n <= 0:
                break
            img = docker_image(c["instance_id"])
            mb = hub_size_mb(img)
            if mb is None or mb > max_mb:
                print(f"  skip {c['instance_id']}: image {img} size={mb} MB", file=sys.stderr)
                continue
            c["_image_mb"] = mb
            picked.append(c)
            n -= 1
    return picked


def manifest_entry(c: dict, specs: dict) -> dict:
    sp = specs[c["repo"]][c["version"]]
    return {
        "task": c["instance_id"],
        "image": c["instance_id"],                      # routing tag -> arn_by_image.json key
        "docker_image": docker_image(c["instance_id"]),
        "image_mb": c["_image_mb"],
        "repo": c["repo"], "version": c["version"], "difficulty": c["difficulty"],
        "base_commit": c["base_commit"],
        "problem_statement": c["problem_statement"],
        "test_patch": c["test_patch"],
        "test_files": c["_files"],
        "test_cmd": " ".join([sp["test_cmd"], *test_directives(c["repo"], c["test_patch"])]),
        "eval_commands": list(sp.get("eval_commands", [])),
        "parser": PARSER_BY_REPO[c["repo"]],
        "FAIL_TO_PASS": c["_f2p"], "PASS_TO_PASS": c["_p2p"],
        "gold_patch": c["patch"],                       # swebench_oracle_check.py only
    }


def write_data(instances: list[dict]) -> None:
    pool_p, oracle_p = OUT / "arn_by_image.json", OUT / "oracle.csv"
    keep = instances
    if pool_p.exists():
        pool = json.loads(pool_p.read_text())
        keep = [e for e in keep if e["image"] in pool]
        print(f"pool filter: {len(keep)}/{len(instances)} deployed")
    if oracle_p.exists():
        with oracle_p.open() as f:
            usable = {r["task"] for r in csv.DictReader(f) if r.get("usable") == "1"}
        keep = [e for e in keep if e["task"] in usable]
        print(f"oracle filter: {len(keep)} usable (gold solves, empty patch does not)")
    max_turns = int(os.environ.get("SWE_MAX_TURNS", "12"))
    rows = []
    for e in keep:
        t = SWEBenchTask(task_id=e["task"], prompt=e["problem_statement"], max_turns=max_turns, spec={
            "image": e["image"], "repo": e["repo"], "difficulty": e["difficulty"], "category": e["repo"],
            "max_test_timeout_sec": float(os.environ.get("SWE_TEST_TIMEOUT", "600"))})
        rows.append({"prompt": t.initial_user_prompt(), "label": t.task_id,
                     "metadata": {"task": t.to_row_kwargs(max_turns=max_turns)}})
    rows.sort(key=lambda r: r["label"])
    eval_rows = rows[4::5]                             # disjoint held-out instances
    train_rows = [r for r in rows if r not in eval_rows]
    DATA.mkdir(exist_ok=True)
    for name, rs in (("swebench_train.jsonl", train_rows), ("swebench_eval.jsonl", eval_rows)):
        (DATA / name).write_text("".join(json.dumps(r) + "\n" for r in rs))
    print(f"wrote {len(train_rows)} train + {len(eval_rows)} eval rows -> {DATA}/swebench_*.jsonl")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-only", action="store_true", help="rebuild data/ from swebench/manifest.json")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    mpath = OUT / "manifest.json"
    if not a.data_only:
        specs = load_specs()
        picked = select(load_rows(), specs)
        instances = [manifest_entry(c, specs) for c in picked]
        mpath.write_text(json.dumps({"source": DATASET, "specs_from": SWEBENCH_PKG,
                                     "instances": instances}, indent=1))
        print(f"selected {len(instances)} instances -> {mpath}")
        for e in instances:
            print(f"  {e['task']:34s} {e['repo']:22s} v{e['version']:6s} F2P={len(e['FAIL_TO_PASS'])} "
                  f"P2P={len(e['PASS_TO_PASS']):3d} image={e['image_mb']} MB")
    write_data(json.loads(mpath.read_text())["instances"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
