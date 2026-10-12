#!/usr/bin/env python3
"""Stage what the cluster runs from into s3://<bucket>/work/ (INSIDE slime_trainer: the HF dir's symlinks
point into /root/.cache/huggingface, which only resolves in the container). `hyperpod.py pull` then
syncs it to /fsx/work on the cluster, so every path run.sh uses (/work/code/...) is the same there.

    python3 deploy/hyperpod/stage_s3.py            # this demo (no ckpt/ logs/) + the 9B HF dirs + base ckpt

Uploads only what changed (same key and size, and the object is newer than the file -> skipped), so a
re-run after a code edit uploads just the edited files.
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig

HERE = Path(__file__).resolve().parent
DEMO_DIR = HERE.parent.parent
WORK = Path(os.environ.get("CONTAINER_REPO", "/work"))
P4 = WORK / "code" / "p4_slime_train"
# Relative to /work. The HF + base-ckpt paths are run.sh's qwen3.5-9B defaults.
TREES = [
    (DEMO_DIR, {"ckpt", "logs", "__pycache__"}),
    (P4 / "hf" / "Qwen3.5-9B-nothink", set()),
    (P4 / "hf" / "Qwen3.5-9B-nothink-sglang", set()),
    (P4 / "ckpt" / "Qwen3.5-9B_torch_dist", set()),
]
REGION = os.environ.get("AWS_REGION", "us-west-2")


def files(root: Path, skip: set[str]):
    for dirpath, dirnames, names in os.walk(root, followlinks=True):
        dirnames[:] = [n for n in dirnames if n not in skip]
        for n in names:
            p = Path(dirpath) / n
            if p.is_file():                            # follows symlinks (the HF snapshot files)
                yield p


def main() -> int:
    state = json.loads((HERE / "hyperpod.json").read_text())
    bucket = state["bucket"]
    s3 = boto3.client("s3", region_name=REGION)
    have: dict[str, tuple[int, float]] = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix="work/"):
        for o in page.get("Contents", []):
            have[o["Key"]] = (o["Size"], o["LastModified"].timestamp())
    todo = []
    for root, skip in TREES:
        if not root.exists():
            print(f"!! missing {root} (prepare_qwen35.sh / the 9B ckpt conversion first)", file=sys.stderr)
            return 2
        for p in files(root, skip):
            key = "work/" + str(p.relative_to(WORK))
            st = p.stat()
            got = have.get(key)
            if got and got[0] == st.st_size and got[1] >= st.st_mtime:
                continue
            todo.append((p, key, st.st_size))
    total = sum(s for _p, _k, s in todo)
    print(f"# {len(todo)} file(s), {total / 2**30:.1f} GiB to s3://{bucket}/work/", flush=True)
    cfg = TransferConfig(multipart_chunksize=64 * 2**20, max_concurrency=8)

    def up(item):
        p, key, _s = item
        s3.upload_file(str(p), bucket, key, Config=cfg)
        return key

    with ThreadPoolExecutor(8) as ex:
        for i, _key in enumerate(ex.map(up, todo), 1):
            if i % 200 == 0 or i == len(todo):
                print(f"#   {i}/{len(todo)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
