#!/usr/bin/env python3
"""Survey the prebuilt images of EVERY SWE-bench Verified instance vs the AgentCore microVM limits
(arm64 only, 2 GB image cap).

For each of the 500 rows, looks up four prebuilt images and records the compressed size of each
(what a pull downloads; the number ECR reports as `imageSizeInBytes`), or blank if it doesn't exist:

    hub_x86_mb    Docker Hub  swebench/sweb.eval.x86_64.<id __ -> _1776_>   (official harness images)
    hub_arm64_mb  Docker Hub  swebench/sweb.eval.arm64.<id __ -> _1776_>
    ghcr_arm64_mb GHCR        epoch-research/swe-bench.eval.arm64.<id>       (Epoch AI rebuilds)
    ghcr_x86_mb   GHCR        epoch-research/swe-bench.eval.x86_64.<id>

Hub sizes come from the tags API (per-architecture entry). GHCR sizes are the sum of the manifest's
layer sizes; the architecture is read from the image config, not trusted from the name
(`ghcr_arm64_arch`). Docker Hub rate-limits anonymous reads (429), so failed Hub rows are retried
serially with the server's Retry-After.

    python3 survey_image_sizes.py            # -> swebench/verified_image_sizes.csv + summary

No AWS, no Docker. Network: huggingface.co, hub.docker.com, ghcr.io.
"""
import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.error
import urllib.request

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

DATASET = "princeton-nlp/SWE-bench_Verified"
CAP_MB = 2048                                               # AgentCore quota L-0A9E32B3 (not adjustable)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "swebench", "verified_image_sizes.csv")
ERR = "error"                                               # retryable (rate limit / network)
MANIFEST_ACCEPT = ", ".join([
    "application/vnd.oci.image.index.v1+json", "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json", "application/vnd.docker.distribution.manifest.v2+json"])


def load_rows() -> pd.DataFrame:
    f = [x for x in list_repo_files(DATASET, repo_type="dataset") if x.endswith(".parquet")][0]
    return pd.read_parquet(hf_hub_download(DATASET, f, repo_type="dataset"))[["instance_id", "repo", "version"]]


def _get(url: str, headers: dict | None = None) -> dict:
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=30))


def hub_size(iid: str, arch: str, patient: bool):
    """Compressed MB of swebench/sweb.eval.<arch>.<id> on Docker Hub; None if absent; ERR if unknown."""
    tag = iid.replace("__", "_1776_").lower()
    want = "amd64" if arch == "x86_64" else arch
    url = f"https://hub.docker.com/v2/repositories/swebench/sweb.eval.{arch}.{tag}/tags/latest"
    for _ in range(6 if patient else 1):
        try:
            imgs = _get(url).get("images", [])
            s = [i.get("size", 0) for i in imgs if i.get("architecture") == want]
            return round(s[0] / 2**20, 1) if s else None
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if patient:
                time.sleep(int(e.headers.get("Retry-After") or 20))
        except Exception:
            if patient:
                time.sleep(5)
    return ERR


def ghcr_size(iid: str, arch: str):
    """(compressed MB, config architecture) of epoch-research/swe-bench.eval.<arch>.<id> on GHCR."""
    repo = f"epoch-research/swe-bench.eval.{arch}.{iid}"
    for _ in range(4):
        try:
            tok = _get(f"https://ghcr.io/token?scope=repository:{repo}:pull")["token"]
            h = {"Authorization": f"Bearer {tok}", "Accept": MANIFEST_ACCEPT}
            m = _get(f"https://ghcr.io/v2/{repo}/manifests/latest", h)
            if "manifests" in m:                            # an index: pick the matching platform
                want = "amd64" if arch == "x86_64" else arch
                sub = [x for x in m["manifests"] if (x.get("platform") or {}).get("architecture") == want]
                if not sub:
                    return None, ""
                m = _get(f"https://ghcr.io/v2/{repo}/manifests/{sub[0]['digest']}", h)
            cfg = _get(f"https://ghcr.io/v2/{repo}/blobs/{m['config']['digest']}", {"Authorization": h["Authorization"]})
            return round(sum(layer["size"] for layer in m["layers"]) / 2**20, 1), cfg.get("architecture", "")
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):                   # GHCR answers 401/403 for a missing repo
                return None, ""
            time.sleep(3)
        except Exception:
            time.sleep(3)
    return ERR, ERR


def main() -> None:
    df = load_rows()
    with cf.ThreadPoolExecutor(16) as ex:
        for col, arch in (("ghcr_arm64", "arm64"), ("ghcr_x86", "x86_64")):
            res = list(ex.map(lambda i: ghcr_size(i, arch), df.instance_id))
            df[f"{col}_mb"], df[f"{col}_arch"] = zip(*res)
        for col, arch in (("hub_x86_mb", "x86_64"), ("hub_arm64_mb", "arm64")):   # fast pass; 429s -> ERR
            df[col] = list(ex.map(lambda i: hub_size(i, arch, patient=False), df.instance_id))
    for col, arch in (("hub_x86_mb", "x86_64"), ("hub_arm64_mb", "arm64")):       # slow, rate-limit-aware pass
        for k in df.index[df[col].astype(str) == ERR]:
            df.at[k, col] = hub_size(df.at[k, "instance_id"], arch, patient=True)
            time.sleep(0.7)
    df = df.drop(columns=["ghcr_x86_arch"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"wrote {OUT}; distinct (repo, version) environments: {df.groupby(['repo', 'version']).ngroups}")
    for col in ("hub_x86_mb", "hub_arm64_mb", "ghcr_arm64_mb", "ghcr_x86_mb"):
        v = pd.to_numeric(df[col], errors="coerce").dropna()
        n_err = int((df[col].astype(str) == ERR).sum())
        if len(v):
            print(f"{col:14} present {len(v):3}/{len(df)}  p50={v.median():.0f} p90={v.quantile(.9):.0f} "
                  f"max={v.max():.0f} MB  <= {CAP_MB}: {(v <= CAP_MB).sum()}  unresolved={n_err}")
        else:
            print(f"{col:14} present   0/{len(df)}  unresolved={n_err}")
    arm = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in ("hub_arm64_mb", "ghcr_arm64_mb")})
    best = arm.min(axis=1)
    print(f"rows with ANY arm64 image: {best.notna().sum()}/{len(df)}; with an arm64 image <= {CAP_MB} MB: "
          f"{(best <= CAP_MB).sum()}/{len(df)}")
    print("ghcr arm64 config architectures:", df.ghcr_arm64_arch.value_counts().to_dict())
    if (df[["hub_x86_mb", "hub_arm64_mb", "ghcr_arm64_mb", "ghcr_x86_mb"]].astype(str) == ERR).any().any():
        sys.exit("some lookups unresolved (rate limit / network) — re-run later")


if __name__ == "__main__":
    main()
