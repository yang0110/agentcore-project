"""Application-level snapshot / restore — the ONLY feasible "GPU instance snapshot".

AgentCore has NO VM/instance snapshot (see docs/gpu-snapshot.md): you don't own the managed EC2, an
EBS snapshot is disk-only (not the paused VRAM/CUDA context), and a GPU instance recycle comes back
blank. The workable path is "tier (a)": serialize the sandbox WORKSPACE (files + small state) to a
store YOU own, and restore it onto a fresh session after a recycle. That is what this module does.

A snapshot is a manifest ``{relative_path: text}``. ``snapshot_workspace`` reads a listed set of
files off any `Sandbox`; ``restore_workspace`` writes them back into a fresh one. Backend-agnostic —
LocalSandbox, AgentCoreSandbox (Code Interpreter), or the GPU Runtime sandbox all work.

Two stores:
  * ``LocalSnapshotStore`` — a host directory. Offline demo / CI, no AWS.
  * ``S3SnapshotStore``    — boto3 put/get. The recommended production target: no mount, AZ-agnostic,
                             VPC-optional (see docs/gpu-snapshot.md "S3 vs FSx" — FSx/EFS need a mount
                             AgentCore won't give you). Import is lazy so offline never needs boto3.

This is a *snapshot* only in the application sense; there is no process/GPU state here (that would be
cuda-checkpoint + CRIU, tier (b): heavy, fragile, and unnecessary for RL — re-seeding is cheap).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol, runtime_checkable

Manifest = dict[str, str]


@runtime_checkable
class SnapshotStore(Protocol):
    def save(self, key: str, manifest: Manifest) -> None: ...
    def load(self, key: str) -> Manifest | None: ...


class LocalSnapshotStore:
    """Snapshots as JSON files under a host directory. Offline stand-in for S3."""

    def __init__(self, root: str | os.PathLike | None = None) -> None:
        self.root = Path(root or os.environ.get("SNAPSHOT_DIR", "/tmp/agentcore_snapshots"))
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def save(self, key: str, manifest: Manifest) -> None:
        self._path(key).write_text(json.dumps(manifest))

    def load(self, key: str) -> Manifest | None:
        p = self._path(key)
        if not p.exists():
            return None
        return json.loads(p.read_text())


class S3SnapshotStore:
    """Snapshots as one JSON object per key in S3 — the recommended production store.

    No mount, works from any instance in any AZ. Key layout: ``s3://<bucket>/<prefix>/<key>.json``.
    """

    def __init__(self, bucket: str | None = None, prefix: str = "agentcore-snapshots",
                 region: str | None = None) -> None:
        self.bucket = bucket or os.environ["SNAPSHOT_S3_BUCKET"]
        self.prefix = prefix.strip("/")
        self._region = region or os.environ.get("AGENTCORE_REGION", "us-west-2")
        self._client = None

    def _c(self):
        if self._client is None:
            import boto3  # lazy: offline paths never import boto3
            self._client = boto3.client("s3", region_name=self._region)
        return self._client

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}.json"

    def save(self, key: str, manifest: Manifest) -> None:
        self._c().put_object(Bucket=self.bucket, Key=self._key(key),
                             Body=json.dumps(manifest).encode())

    def load(self, key: str) -> Manifest | None:
        try:
            resp = self._c().get_object(Bucket=self.bucket, Key=self._key(key))
        except Exception as exc:  # noqa: BLE001 — NoSuchKey (and friends) => nothing to restore
            if "NoSuchKey" in type(exc).__name__ or "NoSuchKey" in str(exc):
                return None
            raise
        return json.loads(resp["Body"].read().decode())


async def snapshot_workspace(sb, paths: list[str], store: SnapshotStore, key: str) -> Manifest:
    """Read `paths` off the sandbox and persist them under `key`. Missing files are skipped."""
    manifest: Manifest = {}
    for p in paths:
        try:
            manifest[p] = await sb.read_file(p)
        except Exception:  # noqa: BLE001 — a not-yet-created file is fine to skip
            continue
    store.save(key, manifest)
    return manifest


async def restore_workspace(sb, store: SnapshotStore, key: str) -> list[str]:
    """Write a previously-saved manifest back into a fresh sandbox. Returns restored paths."""
    manifest = store.load(key)
    if not manifest:
        return []
    for p, text in manifest.items():
        await sb.write_file(p, text)
    return list(manifest)
