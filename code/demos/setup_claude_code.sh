#!/bin/bash
# Fetch + build the agent assets used by demo_a2 / demo_b2 (the REAL Claude Code CLI path).
# Produces, in code/demos/_agent_assets/:
#   * node-v${NODE_VERSION}-linux-x64.tar.xz        — Node 22 runtime (baked into the container)
#   * claude-code-bundled-${CC_VERSION}.tgz         — SELF-CONTAINED Claude Code npm bundle
#
# Why a "bundle" and not just `npm pack @anthropic-ai/claude-code`: the published package is a thin
# wrapper whose real binary lives in a per-platform optionalDependency
# (@anthropic-ai/claude-code-linux-x64). A plain wrapper tgz has no binary, and installing it
# offline (no network in the sandbox/build) leaves `claude` a non-working stub. This script bundles
# the linux-x64 native package INTO the wrapper (as a bundleDependency) so `npm install -g` works
# fully offline. NOTE: the wrapper's postinstall keys off optionalDependencies (which we strip), so
# the Dockerfiles ALSO copy the native binary over bin/claude.exe after install — see those files.
#
# Requires: curl, npm, python3, tar. Re-running is idempotent (skips downloads that already exist).
set -euo pipefail

CC_VERSION="${CC_VERSION:-2.1.263}"
NODE_VERSION="${NODE_VERSION:-22.11.0}"
NATIVE_PKG="@anthropic-ai/claude-code-linux-x64"

ASSETS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_agent_assets"
mkdir -p "$ASSETS"
cd "$ASSETS"

# ---- 1. Node 22 runtime tarball ------------------------------------------------------------
NODE_TARBALL="node-v${NODE_VERSION}-linux-x64.tar.xz"
if [ ! -s "$NODE_TARBALL" ]; then
  echo "[setup] downloading $NODE_TARBALL"
  curl -fsSL -o "$NODE_TARBALL" "https://nodejs.org/dist/v${NODE_VERSION}/${NODE_TARBALL}"
else
  echo "[setup] have $NODE_TARBALL"
fi

# ---- 2. npm pack the wrapper + the linux-x64 native package ---------------------------------
WRAPPER_TGZ="anthropic-ai-claude-code-${CC_VERSION}.tgz"
NATIVE_TGZ="anthropic-ai-claude-code-linux-x64-${CC_VERSION}.tgz"
if [ ! -s "$WRAPPER_TGZ" ]; then
  echo "[setup] npm pack @anthropic-ai/claude-code@${CC_VERSION}"
  npm pack "@anthropic-ai/claude-code@${CC_VERSION}" >/dev/null
fi
if [ ! -s "$NATIVE_TGZ" ]; then
  echo "[setup] npm pack ${NATIVE_PKG}@${CC_VERSION}"
  npm pack "${NATIVE_PKG}@${CC_VERSION}" >/dev/null
fi

# ---- 3. Build the self-contained bundle -----------------------------------------------------
BUNDLE="claude-code-bundled-${CC_VERSION}.tgz"
echo "[setup] building $BUNDLE"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
tar -xzf "$WRAPPER_TGZ" -C "$WORK"                                   # -> $WORK/package
mkdir -p "$WORK/package/node_modules/@anthropic-ai/claude-code-linux-x64"
# The native tgz also unpacks to a top-level "package/" dir; land its contents in the nested path.
tar -xzf "$NATIVE_TGZ" -C "$WORK/package/node_modules/@anthropic-ai/claude-code-linux-x64" \
    --strip-components=1

# Rewrite package.json: move the native pkg to dependencies + bundleDependencies, drop the
# other-platform optionalDependencies (so the offline install resolves ONLY the bundled binary).
python3 - "$WORK/package/package.json" "$NATIVE_PKG" "$CC_VERSION" <<'PY'
import json, sys
path, native_pkg, version = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path) as f:
    pkg = json.load(f)
pkg["dependencies"] = {native_pkg: version}
pkg.pop("optionalDependencies", None)
pkg["bundleDependencies"] = [native_pkg]
with open(path, "w") as f:
    json.dump(pkg, f, indent=2)
PY

# Repack. `npm pack` on the staged dir bundles node_modules listed in bundleDependencies.
OUT="$(cd "$WORK/package" && npm pack --pack-destination "$ASSETS" 2>/dev/null | tail -1)"
mv -f "$ASSETS/$OUT" "$ASSETS/$BUNDLE"

echo "[setup] done:"
ls -la "$ASSETS/$NODE_TARBALL" "$ASSETS/$BUNDLE"
echo "[setup] verify offline install with: docker build -f demo_a2/Dockerfile -t agentcore-demo-a2 _agent_assets"
