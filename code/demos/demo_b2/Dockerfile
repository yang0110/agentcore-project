# Demo B2 sandbox container — the CUSTOM AgentCore Runtime image the Claude Code agent runs in.
#
# One of these is spun up PER EPISODE. Unlike the stock Code Interpreter microVM, the Claude Code
# harness needs: Node 22 + the `claude` CLI, a non-root multi-user setup (claude -p runs under an
# unprivileged user via setsid), and — because this is the GPU demo — CUDA + a CUDA build of torch,
# since the KernelBench checker (`python3 run_checks.py`) imports torch and runs the candidate
# kernel on `cuda`. (Demo A2's container is the same idea WITHOUT the CUDA/torch layer.)
#
# The Claude Code tarball is passed at build time so the exact pinned version is reproducible.
# ANTHROPIC_BASE_URL is NOT baked here; the harness sets it per-episode to the slime adapter.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ARG SLIME_AGENT_CC_TARBALL=claude-code-bundled-2.1.263.tgz  # self-contained bundle from _agent_assets/
ARG TORCH_VERSION=2.5.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124

ENV DEBIAN_FRONTEND=noninteractive

# Base tools + Node 22 (NodeSource) + python for the checker.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg git util-linux python3 python3-pip \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Claude Code CLI from the pinned self-contained bundle (installs offline). The bundle carries the
# linux-x64 native binary as a bundleDependency, but the wrapper's postinstall keys off
# optionalDependencies (which the bundle strips), so we copy the native binary over the bin/ stub
# ourselves — deterministic, no network. (Same fix as demo_a2/Dockerfile.)
COPY ${SLIME_AGENT_CC_TARBALL} /tmp/claude-code.tgz
RUN test -s /tmp/claude-code.tgz \
    && npm install -g --prefix=/usr/local --no-audit --no-fund --offline /tmp/claude-code.tgz \
    && CC=/usr/local/lib/node_modules/@anthropic-ai/claude-code \
    && cp -f "$CC/node_modules/@anthropic-ai/claude-code-linux-x64/claude" "$CC/bin/claude.exe" \
    && chmod +x "$CC/bin/claude.exe" \
    && rm -f /tmp/claude-code.tgz \
    && claude --version

# CUDA build of torch — the KernelBench checker imports torch and runs the kernel on cuda.
RUN pip3 install --no-cache-dir --index-url ${TORCH_INDEX_URL} "torch==${TORCH_VERSION}"

# Non-root multi-user: claude -p runs under an unprivileged account (setsid detaches the session).
RUN useradd --create-home --shell /bin/bash agent
USER agent
WORKDIR /workspace

# Sanity: the checker must be able to import torch (CUDA visible at RUN time in the target host).
RUN python3 -c "import torch; print('torch', torch.__version__)"
