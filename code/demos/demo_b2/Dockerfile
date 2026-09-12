# Demo B2 sandbox image — the CUSTOM AgentCore Runtime that hosts the real Claude Code CLI (GPU / KernelBench).
#
# Same idea as demo_a2/Dockerfile (server + baked Node/claude, everything runs as root) PLUS the
# CUDA/torch layer, because the KernelBench checker (`python3 run_checks.py`) imports torch and runs
# the candidate kernel on `cuda`. This image runs on a g6 CAPACITY PROVIDER instance (x86), reached
# by AgentCoreRuntimeSandbox over invoke_agent_runtime; the Claude Code CLI runs INSIDE it.
#
# Unlike code/deploy/Dockerfile.sandbox.gpu (B3), torch is BAKED IN here rather than installed onto a
# capacity-provider volume: B2 already needs a fat custom image (node+claude), so SANDBOX_PYLIB stays
# UNSET, the volume/`mode=prepare` path stays dormant, and the checker imports torch from system site-packages.
#
# Build context MUST be code/ (so agentcore_slime_demo/ and demos/_agent_assets/ are COPY-able):
#   docker build -f code/demos/demo_b2/Dockerfile -t <ecr>/acs-sandbox-b2:latest code/
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ARG CC_TARBALL=claude-code-bundled-2.1.263.tgz  # self-contained bundle from _agent_assets/
ARG TORCH_VERSION=2.5.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
ENV DEBIAN_FRONTEND=noninteractive

# Base tools + Node 22 (NodeSource) + python + setsid (util-linux).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg git util-linux python3 python3-pip \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# The AgentCore Runtime server framework (same dependency as Dockerfile.sandbox.gpu).
RUN pip3 install --no-cache-dir bedrock-agentcore

# Claude Code CLI from the pinned self-contained bundle (installs offline). Same native-binary fix
# as demo_a2/Dockerfile.
COPY demos/_agent_assets/${CC_TARBALL} /tmp/claude-code.tgz
RUN test -s /tmp/claude-code.tgz \
    && npm install -g --prefix=/usr/local --no-audit --no-fund --offline /tmp/claude-code.tgz \
    && CC=/usr/local/lib/node_modules/@anthropic-ai/claude-code \
    && cp -f "$CC/node_modules/@anthropic-ai/claude-code-linux-x64/claude" "$CC/bin/claude.exe" \
    && chmod +x "$CC/bin/claude.exe" \
    && rm -f /tmp/claude-code.tgz \
    && claude --version

# CUDA build of torch — baked in (not on a volume). The checker imports torch and runs on cuda.
RUN pip3 install --no-cache-dir --index-url ${TORCH_INDEX_URL} "torch==${TORCH_VERSION}"

# Root-only workspace (AgentCore microVM reality). AgentCore runs the container as uid 0 with an
# EMPTY capability set (CapEff/CapBnd=0): no CAP_SETUID/SETGID (`su agent` impossible), no
# CAP_CHOWN/FOWNER/DAC_OVERRIDE (root cannot chown or write files it does not own). So EVERYTHING runs
# as root; keep an `agent` home only for slime's harness config writes, both it and /workspace 777.
RUN useradd -m -s /bin/bash agent \
    && mkdir -p /workspace \
    && chmod 777 /workspace /home/agent
# Pre-ack onboarding in ROOT's HOME (claude runs as root) so `claude -p` starts headless.
RUN mkdir -p /root/.claude \
    && printf '%s' '{"hasCompletedOnboarding": true, "bypassPermissionsModeAccepted": true}' \
       | tee /root/.claude.json /root/.claude/settings.json >/dev/null
# No-op `chown` shim ahead of /bin on PATH: slime's harness runs `chown agent:agent ...` (check=True),
# which fails under zero caps; ownership is irrelevant when all is root, so exit 0.
RUN printf '#!/bin/sh\nexit 0\n' > /usr/local/bin/chown && chmod +x /usr/local/bin/chown
# slime's ensure_agent_user runs `git config --system --add safe.directory '*'` -> /etc/gitconfig, but
# /etc is unwritable to our zero-cap root. Redirect git's system config to a root-owned path via
# GIT_CONFIG_SYSTEM (git >=2.32) and seed it at build time; the runtime re-run just appends. See ENV.
RUN GIT_CONFIG_SYSTEM=/root/gitconfig-system git config --system --add safe.directory '*'

# Sanity: torch must import at build (CUDA availability is only true on the g6 host at runtime).
RUN python3 -c "import torch; print('torch', torch.__version__)"

# ---- The AgentCore Runtime server -----------------------------------------------------------
WORKDIR /app
COPY agentcore_slime_demo/sandbox_server_agent.py /app/sandbox_server_agent.py

# libcuda.so.1 is injected by the host into /usr/lib64; LD_LIBRARY_PATH MUST be set at PROCESS START
# (the loader caches its search path), so bake it into ENV — same fix as Dockerfile.sandbox.gpu.
# SANDBOX_PYLIB stays UNSET (torch is baked, not on a volume), so the prepare path stays dormant.
# IS_SANDBOX=1 lets `claude` run as root with --permission-mode bypassPermissions (it otherwise refuses).
ENV LD_LIBRARY_PATH=/usr/lib64
ENV SANDBOX_WORKDIR=/workspace
ENV AGENTCORE_WORKDIR=/workspace
ENV IS_SANDBOX=1
# git's system config -> root-owned path (see the seed RUN above); /etc is unwritable to zero-cap root.
ENV GIT_CONFIG_SYSTEM=/root/gitconfig-system
EXPOSE 8080
CMD ["python3", "-m", "sandbox_server_agent"]
