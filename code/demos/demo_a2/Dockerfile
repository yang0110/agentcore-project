# Demo A2 sandbox image — the CUSTOM AgentCore Runtime that hosts the real Claude Code CLI (CPU / MBPP).
#
# A2/B2 target: the sandbox is its OWN AgentCore Runtime (AgentCoreRuntimeSandbox), NOT the managed
# Code Interpreter. The trainer-side harness (agentcore_slime_demo.claude_code_generate) drives this
# image over invoke_agent_runtime, and the Claude Code CLI runs INSIDE it. So this image = the proven
# sandbox_server_agent.py server (like code/deploy/Dockerfile.sandbox.cpu) PLUS everything the CLI
# needs: Node 22 + the `claude` CLI + setsid, all BAKED IN.
#
# Two things this image does that the plain CPU sandbox does not:
#   1. Bakes Node 22 + Claude Code from the offline tarballs in code/demos/_agent_assets/ (fetch them
#      with code/demos/setup_claude_code.sh). The trainer SKIPS harness.install_cli for the runtime
#      sandbox (streaming the multi-MB binaries as text over invoke corrupts them), so they MUST be here.
#   2. Runs EVERYTHING as root. AgentCore runs the container as uid 0 with an EMPTY capability set, so
#      `su`/`chown` are impossible; the sandbox is root-only and the CLI runs as root (IS_SANDBOX=1).
# No GPU: MBPP checks are pure-CPU assert tests.
#
# Build context MUST be code/ (so both agentcore_slime_demo/ and demos/_agent_assets/ are COPY-able):
#   docker build -f code/demos/demo_a2/Dockerfile -t <ecr>/acs-sandbox-a2:latest code/
FROM debian:bookworm-slim

# util-linux -> setsid (the harness detaches the agent with setsid); tar/xz for the node tarball;
# python3 + pip for the server + MBPP checker; ca-certificates for any TLS the agent does at runtime;
# git -> slime's ensure_agent_user runs `git config --system ...` and claude may `git diff` the workspace.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates tar xz-utils python3 python3-pip util-linux git \
    && rm -rf /var/lib/apt/lists/*

# The AgentCore Runtime server framework (same dependency as Dockerfile.sandbox.cpu).
RUN pip3 install --no-cache-dir --break-system-packages bedrock-agentcore

# ---- Node 22 (from the offline host tarball) -----------------------------------------------
ARG NODE_TARBALL=node-v22.11.0-linux-x64.tar.xz
COPY demos/_agent_assets/${NODE_TARBALL} /tmp/node.tar.xz
RUN mkdir -p /opt/node && tar -xJf /tmp/node.tar.xz -C /opt/node --strip-components=1 \
    && ln -s /opt/node/bin/node /usr/local/bin/node \
    && ln -s /opt/node/bin/npm  /usr/local/bin/npm \
    && rm /tmp/node.tar.xz \
    && node --version && npm --version

# ---- Claude Code (self-contained bundle; installs offline) ---------------------------------
# The bundle carries the wrapper package + the linux-x64 native binary as a bundleDependency.
# npm installs it offline, but the wrapper's postinstall keys off optionalDependencies (which the
# bundle strips) so it won't materialize the native binary — we copy it over the bin/ stub
# ourselves. Deterministic, no network.
ARG CC_TARBALL=claude-code-bundled-2.1.263.tgz
COPY demos/_agent_assets/${CC_TARBALL} /tmp/claude-code.tgz
RUN npm install -g --prefix=/usr/local --no-audit --no-fund --offline /tmp/claude-code.tgz \
    && CC=/usr/local/lib/node_modules/@anthropic-ai/claude-code \
    && cp -f "$CC/node_modules/@anthropic-ai/claude-code-linux-x64/claude" "$CC/bin/claude.exe" \
    && chmod +x "$CC/bin/claude.exe" \
    && rm -f /tmp/claude-code.tgz \
    && claude --version

# ---- Root-only workspace (AgentCore microVM reality) ---------------------------------------
# AgentCore runs the container as uid 0 but with an EMPTY Linux capability set (CapEff/CapBnd=0):
# no CAP_SETUID/SETGID (so `su agent` is impossible), no CAP_CHOWN/FOWNER/DAC_OVERRIDE (so root can
# neither chown nor write files it does not own). The multi-user design therefore cannot work here —
# EVERYTHING runs as root. We keep an `agent` home only because slime's harness writes config there,
# and make both it and /workspace world-writable so root can write regardless of the recorded owner.
RUN useradd -m -s /bin/bash agent \
    && mkdir -p /workspace \
    && chmod 777 /workspace /home/agent
# Pre-ack onboarding in ROOT's HOME (claude runs as root => reads /root/.claude*), so `claude -p`
# starts headless. (slime's write_config also writes /home/agent/.claude at runtime; harmless.)
RUN mkdir -p /root/.claude \
    && printf '%s' '{"hasCompletedOnboarding": true, "bypassPermissionsModeAccepted": true}' \
       | tee /root/.claude.json /root/.claude/settings.json >/dev/null
# A no-op `chown` shim ahead of /bin on PATH: slime's harness issues `chown agent:agent ...` with
# check=True, which would fail under zero caps. Ownership is irrelevant when all is root, so exit 0.
RUN printf '#!/bin/sh\nexit 0\n' > /usr/local/bin/chown && chmod +x /usr/local/bin/chown
# slime's ensure_agent_user runs `git config --system --add safe.directory '*'`, which targets
# /etc/gitconfig — but /etc is owned by `nobody` and unwritable to our zero-cap root, so the lock
# fails. Redirect git's system config to a root-owned path via GIT_CONFIG_SYSTEM (git >=2.32 honors
# it) and seed it at build time; the runtime re-run then just appends there. See ENV below.
RUN GIT_CONFIG_SYSTEM=/root/gitconfig-system git config --system --add safe.directory '*'

# ---- The AgentCore Runtime server (self-contained: stdlib + bedrock_agentcore, no relative imports) ----
WORKDIR /app
COPY agentcore_slime_demo/sandbox_server_agent.py /app/sandbox_server_agent.py

# SANDBOX_PYLIB stays UNSET -> the GPU/torch prepare path in sandbox_server_agent.py is dormant (CPU image).
# IS_SANDBOX=1 lets `claude` run as root with --permission-mode bypassPermissions (it otherwise refuses).
ENV SANDBOX_WORKDIR=/workspace
ENV AGENTCORE_WORKDIR=/workspace
ENV IS_SANDBOX=1
# git's system config -> root-owned path (see the seed RUN above); /etc is unwritable to zero-cap root.
ENV GIT_CONFIG_SYSTEM=/root/gitconfig-system
EXPOSE 8080
CMD ["python3", "-m", "sandbox_server_agent"]
