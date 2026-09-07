# Demo A2 — custom container for the CLAUDE CODE harness (CPU / MBPP).
#
# This is the image the sandbox launches to host the Claude Code CLI. The restricted serverless
# Code Interpreter cannot run a full multi-user CLI, so Demo A2 needs this custom container: a real
# multi-user box where an `agent` user + `setsid` + `node`/`npm` work and Claude Code can run.
#
# Node 22 + Claude Code are BAKED IN at build time from the tarballs in code/demos/_agent_assets/
# (fetch them with code/demos/setup_claude_code.sh). Baking them in makes cold starts fast and
# means the sandbox needs NO network to install the agent. The claude-code tarball is the
# SELF-CONTAINED bundle (wrapper + the linux-x64 native binary as a bundleDependency), so the
# `npm install -g` below runs fully OFFLINE. No GPU: MBPP checks are pure-CPU assert tests.
#
# Build (context = the assets dir that holds the two tarballs):
#   docker build -f code/demos/demo_a2/Dockerfile -t agentcore-demo-a2 code/demos/_agent_assets
FROM debian:bookworm-slim

# util-linux -> setsid (the harness detaches the agent with setsid); tar/xz for the node tarball;
# python3 for the MBPP checker; ca-certificates for any TLS the agent does at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates tar xz-utils python3 util-linux \
    && rm -rf /var/lib/apt/lists/*

# ---- Node 22 (from the host tarball) -------------------------------------------------------
ARG NODE_TARBALL=node-v22.11.0-linux-x64.tar.xz
COPY ${NODE_TARBALL} /tmp/node.tar.xz
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
COPY ${CC_TARBALL} /tmp/claude-code.tgz
RUN npm install -g --prefix=/usr/local --no-audit --no-fund --offline /tmp/claude-code.tgz \
    && CC=/usr/local/lib/node_modules/@anthropic-ai/claude-code \
    && cp -f "$CC/node_modules/@anthropic-ai/claude-code-linux-x64/claude" "$CC/bin/claude.exe" \
    && chmod +x "$CC/bin/claude.exe" \
    && rm -f /tmp/claude-code.tgz \
    && claude --version

# ---- Multi-user, non-root agent account ----------------------------------------------------
# Claude Code and the sandbox exec run as this unprivileged user in an isolated workspace.
RUN useradd -m -s /bin/bash agent \
    && mkdir -p /workspace && chown agent:agent /workspace
ENV AGENTCORE_WORKDIR=/workspace
USER agent
WORKDIR /workspace
CMD ["sleep", "infinity"]
