#!/usr/bin/env python3
"""The mini_claude agent loop, run INSIDE the sandbox session (HARNESS_WHERE=session).

Stdlib only and Python 3.6+ so it runs on any task image that has python3 (Terminal-Bench images,
the generic CPU image, the g6 farm) without an image rebuild. `in_session_harness.py` writes this
file plus a config JSON into `.acs_h/` of the episode's working directory, starts it detached with
one exec, and long-polls for `.acs_h/done`.

The loop is MiniClaudeHarness's: POST /v1/messages to slime's AnthropicAdapter (the session id in
the Bearer header keys the trajectory), run each tool_use locally (write_file / bash / read_file,
relative to this working directory), feed the results back, stop on a turn without a tool call or
at the turn / wall budget. The tokens are recorded by the adapter in the trainer, not here.

Outputs in `.acs_h/`: status.json (counters, rewritten every turn; the final one adds the
transcript), then an empty `done` marker. stop == "unreachable" means the adapter could not be
reached before any model call, so the trainer may fall back to running the loop itself.
"""

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


def _clip(text, limit):
    if len(text) <= limit:
        return text
    half = limit // 2
    return "%s\n... [truncated %d chars] ...\n%s" % (text[:half], len(text) - limit, text[-half:])


def _bash(command, timeout):
    proc = subprocess.Popen(["bash", "-lc", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            stdin=subprocess.DEVNULL, universal_newlines=True, start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout)
        return proc.returncode, out, err
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        # A process the command detached into ANOTHER session (nohup/setsid server) survives killpg and
        # keeps the pipe open: a bare communicate() would block until the harness's hard deadline
        # (~1 h, seen 2026-10-04 on play-lord). Wait briefly, then drop the pipes.
        try:
            out, _ = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            for f in (proc.stdout, proc.stderr):
                try:
                    f.close()
                except Exception:  # noqa: BLE001
                    pass
            out = ""
        return 124, out or "", "timeout"


def _tool(name, inp, cfg, status):
    try:
        if name == "write_file":
            path = str(inp.get("path", "")).strip()
            if not path:
                return "error: write_file requires a non-empty 'path'"
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "w") as f:
                f.write(str(inp.get("content", "")))
            return "wrote %s" % path
        if name == "read_file":
            path = str(inp.get("path", "")).strip()
            if not path:
                return "error: read_file requires a non-empty 'path'"
            with open(path, errors="replace") as f:
                return _clip(f.read(), cfg["max_tool_chars"])
        if name == "bash":
            command = str(inp.get("command", "")).strip()
            if not command:
                return "error: bash requires a non-empty 'command'"
            code, out, err = _bash(command, cfg["exec_timeout"])
            status["last_exit_code"] = code
            return _clip("exit_code: %s\n--- stdout ---\n%s\n--- stderr ---\n%s" % (code, out, err),
                         cfg["max_tool_chars"])
        return "error: unknown tool %r" % (name,)
    except Exception as e:  # surface tool failures to the model, like MiniClaudeHarness
        return "error: tool %s raised %s: %s" % (name, type(e).__name__, e)


def _save(state_dir, status, done=False):
    tmp = os.path.join(state_dir, "status.json.tmp")
    with open(tmp, "w") as f:
        json.dump(status, f)
    os.replace(tmp, os.path.join(state_dir, "status.json"))
    if done:
        open(os.path.join(state_dir, "done"), "w").close()


def main(cfg_path):
    state_dir = os.path.dirname(os.path.abspath(cfg_path))
    with open(cfg_path) as f:
        cfg = json.load(f)
    os.remove(cfg_path)   # the prompt + session id need not linger where the agent can read them
    status = {"turns": 0, "tool_calls": 0, "stop": "none", "last_exit_code": None, "error": ""}
    messages = [{"role": "user", "content": cfg["prompt"]}]
    deadline = time.time() + cfg["wall_sec"]
    url = cfg["adapter_url"].rstrip("/") + "/v1/messages"
    headers = {"Authorization": "Bearer " + cfg["session_id"], "anthropic-version": "2023-06-01",
               "content-type": "application/json"}

    # Fail fast (not after request_timeout) when the adapter's port is not reachable from here.
    u = urllib.parse.urlparse(cfg["adapter_url"])
    try:
        socket.create_connection((u.hostname, u.port or 80), timeout=cfg.get("connect_timeout", 5)).close()
    except OSError as e:
        status.update(stop="unreachable", error="%s: %s" % (type(e).__name__, e))
        status["transcript"] = messages
        _save(state_dir, status, done=True)
        return

    try:
        for turn in range(cfg["max_turns"]):
            if time.time() > deadline:
                status["stop"] = "deadline"
                break
            body = {"model": cfg.get("model", "slime-actor"), "max_tokens": cfg["max_tokens"],
                    "system": cfg["system"], "messages": messages, "tools": cfg["tools"]}
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=cfg["request_timeout"]) as resp:
                    data = json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                raise RuntimeError("adapter /v1/messages returned %s: %s" % (e.code, e.read()[:500]))
            blocks = data.get("content") or []
            status["turns"] = turn + 1
            status["stop"] = data.get("stop_reason") or "end_turn"
            messages.append({"role": "assistant", "content": blocks})
            uses = [b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_use"]
            if not uses:
                break
            results = []
            for tu in uses:
                status["tool_calls"] += 1
                results.append({"type": "tool_result", "tool_use_id": tu.get("id", ""),
                                "content": _tool(tu.get("name", ""), tu.get("input") or {}, cfg, status)})
            messages.append({"role": "user", "content": results})
            _save(state_dir, status)
        else:
            status["stop"] = "budget"
    except Exception as e:
        status.update(stop="error", error="%s: %s" % (type(e).__name__, e))
    status["transcript"] = messages
    _save(state_dir, status, done=True)


if __name__ == "__main__":
    main(sys.argv[1])
