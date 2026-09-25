#!/usr/bin/env python3
# Live isolation probe for the Ray farm (run on the HOST while `bash 4_deploy_runtime.sh` has the local
# farm up on 127.0.0.1:8080). Output of the 2026-09-25 run: logs/isolation_probe_20260925.log.
# Note: the slim image has no `ps`, so the "ps"/"procs" lines print nothing; the farm's own
# `left_procs` (a /proc scan by uid) in the close result is the authoritative leftover count.
"""Live isolation probe against the local farm (fractional mode) at 127.0.0.1:8080."""
import json, subprocess, threading, urllib.request
URL = "http://127.0.0.1:8080/invocations"
def inv(**req):
    r = urllib.request.Request(URL, json.dumps(req).encode(), {"Content-Type": "application/json",
        "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": "probe-isolation-0000000000000000000000"})
    try:
        return json.loads(urllib.request.urlopen(r, timeout=600).read())
    except urllib.error.HTTPError as e:
        return {"HTTPError": e.code, "body": e.read().decode()[:300]}
def ex(sub, cmd, t=300): return inv(op="exec", sub_session=sub, command=cmd, timeout=t)
def dx(cmd): return subprocess.run(["docker", "exec", "acs_local_gpu_sbx", "sh", "-c", cmd], capture_output=True, text=True).stdout
P = lambda k, v: print(f"== {k}: {json.dumps(v)[:900]}", flush=True)
P("ping", inv(op="ping", sub_session="probe"))
A, B = "probeA", "probeB"
P("open A", inv(op="open", sub_session=A)); P("open B", inv(op="open", sub_session=B))
P("A id", ex(A, "id; echo HOME=$HOME TMPDIR=$TMPDIR TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR; pwd; ls -ld . ..; echo TOPSECRET > secret.txt; ls -l secret.txt; nproc"))
P("B id", ex(B, "id; echo HOME=$HOME; pwd; ls -ld ."))
apath = ex(A, "pwd")["stdout"].strip()
P("B cat A", ex(B, f"cat {apath}/secret.txt; ls {apath}; ls $(dirname {apath}); ls $(dirname $(dirname {apath}))"))
P("B cat A home", ex(B, "ls -la /var/tmp/b4-home/ ; ls /var/tmp/b4-home/uid-20000"))
for p in (f"../../actor-0/{A}/secret.txt", f"{apath}/secret.txt"):
    P(f"B read_file {p}", inv(op="read_file", sub_session=B, path=p))
P("B symlink", ex(B, f"ln -s {apath} link; cat link/secret.txt"))
P("B read_file link", inv(op="read_file", sub_session=B, path="link/secret.txt"))
P("B kill A procs", ex(B, "ps -eo uid,pid,args | head -30; kill -9 $(ps -u 20000 -o pid= | head -1) 2>&1 || true"))
P("nvcc/lock", ex(A, "which nvcc; ls -l /dev/shm/b4-nvcc.lock; python3 -c \"import sys;print(open('/dev/shm/b4-nvcc.lock').read()=='' )\"; python3 -c 'import sitecustomize,inspect;print(sitecustomize.__file__)'"))
LOAD = r'''
import torch, time
from torch.utils.cpp_extension import load_inline
src = "torch::Tensor add1(torch::Tensor x){return x+1;}"
t=time.time(); m = load_inline(name="probe_ext", cpp_sources=src, functions=["add1"], verbose=False)
print("load_inline ok", round(time.time()-t,1), "s", m.add1(torch.zeros(2)).tolist())
import os; print("ext dir", os.environ.get("TORCH_EXTENSIONS_DIR")); os.system("ls -ld $TORCH_EXTENSIONS_DIR $TORCH_EXTENSIONS_DIR/*")
'''
P("A load_inline (cpp, uid)", ex(A, f"python3 - <<'PY'\n{LOAD}\nPY", 600))
CUDA = r'''
import torch, os, time
free, total = torch.cuda.mem_get_info()
print("dev", torch.cuda.get_device_name(0), "CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"), "pid", os.getpid(), "uid", os.getuid())
x = torch.empty(int(0.3*total), dtype=torch.uint8, device="cuda"); print("0.3x total alloc OK")
try:
    y = torch.empty(int(0.3*total), dtype=torch.uint8, device="cuda"); print("0.6x total alloc OK  (cap NOT applied)")
except torch.OutOfMemoryError as e: print("0.6x total -> OOM (cap applied):", str(e).split(".")[0][:160])
time.sleep(20); print("done")
'''
res = {}
def run(s): res[s] = ex(s, f"python3 - <<'PY'\n{CUDA}\nPY", 300)
ts = [threading.Thread(target=run, args=(s,)) for s in (A, B)]
[t.start() for t in ts]
import time; time.sleep(12)
print("== nvidia-smi compute apps (host):", subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader", "-i", "6"], capture_output=True, text=True).stdout)
print("== container ps during CUDA:", dx("ps -eo uid,pid,rss,args | grep -v ' 0 ' | head; ps -eo uid,args | grep python3 | grep -v grep | awk '{print $1}' | sort | uniq -c"))
[t.join() for t in ts]
P("A cuda", res[A]); P("B cuda", res[B])
P("A spawn escapee", ex(A, "setsid nohup sleep 1000 >/dev/null 2>&1 & (sleep 1001 &) ; sleep 1; ps -u 20000 -o pid,args"))
P("A timeout", ex(A, "sleep 30", 3))
print("== uid 20000 procs before close:", dx("ps -u 20000 -o pid,stat,args"))
P("close A", inv(op="close", sub_session=A))
print("== uid 20000 procs after close:", repr(dx("ps -u 20000 -o pid,stat,args --no-headers")))
print("== A dir after close:", dx(f"ls -ld {apath} 2>&1"))
P("close B", inv(op="close", sub_session=B))
P("reopen A same slot?", inv(op="open", sub_session="probeA2")); P("A2 id", ex("probeA2", "id -u; pwd; ls -a")); P("close A2", inv(op="close", sub_session="probeA2"))
