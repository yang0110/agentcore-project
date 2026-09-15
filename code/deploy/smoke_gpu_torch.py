#!/usr/bin/env python3
"""Cheap validation that the redeployed GPU sandbox (startup torch install) yields a working torch.
Warms the cold g6 (triggers the CMD pip install) and runs `import torch; cuda.is_available()`.
Retries transient 5xx during the g6 cold launch. Usage: AGENTCORE_SANDBOX_GPU_RUNTIME_ARN=... python3 smoke_gpu_torch.py
"""
import json
import os
import time

import boto3

REGION = os.environ.get("AWS_REGION", "us-west-2")
ARN = os.environ["AGENTCORE_SANDBOX_GPU_RUNTIME_ARN"]
SESSION = "smoke-gpu-torch-" + "x" * 40  # runtimeSessionId must be [33,128] chars

rt = boto3.client("bedrock-agentcore", region_name=REGION)
payload = {"op": "exec",
           "command": "python3 -c \"import torch; print('TORCH', torch.__version__, 'CUDA', torch.cuda.is_available())\"",
           "timeout": 300}

for attempt in range(40):  # cold g6 launch + ~90s pip install can take several minutes
    try:
        resp = rt.invoke_agent_runtime(agentRuntimeArn=ARN, runtimeSessionId=SESSION,
                                       payload=json.dumps(payload).encode())
        body = resp["response"].read()
        print("RESULT:", body.decode(errors="replace"))
        break
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        transient = any(s in msg for s in ("500", "503", "throttl", "ServiceUnavailable",
                                           "RuntimeClientError", "not ready", "Timeout", "timeout"))
        print(f"attempt {attempt}: {type(e).__name__}: {msg[:160]}")
        if not transient:
            raise
        time.sleep(15)
else:
    raise SystemExit("gpu sandbox never became ready")
