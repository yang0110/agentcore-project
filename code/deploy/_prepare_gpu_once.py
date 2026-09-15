"""One-off: invoke the GPU sandbox Runtime with {"mode":"prepare"} to install the CUDA torch
stack onto its volume. Uses a >=33-char runtimeSessionId (AgentCore requirement; the shipped
deploy_two_runtime.prepare_gpu_volume used an 18-char id and fails ParamValidation)."""
import os
import boto3
from botocore.config import Config

REGION = os.environ.get("AWS_REGION", "us-west-2")
ARN = os.environ["GPU_SANDBOX_ARN"]
SID = "prepare-gpu-volume-padded-000000000000"  # 38 chars

cfg = Config(read_timeout=int(os.environ.get("INVOKE_READ_TIMEOUT", "1500")),
             connect_timeout=60, retries={"max_attempts": 0})
c = boto3.client("bedrock-agentcore", region_name=REGION, config=cfg)
print(f"invoking mode=prepare on {ARN} (sid len={len(SID)}) ...", flush=True)
r = c.invoke_agent_runtime(agentRuntimeArn=ARN, runtimeSessionId=SID,
                           payload=b'{"mode":"prepare"}')
print(r["response"].read().decode())
print("PREPARE_DONE")
