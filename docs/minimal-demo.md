# Minimal Demo: slime RL with an AgentCore sandbox

> **Historical (2026-09-30).** This doc was written for demos and code dirs that were removed on 2026-09-30; the only live demo is [`code/demos/demo_cpu_terminalbench`](../code/demos/demo_cpu_terminalbench/README.md).

Goal: the smallest end-to-end thing that proves the loop works —
**SGLang generates code → AgentCore runs it → exit code becomes the reward.**

Single-turn, one file. Full design + citations are in
[`agentcore-slime-integration.md`](./agentcore-slime-integration.md).

---

## What the demo does

1. Task: "write Python that prints the answer." Prompt carries an expected stdout in `sample.label`.
2. slime's policy (SGLang) generates a code block.
3. That code runs in an AgentCore Code Interpreter session (isolated microVM).
4. Reward = `1.0` if stdout matches the label and exit code is 0, else `0.0`.
5. Return one `Sample` with `tokens / response_length / reward / status` set.

That's the whole point — a working reward signal sourced from real sandboxed execution.

---

## Step 1 — sandbox adapter

Use `AgentCoreSandbox` from the integration doc (§4). Save it as
`slime_plugins/agentcore_sandbox.py`. It implements slime's `Sandbox` Protocol
(`slime/agent/sandbox.py:27-59`) over the Code Interpreter client and adds `run_python`.

## Step 2 — the custom generate function

```python
# slime_plugins/agentcore_generate.py
import re
from slime.utils.http_utils import post
from slime.utils.types import Sample, Status
from slime_plugins.agentcore_sandbox import AgentCoreSandbox

CODE_RE = re.compile(r"```(?:python)?\n(.*?)```", re.S)

async def generate(args, sample: Sample, sampling_params, evaluation: bool = False) -> Sample:
    # 1) policy generates an action (code) — SGLang, pattern from sglang_rollout.py:154-223
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    out = await post(url, {"input_ids": sample.tokens,
                           "sampling_params": sampling_params,
                           "return_logprob": True})

    meta = out["meta_info"]
    resp_logprobs = meta["output_token_logprobs"]          # [[logprob, token_id, ...], ...]
    resp_ids   = [tok for _, tok, *_ in resp_logprobs]
    resp_lps   = [lp  for lp, *_    in resp_logprobs]
    resp_text  = out["text"]

    # record the model's tokens as trainable (types.py:253-314)
    sample.append_response_tokens(args, tokens=resp_ids, log_probs=resp_lps, trainable=True)
    sample.response = resp_text
    sample.status = (Status.TRUNCATED if meta["finish_reason"]["type"] == "length"
                     else Status.COMPLETED)

    # 2) execute the action in the AgentCore sandbox (isolated microVM per rollout)
    m = CODE_RE.search(resp_text)
    code = m.group(1) if m else resp_text
    async with AgentCoreSandbox(region=args.agentcore_region) as sb:
        exit_code, stdout, _stderr = await sb.run_python(code, clear_context=True)

    # 3) reward straight from the sandbox result (no separate RM needed)
    expected = (sample.label or "").strip()
    sample.reward = 1.0 if (exit_code == 0 and stdout.strip() == expected) else 0.0
    return sample
```

Notes:
- No `--custom-rm-path` needed: filling `sample.reward` here makes slime skip the RM
  (`sglang_rollout.py:317`).
- Single-turn, so no observation tokens. For multi-turn, append the sandbox stdout back with
  `sample.append_response_tokens(..., trainable=False)` (masked from loss) and loop — see
  `examples/search-r1/generate_with_search.py:244`.
- `args.agentcore_region` — read from an env var in your launch script, or hardcode a region.

## Step 3 — data

A tiny JSONL, one prompt per line, with the expected stdout in `label`:

```json
{"prompt": "Write Python that prints the 10th Fibonacci number.", "label": "55"}
{"prompt": "Write Python that prints 2 to the power 10.", "label": "1024"}
```

## Step 4 — launch

```bash
export AGENTCORE_REGION=us-west-2   # + AWS creds with Code Interpreter IAM actions

python train.py \
  --hf-checkpoint  /path/to/policy \
  --prompt-data    demo.jsonl \
  --input-key prompt --label-key label \
  --n-samples-per-prompt 8 \
  --rollout-batch-size 32 \
  --custom-generate-function-path slime_plugins.agentcore_generate.generate \
  --rm-type none \
  ... your usual megatron / optim / GRPO args ...
```

`--custom-generate-function-path` is picked up at `sglang_rollout.py:283-294`; slime's default
rollout loop, batching, grouping and GRPO advantage all keep working — you only replaced how a
single sample is produced.

---

## How to know it worked

- slime logs a non-zero mean reward that climbs over steps (the policy learns to emit code that
  runs clean and prints the expected value).
- Each rollout shows a distinct AgentCore Code Interpreter session in CloudWatch.
- Sanity check the adapter alone before training:

```python
import asyncio
from slime_plugins.agentcore_sandbox import AgentCoreSandbox

async def main():
    async with AgentCoreSandbox(region="us-west-2") as sb:
        print(await sb.run_python("print(sum(range(11)))"))   # -> (0, "55\n", "")
asyncio.run(main())
```

---

## Scope / limits (kept honest)

- Single-turn, exact-match reward — deliberately trivial to isolate the sandbox↔RL seam.
- One microVM session per rollout; mind AgentCore concurrent-session quotas, not GPUs.
- Verify `executeCode` result field shapes against your installed SDK (see integration doc §6).
