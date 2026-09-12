"""The policy: the thing being trained (the agent's *brain*).

Agent Lightning principle (arXiv 2608.17528): the agent talks to the model through a
proxy/gateway with an OpenAI-compatible interface, and the gateway returns token ids +
logprobs so there's no retokenization drift (see the vLLM blog cited in the AL README).
We mirror that contract here:

    Completion(text, token_ids, logprobs)  ← everything training needs
    async def complete(messages, sampling_params) -> Completion

Two implementations:
  * MockPolicy   — scripted, deterministic. No model, no GPU. Drives the offline smoke:
                   returns a buggy attempt first, then a correct one, so the multi-turn
                   loop is actually exercised and ends solved.
  * SGLangPolicy — HTTP to slime's SGLang router (/generate) with return_logprob, exactly
                   how slime's rollout gets tokens back. Used under real training.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol

# A chat message: {"role": "system"|"user"|"assistant", "content": str}
Message = dict[str, str]

# --- Optional INFERENCE-concurrency gate (decouples inference load from sandbox count) ----------
# slime already gates the number of concurrent episodes (= concurrent sandboxes) with ONE semaphore,
# Semaphore(sglang_server_concurrency * num_engines). That same knob therefore also bounds how many
# decode requests hit SGLang at once, so episode/sandbox concurrency and inference concurrency are
# COUPLED. For the sandbox-concurrency ablation (docs/sandbox-concurrency-ablation.md) we want to
# vary the sandbox count S while holding inference concurrency I FIXED. INFER_CONC adds a SECOND,
# finer-grained gate around ONLY the SGLang /generate POST: with S episodes in flight, at most I are
# ever decoding; the rest are in sandbox ops or waiting here. It is a MODULE-GLOBAL semaphore so it
# is shared across every episode coroutine in the single rollout event loop (generate() makes a
# fresh SGLangPolicy per episode, so a per-instance gate would not bind them together).
# Unset / 0 => disabled (original behavior; every other demo is unaffected).
_INFER_SEM = None            # created lazily inside the running loop (see _infer_sem)
_INFER_SEM_CAP = None        # remembers the cap the live semaphore was built with


def _infer_sem():
    """Return the process-global inference semaphore, or None when INFER_CONC is unset/0.

    Built on first use (inside an async context, so there is a running loop). The `if is None`
    build has no await, so concurrent coroutines cannot race to double-create it.
    """
    global _INFER_SEM, _INFER_SEM_CAP
    if _INFER_SEM_CAP is None:
        import asyncio
        cap = int(os.environ.get("INFER_CONC", "0") or "0")
        _INFER_SEM_CAP = cap
        _INFER_SEM = asyncio.Semaphore(cap) if cap > 0 else None
    return _INFER_SEM


@dataclass
class Completion:
    text: str
    token_ids: list[int] = field(default_factory=list)   # empty for MockPolicy
    logprobs: list[float] = field(default_factory=list)  # empty for MockPolicy


class Policy(Protocol):
    async def complete(self, messages: list[Message], sampling_params: dict) -> Completion: ...


class MockPolicy:
    """Deterministic, model-free policy for offline runs.

    It ignores the prompt content and instead replays `task.mock_attempts` in order,
    indexed by how many assistant turns have already happened. This makes the smoke test
    reproducible while still going through the real write→run→observe→fix loop.
    """

    def __init__(self, attempts: list[str]) -> None:
        self._attempts = attempts
        self._turn = 0

    async def complete(self, messages: list[Message], sampling_params: dict) -> Completion:
        idx = min(self._turn, len(self._attempts) - 1)
        code = self._attempts[idx]
        self._turn += 1
        return Completion(text=f"Here is solution.py:\n```python\n{code}\n```")


class LocalHFPolicy:
    """Real policy that runs a HF model locally on the GPU (transformers .generate).

    A robust alternative to SGLangPolicy for environments where the SGLang server can't
    start (e.g. an sgl_kernel/torch ABI mismatch — a documented pitfall). It loads the model
    ONCE and serves every turn, returning generated text + token_ids + per-token logprobs
    (from generation scores) — the same Completion contract training needs. Production RL
    still uses SGLang for throughput + weight sync; this is for live single-node demos.
    """

    def __init__(self, model_path: str, device: str = "cuda:0", dtype: str = "bfloat16",
                 enable_thinking: bool = False) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_path = model_path
        self.device = device
        self._torch = torch
        self.enable_thinking = enable_thinking
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=getattr(torch, dtype)
        ).to(device).eval()

    async def complete(self, messages: list[Message], sampling_params: dict) -> Completion:
        import asyncio
        return await asyncio.to_thread(self._generate, messages, sampling_params)

    def _generate(self, messages: list[Message], sampling_params: dict) -> Completion:
        torch = self._torch
        # tokenize=False then encode, robust across tokenizers versions (tokenize=True can
        # return an Encoding object that .generate() rejects).
        text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=self.enable_thinking,
        )
        enc = self.tokenizer(text, return_tensors="pt").to(self.device)
        in_len = enc.input_ids.shape[1]
        temp = sampling_params.get("temperature", 1.0)
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=sampling_params.get("max_new_tokens", 1024),
                do_sample=temp > 0,
                temperature=temp if temp > 0 else None,
                top_p=sampling_params.get("top_p", 1.0) if temp > 0 else None,
                return_dict_in_generate=True, output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        gen_ids = out.sequences[0][in_len:].tolist()
        # Per-token logprob of the sampled token, from the step scores.
        logprobs: list[float] = []
        for step, tid in zip(out.scores, gen_ids):
            lp = torch.log_softmax(step[0].float(), dim=-1)
            logprobs.append(float(lp[tid]))
        text_out = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return Completion(text=text_out, token_ids=gen_ids, logprobs=logprobs)


class SGLangPolicy:
    """Calls slime's SGLang router. Requires `transformers` + a running router.

    Endpoint/params match slime's rollout usage: POST {router}/generate with input_ids and
    return_logprob=True so we get token ids + logprobs back (no retokenization drift). The
    chat template is applied locally via the tokenizer.
    """

    def __init__(self, router_url: str | None = None, model_path: str | None = None,
                 chat_template_kwargs: dict | None = None) -> None:
        self.router_url = (router_url or os.environ["SGLANG_ROUTER_URL"]).rstrip("/")
        self.model_path = model_path or os.environ["POLICY_MODEL_PATH"]
        # Reasoning models (e.g. Qwen3) emit <think> blocks by default, which slows codegen
        # rollouts and bloats the trainable span. Pass {"enable_thinking": False} to suppress.
        self.chat_template_kwargs = chat_template_kwargs or {}
        from transformers import AutoTokenizer  # lazy: offline path never imports this
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    async def complete(self, messages: list[Message], sampling_params: dict) -> Completion:
        import aiohttp

        ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, **self.chat_template_kwargs
        )
        # transformers 5.x can return a BatchEncoding (dict-like) instead of a bare list[int];
        # that object is not JSON-serializable for the SGLang POST. Coerce to a flat id list.
        if hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        input_ids = [int(t) for t in ids]
        payload = {
            "input_ids": input_ids,
            "sampling_params": {
                "max_new_tokens": sampling_params.get("max_new_tokens", 1024),
                "temperature": sampling_params.get("temperature", 1.0),
                "top_p": sampling_params.get("top_p", 1.0),
            },
            "return_logprob": True,
        }
        # Bounded connector + explicit timeout: under RL we may run many episodes at once,
        # and a fresh unbounded session per call storms the SGLang router (connect timeouts).
        # slime also gates overall concurrency via --sglang-server-concurrency.
        timeout = aiohttp.ClientTimeout(total=600, connect=60)
        connector = aiohttp.TCPConnector(limit=0, force_close=True)
        # Hold the inference gate (if enabled) across ONLY the decode round-trip, so at most
        # INFER_CONC episodes are decoding at once regardless of how many sandboxes are in flight.
        sem = _infer_sem()
        if sem is None:
            async with aiohttp.ClientSession(timeout=timeout, connector=connector) as sess:
                async with sess.post(f"{self.router_url}/generate", json=payload) as resp:
                    data = await resp.json()
        else:
            async with sem:
                async with aiohttp.ClientSession(timeout=timeout, connector=connector) as sess:
                    async with sess.post(f"{self.router_url}/generate", json=payload) as resp:
                        data = await resp.json()

        meta = data["meta_info"]
        # SGLang returns [[logprob, token_id, token_text], ...] under output_token_logprobs
        # (3 elements per entry, not 2). Index positionally so it is robust to the trailing text.
        pairs = meta.get("output_token_logprobs", [])
        token_ids = [int(p[1]) for p in pairs]
        logprobs = [float(p[0]) for p in pairs]
        text = data.get("text") or self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return Completion(text=text, token_ids=token_ids, logprobs=logprobs)
