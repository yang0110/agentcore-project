#!/usr/bin/env python3
"""REAL agentic-RL training — not a smoke test, not a dry run.

This runs an actual policy-gradient training loop that UPDATES the model weights and moves
the reward. It reuses the exact demo components (tasks, sandbox, checker reward, loss mask)
so the thing being trained is the same agent the demos run:

  step = collect a batch of ENV rollouts -> reward each in the sandbox -> GRPO advantages
         -> real forward/backward on the policy -> optimizer.step() -> weights change.

Algorithm: **GRPO** (Group Relative Policy Optimization, the DeepSeek/slime default for
verifiable-reward RL). For each prompt we sample a GROUP of G completions; the group mean
reward is the baseline, so advantage_i = (r_i - mean)/(std+eps). The loss is the standard
policy-gradient surrogate over ONLY the action tokens (the loss mask from generate.py):
    L = - mean_i ( advantage_i * sum_t logπ(a_t) )
Real gradients flow into LoRA adapters on Qwen3-4B (full-FT Adam on 4B needs ~56 GB > one
L40S; LoRA is the standard memory-safe RL path and is genuinely real training).

Why LoRA + a local loop instead of slime's Megatron+SGLang trainer: SGLang can't start on
this box (sgl_kernel/torch ABI break — a documented pitfall), which blocks slime's rollout
engine. This loop trains the identical policy/reward/mask contract slime uses, so it is a
faithful stand-in for the weight-update step.

"100 envs per demo" = 100 rollout episodes per training step (P prompts x G group size).

Usage:
    HF_HOME=./.hf_cache python3 scripts/train_rl.py --profile cpu --envs 100 --group 5 --steps 6
    HF_HOME=./.hf_cache python3 scripts/train_rl.py --profile gpu --envs 99  --group 3 --steps 6
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentcore_slime_demo.sandbox import LocalSandbox
from agentcore_slime_demo.task import CodingTask

DEFAULT_MODEL = os.environ.get("POLICY_MODEL_PATH", "Qwen/Qwen3-4B")


def load_tasks(profile: str, n: int) -> list[CodingTask]:
    """A pool of tasks to draw prompts from (cycled to fill the batch)."""
    if profile == "cpu":
        from agentcore_slime_demo.mbpp import load_mbpp
        pool = load_mbpp("train", limit=None)
    else:
        from agentcore_slime_demo.kernelbench import load_kernelbench_tasks
        pool = load_kernelbench_tasks()
    return pool


async def reward_for(task: CodingTask, code: str | None, pool_gate) -> float:
    """Binary verifiable reward: 1.0 iff the checker passes on the generated code."""
    if not code:
        return 0.0
    async with LocalSandbox() as sb:
        if pool_gate is not None:
            from agentcore_slime_demo.gpu_pool import GpuGatedSandbox
            sb2 = GpuGatedSandbox(sb, pool_gate, gate_all=True)
        else:
            sb2 = sb
        await sb2.write_file("run_checks.py", task.runner_source())
        await sb2.write_file("solution.py", code)
        code_exit, _, _ = await sb2.exec("python3 run_checks.py", timeout=60)
    return 1.0 if code_exit == 0 else 0.0


class GRPOTrainer:
    def __init__(self, model_path: str, device: str, lr: float, profile: str):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.profile = profile
        self.tok = AutoTokenizer.from_pretrained(model_path)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"  # left-pad for batched generation
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).to(device)
        lora = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        self.model = get_peft_model(model, lora)
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()
        self.model.print_trainable_parameters()
        self.opt = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=lr
        )
        # GPU pool for the GPU profile so many CUDA rollouts share the device safely.
        self.pool = None
        if profile == "gpu":
            from agentcore_slime_demo.gpu_pool import GpuPool
            self.pool = GpuPool(max_concurrent_gpu=8, per_env_mem_mib=2048)

    def _prompt_text(self, task: CodingTask) -> str:
        msgs = [{"role": "system", "content": task.system_prompt()},
                {"role": "user", "content": task.initial_user_prompt()}]
        return self.tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )

    @property
    def _no_think(self):
        return True

    def generate_group(self, task: CodingTask, group: int, temp: float, max_new: int):
        """Sample `group` completions for one prompt in a single batched generate call."""
        torch = self.torch
        text = self._prompt_text(task)
        enc = self.tok(text, return_tensors="pt").to(self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model.generate(
                **enc, do_sample=True, temperature=temp, top_p=0.95,
                max_new_tokens=max_new, num_return_sequences=group,
                pad_token_id=self.tok.pad_token_id,
            )
        in_len = enc.input_ids.shape[1]
        samples = []
        for row in out:
            resp_ids = row[in_len:].tolist()
            # trim trailing pad
            while resp_ids and resp_ids[-1] == self.tok.pad_token_id:
                resp_ids.pop()
            text_out = self.tok.decode(resp_ids, skip_special_tokens=True)
            samples.append({"prompt_ids": enc.input_ids[0].tolist(), "resp_ids": resp_ids,
                            "text": text_out})
        return samples

    def masked_logprob_sum(self, prompt_ids: list[int], resp_ids: list[int]):
        """log π(response | prompt), summed over response tokens, WITH gradient."""
        torch = self.torch
        ids = torch.tensor([prompt_ids + resp_ids], device=self.device)
        self.model.train()
        out = self.model(ids, use_cache=False)
        logits = out.logits[0]                       # [T, V]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        # token t is predicted from position t-1; response starts at len(prompt_ids).
        start = len(prompt_ids)
        total = logits.new_zeros((), dtype=torch.float32)
        for t in range(start, len(prompt_ids) + len(resp_ids)):
            tok_id = ids[0, t]
            total = total + logprobs[t - 1, tok_id]
        return total

    async def collect(self, tasks: list[CodingTask], envs: int, group: int, temp: float,
                      max_new: int):
        """Roll out `envs` episodes (P prompts x G group), reward each. Returns samples."""
        P = max(1, envs // group)
        chosen = [tasks[i % len(tasks)] for i in range(P)]
        groups = []
        for task in chosen:
            samples = self.generate_group(task, group, temp, max_new)
            groups.append((task, samples))
        # Reward all samples concurrently (bounded).
        sem = asyncio.Semaphore(16 if self.profile == "cpu" else 8)

        async def score(task, s):
            async with sem:
                code = CodingTask.extract_code(s["text"])
                s["reward"] = await reward_for(task, code, self.pool)

        await asyncio.gather(*(score(t, s) for t, ss in groups for s in ss))
        return groups

    def train_step(self, groups) -> dict:
        """GRPO advantages + one real optimizer step. Returns metrics."""
        torch = self.torch
        # advantages per group
        flat = []
        for task, samples in groups:
            rs = [s["reward"] for s in samples]
            mean = sum(rs) / len(rs)
            var = sum((r - mean) ** 2 for r in rs) / len(rs)
            std = var ** 0.5
            for s in samples:
                s["adv"] = (s["reward"] - mean) / (std + 1e-4)
                flat.append(s)

        self.opt.zero_grad()
        n_used = 0
        loss_val = 0.0
        # Only samples with nonzero advantage contribute gradient (GRPO); skip the rest to
        # save compute. Accumulate loss over the batch, then a single optimizer step.
        contributing = [s for s in flat if abs(s["adv"]) > 1e-6]
        denom = max(1, len(contributing))
        for s in contributing:
            lp = self.masked_logprob_sum(s["prompt_ids"], s["resp_ids"])
            loss = -(s["adv"] * lp) / denom
            loss.backward()
            loss_val += float(loss.detach())
            n_used += 1
        if n_used > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], 1.0)
            self.opt.step()
        rewards = [s["reward"] for s in flat]
        return {
            "n_env": len(flat),
            "mean_reward": sum(rewards) / len(rewards),
            "solve_rate": sum(1 for r in rewards if r >= 1.0) / len(rewards),
            "n_grad_samples": n_used,
            "loss": loss_val,
        }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["cpu", "gpu"], required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--envs", type=int, default=100, help="rollout episodes per step (P*G)")
    ap.add_argument("--group", type=int, default=5, help="GRPO group size G")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--log", default="train_rl_log.jsonl")
    args = ap.parse_args()

    print("=" * 78)
    print(f"REAL GRPO training | profile={args.profile} model={args.model}")
    print(f"envs/step={args.envs} group={args.group} steps={args.steps} lr={args.lr} temp={args.temp}")
    print("=" * 78)

    tasks = load_tasks(args.profile, args.envs)
    print(f"task pool: {len(tasks)} task(s)")
    trainer = GRPOTrainer(args.model, args.device, args.lr, args.profile)

    log_path = Path(args.log)
    history = []
    for step in range(1, args.steps + 1):
        t0 = time.perf_counter()
        groups = await trainer.collect(tasks, args.envs, args.group, args.temp, args.max_new_tokens)
        m = trainer.train_step(groups)
        m["step"] = step
        m["sec"] = round(time.perf_counter() - t0, 1)
        history.append(m)
        with log_path.open("a") as f:
            f.write(json.dumps(m) + "\n")
        print(f"step {step:>2}/{args.steps} | envs={m['n_env']} "
              f"mean_reward={m['mean_reward']:.3f} solve_rate={m['solve_rate']:.3f} "
              f"grad_samples={m['n_grad_samples']} loss={m['loss']:+.4f} ({m['sec']}s)")

    r0, rN = history[0]["mean_reward"], history[-1]["mean_reward"]
    print("\n" + "=" * 78)
    print(f"DONE. mean_reward {r0:.3f} -> {rN:.3f}  (delta {rN - r0:+.3f})   log: {log_path}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
