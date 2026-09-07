"""Offline tests for the demo. Run: `python -m pytest code/tests -q` (or plain `python code/tests/test_smoke.py`)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentcore_slime_demo.harness import run_episode
from agentcore_slime_demo.policy import MockPolicy
from agentcore_slime_demo.sandbox import LocalSandbox
from agentcore_slime_demo.task import CodingTask, load_tasks

TASKS = Path(__file__).resolve().parents[1] / "agentcore_slime_demo" / "data" / "tasks.jsonl"
GPU_TASKS = Path(__file__).resolve().parents[1] / "agentcore_slime_demo" / "data" / "gpu_tasks.jsonl"


async def _episode(task: CodingTask):
    async with LocalSandbox() as sb:
        return await run_episode(MockPolicy(task.mock_attempts), sb, task)


def test_all_tasks_solved_and_multi_turn():
    for task in load_tasks(TASKS):
        traj = asyncio.run(_episode(task))
        assert traj.solved, f"{task.task_id} not solved"
        assert traj.reward == 1.0
        # The first attempt is buggy by construction -> the loop must take >1 turn.
        assert traj.num_turns >= 2, f"{task.task_id} solved in one turn (not multi-turn)"
        assert traj.turns[0].exit_code != 0, f"{task.task_id} first attempt unexpectedly passed"
        assert traj.turns[-1].exit_code == 0


def test_extract_code_takes_last_block():
    text = "```python\nx=1\n```\nthen\n```python\ny=2\n```"
    assert CodingTask.extract_code(text) == "y=2"
    assert CodingTask.extract_code("no code here") is None


def test_build_token_stream_masks_only_actions():
    # Pure-function check of the loss mask, with a fake tokenizer (no transformers needed).
    from agentcore_slime_demo.generate import build_token_stream
    from agentcore_slime_demo.harness import Trajectory, Turn
    from agentcore_slime_demo.policy import Completion

    class FakeTok:
        # Realistic re-tokenizer: turn-0 prompt (2 msgs) -> [100,101]; turn-1 prompt (4 msgs)
        # -> the initial prompt + the turn-0 assistant tokens re-tokenized identically
        # ([1,2,3]) + the injected observation user turn ([90,91]). This exercises both the
        # trainable (assistant) and masked (observation) regions.
        def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
            if len(messages) == 2:
                return [100, 101]
            return [100, 101, 1, 2, 3, 90, 91]

    traj = Trajectory(task_id="t")
    traj.turns = [
        Turn(prompt_messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
             completion=Completion(text="", token_ids=[1, 2, 3]), observation="fail",
             exit_code=1, submitted_code=True),
        Turn(prompt_messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                              {"role": "assistant", "content": "a"}, {"role": "user", "content": "fail"}],
             completion=Completion(text="", token_ids=[4, 5]), observation="pass",
             exit_code=0, submitted_code=True),
    ]
    tokens, response_length, mask, rollout_lp = build_token_stream(traj, FakeTok())
    # slime contract: loss_mask + rollout_log_probs cover only the response region,
    # len == response_length.
    assert len(mask) == response_length
    assert len(rollout_lp) == response_length
    prompt_len = len(tokens) - response_length
    assert prompt_len == 2  # the initial [100,101] prompt is the non-trainable prefix
    response_region = tokens[prompt_len:]
    # Trainable positions (mask==1) must be exactly the assistant/action tokens.
    trainable = [t for t, m in zip(response_region, mask) if m == 1]
    assert trainable == [1, 2, 3, 4, 5]
    # The injected observation between turns must be present but masked out.
    masked = [t for t, m in zip(response_region, mask) if m == 0]
    assert masked == [90, 91]


def test_sandboxes_conform_to_slime_protocol():
    # Ties this demo to the REAL slime contract: our sandboxes must be drop-in for E2BSandbox.
    import inspect

    from agentcore_slime_demo.sandbox import AgentCoreSandbox, LocalSandbox

    # slime is a cloned repo, not pip-installed — add it to the path if present.
    slime_repo = Path(__file__).resolve().parents[2] / "repos" / "slime"
    if slime_repo.is_dir() and str(slime_repo) not in sys.path:
        sys.path.insert(0, str(slime_repo))
    try:
        from slime.agent.sandbox import Sandbox as SlimeSandbox
    except Exception as e:
        try:
            import pytest  # type: ignore
            pytest.skip(f"slime not importable: {e}")
        except ImportError:
            print(f"SKIP test_sandboxes_conform_to_slime_protocol: slime not importable ({e})")
        return

    for cls in (LocalSandbox, AgentCoreSandbox):
        assert isinstance(cls(), SlimeSandbox), f"{cls.__name__} not a slime Sandbox"
        # runtime_checkable only checks names; also assert exec signature compatibility.
        params = inspect.signature(cls.exec).parameters
        for kw in ("user", "env", "timeout", "check", "idempotent"):
            assert kw in params, f"{cls.__name__}.exec missing kw-only arg {kw!r}"


def test_real_claude_code_path_imports():
    # The real-harness module must import even without slime/AWS (all heavy imports are lazy).
    import agentcore_slime_demo.claude_code_generate as ccg
    assert hasattr(ccg, "generate")


def test_recovery_from_transient_boot_failures():
    from agentcore_slime_demo.recovery import FlakySandbox, resilient_episode
    task = load_tasks(TASKS)[0]
    FlakySandbox.reset_counters()
    traj = asyncio.run(resilient_episode(
        task,
        make_policy=lambda: MockPolicy(task.mock_attempts),
        make_sandbox=lambda: FlakySandbox(LocalSandbox(), fail_boots=2),
        max_boot_retries=4, backoff_base_sec=0.0,
    ))
    assert traj.status == "completed" and traj.solved
    assert traj.boot_attempts == 3   # 2 failed boots + 1 success


def test_recovery_from_mid_episode_session_death():
    from agentcore_slime_demo.recovery import FlakySandbox, resilient_episode
    task = load_tasks(TASKS)[0]
    FlakySandbox.reset_counters()
    traj = asyncio.run(resilient_episode(
        task,
        make_policy=lambda: MockPolicy(task.mock_attempts),
        make_sandbox=lambda: FlakySandbox(LocalSandbox(), fail_exec_after=1),
        max_boot_retries=4, backoff_base_sec=0.0,
    ))
    # The session dies after the 1st exec on the first attempt; a fresh session solves it.
    assert traj.status == "completed" and traj.solved
    assert traj.boot_attempts >= 2


def test_unrecoverable_aborts_cleanly_without_raising():
    from agentcore_slime_demo.recovery import FlakySandbox, resilient_episode
    task = load_tasks(TASKS)[0]
    FlakySandbox.reset_counters()
    traj = asyncio.run(resilient_episode(
        task,
        make_policy=lambda: MockPolicy(task.mock_attempts),
        make_sandbox=lambda: FlakySandbox(LocalSandbox(), fail_boots=1, fatal=True),
        max_boot_retries=4, backoff_base_sec=0.0,
    ))
    assert traj.status == "aborted" and not traj.solved and traj.reward == 0.0
    assert "Fatal" in traj.abort_reason or "fatal" in traj.abort_reason


def test_profiles_describe_two_demo_cases():
    # The two demo cases differ exactly in compute + AgentCore provisioning.
    from agentcore_slime_demo.profiles import CPU_SANDBOX, GPU_SANDBOX, get_profile

    assert CPU_SANDBOX.compute == "cpu" and not CPU_SANDBOX.needs_gpu
    assert CPU_SANDBOX.agentcore_mode == "code_interpreter"
    assert CPU_SANDBOX.instance_types == ()          # serverless, no capacity provider

    assert GPU_SANDBOX.compute == "gpu" and GPU_SANDBOX.needs_gpu
    assert GPU_SANDBOX.agentcore_mode == "runtime_instance"
    assert GPU_SANDBOX.capacity_provider and GPU_SANDBOX.instance_types  # GPU pool required
    assert any("capacity-provider" in c for c in GPU_SANDBOX.provision_cli)
    assert get_profile("gpu-sandbox") is GPU_SANDBOX


def test_gpu_task_needs_real_cuda_inside_sandbox():
    # The GPU demo case must actually run on CUDA: the buggy (CPU) attempt fails the
    # 'ran on cuda' check, the fixed (.cuda()) attempt passes. Skip when no GPU is present.
    try:
        import torch  # noqa: F401
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False
    if not has_cuda:
        try:
            import pytest  # type: ignore
            pytest.skip("no CUDA GPU available in the offline sandbox")
        except ImportError:
            print("SKIP test_gpu_task_needs_real_cuda_inside_sandbox: no CUDA")
        return

    task = load_tasks(GPU_TASKS)[0]
    traj = asyncio.run(_episode(task))
    assert traj.solved and traj.reward == 1.0
    assert traj.num_turns >= 2                       # CPU attempt fails, GPU attempt passes
    assert traj.turns[0].exit_code != 0              # buggy CPU attempt rejected
    assert traj.turns[-1].exit_code == 0


def test_error_classifier():
    from agentcore_slime_demo.recovery import FatalSandboxError, is_recoverable
    assert is_recoverable(RuntimeError("session terminated"))
    assert is_recoverable(RuntimeError("ThrottlingException: rate exceeded"))
    assert not is_recoverable(RuntimeError("AccessDeniedException"))
    assert not is_recoverable(FatalSandboxError("bad config"))


def test_gpu_pool_memory_safety_math():
    # The static check must catch overcommit: 8 concurrent x 8 GiB on 2x 16 GiB GPUs
    # is 4/device x 8 GiB = 32 GiB/GPU > 16 GiB -> must NOT "fit".
    from agentcore_slime_demo.gpu_pool import GpuPool
    over = GpuPool(num_gpus=2, gpu_mem_mib=16384, max_concurrent_gpu=8, per_env_mem_mib=8192)
    assert over.max_leases_per_device() == 4
    assert not over.memory_safety()["fits"]
    # Same GPUs, modest cap -> fits with headroom.
    ok = GpuPool(num_gpus=2, gpu_mem_mib=16384, max_concurrent_gpu=8, per_env_mem_mib=1024)
    assert ok.memory_safety()["fits"]
    assert 0.0 < ok.mem_fraction <= 1.0


def test_gpu_pool_never_exceeds_gate_and_spreads_devices():
    # Many envs, small gate: peak concurrent GPU workers must never exceed the gate, and
    # work must spread across all devices. Pure scheduling — no torch/CUDA needed.
    from agentcore_slime_demo.gpu_pool import GpuPool

    pool = GpuPool(num_gpus=4, gpu_mem_mib=46068, max_concurrent_gpu=6, per_env_mem_mib=1024)

    async def worker():
        async with pool.slot() as lease:
            await asyncio.sleep(0.01)
            assert lease.env["CUDA_VISIBLE_DEVICES"] == str(lease.device)
            assert "ENV_GPU_MEM_FRACTION" in lease.env

    async def drive():
        await asyncio.gather(*(worker() for _ in range(50)))

    asyncio.run(drive())
    assert pool.stats.admitted == 50
    assert pool.stats.peak_concurrent <= 6            # gate honoured
    assert set(pool.stats.per_device_admits) == {0, 1, 2, 3}   # all devices used


def test_gpu_pool_leases_only_reserved_physical_devices():
    # Demo B split: trainer + SGLang own physical GPUs 0-5; envs may use ONLY 6,7. The pool
    # must lease exactly those physical ids (never 0-5) and still honour the gate + spread.
    from agentcore_slime_demo.gpu_pool import GpuPool

    pool = GpuPool(devices=[6, 7], gpu_mem_mib=46068, max_concurrent_gpu=8, per_env_mem_mib=2048)
    assert pool.num_gpus == 2

    seen = set()

    async def worker():
        async with pool.slot() as lease:
            await asyncio.sleep(0.01)
            assert lease.device in (6, 7)
            assert lease.env["CUDA_VISIBLE_DEVICES"] in ("6", "7")
            seen.add(lease.device)

    async def drive():
        await asyncio.gather(*(worker() for _ in range(40)))

    asyncio.run(drive())
    assert pool.stats.admitted == 40
    assert pool.stats.peak_concurrent <= 8
    assert seen == {6, 7}                                  # both reserved GPUs used
    assert set(pool.stats.per_device_admits) == {6, 7}     # stats keyed by physical id


def test_gpu_gated_sandbox_gates_only_gpu_commands():
    # The wrapper must (a) route GPU commands through the pool and inject the lease env,
    # (b) let non-GPU commands pass straight through ungated. Uses LocalSandbox + a
    # command that echoes CUDA_VISIBLE_DEVICES so we can see the lease landed.
    from agentcore_slime_demo.gpu_pool import GpuGatedSandbox, GpuPool

    pool = GpuPool(num_gpus=2, gpu_mem_mib=46068, max_concurrent_gpu=2, per_env_mem_mib=1024)

    async def run():
        async with LocalSandbox() as inner:
            sb = GpuGatedSandbox(inner, pool)   # heuristic mode
            # GPU-looking command -> gated -> lease pins the device the pool assigned (0 or 1),
            # overriding any ambient CUDA_VISIBLE_DEVICES on the host.
            code, out, _ = await sb.exec("echo cuda device=$CUDA_VISIBLE_DEVICES")
            assert code == 0
            assert out.strip().split("device=")[1].strip() in {"0", "1"}
            # Plain command -> not gated (no pool admission).
            await sb.exec("echo plain hello")
        return pool.stats.admitted

    admitted = asyncio.run(run())
    assert admitted == 1   # exactly the one GPU command went through the pool


def test_run_episode_with_gpu_pool_still_solves():
    # Wiring check: passing a gpu_pool routes the checker exec through admission control
    # while the multi-turn episode still solves. Works without CUDA (LocalSandbox); the
    # CPU-repair task's checker is CPU, but gate_all forces it through the pool anyway.
    from agentcore_slime_demo.gpu_pool import GpuPool

    task = load_tasks(TASKS)[0]
    pool = GpuPool(num_gpus=2, gpu_mem_mib=46068, max_concurrent_gpu=4, per_env_mem_mib=1024)

    async def run():
        async with LocalSandbox() as sb:
            return await run_episode(MockPolicy(task.mock_attempts), sb, task, gpu_pool=pool)

    traj = asyncio.run(run())
    assert traj.solved and traj.reward == 1.0
    assert pool.stats.admitted == traj.num_turns   # one gated checker exec per turn


def _skip(msg):
    try:
        import pytest  # type: ignore
        pytest.skip(msg)
    except ImportError:
        print(f"SKIP: {msg}")


def test_mbpp_loads_and_solves_offline():
    # MBPP -> assert-mode CodingTask; the failing stub (turn 0) then reference (turn 1)
    # must exercise a genuine multi-turn fail->fix. Skips cleanly if HF/network is absent.
    try:
        from agentcore_slime_demo.mbpp import load_mbpp
        tasks = load_mbpp("train", limit=3)
    except Exception as e:  # noqa: BLE001
        return _skip(f"MBPP not loadable (network/datasets): {e}")

    assert len(tasks) == 3
    for t in tasks:
        assert t.check_mode == "assert" and t.checks and "assert" in t.checks[0]
        traj = asyncio.run(_episode(t))
        assert traj.solved and traj.reward == 1.0, f"{t.task_id} not solved"
        assert traj.num_turns >= 2 and traj.turns[0].exit_code != 0   # stub fails, ref passes


def test_kernelbench_tasks_need_real_cuda():
    # Demo B: each KernelBench-style task must run on CUDA. Buggy CPU attempt fails the
    # device check; the GPU attempt passes. Skips without a CUDA GPU.
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False
    if not has_cuda:
        return _skip("no CUDA GPU for KernelBench tasks")

    from agentcore_slime_demo.kernelbench import load_kernelbench_tasks
    tasks = load_kernelbench_tasks()
    assert len(tasks) >= 3
    for t in tasks:
        assert t.check_mode == "assert"
        assert any("cuda" in c for c in t.checks)          # correctness+device gated
        traj = asyncio.run(_episode(t))
        assert traj.solved and traj.reward == 1.0, f"{t.task_id} not solved"
        assert traj.turns[0].exit_code != 0                # CPU attempt rejected
        assert traj.turns[-1].exit_code == 0


def test_assert_mode_runner_distinguishes_pass_fail():
    # Pure-function check of the new assert check_mode: a right and a wrong solution.
    from agentcore_slime_demo.task import CodingTask
    t = CodingTask(task_id="am", prompt="add", checks=["assert add(2, 3) == 5"],
                   check_mode="assert")
    assert "assert add(2, 3) == 5" in t.runner_source()   # stmt embedded in checker

    async def run(code):
        async with LocalSandbox() as sb:
            await sb.write_file("run_checks.py", t.runner_source())
            await sb.write_file("solution.py", code)
            return (await sb.exec("python3 run_checks.py"))[0]

    assert asyncio.run(run("def add(a, b):\n    return a + b\n")) == 0
    assert asyncio.run(run("def add(a, b):\n    return a - b\n")) == 1


if __name__ == "__main__":
    test_all_tasks_solved_and_multi_turn()
    test_extract_code_takes_last_block()
    test_build_token_stream_masks_only_actions()
    test_sandboxes_conform_to_slime_protocol()
    test_real_claude_code_path_imports()
    test_recovery_from_transient_boot_failures()
    test_recovery_from_mid_episode_session_death()
    test_unrecoverable_aborts_cleanly_without_raising()
    test_profiles_describe_two_demo_cases()
    test_gpu_task_needs_real_cuda_inside_sandbox()
    test_error_classifier()
    test_gpu_pool_memory_safety_math()
    test_gpu_pool_never_exceeds_gate_and_spreads_devices()
    test_gpu_gated_sandbox_gates_only_gpu_commands()
    test_run_episode_with_gpu_pool_still_solves()
    test_mbpp_loads_and_solves_offline()
    test_kernelbench_tasks_need_real_cuda()
    test_assert_mode_runner_distinguishes_pass_fail()
    print("ok")
