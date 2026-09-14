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


def test_task_families_solve_multi_turn_offline():
    """The demo_a task-family zoo (SWE, tool-use, SQL, math, CTF, RAG, regex, planning,
    config, data) all run on the SAME in-process harness + CPU sandbox, with a deterministic
    checker per family. Every seed must (a) fail its first (buggy) attempt and (b) pass after
    the fix — proving both the reward and that the checker genuinely discriminates."""
    from agentcore_slime_demo.task_families import FAMILY_CLASSES
    from agentcore_slime_demo.task import task_from_dict

    families_seen = set()
    for cls in FAMILY_CLASSES:
        seeds = cls.seeds()
        assert len(seeds) >= 6, f"{cls.family}: too few seeds ({len(seeds)})"
        families_seen.add(cls.family)
        for t in seeds:
            # Round-trip through the dataset serialization the trainer actually uses.
            rebuilt = task_from_dict({**t.to_row_kwargs(), "mock_attempts": t.mock_attempts})
            traj = asyncio.run(_episode(rebuilt))
            assert traj.solved, f"{t.task_id} not solved"
            assert traj.reward == 1.0
            assert traj.num_turns >= 2, f"{t.task_id} not multi-turn"
            assert traj.turns[0].exit_code != 0, f"{t.task_id} first (buggy) attempt passed"
            assert traj.turns[-1].exit_code == 0
    assert families_seen == {"swe", "tool_use", "sql", "math", "ctf", "rag",
                             "regex", "planning", "config", "data"}


def test_family_dispatch_and_backward_compat():
    """`task_from_dict` routes on `family`; a legacy row without one stays CodingTask."""
    from agentcore_slime_demo.task import task_from_dict, CodingTask
    from agentcore_slime_demo.task_families import SqlTask

    sql = task_from_dict({"family": "sql", "task_id": "x", "prompt": "q",
                          "spec": {"schema": ["CREATE TABLE t (a INTEGER)"],
                                   "inserts": ["INSERT INTO t VALUES (1)"], "expected": [[1]]}})
    assert isinstance(sql, SqlTask)
    assert sql.artifact_filename == "query.sql" and sql.fence_lang == "sql"
    legacy = task_from_dict({"task_id": "m", "prompt": "p",
                             "checks": ["assert f(1) == 1"], "check_mode": "assert"})
    assert isinstance(legacy, CodingTask)


def test_pyc_cache_does_not_shadow_same_length_rewrite():
    """Regression: a same-byte-length fix rewritten within the same second must NOT be
    shadowed by a stale __pycache__/*.pyc from the previous turn (dont_write_bytecode)."""
    from agentcore_slime_demo.task import CodingTask
    t = CodingTask(task_id="pyc", prompt="add", checks=["assert add(2, 3) == 5"],
                   check_mode="assert",
                   mock_attempts=["def add(a, b):\n    return a - b\n",     # buggy, len N
                                  "def add(a, b):\n    return a + b\n"])    # fix, SAME len N
    assert len(t.mock_attempts[0]) == len(t.mock_attempts[1])
    traj = asyncio.run(_episode(t))
    assert traj.solved and traj.turns[0].exit_code != 0 and traj.turns[-1].exit_code == 0


if __name__ == "__main__":
    test_all_tasks_solved_and_multi_turn()
    test_extract_code_takes_last_block()
    test_build_token_stream_masks_only_actions()
    test_sandboxes_conform_to_slime_protocol()
    test_recovery_from_transient_boot_failures()
    test_recovery_from_mid_episode_session_death()
    test_unrecoverable_aborts_cleanly_without_raising()
    test_profiles_describe_two_demo_cases()
    test_gpu_task_needs_real_cuda_inside_sandbox()
    test_error_classifier()
    test_mbpp_loads_and_solves_offline()
    test_kernelbench_tasks_need_real_cuda()
    test_assert_mode_runner_distinguishes_pass_fail()
    test_task_families_solve_multi_turn_offline()
    test_family_dispatch_and_backward_compat()
    test_pyc_cache_does_not_shadow_same_length_rewrite()
    print("ok")
