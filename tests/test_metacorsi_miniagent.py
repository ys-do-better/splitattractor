import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest

from flashrsi.agents import MiniAgent
from flashrsi.core import Task, Trajectory


def test_hosted_shell_uses_harbor_and_reports_native_exit_code():
    from flashrsi.integrations._hosted_miniagent import HostedShell

    calls = []

    async def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout="out", stderr="err", return_code=7)

    shell = HostedShell(SimpleNamespace(exec=run), "/app")

    result = json.loads(
        asyncio.run(shell({"command": "printf hello", "cwd": "sub"}, None))
    )
    assert result["exit_code"] == 7
    assert calls[0][1] == {"cwd": "/app/sub", "timeout_sec": 60}
    with pytest.raises(ValueError, match="outside"):
        asyncio.run(shell({"command": "pwd", "cwd": "../.."}, None))


def test_hosted_miniagent_pins_executable_snapshot_resources(tmp_path):
    from flashrsi.integrations._hosted_miniagent import validate_resources

    target = MiniAgent({})
    target.set_scope({"code": "rw", "prompt": "rw"})
    with target.snapshot.check_out() as workspace:
        prompt = workspace / "prompts/system.md"
        prompt.write_text("A revised instruction", encoding="utf-8")
        validate_resources(target.snapshot.commit(workspace))
        runtime = workspace / "runtime/tools.py"
        runtime.write_text("raise RuntimeError('changed')", encoding="utf-8")
        with pytest.raises(ValueError, match="bundled"):
            validate_resources(target.snapshot.commit(workspace))


def test_episode_conversion_rejects_untrusted_or_unverified_results(tmp_path):
    from flashrsi.algorithms.metacorsi.episode_adapter import verified_episode

    task = Task(
        "datacurve/synthetic", "Implement a feature.", metadata={"task_checksum": "abc"}
    )
    data = {
        "type": "HarborVerifier",
        "task_id": task.task_id,
        "task_checksum": "abc",
        "snapshot_id": "other",
        "rewards": {"reward": 1},
    }
    assert verified_episode(Trajectory(task=task, messages=(data,)), "expected") is None
    data.update(snapshot_id="expected", verifier_environment_mode="shared")
    assert verified_episode(Trajectory(task=task, messages=(data,)), "expected") is None


def test_verified_episode_uses_native_score_and_preserves_missing_counts(tmp_path):

    from flashrsi.algorithms.metacorsi.episode_adapter import verified_episode

    task = Task("datacurve/test", "Example", metadata={"task_checksum": "abc"})
    rewards = {
        "reward": 0,
        "f2p_passed": 1,
        "f2p_total": 2,
        "p2p_passed": 3,
        "p2p_total": 3,
    }
    event = {
        "type": "HarborVerifier",
        "task_id": task.task_id,
        "task_checksum": "abc",
        "trial_id": "native",
        "snapshot_id": "expected",
        "verifier_environment_mode": "separate",
        "rewards": rewards,
        "artifacts": {"directory": str(tmp_path)},
    }
    result = {
        "task_name": task.task_id,
        "task_checksum": "abc",
        "id": "native",
        "verifier_result": {"rewards": rewards},
    }
    (tmp_path / "result.json").write_text(json.dumps(result))
    trajectory = Trajectory(task=task, messages=(event,), output="Self-reported PASS")
    episode = verified_episode(trajectory, "expected")
    assert episode["reward"] == 0
    assert episode["f2p_passed"] == 1
    assert episode["patch"] == ""
    assert verified_episode(trajectory, "different") is None
    del rewards["p2p_total"]
    assert verified_episode(trajectory, "expected") is None


def test_native_runtime_seed_matches_v23_and_evolves_twice(tmp_path):
    from pathlib import Path

    from flashrsi.agents import MetaCoRSIMiniAgent

    project = Path(__file__).resolve().parents[2] / "metacorsi/metacorsi"
    if not (project / "metacorsi").is_dir():
        pytest.skip("native MetaCoRSI source is not installed")
    target = MetaCoRSIMiniAgent(
        {"project_dir": str(project), "llm": {"model_name": "mock"}}
    )
    assert (
        target.snapshot.id
        == MetaCoRSIMiniAgent(
            {"project_dir": str(project), "llm": {"model_name": "mock"}}
        ).snapshot.id
    )

    async def verify():
        original = await target.check_native()
        assert original == {"version": 0, "tasks": [], "atoms": 5, "slow": 5, "fast": 0}
        target.set_scope({"memory": "rw"})
        snapshot = target.snapshot
        for step, reward in enumerate((0, 1), 1):
            task_id = f"test-episode-{step}"
            with snapshot.check_out() as checkout:
                patch = "diff --git a/example.py b/example.py\n+++ b/example.py\n+def demo(): ...\n"
                data = {
                    "memory": str(checkout / "memory"),
                    "policy": "standard",
                    "episodes": [
                        {
                            "task": task_id,
                            "reward": reward,
                            "f2p_passed": reward + 1,
                            "f2p_total": 2,
                            "p2p_passed": 1,
                            "p2p_total": 1,
                            "patch": patch,
                            "source": str(tmp_path),
                            "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
                            "final_analysis": "Preserve the existing entry point.",
                            "observed_at": float(step),
                        }
                    ],
                }
                receipt = await target.native("evolve", data)
                assert receipt["version"] == step
                current = checkout / "memory/current.json"

                atoms = json.loads(current.read_bytes())["atoms"]
                if not reward:
                    failed = [
                        a
                        for a in atoms
                        if a["atom_id"].startswith("deepswe.test-episode-1")
                    ]
                    assert {"P", "R", "F", "S"} <= {a["kind"] for a in failed}
                    assert all(a["kind"] != "M" for a in failed)
                    registry = json.loads(
                        (checkout / "memory/candidates.json").read_bytes()
                    )
                    assert task_id in registry["tasks"]
                (checkout / "memory/candidates.json").unlink()
                snapshot = snapshot.commit(checkout)
                assert "memory/candidates.json" not in snapshot.files
            inspected = await target.check_native(snapshot=snapshot)
            assert inspected["slow"] == original["slow"]
            assert inspected["fast"] > 0
            assert inspected["version"] == step
            assert task_id in inspected["tasks"]
            with snapshot._export(full=True) as active:
                retrieved = await target.native(
                    "ground",
                    {
                        "memory": str(active / "memory"),
                        "task_id": f"datacurve/{task_id}",
                        "question": f"{task_id}: Preserve the existing entry point.",
                        "evidence": "example.py",
                    },
                )
            assert retrieved["version"] == step
            assert task_id in retrieved["prompt"]
            task = Task(f"datacurve/{task_id}", "Example")
            with pytest.raises(ValueError, match="Held-out"):
                await target.check_native(snapshot=snapshot, forbidden_tasks=(task,))

    asyncio.run(verify())


def test_miniagent_algorithm_uses_only_training_receipts_in_two_rounds(tmp_path):
    from pathlib import Path

    from flashrsi.agents import MetaCoRSIMiniAgent
    from flashrsi.algorithms.metacorsi import MetaCoRSI
    from flashrsi.benchmarks import DeepSWE

    project = Path(__file__).resolve().parents[2] / "metacorsi/metacorsi"
    if not (project / "metacorsi").is_dir():
        pytest.skip("native MetaCoRSI source is not installed")
    target = MetaCoRSIMiniAgent({"project_dir": str(project)})
    benchmark = DeepSWE({})
    benchmark._splits = {
        role: (
            Task(
                f"datacurve/example-{role}",
                "Example",
                env="/test-fixture",
                metadata={"task_checksum": role},
            ),
        )
        for role in ("train", "dev", "test")
    }
    benchmark._tasks_by_id = {
        task.task_id: task for split in benchmark._splits.values() for task in split
    }
    calls = []

    async def rollout(agent, task):
        calls.append(task.task_id)
        trial = tmp_path / str(len(calls))
        trial.mkdir()
        patch = trial / "artifacts/logs/artifacts/model.patch"
        patch.parent.mkdir(parents=True)
        patch.write_text(
            "diff --git a/example.py b/example.py\n+++ b/example.py\n+version = "
            + str(len(calls))
            + "\n"
        )
        result = {
            "task_name": task.task_id,
            "task_checksum": task.metadata["task_checksum"],
            "id": trial.name,
        }
        reward = 0 if len(calls) == 1 else 1
        rewards = {
            "reward": reward,
            "f2p_passed": reward + 1,
            "f2p_total": 2,
            "p2p_passed": 1,
            "p2p_total": 1,
        }
        result["verifier_result"] = {"rewards": rewards}
        (trial / "result.json").write_text(json.dumps(result))
        return Trajectory(
            output="Example outcome",
            messages=(
                {
                    "type": "HarborVerifier",
                    "task_id": task.task_id,
                    "task_checksum": task.metadata["task_checksum"],
                    "trial_id": trial.name,
                    "verifier_environment_mode": "separate",
                    "snapshot_id": agent.snapshot.id,
                    "rewards": rewards,
                    "artifacts": {"directory": str(trial)},
                },
            ),
        )

    target._env_runner = rollout
    method = MetaCoRSI(
        {"max_steps": 2, "evolution_policy": "standard"},
        target=target,
        worker=None,
        benchmark=benchmark,
    )
    method._run_dir = tmp_path / "experiment"
    for child in ("snapshots", "trajectories", "working"):
        (method._run_dir / child).mkdir(parents=True)

    async def verify():
        final = await method.evolve()
        memory = await target.check_native(snapshot=final)
        assert memory["version"] == 2
        assert memory["tasks"] == ["example-train"]
        assert calls.count("datacurve/example-train") == 2
        assert "datacurve/example-test" not in calls
        quarantine = json.loads((method._run_dir / "quarantine/0001.json").read_bytes())
        assert "example-train" in quarantine["task_candidates"]["tasks"]
        assert "memory/candidates.json" not in final.files

    asyncio.run(verify())
