"""Offline checks for the host verifier and Snapshot boundaries."""

import asyncio
from pathlib import Path

import pytest

from flashrsi.benchmarks import DeepSWE
from flashrsi.config import load_config
from flashrsi.core import Task
from flashrsi.runner import _AGENTS, _ALGORITHMS


def test_registered_experiment_does_not_require_worker_credentials():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples/quickstarts/deepswe_metacorsi.yaml"
    )
    config = load_config(path)
    assert config.agent.name in _AGENTS
    assert not _ALGORITHMS[config.algorithm.name].requires_worker


@pytest.mark.parametrize("mutation", [{"task_checksum": "other"}, {"reward": True}])
def test_invalid_verifier_evidence_does_not_score(mutation):
    from flashrsi.agents.metacorsi import verifier_trajectory

    task = Task("datacurve/test", "Task", metadata={"task_checksum": "digest"})
    report = {
        "task_id": task.task_id,
        "task_checksum": "digest",
        "rewards": {"reward": 1, "f2p_passed": 4},
        "trial_dir": "/fake/trial",
    }
    if "reward" in mutation:
        report["rewards"].update(mutation)
    else:
        report.update(mutation)
    assert verifier_trajectory(report, task, "snapshot").status == "error"


def test_verified_failed_repair_retains_native_evidence():
    from dataclasses import replace

    from flashrsi.agents.metacorsi import verifier_trajectory

    task = Task("datacurve/test", "Task", metadata={"task_checksum": "digest"})
    report = {
        "task_id": task.task_id,
        "task_checksum": "digest",
        "rewards": {"reward": 0, "f2p_passed": 4},
        "trial_dir": "/fake/trial",
    }
    trajectory = verifier_trajectory(report, task, "snapshot")
    benchmark = DeepSWE({})
    benchmark._splits = {"test": (task,)}
    benchmark._tasks_by_id = {task.task_id: task}
    (graded,) = benchmark.grade((replace(trajectory, task=task),))
    assert graded.metrics["reward"] == 0
    assert graded.status == "finished"
    assert graded.messages[-1]["rewards"]["f2p_passed"] == 4


@pytest.mark.parametrize(("candidate_reward", "accepted"), [(0, False), (1, True)])
def test_evolution_publishes_only_candidates_that_preserve_dev(
    tmp_path, monkeypatch, candidate_reward, accepted
):
    import hashlib
    import json

    from flashrsi.agents.metacorsi import MetaCoRSITarget, verifier_trajectory
    from flashrsi.algorithms.metacorsi import MetaCoRSI
    from flashrsi.core import Agent, Snapshot

    memory = tmp_path / "memory"
    generation = memory / "ltm/versions/v1"
    generation.mkdir(parents=True)
    (memory / "current.json").write_text("original")
    manifest = generation / "manifest.json"
    manifest.write_text('{"files": {}}')
    (memory / "ltm/CURRENT.json").write_text(
        json.dumps(
            {
                "path": "versions/v1",
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            }
        )
    )
    seed = Snapshot.from_directory(
        tmp_path,
        agent_type="metacorsi",
        component_rules={"memory": ("memory/**",), "config": ("config/**",)},
    )
    monkeypatch.setattr(
        MetaCoRSITarget, "__init__", lambda agent: Agent.__init__(agent, {})
    )
    target = MetaCoRSITarget()
    target._snapshot = seed
    benchmark = DeepSWE({})
    benchmark._splits = {
        role: (
            Task(
                f"datacurve/{role}",
                role,
                env="/fixture",
                metadata={"task_checksum": "digest"},
            ),
        )
        for role in ("train", "dev", "test")
    }
    benchmark._tasks_by_id = {
        task.task_id: task for group in benchmark._splits.values() for task in group
    }
    calls = []

    async def rollout(agent, task):
        calls.append((task.question, agent.snapshot.id))
        reward = candidate_reward if agent.snapshot.id != seed.id else 1
        return verifier_trajectory(
            {
                "task_id": task.task_id,
                "task_checksum": "digest",
                "rewards": {"reward": reward},
                "trial_dir": f"/trials/{task.question}",
            },
            task,
            agent.snapshot.id,
        )

    async def native(operation, data):
        assert operation == "evolve"
        assert data["trials"] == ["/trials/train"]
        (Path(data["memory"]) / "current.json").write_text("candidate")
        # The quarantined patch registry must stay outside the snapshot.
        (Path(data["memory"]) / "candidates.json").write_text("patch")
        return {"status": "promoted"}

    target._env_runner = rollout
    monkeypatch.setattr(target, "native", native)
    method = MetaCoRSI({}, target=target, worker=None, benchmark=benchmark)
    result = asyncio.run(method.evolve())
    assert (result.id != seed.id) is accepted
    assert target.snapshot.id == result.id
    assert "memory/candidates.json" not in result.files
    assert (memory / "current.json").read_text() == "original"
    assert [role for role, _ in calls] == ["train", "dev", "dev"]
