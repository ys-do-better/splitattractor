"""Unit and integration tests for the DeepSeek Harness (dsh) adapter on DeepSWE."""

from __future__ import annotations

import asyncio
import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from flashrsi._run_summary import _add_usage, _empty
from flashrsi.agents import DSH, DeepSeekHarness
from flashrsi.agents.deepseek_harness import (
    _parse_dsh_events,
    _render_snapshot_resources,
    _summarize_usage,
    _validate,
    _write_cordis_patch,
)
from flashrsi.agents.metacorsi import verifier_trajectory
from flashrsi.agents.metacorsi._native import _extract_trial_evidence
from flashrsi.algorithms import ACE, OPRO, MetaCoRSI, SkillOpt
from flashrsi.algorithms.rrsi import _tokens as rrsi_tokens
from flashrsi.benchmarks import DeepSWE
from flashrsi.config import load_config
from flashrsi.core import Agent, Task, Trajectory
from flashrsi.integrations._hosted_dsh import HostedDSHBridge
from flashrsi.integrations._hosted_miniagent import _TRIAL, HostedTrial
from flashrsi.runner import _AGENTS


@pytest.fixture(scope="module")
def native_project():
    root = Path(__file__).resolve().parents[2]
    for project in (root / "metacorsi", root / "metacorsi/metacorsi"):
        if (project / "metacorsi/knowledge/evolution.py").is_file():
            return project
    pytest.skip("native MetaCoRSI source is not installed")


def _dsh_config(**overrides):
    return {
        "timeout": 30,
        "max_turns": 4,
        "max_extension": 2,
        "llm": {
            "model_name": "deepseek-flash",
            "api_key": "secret-dsh-key",
            "base_url": "https://api.deepseek.com/v1",
            "max_tokens": 4096,
        },
        **overrides,
    }


def _make_fake_dsh(tmp_path: Path, responses: list[dict]) -> Path:
    """Create an executable script that mimics `dsh --profile headless --json -`."""
    state_file = tmp_path / "dsh_state.json"
    state_file.write_text(json.dumps({"index": 0, "responses": responses}), encoding="utf-8")
    script = tmp_path / "fake_dsh.py"
    script.write_text(
        f"""#!/usr/bin/env python3
import json, sys
from pathlib import Path

state_path = Path({str(state_file)!r})
state = json.loads(state_path.read_text(encoding="utf-8"))
idx = min(state["index"], len(state["responses"]) - 1)
item = state["responses"][idx]
state["index"] += 1
state_path.write_text(json.dumps(state), encoding="utf-8")
_ = sys.stdin.read()

session_id = item.get("session_id", "session-fake-123")
text = item["text"]
usage = item.get("usage", {{"inputTokens": 1200, "outputTokens": 80, "cacheReadTokens": 300, "totalTokens": 1580}})
events = [
    {{"type": "session", "sessionId": session_id, "cwd": "."}},
    {{"type": "status", "phase": "turn_start", "turn": 1}},
    {{"type": "status", "phase": "step_start", "turn": 1, "step": 1}},
    {{"type": "text", "text": text}},
    {{"type": "status", "phase": "step_end", "turn": 1, "step": 1, "usage": usage}},
    {{"type": "status", "phase": "turn_end", "turn": 1, "reason": {{"kind": "completed"}}}},
    {{"type": "final", "sessionId": session_id, "reason": {{"kind": "completed"}}, "text": text}},
]
for ev in events:
    print(json.dumps(ev))
sys.exit(item.get("returncode", 0))
""",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def test_deepseek_harness_and_dsh_expose_snapshot_and_validate():
    for name, cls in (("deepseek_harness", DeepSeekHarness), ("dsh", DSH)):
        assert _AGENTS[name] is cls
        agent = cls(_dsh_config())
        assert agent.name == name
        assert agent.instruction_file == "AGENTS.md"
        assert "AGENTS.md" in agent.snapshot.files
        assert "prompts/system.md" in agent.snapshot.files
        assert "skills/main.md" in agent.snapshot.files
        assert "config.yaml" in agent.snapshot.files
        assert "memory/current.json" in agent.snapshot.files
        assert set(agent.component_rules) == {"prompt", "skill", "config", "memory"}

    for bad_field in ("node_path", "bun_path", "gateway_path"):
        with pytest.raises(ValueError, match=bad_field):
            _validate(_dsh_config(**{bad_field: "/usr/bin/false"}))
    with pytest.raises(ValueError, match="model_name"):
        _validate(_dsh_config(llm={"api_key": "secret"}))
    with pytest.raises(ValueError, match="api_key"):
        _validate(_dsh_config(llm={"model_name": "deepseek-flash"}))


def test_dsh_quickstart_loads_and_validates():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples/quickstarts/deepswe_dsh.yaml"
    )
    cfg = load_config(path)
    assert cfg.agent.name == "dsh"
    assert cfg.benchmark.name == "deepswe"
    assert cfg.algorithm.name == "skillopt"


def test_dsh_event_parsing_and_canonical_token_accounting(tmp_path):
    raw_events = "\n".join(
        json.dumps(ev)
        for ev in (
            {"type": "session", "sessionId": "session-abc", "cwd": str(tmp_path)},
            {"type": "status", "phase": "step_start", "turn": 1, "step": 1},
            {"type": "thinking", "text": "Inspecting slice evaluation"},
            {"type": "text", "text": '{"analysis":"ok","commands":["pwd"],"done":true}'},
            {
                "type": "status",
                "phase": "step_end",
                "turn": 1,
                "step": 1,
                "usage": {
                    "inputTokens": 22051,
                    "outputTokens": 120,
                    "totalTokens": 26395,
                    "cacheReadTokens": 4224,
                },
            },
            {
                "type": "final",
                "sessionId": "session-abc",
                "reason": {"kind": "completed"},
                "text": '{"analysis":"ok","commands":["pwd"],"done":true}',
            },
        )
    )
    final_text, session_id, events, calls = _parse_dsh_events(raw_events)
    assert session_id == "session-abc"
    assert json.loads(final_text)["done"] is True
    assert len(events) == 6
    usage = _summarize_usage(calls, turns=1, tool_calls=1, session_id=session_id)
    assert usage["input_tokens"] == 22051
    assert usage["output_tokens"] == 120
    assert usage["cache_read_input_tokens"] == 4224
    assert usage["total_tokens"] == 26395
    assert rrsi_tokens(usage) == 26395

    total = _empty()
    _add_usage(total, usage, set(), "target")
    assert total["input_tokens"] == 22051 + 4224
    assert total["output_tokens"] == 120
    assert total["cache_read_tokens"] == 4224
    assert total["usage_entries_missing_tokens"] == 0


def test_cordis_patch_includes_evolved_prompts_and_skills(tmp_path):
    agent = DSH(_dsh_config())
    with agent.snapshot.check_out() as checkout:
        (checkout / "AGENTS.md").write_text("CUSTOM_EVOLVED_AGENTS_MD", encoding="utf-8")
        (checkout / "skills/main.md").write_text("CUSTOM_EVOLVED_SKILL_MD", encoding="utf-8")
        sys_prompt, extra, skill_dirs = _render_snapshot_resources(checkout)
        assert "CUSTOM_EVOLVED_AGENTS_MD" in extra
        assert "CUSTOM_EVOLVED_SKILL_MD" in extra
        assert len(skill_dirs) == 1
        patch_file = _write_cordis_patch(
            tmp_path / "test.patch.yml",
            model="deepseek-flash",
            max_tokens=4096,
            persona_prefix=sys_prompt,
            persona_suffix=extra,
            skill_dirs=skill_dirs,
            disable_host_tools=True,
        )
        content = patch_file.read_text(encoding="utf-8")
        assert "CUSTOM_EVOLVED_AGENTS_MD" in content
        assert "CUSTOM_EVOLVED_SKILL_MD" in content
        assert "tool-bash" in content


def test_direct_run_executes_dsh_and_records_usage(tmp_path):
    fake_dsh = _make_fake_dsh(
        tmp_path,
        [
            {
                "text": "Direct answer from DSH",
                "usage": {
                    "inputTokens": 500,
                    "outputTokens": 40,
                    "cacheReadTokens": 100,
                    "totalTokens": 640,
                },
            }
        ],
    )
    agent = DSH(_dsh_config(cli_path=str(fake_dsh)))
    trajectory = asyncio.run(agent.run("Solve the question"))
    assert trajectory.status == "finished"
    assert trajectory.output == "Direct answer from DSH"
    assert trajectory.usage["total_tokens"] == 640
    assert trajectory.usage["turns"] == 1
    assert rrsi_tokens(trajectory.usage) == 640


def test_hosted_dsh_bridge_executes_container_commands_rca_and_budget_extension(
    tmp_path, native_project
):
    fake_dsh = _make_fake_dsh(
        tmp_path,
        [
            {
                "text": json.dumps(
                    {
                        "analysis": "Run failing test first and check blocked command",
                        "commands": [
                            "cd /app && pytest -q",
                            "git reflog",
                        ],
                        "done": False,
                    }
                ),
                "usage": {
                    "inputTokens": 1000,
                    "outputTokens": 100,
                    "cacheReadTokens": 200,
                    "totalTokens": 1300,
                },
            },
            {
                "text": json.dumps(
                    {
                        "analysis": "Apply fix and pass focused pytest",
                        "commands": ["cd /app && pytest -q"],
                        "done": False,
                    }
                ),
                "usage": {
                    "inputTokens": 1400,
                    "outputTokens": 110,
                    "cacheReadTokens": 400,
                    "totalTokens": 1910,
                },
            },
            {
                "text": json.dumps(
                    {
                        "analysis": "Extension step unlocked after focused test pass",
                        "commands": ["cd /app && git status --short"],
                        "done": True,
                    }
                ),
                "usage": {
                    "inputTokens": 1600,
                    "outputTokens": 90,
                    "cacheReadTokens": 500,
                    "totalTokens": 2190,
                },
            },
        ],
    )
    target = DSH(
        _dsh_config(
            cli_path=str(fake_dsh),
            project_dir=str(native_project),
            enable_memory=True,
            max_turns=2,
            max_extension=1,
        )
    )
    task = Task(
        "datacurve/abs-stepped-slices",
        "Implement stepped slices in ABS.",
        env="/fake/env",
        metadata={"task_checksum": "digest"},
    )
    pytest_calls = 0
    executed_commands: list[str] = []

    async def fake_exec(command: str, timeout_sec: int = 60, **kwargs):
        nonlocal pytest_calls
        executed_commands.append(command)
        if "pytest" in command:
            pytest_calls += 1
            if pytest_calls == 1:
                return SimpleNamespace(
                    return_code=1,
                    stdout="FAILED test_slice.py::test_step - AssertionError: expected [1, 3]",
                    stderr="",
                )
            return SimpleNamespace(return_code=0, stdout="1 passed", stderr="")
        return SimpleNamespace(return_code=0, stdout="/app\nmain.go\n", stderr="")

    environment = SimpleNamespace(exec=fake_exec)
    bridge = HostedDSHBridge(logs_dir=tmp_path / "agent_logs")
    asyncio.run(bridge.setup(environment))
    hosted_trial = HostedTrial(target, task, task.question)
    context = SimpleNamespace(metadata={})

    token = _TRIAL.set(hosted_trial)
    try:
        asyncio.run(bridge.run(task.question, environment, context))
    finally:
        _TRIAL.reset(token)

    traj = hosted_trial.trajectory
    assert traj is not None
    assert traj.status == "finished"
    assert "Extension step unlocked" in traj.output
    assert traj.usage["turns"] == 3
    assert traj.usage["tool_calls"] == 4
    assert traj.usage["total_tokens"] == 1300 + 1910 + 2190
    assert rrsi_tokens(traj.usage) == 1300 + 1910 + 2190
    # Blocked answer-search command ("git reflog") must never reach environment.exec
    assert not any("git reflog" in cmd for cmd in executed_commands)
    memory_msg = next(m for m in traj.messages if m.get("type") == "MetaCoRSIMemory")
    assert memory_msg["focused_test_seen"] is True
    assert len(memory_msg["rca"]) == 1
    assert (bridge.logs_dir / "dsh_raw.jsonl").is_file()
    assert (bridge.logs_dir / "dsh_bridge.json").is_file()


def test_native_trial_evidence_extractor_populates_metacorsi_target_usage(tmp_path):
    (tmp_path / "dsh_raw.jsonl").write_text(
        json.dumps(
            {
                "session_id": "session-xyz",
                "attempts": [
                    {
                        "attempt": 1,
                        "returncode": 0,
                        "events": [
                            {"type": "session", "sessionId": "session-xyz"},
                            {
                                "type": "status",
                                "phase": "step_end",
                                "turn": 1,
                                "step": 1,
                                "usage": {
                                    "inputTokens": 3000,
                                    "outputTokens": 200,
                                    "cacheReadTokens": 500,
                                    "totalTokens": 3700,
                                },
                            },
                        ],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "dsh_bridge.json").write_text(
        json.dumps(
            {
                "model": "deepseek-flash",
                "grounding": [{"abstract_id": "abstract.1"}],
                "evolved_knowledge": "Grounded schema",
                "brain": {"indexed_ltm": {"version": 29}},
                "history": [
                    {
                        "step": 1,
                        "analysis": "Fixed parser boundary",
                        "commands": ["cd /app && pytest -q"],
                        "return_codes": [0],
                        "rca": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output, messages, usage = _extract_trial_evidence(tmp_path)
    assert output == "Fixed parser boundary"
    assert usage["total_tokens"] == 3700
    assert usage["turns"] == 1
    assert usage["tool_calls"] == 1
    task = Task("datacurve/test", "Task", metadata={"task_checksum": "digest"})
    traj = verifier_trajectory(
        {
            "task_id": task.task_id,
            "task_checksum": "digest",
            "rewards": {"reward": 1},
            "trial_dir": str(tmp_path),
            "output": output,
            "messages": messages,
            "usage": usage,
        },
        task,
        "snap-1",
    )
    assert traj.status == "finished"
    assert traj.output == "Fixed parser boundary"
    assert traj.usage["total_tokens"] == 3700
    assert any(m.get("type") == "MetaCoRSIMemory" for m in traj.messages)


def test_dsh_works_with_opro_ace_skillopt_and_metacorsi_algorithms(
    tmp_path, native_project
):
    benchmark = DeepSWE({})
    benchmark._splits = {
        role: (
            Task(
                f"datacurve/dsh-{role}",
                f"Solve {role}",
                env="/fixture",
                metadata={"task_checksum": f"chk-{role}"},
            ),
        )
        for role in ("train", "dev", "test")
    }
    benchmark._tasks_by_id = {
        task.task_id: task for split in benchmark._splits.values() for task in split
    }

    # 1. OPRO evolves AGENTS.md on DSH
    target_opro = DSH(_dsh_config())
    seed_id = target_opro.snapshot.id

    async def opro_rollout(agent, task):
        content = agent.snapshot.files["AGENTS.md"].content.decode("utf-8")
        reward = 1 if "Improved DSH instruction" in content else 0
        return Trajectory(
            output="done",
            messages=(
                {
                    "type": "HarborVerifier",
                    "task_id": task.task_id,
                    "task_checksum": task.metadata["task_checksum"],
                    "rewards": {"reward": reward},
                },
            ),
            usage={
                "model_calls": [
                    {
                        "call_id": f"{agent.snapshot.id}:{task.task_id}",
                        "usage": {
                            "input_tokens": 100,
                            "output_tokens": 20,
                            "total_tokens": 120,
                        },
                    }
                ],
                "turns": 1,
            },
        )

    target_opro._env_runner = opro_rollout

    class FakeWorker(Agent):
        name = "fake_worker"
        description = "Fake worker"

        def __init__(self, reply_fn):
            super().__init__({})
            self._reply_fn = reply_fn

        async def _run(self, prompt, working_dir=None):
            return Trajectory(output=self._reply_fn(prompt))

    opro = OPRO(
        {"max_steps": 1, "num_candidates": 1, "few_shot_size": 1},
        target=target_opro,
        worker=FakeWorker(lambda _: "<INS>Improved DSH instruction</INS>"),
        benchmark=benchmark,
    )
    evolved_opro = asyncio.run(opro.evolve())
    assert evolved_opro.id != seed_id
    assert b"Improved DSH instruction" in evolved_opro.files["AGENTS.md"].content

    # 2. ACE evolves AGENTS.md playbook on DSH
    target_ace = DSH(_dsh_config())
    target_ace._env_runner = opro_rollout
    ace_replies = iter(
        [
            json.dumps(
                {
                    "reasoning": "Need better slice check",
                    "error_identification": "Missing negative step branch",
                    "root_cause_analysis": "Parser skipped third colon",
                    "correct_approach": "Parse third slice expression",
                    "key_insight": "Always check 3-part slice syntax",
                    "bullet_tags": [],
                }
            ),
            json.dumps(
                {
                    "reasoning": "Add slice bullet",
                    "operations": [
                        {
                            "type": "ADD",
                            "section": " strategies_and_hard_rules",
                            "content": "Verify 3-part slice syntax in parser and evaluator.",
                        }
                    ],
                }
            ),
        ]
    )
    ace = ACE(
        {"mode": "online", "max_steps": 1, "max_reflection_rounds": 1},
        target=target_ace,
        worker=FakeWorker(lambda _: next(ace_replies)),
        benchmark=benchmark,
    )
    evolved_ace = asyncio.run(ace.evolve())
    assert evolved_ace.id != target_ace.snapshot.root_id
    assert (
        b"Verify 3-part slice syntax" in evolved_ace.files["AGENTS.md"].content
    )

    # 3. SkillOpt evolves skills/main.md on DSH
    target_skillopt = DSH(_dsh_config())

    async def skillopt_rollout(agent, task):
        skill_text = agent.snapshot.files["skills/main.md"].content.decode("utf-8")
        reward = 1 if "Check AST slice bounds" in skill_text else 0
        return Trajectory(
            output="done",
            messages=(
                {
                    "type": "HarborVerifier",
                    "task_id": task.task_id,
                    "task_checksum": task.metadata["task_checksum"],
                    "rewards": {"reward": reward},
                },
            ),
        )

    target_skillopt._env_runner = skillopt_rollout
    skillopt = SkillOpt(
        {
            "num_epochs": 1,
            "batch_size": 1,
            "max_steps": 1,
            "use_slow_update": False,
            "use_meta_skill": False,
        },
        target=target_skillopt,
        worker=FakeWorker(
            lambda _: json.dumps(
                {
                    "reasoning": "Add AST slice bound guideline",
                    "edits": [
                        {
                            "op": "append",
                            "path": "skills/main.md",
                            "content": "- Check AST slice bounds before evaluation.",
                        }
                    ],
                }
            )
        ),
        benchmark=benchmark,
    )
    evolved_skill = asyncio.run(skillopt.evolve())
    assert b"Check AST slice bounds" in evolved_skill.files["skills/main.md"].content

    # 4. MetaCoRSI evolves memory/** on DSH across two rounds (fail -> pass)
    target_meta = DSH(_dsh_config(project_dir=str(native_project)))
    calls = []

    async def meta_rollout(agent, task):
        calls.append(task.task_id)
        trial = tmp_path / f"trial-{len(calls)}"
        trial.mkdir()
        patch = trial / "artifacts/logs/artifacts/model.patch"
        patch.parent.mkdir(parents=True)
        patch.write_text(
            f"diff --git a/app.py b/app.py\n+++ b/app.py\n+step = {len(calls)}\n",
            encoding="utf-8",
        )
        reward = 0 if len(calls) == 1 else 1
        rewards = {
            "reward": reward,
            "f2p_passed": reward + 1,
            "f2p_total": 2,
            "p2p_passed": 1,
            "p2p_total": 1,
        }
        (trial / "result.json").write_text(
            json.dumps(
                {
                    "task_name": task.task_id,
                    "task_checksum": task.metadata["task_checksum"],
                    "id": trial.name,
                    "verifier_result": {"rewards": rewards},
                }
            ),
            encoding="utf-8",
        )
        return Trajectory(
            output="Solved with DSH",
            messages=(
                {
                    "type": "MetaCoRSIMemory",
                    "version": 0,
                    "groundings": [],
                },
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
            usage={
                "model_calls": [
                    {
                        "call_id": f"call-{len(calls)}",
                        "usage": {
                            "input_tokens": 1000,
                            "output_tokens": 100,
                            "total_tokens": 1100,
                        },
                    }
                ],
                "turns": 1,
            },
        )

    target_meta._env_runner = meta_rollout
    meta_algo = MetaCoRSI(
        {"max_steps": 2, "evolution_policy": "standard"},
        target=target_meta,
        worker=None,
        benchmark=benchmark,
    )
    meta_algo._run_dir = tmp_path / "meta_run"
    for child in ("snapshots", "trajectories", "working"):
        (meta_algo._run_dir / child).mkdir(parents=True)

    final_snapshot = asyncio.run(meta_algo.evolve())
    inspected = asyncio.run(target_meta.check_native(snapshot=final_snapshot))
    assert inspected["version"] == 2
    assert inspected["tasks"] == ["dsh-train"]
    assert inspected["slow"] == 5
    assert inspected["fast"] > 0
    assert "memory/candidates.json" not in final_snapshot.files
    assert (meta_algo._run_dir / "quarantine/0001.json").is_file()
