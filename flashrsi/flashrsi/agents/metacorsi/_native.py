"""Child process for MetaCoRSI's Harbor 0.24 runtime and native evolution."""

import asyncio
import hashlib
import json
import os
import shutil
import sys
import tempfile
import warnings
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path


def load_memory(memory):
    from metacorsi.knowledge.indexed_ltm import IndexedLTM
    from metacorsi.knowledge.store import VersionedKnowledgeStore

    store = VersionedKnowledgeStore.from_file(memory / "current.json")
    index = IndexedLTM.load(memory / "ltm", expected_version=store.active.version)
    if {a.atom_id: a for a in store.active.atoms} != {
        key: record.atom for key, record in index.records.items()
    }:
        raise ValueError("Active knowledge atoms differ from the indexed LTM")
    return index


def _extract_trial_evidence(agent_dir: Path) -> tuple[str, list[dict], dict]:
    messages: list[dict] = []
    model_calls: list[dict] = []
    session_id: str | None = None
    output = ""
    tool_calls = 0
    turns = 0

    raw_path = agent_dir / "dsh_raw.jsonl"
    if raw_path.is_file():
        for line_idx, line in enumerate(
            raw_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("session_id"):
                session_id = str(record["session_id"])
            for att in record.get("attempts") or ():
                att_no = att.get("attempt", 1)
                for ev in att.get("events") or ():
                    if not isinstance(ev, dict):
                        continue
                    if ev.get("type") == "session" and ev.get("sessionId"):
                        session_id = str(ev["sessionId"])
                    elif (
                        ev.get("type") == "status"
                        and ev.get("phase") == "step_end"
                        and isinstance(ev.get("usage"), dict)
                    ):
                        u = ev["usage"]
                        inp = int(u.get("inputTokens", u.get("input_tokens", 0)) or 0)
                        out = int(u.get("outputTokens", u.get("output_tokens", 0)) or 0)
                        cache_read = int(
                            u.get(
                                "cacheReadTokens",
                                u.get(
                                    "cacheReadInputTokens",
                                    u.get("cache_read_input_tokens", 0),
                                ),
                            )
                            or 0
                        )
                        cache_write = int(
                            u.get(
                                "cacheWriteTokens",
                                u.get(
                                    "cacheCreationInputTokens",
                                    u.get("cache_creation_input_tokens", 0),
                                ),
                            )
                            or 0
                        )
                        total = int(
                            u.get(
                                "totalTokens",
                                u.get(
                                    "total_tokens",
                                    inp + out + cache_read + cache_write,
                                ),
                            )
                            or (inp + out + cache_read + cache_write)
                        )
                        model_calls.append(
                            {
                                "call_id": f"{session_id or 'dsh'}:L{line_idx}:A{att_no}:{len(model_calls) + 1}",
                                "usage": {
                                    "input_tokens": inp,
                                    "output_tokens": out,
                                    "cache_read_input_tokens": cache_read,
                                    "cache_creation_input_tokens": cache_write,
                                    "total_tokens": total,
                                },
                            }
                        )

    bridge_path = agent_dir / "dsh_bridge.json"
    if bridge_path.is_file():
        try:
            bridge = json.loads(
                bridge_path.read_text(encoding="utf-8", errors="replace")
            )
        except json.JSONDecodeError:
            bridge = {}
        history = bridge.get("history") or []
        turns = len(history)
        for item in history:
            if not isinstance(item, dict):
                continue
            cmds = item.get("commands") or []
            tool_calls += len(cmds)
            if item.get("analysis"):
                output = str(item["analysis"])
            messages.append(
                {
                    "role": "assistant",
                    "step": item.get("step"),
                    "analysis": item.get("analysis", ""),
                    "commands": cmds,
                    "return_codes": item.get("return_codes", []),
                    "rca": item.get("rca"),
                }
            )
        messages.append(
            {
                "type": "MetaCoRSIMemory",
                "strategy": bridge.get("memory_strategy", "cls_dual_store"),
                "policy": bridge.get("ltm_policy", "modular_reverse_v23"),
                "version": (bridge.get("brain") or {})
                .get("indexed_ltm", {})
                .get("version"),
                "groundings": bridge.get("grounding") or [],
                "split_attractor_metrics": bridge.get("split_attractor_metrics"),
                "initial_retrieval": bridge.get("evolved_knowledge") or "",
                "rca": [
                    {"step": h.get("step"), "diagnosis": h.get("rca")}
                    for h in history
                    if isinstance(h, dict) and h.get("rca")
                ],
            }
        )

    inp_sum = sum(c["usage"]["input_tokens"] for c in model_calls)
    out_sum = sum(c["usage"]["output_tokens"] for c in model_calls)
    cr_sum = sum(c["usage"]["cache_read_input_tokens"] for c in model_calls)
    cw_sum = sum(c["usage"]["cache_creation_input_tokens"] for c in model_calls)
    tot_sum = sum(c["usage"]["total_tokens"] for c in model_calls)
    usage = (
        {
            "model_calls": model_calls,
            "turns": turns or len(model_calls),
            "tool_calls": tool_calls,
            "input_tokens": inp_sum,
            "output_tokens": out_sum,
            "cache_read_input_tokens": cr_sum,
            "cache_creation_input_tokens": cw_sum,
            "total_tokens": tot_sum,
            **({"session_id": session_id} if session_id else {}),
        }
        if (model_calls or turns)
        else {}
    )
    return output, messages, usage


async def run(data):
    from harbor.models.task.task import Task
    from harbor.models.trial.config import (
        AgentConfig,
        EnvironmentConfig,
        TaskConfig,
        TrialConfig,
    )
    from harbor.trial.trial import Trial

    task = Task(Path(data["task_path"]))
    with warnings.catch_warnings(action="ignore", category=DeprecationWarning):
        checksum = task.checksum
    has_steps = bool(getattr(task, "has_steps", getattr(task.config, "steps", None)))
    if (
        task.name != data["task_id"]
        or checksum != data["checksum"]
        or task.instruction != data["instruction"]
        or has_steps
        or task.config.environment.os.value != "linux"
        or task.config.verifier.environment_mode != "separate"
        or not task.config.verifier.collect
        or task.config.agent.network_mode != "no-network"
        or task.config.verifier.network_mode != "no-network"
    ):
        raise ValueError("DeepSWE task identity or verifier protocol changed")
    snapshot, output = Path(data["snapshot"]), Path(data["output"])
    os.environ.update(
        {
            "METACORSI_DSH_EXE": data["dsh"],
            "METACORSI_DSH_BASE_URL": data["base_url"],
            "METACORSI_DSH_PATCH": str(snapshot / "config/dsh.panel-frozen.patch.yml"),
            "METACORSI_DSH_MATERIALIZE_PATCH": str(
                snapshot / "config/dsh.panel-frozen-materialize.patch.yml"
            ),
            "METACORSI_DSH_COMMAND_MAX_CHARS": "1600",
            "METACORSI_DSH_RUNTIME_ROOT": str(output / "dsh-runtime"),
            "METACORSI_DSH_MAX_STEPS": str(data["max_steps"]),
            "METACORSI_DSH_MAX_EXTENSION": str(data["max_extension"]),
            "METACORSI_KNOWLEDGE_STORE": str(snapshot / "memory/current.json"),
            "METACORSI_DISABLE_CANDIDATE_REPLAY": "1",
            "METACORSI_CONTROL_MODE": data.get("control_mode", "rca_only"),
            "METACORSI_LTM_POLICY": data.get("ltm_policy", "modular_reverse_v23"),
            "METACORSI_MEMORY_STRATEGY": data.get("memory_strategy", "cls_dual_store"),
            "DSH_TELEMETRY_MODE": "DISABLED",
        }
    )
    # DSH runs on the host. The native repository and verifier stay offline.
    trial = await Trial.create(
        TrialConfig(
            task=TaskConfig(path=task.task_dir),
            trials_dir=output / "trials",
            agent=AgentConfig(
                import_path="metacorsi_dsh_agent:AgentHarness",
                model_name="openai/" + data["model"],
                override_timeout_sec=data["timeout"],
            ),
            environment=EnvironmentConfig(type="docker", delete=True),
        )
    )
    try:
        result = await trial.run()
    finally:
        shutil.rmtree(output / "dsh-runtime", ignore_errors=True)
    final_output, trace_messages, usage = _extract_trial_evidence(
        trial.paths.agent_dir
    )
    return {
        "task_id": result.task_name,
        "task_checksum": result.task_checksum,
        "rewards": (
            result.verifier_result.rewards
            if result.verifier_result and not result.exception_info
            else None
        ),
        "trial_dir": str(trial.paths.trial_dir),
        "output": final_output,
        "messages": trace_messages,
        "usage": usage,
    }


def evolve(data):
    from metacorsi.knowledge.evolution import evolve as native_evolve

    memory = Path(data["memory"])
    before = load_memory(memory)
    layers = before.manifest.get("memory_layers")
    if "episodes" in data:
        from metacorsi.knowledge.evolution import (
            VerifiedEpisode,
            _failed_tests,
            _patch_blueprint,
            evolve_from_episodes,
        )

        episodes = []
        for row in data["episodes"]:
            row = dict(row)
            digest = row.pop("patch_sha256")
            if digest and hashlib.sha256(row["patch"].encode()).hexdigest() != digest:
                raise ValueError(
                    "Archived patch checksum changed during episode conversion"
                )
            episodes.append(
                VerifiedEpisode(
                    **row,
                    failed_tests=_failed_tests(Path(row["source"])),
                    implementation_blueprint=_patch_blueprint(row["patch"]),
                )
            )
        receipt = evolve_from_episodes(
            memory / "current.json",
            episodes,
            policy=data["policy"],
            sources=tuple(ep.source for ep in episodes),
        )
    else:
        receipt = native_evolve(
            memory / "current.json",
            [Path(path) for path in data["trials"]],
            policy=data.get("policy", "grounding_feedback"),
        )
    after = load_memory(memory)
    if layers:
        backbone = set(layers["backbone_atom_ids"])
        if any(
            key not in after.records
            or before.records[key].atom != after.records[key].atom
            for key in backbone
        ):
            raise ValueError("Evolved MetaCoRSI memory changed stable backbone atoms")
        delta = sorted(after.records.keys() - backbone)
        by_task = Counter(after.records[key].task for key in delta)
        by_kind = Counter(after.records[key].atom.kind for key in delta)
        after.manifest["memory_layers"] = {
            **layers,
            "delta_atom_ids": delta,
            "delta_atom_count": len(delta),
            "delta_tasks": sorted(by_task),
            "fast_store": {
                **layers.get("fast_store", {}),
                "source_version": after.version,
                "atom_count": len(delta),
                "tasks": sorted(by_task),
                "by_task": dict(sorted(by_task.items())),
                "by_kind": dict(sorted(by_kind.items())),
            },
        }
        # Native commit reuses a generation with the same atom/grounding digest.
        # Commit to an empty index so the updated layer metadata is persisted.
        with tempfile.TemporaryDirectory(dir=memory) as staging:
            index = Path(staging) / "ltm"
            after.commit(index)
            shutil.rmtree(memory / "ltm")
            index.rename(memory / "ltm")
    load_memory(memory)
    return {
        key: receipt.get(key)
        for key in (
            "status",
            "policy",
            "version",
            "episodes",
            "new_atoms",
            "candidate_updates",
        )
    }


def seed_memory(memory):
    from metacorsi.io import atomic_json
    from metacorsi.knowledge.grounding import GroundedKnowledgeGraph
    from metacorsi.knowledge.indexed_ltm import IndexedLTM
    from metacorsi.knowledge.store import VersionedKnowledgeStore, k7_seed_atoms

    atoms = k7_seed_atoms()
    stamp = "1970-01-01T00:00:00+00:00"
    seed = asdict(VersionedKnowledgeStore(atoms, version=0).active)
    seed.update(
        committed_at=stamp, metadata={"profile": "native-K7-priors-no-episodes"}
    )
    atomic_json(memory / "current.json", seed)
    index = IndexedLTM.build(
        atoms=atoms, version=0, parent=None, graph=GroundedKnowledgeGraph()
    )
    index.manifest.update(
        built_at=stamp,
        memory_layers={
            "profile": "native-K7-priors-no-episodes",
            "backbone_version": 0,
            "backbone_atom_ids": sorted(index.records),
            "backbone_atom_count": len(atoms),
            "delta_atom_ids": [],
            "delta_atom_count": 0,
            "delta_tasks": [],
        },
    )
    index.commit(memory / "ltm")


def memory_operation(operation, data):
    memory = Path(data["memory"])
    if operation == "seed":
        seed_memory(memory)
        return {"status": "seeded"}
    if operation == "normalize_seed":
        index = load_memory(memory)
        if index.manifest.get("memory_layers"):
            return {"status": "verified"}
        if index.version != 20 or len(index.records) != 215:
            raise ValueError(
                "Expected a layered K28/K29 package or the original 215-atom K20"
            )
        # The established K20 source is the full stable backbone with no fast delta.
        index.manifest["memory_layers"] = {
            "profile": "native-K20-stable-backbone",
            "backbone_version": 20,
            "backbone_atom_ids": sorted(index.records),
            "backbone_atom_count": 215,
            "delta_atom_ids": [],
            "delta_atom_count": 0,
            "delta_tasks": [],
        }
        with tempfile.TemporaryDirectory(dir=memory) as staging:
            destination = Path(staging) / "ltm"
            index.commit(destination)
            shutil.rmtree(memory / "ltm")
            destination.rename(memory / "ltm")
        return {"status": "normalized-k20"}
    if operation == "inspect":
        index = load_memory(memory)
        if not index.manifest.get("memory_layers"):
            raise ValueError("Indexed memory is missing CLS memory_layers")
        layers = index.manifest["memory_layers"]
        stable, fast = set(layers["backbone_atom_ids"]), set(layers["delta_atom_ids"])
        if (
            stable & fast
            or stable | fast != index.records.keys()
            or len(stable) != layers["backbone_atom_count"]
            or len(fast) != layers["delta_atom_count"]
        ):
            raise ValueError("CLS memory_layers must partition the indexed atoms")
        if any(
            set(atom_ids) - index.records.keys() for atom_ids in index.anchors.values()
        ):
            raise ValueError("Retention anchors refer to absent knowledge atoms")
        tasks = set(index.indexes["task"])
        for abstract in index.abstractions.values():
            tasks.update(abstract.positive_tasks)
            tasks.update(abstract.negative_tasks)
        for item in index.groundings:
            if item.task:
                tasks.add(item.task)
            tasks.update(
                index.records[atom_id].task
                for atom_id in item.source_atom_ids
                if atom_id in index.records and index.records[atom_id].task
            )
        return {
            "version": index.version,
            "tasks": sorted(tasks),
            "atoms": len(index.records),
            "slow": len(stable),
            "fast": len(fast),
        }
    from metacorsi.brain import MetaCoRSIBrain
    from metacorsi.contracts import FailureEvidence

    os.environ.update(
        METACORSI_LTM_POLICY=data.get("ltm_policy", "modular_reverse_v23"),
        METACORSI_MEMORY_STRATEGY=data.get("memory_strategy", "cls_dual_store"),
    )
    brain = MetaCoRSIBrain(
        control_mode="rca_only", knowledge_path=memory / "current.json"
    )
    if operation == "ground":
        # The native store uses the unqualified task slug for episodic retrieval.
        task_id = data["task_id"].removeprefix("datacurve/")
        prompt = brain.evolved_knowledge_prompt(
            data["question"],
            task_id=task_id,
            repository_evidence=data["evidence"],
        )
        generated = brain.generate_agent("dsh", task_id, data["question"])
        return {
            "prompt": prompt,
            "k7_context": brain.skill_router.prompt(
                data["question"], include_validation=False
            ),
            "working_memory": generated.working_memory.receipt(),
            "processor_policy": generated.processor_policy,
            "groundings": brain.grounding_receipt(),
            "version": brain.ltm.active.version,
        }
    diagnosis = brain.analyze(
        FailureEvidence(
            task_id=data["task_id"],
            assertion=data["observation"][-6000:],
            trace=data["observation"][-12000:],
            task_text=data["question"],
            failed_tests=(data["command"],),
            metadata={"impacted_tests": [data["command"]]},
        )
    )
    return asdict(diagnosis)


def main(operation, data):
    if operation in {"seed", "normalize_seed", "inspect", "ground", "analyze"}:
        return memory_operation(operation, data)
    if operation in {"check", "run"} and not version("harbor").startswith(
        ("0.23.", "0.24.")
    ):
        raise ValueError(
            "Set agent.python to a Python environment with Harbor 0.23.x or 0.24.x"
        )
    if operation == "check":
        from metacorsi_dsh_agent import AgentHarness

        load_memory(Path(data["memory"]))
        return {"harness": AgentHarness.name(), "harbor": version("harbor")}
    if operation == "run":
        return asyncio.run(run(data))
    if operation == "evolve":
        return evolve(data)
    raise ValueError(f"Unexpected MetaCoRSI operation: {operation}")


if __name__ == "__main__":
    data = json.load(sys.stdin)
    with redirect_stdout(sys.stderr):
        outcome = main(sys.argv[1], data)
    print(json.dumps(outcome))
