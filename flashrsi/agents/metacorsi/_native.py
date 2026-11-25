"""Child process for MetaCoRSI's Harbor 0.24 runtime and native evolution."""

import asyncio
import json
import os
import shutil
import sys
import tempfile
import warnings
from collections import Counter
from contextlib import redirect_stdout
from importlib.metadata import version
from pathlib import Path


def load_memory(memory):
    from metacorsi.knowledge.indexed_ltm import IndexedLTM
    from metacorsi.knowledge.store import VersionedKnowledgeStore

    store = VersionedKnowledgeStore.from_file(memory / "current.json")
    return IndexedLTM.load(memory / "ltm", expected_version=store.active.version)


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
    if (
        task.name != data["task_id"]
        or checksum != data["checksum"]
        or task.instruction != data["instruction"]
        or task.has_steps
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
            "METACORSI_CONTROL_MODE": "rca_only",
            "METACORSI_LTM_POLICY": "modular_reverse_v23",
            "METACORSI_MEMORY_STRATEGY": "cls_dual_store",
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
    return {
        "task_id": result.task_name,
        "task_checksum": result.task_checksum,
        "rewards": (
            result.verifier_result.rewards
            if result.verifier_result and not result.exception_info
            else None
        ),
        "trial_dir": str(trial.paths.trial_dir),
    }


def evolve(data):
    from metacorsi.knowledge.evolution import evolve as native_evolve

    memory = Path(data["memory"])
    before = load_memory(memory)
    layers = before.manifest.get("memory_layers")
    receipt = native_evolve(
        memory / "current.json",
        [Path(path) for path in data["trials"]],
        policy="grounding_feedback",
    )
    after = load_memory(memory)
    if layers:
        backbone = set(layers["backbone_atom_ids"])
        if not backbone <= after.records.keys():
            raise ValueError("Evolved MetaCoRSI memory lost stable backbone atoms")
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
    return {key: receipt[key] for key in ("status", "version", "episodes", "new_atoms")}


def main(operation, data):
    if version("harbor") != "0.24.0":
        raise ValueError("Set agent.python to a Python environment with Harbor 0.24.0")
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
