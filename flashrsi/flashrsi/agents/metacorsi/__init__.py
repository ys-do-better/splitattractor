"""Run the local MetaCoRSI DSH harness against a FlashRSI DeepSWE task."""

import asyncio
import json
import os
import re
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp

from flashrsi._records import _config_record, _credential_key
from flashrsi.core import Agent, Trajectory

_NATIVE = Path(__file__).with_name("_native.py")
_PROFILES = ("dsh.panel-frozen.patch.yml", "dsh.panel-frozen-materialize.patch.yml")
_MEMORY = "memory"


def copy_active_memory(source, destination):
    """Keep one hash-verified indexed generation, never the patch registry."""
    source, destination = Path(source), Path(destination)
    pointer = json.loads((source / "ltm/CURRENT.json").read_bytes())
    generation = Path(pointer["path"])
    if (
        len(generation.parts) != 2
        or generation.parts[0] != "versions"
        or not generation.name.startswith("v")
    ):
        raise ValueError("MetaCoRSI LTM pointer must refer to ltm/versions/v*")
    source_gen = source / "ltm" / generation
    manifest = json.loads((source_gen / "manifest.json").read_bytes())
    paths = [Path("current.json"), Path("ltm/CURRENT.json")]
    if (source / "grounding_graph.json").is_file():
        paths.append(Path("grounding_graph.json"))
    paths.append(Path("ltm") / generation / "manifest.json")
    for name in manifest["files"]:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid path in MetaCoRSI LTM manifest")
        paths.append(Path("ltm") / generation / relative)
    for path in paths:
        dest = destination / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / path, dest)


class MetaCoRSITarget(Agent):
    name = "metacorsi"
    description = "Local MetaCoRSI DSH harness with native DeepSWE verification."

    def __init__(self, config):
        root = config.get("project_dir") or os.environ.get("METACORSI_PROJECT_DIR")
        if not root:
            raise ValueError("Set agent.project_dir to the local MetaCoRSI repository")
        self.project = Path(root).expanduser().resolve()
        if not (self.project / "metacorsi_dsh_agent/__init__.py").is_file():
            raise ValueError(
                "agent.project_dir does not contain the MetaCoRSI DSH harness"
            )
        python = config.get("python") or os.environ.get("METACORSI_PYTHON")
        self.python = (
            Path(python).expanduser().absolute() if python else Path(sys.executable)
        )
        if not self.python.is_file():
            raise ValueError(
                "agent.python must point to the MetaCoRSI Python environment"
            )
        seed = config.get("knowledge_dir")
        seed = (
            Path(seed).expanduser().resolve()
            if seed
            else self.project / "knowledge/releases/k29_cls_nearmiss3"
        )
        self.dsh = (
            config.get("cli_path")
            or os.environ.get("METACORSI_DSH_EXE")
            or shutil.which("dsh")
            or "/Applications/DeepSeek Harness.app/Contents/Resources/runtime/cli/bin/dsh"
        )
        if not os.path.isfile(self.dsh) or not os.access(self.dsh, os.X_OK):
            raise ValueError("Set agent.cli_path to the executable DSH CLI")
        if "source_dir" in config:
            raise ValueError("MetaCoRSI uses agent.project_dir and agent.knowledge_dir")
        for field, default, minimum in (
            ("max_turns", 24, 1),
            ("max_extension", 6, 0),
            ("timeout", 5400, 1),
        ):
            value = config.get(field, default)
            if type(value) is not int or value < minimum:
                raise ValueError(f"agent.{field} must be an integer >= {minimum}")
        with TemporaryDirectory(prefix="flashrsi-metacorsi-root-") as temporary:
            source = Path(temporary)
            (source / "components.yaml").write_text(
                "memory:\n  - memory/**\nconfig:\n  - config/**\n", encoding="utf-8"
            )
            copy_active_memory(seed, source / _MEMORY)
            (source / "config").mkdir()
            model = config.get("llm", {}).get("model_name")
            if not isinstance(model, str) or not model.strip():
                raise ValueError(
                    "Set agent.llm.model_name to the DSH provider model ID"
                )
            for profile in _PROFILES:
                path = self.project / "configs" / profile
                text, count = re.subn(
                    r"(?m)^( +(?:- id|model): )deepseek-flash[ \t]*$",
                    lambda match: match[1] + json.dumps(model),
                    path.read_text(encoding="utf-8"),
                )
                if count != 2:
                    raise ValueError(f"Cannot bind the chosen DSH model in {profile}")
                (source / "config" / profile).write_text(text, encoding="utf-8")
            super().__init__({**config, "source_dir": str(source)})
        del self.config["source_dir"]

    async def _run(self, prompt, working_dir=None):
        raise ValueError("MetaCoRSI runs as a DeepSWE Target")

    async def native(self, operation, data, *, log=None, timeout=120):
        env = {
            key: val
            for key, val in os.environ.items()
            if not key.startswith("METACORSI_") and not _credential_key(key)
        }
        env.update(
            PYTHONPATH=os.pathsep.join(
                str(path) for path in (self.project, self.project / "vendor/rca-repair")
            ),
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONUTF8="1",
        )
        if operation == "run":
            env["METACORSI_DSH_API_KEY"] = (
                self.config["llm"].get("api_key") or os.environ["METACORSI_DSH_API_KEY"]
            )
        stdout = asyncio.subprocess.PIPE
        if log is None:
            stderr = asyncio.subprocess.PIPE
        else:
            stderr = log
        process = await asyncio.create_subprocess_exec(
            str(self.python),
            str(_NATIVE),
            operation,
            stdin=asyncio.subprocess.PIPE,
            stdout=stdout,
            stderr=stderr,
            env=env,
        )
        try:
            output, errors = await asyncio.wait_for(
                process.communicate(json.dumps(data).encode()), timeout
            )
        except (asyncio.CancelledError, TimeoutError):
            if process.returncode is None:
                process.terminate()
            try:
                await asyncio.wait_for(process.communicate(), 30)
            except TimeoutError:
                process.kill()
                await process.communicate()
            raise
        if process.returncode:
            detail = (
                errors.decode(errors="replace").strip().splitlines()[-1:]
                if errors
                else []
            )
            raise ValueError(
                "MetaCoRSI native "
                + operation
                + " failed"
                + (f": {detail[0]}" if detail else "; see native.log")
            )
        return json.loads(output)

    async def check_native(self):
        with self.snapshot._export(full=True) as directory:
            await self.native("check", {"memory": str(directory / _MEMORY)})

    async def run_task(self, task):
        event = {
            "type": "RunConfig",
            "config": _config_record(self.config),
            "snapshot_id": self.snapshot.id,
        }
        parent = self.config.get("harbor", {}).get("artifacts_dir")
        if parent:
            Path(parent).mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="flashrsi-metacorsi-") as temporary:
            output = (
                Path(mkdtemp(prefix="metacorsi-", dir=parent))
                if parent
                else Path(temporary)
            )
            with self.snapshot._export(full=True) as directory:
                data = {
                    "snapshot": str(directory),
                    "output": str(output),
                    "task_path": task.env,
                    "task_id": task.task_id,
                    "checksum": task.metadata["task_checksum"],
                    "instruction": task.question,
                    "dsh": str(Path(self.dsh).resolve()),
                    "model": self.config["llm"]["model_name"],
                    "base_url": self.config["llm"]["base_url"],
                    "max_steps": self.config.get("max_turns", 24),
                    "max_extension": self.config.get("max_extension", 6),
                    "timeout": self.config.get("timeout", 5400),
                }
                try:
                    with (output / "native.log").open("wb") as log:
                        report = await self.native(
                            "run", data, log=log, timeout=data["timeout"] + 3600
                        )
                except (ValueError, TimeoutError, OSError):
                    return Trajectory(
                        status="error",
                        messages=(event,),
                        error_message=f"MetaCoRSI trial failed; see {output / 'native.log'}"
                        if parent
                        else "MetaCoRSI trial failed",
                    )
            return verifier_trajectory(report, task, self.snapshot.id, event)


def verifier_trajectory(report, task, snapshot_id, *messages):
    rewards = report.get("rewards") or {}
    reward = rewards.get("reward")
    if (
        report.get("task_id") != task.task_id
        or report.get("task_checksum") != task.metadata["task_checksum"]
        or type(reward) not in (int, float)
        or not 0 <= reward <= 1
    ):
        return Trajectory(
            status="error",
            messages=messages,
            error_message="No valid native verifier reward for the requested DeepSWE task",
        )
    event = {
        "type": "HarborVerifier",
        "task_id": task.task_id,
        "task_checksum": report["task_checksum"],
        "snapshot_id": snapshot_id,
        "verifier_environment_mode": "separate",
        "rewards": rewards,
        "trial_dir": report["trial_dir"],
    }
    return Trajectory(messages=(*messages, event))
