"""Structured MetaCoRSI memory for the common hosted MiniAgent runtime."""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from flashrsi._records import _credential_key
from flashrsi.agents.metacorsi import _NATIVE, MetaCoRSITarget, copy_active_memory
from flashrsi.agents.miniagent import MiniAgent
from flashrsi.agents.miniagent.runtime.workspace import Middleware

_FOCUSED = re.compile(
    r"\b(pytest|go test|cargo test|npm test|pnpm test|yarn test|"
    r"tox|unittest|make\s+(?:\w*test\w*))\b"
)


class MemoryRuntime:
    def __init__(self, target, task, memory, grounding, evidence):
        self.target = target
        self.base = {
            "memory": str(memory),
            "question": task.question,
            "task_id": task.task_id,
        }
        self.grounding = grounding
        self.evidence = evidence
        self.rca = []
        self.focused_success = False
        self.system = None
        self.injection_bytes = 0
        self.middleware = Middleware(
            name="metacorsi_v23", before_llm=self.before_llm, after_tool=self.after_tool
        )

    async def before_llm(self, messages, ctx):
        if self.system is None:
            self.system = messages[0]["content"]
        budget = self.target.config.get("max_turns", 30)
        base = min(budget, 24)
        ctx.max_turns = budget if self.focused_success else base
        if ctx.turn <= 4:
            stage = "DIAGNOSE. Locate the first causal decision point."
        elif ctx.turn <= 12 and not self.focused_success:
            stage = "MATERIALIZE. Implement and wire a minimal vertical slice."
        elif ctx.turn <= 18:
            stage = "VERIFY. Run a focused test and fix observed boundaries."
        else:
            stage = "CLOSE. Finish integration, run decisive tests and commit according to the task."
        memory = "\n\n".join(
            filter(
                None,
                (
                    self.grounding.get("k7_context"),
                    self.grounding["prompt"].strip(),
                    "\n".join(self.grounding.get("processor_policy", ())),
                ),
            )
        )
        self.injection_bytes = len(memory.encode("utf-8"))
        messages[0] = {
            "role": "system",
            "content": f"{self.system}\n\n{memory}\n\nMetaCoRSI phase: {stage}\n"
            f"Turn {ctx.turn} of {ctx.max_turns}.",
        }
        return messages

    async def after_tool(self, name, args, result, ctx):
        if name != "shell":
            return None
        try:
            observation = json.loads(result)
        except (ValueError, TypeError):
            return None
        code = observation.get("exit_code")
        command = args.get("command", "")
        if code == 0 and _FOCUSED.search(command):
            self.focused_success = True
            ctx.max_turns = self.target.config.get("max_turns", 30)
        if type(code) is not int or code == 0:
            return {
                "metadata": {
                    "exit_code": code,
                    "timed_out": observation.get("timed_out"),
                }
            }
        diagnosis = await self.target.native(
            "analyze",
            {
                **self.base,
                "command": command,
                "observation": result,
            },
        )
        self.rca.append({"turn": ctx.turn, "exit_code": code, "diagnosis": diagnosis})
        return {
            "metadata": {"exit_code": code, "root_step": diagnosis["root_step"]},
            "result": result + "\nMetaCoRSI RCA: " + diagnosis["explanation"],
        }

    def receipt(self):
        return {
            "type": "MetaCoRSIMemory",
            "strategy": "cls_dual_store",
            "policy": "modular_reverse_v23",
            "version": self.grounding["version"],
            "groundings": self.grounding["groundings"],
            "initial_evidence": self.evidence,
            "initial_retrieval": self.grounding["prompt"],
            "working_memory": self.grounding.get("working_memory"),
            "repository_probe_count": 1,
            "injection_bytes_per_call": self.injection_bytes,
            "rca": self.rca,
            "focused_test_seen": self.focused_success,
        }


class MetaCoRSIMiniAgent(MiniAgent):
    name = "metacorsi_miniagent"
    description = "MiniAgent with host-side MetaCoRSI CLS memory and RCA."
    native = MetaCoRSITarget.native

    def __init__(self, config):
        if config.get("source_dir"):
            raise ValueError("MetaCoRSI MiniAgent uses bundled MiniAgent resources")
        project = config.get("project_dir") or os.environ.get("METACORSI_PROJECT_DIR")
        if not project:
            raise ValueError("Set agent.project_dir to the MetaCoRSI source")
        self.project = Path(project).expanduser().resolve()
        if not (self.project / "metacorsi/knowledge/evolution.py").is_file():
            raise ValueError("agent.project_dir does not contain MetaCoRSI evolution")
        self.python = (
            Path(config.get("python") or sys.executable).expanduser().absolute()
        )
        budget = config.get("max_turns", 30)
        if type(budget) is not int or not 1 <= budget <= 30:
            raise ValueError(
                "MetaCoRSI MiniAgent max_turns must be an integer from 1 to 30"
            )
        scope = config.get("evaluation_scope", "heldout")
        if scope not in {"heldout", "historical"}:
            raise ValueError("agent.evaluation_scope must be heldout or historical")
        harbor = {**config.get("harbor", {}), "hosted_miniagent": True}
        cfg = {**config, "max_turns": budget, "harbor": harbor}
        stock = MiniAgent({}).snapshot
        with stock._export(full=True) as directory, TemporaryDirectory() as temporary:
            path = Path(temporary)
            shutil.copytree(directory, path, dirs_exist_ok=True)
            components = Path(sys.modules[MiniAgent.__module__].__file__).with_name(
                "components.yaml"
            )
            (path / "components.yaml").write_text(
                components.read_text() + "\nmemory:\n  - memory/**\n"
            )
            seed = config.get("knowledge_dir")
            if seed:
                copy_active_memory(seed, path / "memory")
            env = {
                key: value
                for key, value in os.environ.items()
                if not _credential_key(key) and not key.startswith("METACORSI_")
            }
            env.update(
                PYTHONPATH=os.pathsep.join(
                    str(p) for p in (self.project, self.project / "vendor/rca-repair")
                ),
                PYTHONDONTWRITEBYTECODE="1",
                PYTHONUTF8="1",
            )
            operation = "normalize_seed" if seed else "seed"
            result = subprocess.run(
                [str(self.python), str(_NATIVE), operation],
                input=json.dumps({"memory": str(path / "memory")}),
                text=True,
                capture_output=True,
                env=env,
                timeout=30,
                check=False,
            )
            if result.returncode:
                detail = result.stderr.strip().splitlines()
                raise ValueError(
                    "Cannot initialize MiniAgent memory"
                    + (f": {detail[-1]}" if detail else "")
                )
            super().__init__({**cfg, "source_dir": str(path)})
        del self.config["source_dir"]

    async def _run(self, prompt, working_dir=None):
        raise ValueError("Use MetaCoRSI MiniAgent with the hosted DeepSWE runner")

    async def check_native(self, *, snapshot=None, forbidden_tasks=()):
        with (snapshot or self.snapshot)._export(full=True) as directory:
            inspected = await self.native(
                "inspect", {"memory": str(directory / "memory")}
            )
        if self.config.get("evaluation_scope", "heldout") == "heldout":
            overlap = set(inspected["tasks"]).intersection(
                task.task_id.removeprefix("datacurve/") for task in forbidden_tasks
            )
            if overlap:
                raise ValueError(
                    f"Held-out tasks appear in memory: {', '.join(sorted(overlap))}"
                )
        return inspected

    async def memory_runtime(self, task, directory, shell):
        memory = directory / "memory"
        output = await shell(
            {"command": "pwd && find . -maxdepth 2 -type f | head -80"}, None
        )
        evidence = json.loads(output)["output"]
        grounding = await self.native(
            "ground",
            {
                "memory": str(memory),
                "task_id": task.task_id,
                "question": task.question,
                "evidence": evidence,
            },
        )
        return MemoryRuntime(self, task, memory, grounding, evidence)
