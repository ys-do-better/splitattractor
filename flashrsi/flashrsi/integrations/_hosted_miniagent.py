"""Run trusted MiniAgent code on the host, with only its shell in Harbor."""

import asyncio
import json
import math
import posixpath
from contextvars import ContextVar
from dataclasses import dataclass
from tempfile import TemporaryDirectory

from harbor.agents.base import BaseAgent

from flashrsi._records import _config_record
from flashrsi.agents.miniagent import MiniAgent
from flashrsi.agents.miniagent.runtime.agent import run
from flashrsi.core import Trajectory

# The stock MiniAgent runtime configures its SDK clients in module globals.
# Serialize host invocations so concurrent trials cannot share those clients.
_RUN_LOCK = asyncio.Lock()
_TRIAL = ContextVar("flashrsi_hosted_miniagent_trial")


@dataclass
class HostedTrial:
    agent: MiniAgent
    task: object
    prompt: str
    trajectory: Trajectory | None = None


def validate_resources(snapshot):
    allowed = {"prompt", "skill", "memory"}

    def executable_resources(files):
        return {
            path: entry
            for path, entry in files.items()
            if entry.component not in allowed
        }

    stock = MiniAgent({}).snapshot
    if executable_resources(snapshot.files) != executable_resources(stock.files):
        raise ValueError(
            "Hosted MiniAgent requires bundled runtime, tools and configuration"
        )
    for path, item in snapshot.files.items():
        if item.component not in allowed:
            continue
        directory = {"prompt": "prompts/", "skill": "skills/", "memory": "memory/"}[
            item.component
        ]
        if not path.startswith(directory) or item.executable:
            raise ValueError(
                "Hosted MiniAgent resource layout does not match its component"
            )


class HostedShell:
    def __init__(self, environment, workdir):
        self.environment = environment
        self.workdir = workdir

    async def __call__(self, args, _ctx):
        command = args.get("command")
        cwd = args.get("cwd", ".")
        timeout = args.get("timeout", 60)
        if not isinstance(command, str) or not command.strip():
            raise ValueError("command must be a non-empty string")
        if not isinstance(cwd, str) or not cwd or type(timeout) not in (int, float):
            raise ValueError("cwd and timeout have invalid types")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        directory = posixpath.normpath(posixpath.join(self.workdir, cwd))
        if posixpath.commonpath([self.workdir, directory]) != self.workdir:
            raise ValueError("cwd is outside the task working directory")
        try:
            result = await self.environment.exec(
                command, cwd=directory, timeout_sec=timeout
            )
        except TimeoutError:
            return json.dumps(
                {"exit_code": None, "output": "", "timed_out": True, "truncated": False}
            )
        data = ((result.stdout or "") + (result.stderr or "")).encode()
        return json.dumps(
            {
                "exit_code": result.return_code,
                "output": data[:32768].decode(errors="replace"),
                "timed_out": False,
                "truncated": len(data) > 32768,
            },
            ensure_ascii=False,
        )


class HostedMiniAgentBridge(BaseAgent):
    @staticmethod
    def name():
        return "flashrsi-hosted-miniagent"

    def version(self):
        return "1"

    async def setup(self, environment):
        pass

    async def run(self, instruction, environment, context):
        trial = _TRIAL.get()
        target = trial.agent
        validate_resources(target.snapshot)
        working = await environment.exec("pwd", timeout_sec=10)
        workdir = (working.stdout or "").strip()
        if working.return_code or not workdir.startswith("/") or "\n" in workdir:
            raise ValueError(
                "Harbor did not provide an absolute task working directory"
            )
        shell = HostedShell(environment, workdir)
        async with _RUN_LOCK:
            with TemporaryDirectory(prefix="flashrsi-hosted-miniagent-") as temporary:
                with target.snapshot._export(full=True) as directory:
                    runtime = (
                        await target.memory_runtime(trial.task, directory, shell)
                        if hasattr(target, "memory_runtime")
                        else None
                    )

                    def configure_bundle(bundle):
                        bundle.extra_tools["shell"] = shell
                        if runtime is not None:
                            bundle.middlewares.append(runtime.middleware)

                    result = await asyncio.wait_for(
                        run(
                            trial.prompt,
                            workspace_dir=directory,
                            runs_root=temporary,
                            llm_config=dict(target.config["llm"]),
                            max_turns=target.config.get("max_turns", 24),
                            bundle_transform=configure_bundle,
                            force_answer=False,
                        ),
                        timeout=target.config.get("timeout", 1800),
                    )
                    success = result["stop_reason"] == "completed"
                    messages = [
                        {
                            "type": "RunConfig",
                            "config": _config_record(target.config),
                            "snapshot_id": target.snapshot.id,
                        },
                        *result["messages"],
                    ]
                    messages.append(
                        {
                            "type": "MiniAgentTermination",
                            "reason": result["termination_reason"],
                        }
                    )
                    if runtime is not None:
                        messages.append(runtime.receipt())
                    trial.trajectory = Trajectory(
                        status="finished" if success else "error",
                        output=result["answer"] if success else "",
                        messages=tuple(messages),
                        usage={
                            "model_calls": result["usage"],
                            "turns": result["turns"],
                            "tool_calls": result["tool_calls"],
                        },
                        error_message=None if success else result["stop_reason"],
                    )
