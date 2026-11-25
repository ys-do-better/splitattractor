from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .config import load
from .io import atomic_json, read_dotenv
from .targets import registry


def _task_path(settings) -> Path:
    p = Path(settings.raw["benchmark"]["tasks_file"])
    return p if p.is_absolute() else settings.root / p


def plan(settings, agents: list[str] | None = None) -> dict:
    agents = agents or [settings.target_agent]
    tasks = _task_path(settings)
    ids = [x.strip() for x in tasks.read_text(encoding="utf-8").splitlines()
           if x.strip() and not x.lstrip().startswith("#")]
    if len(ids) != 20:
        raise ValueError(f"formal task set must contain 20 IDs, found {len(ids)}")
    unknown = [agent for agent in agents if agent not in registry]
    if unknown:
        raise ValueError(f"unknown Target Agent(s): {', '.join(unknown)}")
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "model": settings.model,
            "dataset": settings.raw["benchmark"]["dataset"], "task_ids": ids, "agents": agents,
            "control_mode": settings.control_mode,
            "commands": {a: registry[a].command(tasks, settings.result_dir, settings.raw) for a in agents}}


def run_agent(settings, agent: str, tasks_file: str | None = None,
              job_name: str | None = None) -> int:
    # The protocol is the shared two-arm experiment contract, even when one
    # arm is launched in a separate process.
    receipt = plan(settings, list(settings.raw.get("agents", [agent])))
    out = settings.result_dir
    atomic_json(out / "protocol.json", receipt)
    dotenv = (settings.root / settings.raw["llm"]["dotenv"]).resolve()
    secret = read_dotenv(dotenv)
    env = os.environ.copy()
    env.update({"OPENAI_API_KEY": secret[settings.raw["llm"]["api_key_field"]],
                "OPENAI_API_BASE": secret[settings.raw["llm"]["base_url_field"]],
                "OPENAI_API_BASE_URL": secret[settings.raw["llm"]["base_url_field"]],
                "LITELLM_PROXY_API_KEY": secret[settings.raw["llm"]["api_key_field"]],
                "LITELLM_PROXY_API_BASE": secret[settings.raw["llm"]["base_url_field"]],
                "PYTHONPATH": os.pathsep.join((str(settings.root), env.get("PYTHONPATH", "")))})
    command = list(receipt["commands"].get(agent) or
                   registry[agent].command(_task_path(settings), settings.result_dir, settings.raw))
    if tasks_file:
        task_path = Path(tasks_file).resolve()
        command = registry[agent].command(task_path, settings.result_dir, settings.raw)
    if job_name:
        command += ["--job-name", job_name]
    suffix = f".{job_name}" if job_name else ""
    log = out / agent / f"runner{suffix}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    with log.open("a", encoding="utf-8") as fh:
        result = subprocess.run(command, cwd=settings.root, env=env,
                                stdout=fh, stderr=subprocess.STDOUT)
    receipt_name = f"run_receipt{suffix}.json"
    atomic_json(out / agent / receipt_name, {"agent": agent, "model": settings.model,
                "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
                "exit_code": result.returncode, "log": str(log),
                "tasks_file": str(tasks_file or _task_path(settings)), "job_name": job_name})
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--agents", default=None,
                   help="comma-separated override; defaults to experiment.target_agent")
    r = sub.add_parser("run")
    r.add_argument("--agent", choices=tuple(registry),
                   help="override experiment.target_agent from the config")
    r.add_argument("--tasks-file")
    r.add_argument("--job-name")
    args = parser.parse_args()
    settings = load(args.config)
    try:
        if args.action == "plan":
            selected = args.agents.split(",") if args.agents else None
            print(json.dumps(plan(settings, selected), ensure_ascii=False, indent=2))
        else:
            raise SystemExit(run_agent(
                settings, args.agent or settings.target_agent, args.tasks_file, args.job_name
            ))
    except (KeyError, ValueError, NotImplementedError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
