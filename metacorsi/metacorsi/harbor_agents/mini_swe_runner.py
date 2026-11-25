from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from metacorsi.harbor_agents.common import harbor_executable
from metacorsi.io import atomic_json


def _reward(result: dict) -> float | None:
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    value = rewards.get("reward") if isinstance(rewards, dict) else None
    return float(value) if isinstance(value, (int, float)) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-file", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--control-mode", choices=("rca_only", "rca_with_regression_gates"),
                        default="rca_only")
    parser.add_argument("--job-name", default="deepswe_mini_swe_metaco_rca_1017")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    task_ids = [line.strip() for line in args.tasks_file.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
    if not task_ids:
        raise SystemExit("task file is empty")

    jobs = args.result_dir / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    command = [
        harbor_executable(), "run",
        "-d", "datacurve/deep-swe-1-1",
        "--agent", "mini-swe-agent",
        "-m", f"openai/{args.model}",
        "--agent-kwarg", "version=2.4.6",
        "--agent-kwarg", "cost_limit=0",
        "--agent-kwarg", "max_tokens=32768",
        "--agent-kwarg", f"prompt_template_path={root / 'configs' / 'mini_swe_metacorsi_prompt.j2'}",
        "-k", "1", "-n", str(args.concurrency),
        "--jobs-dir", str(jobs), "--job-name", args.job_name,
    ]
    api_host = urlparse(os.environ["OPENAI_API_BASE"]).hostname
    if api_host:
        command += ["--allow-agent-host", api_host]
    for task_id in task_ids:
        command += ["--include-task-name", f"datacurve/{task_id}"]

    env = os.environ.copy()
    env.update({
        "OPENAI_API_BASE_URL": env["OPENAI_API_BASE"],
        "METACORSI_CONTROL_MODE": args.control_mode,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    })
    log = args.result_dir / "harbor.log"
    with log.open("a", encoding="utf-8") as stream:
        exit_code = subprocess.run(
            command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT
        ).returncode

    per_task = {}
    for result_path in (jobs / args.job_name).glob("*/result.json"):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        task_id = str(result.get("task_name") or
                      result.get("config", {}).get("task", {}).get("name") or
                      result_path.parent.name)
        per_task[task_id] = {
            "reward": _reward(result),
            "trial": str(result_path.parent),
            "exception": bool(result.get("exception_info")),
        }
    passes = sum(item["reward"] == 1.0 for item in per_task.values())
    aggregate = {
        "schema_version": 1,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "framework": "mini-swe-agent-metaco-k7",
        "model": args.model,
        "control_mode": args.control_mode,
        "tasks_requested": len(task_ids),
        "tasks_found": len(per_task),
        "passes": passes,
        "pass_rate": passes / len(task_ids),
        "exit_code": exit_code,
        "per_task": per_task,
    }
    atomic_json(args.result_dir / f"aggregate.{args.job_name}.json", aggregate)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
