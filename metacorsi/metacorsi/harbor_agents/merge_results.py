from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from metacorsi.io import atomic_json


def _task_id(data: dict[str, Any]) -> str:
    value = data.get("task_name") or (data.get("config") or {}).get("task", {}).get("name")
    return str(value or "").removeprefix("datacurve/")


def _reward(data: dict[str, Any]) -> float | None:
    rewards = (data.get("verifier_result") or {}).get("rewards") or {}
    value = rewards.get("reward") if isinstance(rewards, dict) else None
    return float(value) if isinstance(value, (int, float)) else None


def _rca_count(trial: Path) -> int:
    count = 0
    for candidate in (trial / "agent" / "metacorsi_rca.jsonl", trial / "agent" / "dsh_bridge.json"):
        if not candidate.exists():
            continue
        if candidate.suffix == ".jsonl":
            count += sum(1 for line in candidate.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip())
        else:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            count += sum(1 for item in payload.get("history", []) if item.get("rca"))
    return count


def merge(
    tasks_file: Path,
    job_dirs: list[Path],
    framework: str,
    model: str,
) -> dict[str, Any]:
    requested = [line.strip() for line in tasks_file.read_text(encoding="utf-8").splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
    requested_set = set(requested)
    valid: dict[str, dict[str, Any]] = {}
    infrastructure: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []

    for job_dir in job_dirs:
        for result_path in sorted(job_dir.glob("*/result.json")):
            data = json.loads(result_path.read_text(encoding="utf-8"))
            task = _task_id(data)
            if task not in requested_set:
                continue
            trial = result_path.parent
            reward = _reward(data)
            agent_model = str(((data.get("config") or {}).get("agent") or {}).get("model_name") or "")
            item = {
                "task_id": task,
                "reward": reward,
                "f2p_passed": ((data.get("verifier_result") or {}).get("rewards") or {}).get("f2p_passed"),
                "f2p_total": ((data.get("verifier_result") or {}).get("rewards") or {}).get("f2p_total"),
                "p2p_passed": ((data.get("verifier_result") or {}).get("rewards") or {}).get("p2p_passed"),
                "p2p_total": ((data.get("verifier_result") or {}).get("rewards") or {}).get("p2p_total"),
                "trial": str(trial.resolve()),
                "job": str(job_dir.resolve()),
                "model": agent_model.removeprefix("openai/"),
                "rca_events": _rca_count(trial),
                "finished_at": data.get("finished_at"),
            }
            if reward is None:
                item["exception"] = data.get("exception_info")
                infrastructure.append(item)
                continue
            if item["model"] != model:
                raise ValueError(f"model mismatch for {task}: {item['model']!r} != {model!r}")
            if task in valid:
                duplicates.append(valid[task])
            valid[task] = item

    missing = [task for task in requested if task not in valid]
    ordered = {task: valid[task] for task in requested if task in valid}
    passes = sum(item["reward"] == 1 for item in ordered.values())
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "framework": framework,
        "model": model,
        "tasks_requested": len(requested),
        "valid_results": len(ordered),
        "passes": passes,
        "pass_rate": passes / len(requested),
        "complete": not missing,
        "missing_valid_results": missing,
        "per_task": ordered,
        "excluded_infrastructure_attempts": infrastructure,
        "superseded_valid_attempts": duplicates,
        "source_jobs": [str(path.resolve()) for path in job_dirs],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-file", type=Path, required=True)
    parser.add_argument("--job-dir", action="append", type=Path, required=True)
    parser.add_argument("--framework", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_json(args.output, merge(args.tasks_file, args.job_dir, args.framework, args.model))


if __name__ == "__main__":
    main()
