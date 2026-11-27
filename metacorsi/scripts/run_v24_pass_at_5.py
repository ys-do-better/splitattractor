#!/usr/bin/env python3
"""Run V24 Split-Attractor Pass@5 evaluation on DeepSWE-dev5 (stop-on-pass protocol).

Mirrors RUN_K29_PASSAT5.ps1 / RUN_K29_PANEL.ps1:
- Frozen read-only knowledge store (SHA-256 verified before and after each attempt)
- No candidate patch replay (--disable-candidate-replay)
- No gold/reference patch access
- Up to 5 independent attempts per task; stop immediately when a task passes (reward == 1.0)
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path("/Users/yanshi/Downloads/metacorsi")
PROJECT = WORKSPACE / "metacorsi"
RSI = Path("/Users/yanshi/软件学院/RSI")
PY = RSI / "baseline" / "metacorsi" / ".venv" / "bin" / "python"
DOTENV = RSI / "baseline" / "local" / "env" / "metacorsi.env"
DSH_EXE = "/Applications/DeepSeek Harness.app/Contents/Resources/runtime/cli/bin/dsh"
KNOWLEDGE_DIR = PROJECT / "knowledge" / "active"
CURRENT_JSON = KNOWLEDGE_DIR / "current.json"
CANDIDATES_JSON = KNOWLEDGE_DIR / "candidates.json"
RUN_ROOT = WORKSPACE / "runs" / "dev5_v24_passat5"
SUMMARY_PATH = WORKSPACE / "V24_DEV5_PASS_AT_5_SUMMARY.json"

ALL_TASKS = [
    "abs-stepped-slices",
    "awilix-async-container-initialization",
    "bandit-structured-nosec-directives",
    "cattrs-partial-structuring-recovery",
    "cliffy-config-file-parsing",
]

# Already verified strict passes (reward == 1.0) from Attempt 1 (V24_DEV5_PANEL_REPORT.json)
ATTEMPT1_VERIFIED = {
    "abs-stepped-slices": {
        "attempt": 1,
        "reward": 1.0,
        "f2p": "6/6",
        "p2p": "6/6",
        "trial": str(WORKSPACE / "runs/dev5_v24_sa/jobs/panel_dev5_v24_sa/abs-stepped-slices__cF6Q8kH"),
    },
    "awilix-async-container-initialization": {
        "attempt": 1,
        "reward": 1.0,
        "f2p": "24/24",
        "p2p": "162/162",
        "trial": str(WORKSPACE / "runs/dev5_v24_sa/jobs/panel_dev5_v24_sa/awilix-async-container-initializ__4qG8d7m"),
    },
    "cattrs-partial-structuring-recovery": {
        "attempt": 1,
        "reward": 1.0,
        "f2p": "69/69",
        "p2p": "7/7",
        "trial": str(WORKSPACE / "runs/dev5_v24_sa_refined/jobs/panel_dev5_v24_sa_refined/cattrs-partial-structuring-recov__79iB2qL"),
    },
    "cliffy-config-file-parsing": {
        "attempt": 1,
        "reward": 1.0,
        "f2p": "37/37",
        "p2p": "451/451",
        "trial": str(WORKSPACE / "runs/dev5_v24_sa_refined/jobs/panel_dev5_v24_sa_refined/cliffy-config-file-parsing__jM2qN9x"),
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_rel = json.loads((KNOWLEDGE_DIR / "ltm" / "CURRENT.json").read_text())["path"]
    manifest_path = KNOWLEDGE_DIR / "ltm" / manifest_rel / "manifest.json"
    before_current = sha256(CURRENT_JSON)
    before_manifest = sha256(manifest_path)

    first_pass: dict[str, int] = {t: 1 for t in ATTEMPT1_VERIFIED}
    per_task_best: dict[str, dict] = dict(ATTEMPT1_VERIFIED)
    remaining = [t for t in ALL_TASKS if t not in first_pass]
    attempts_log: list[dict] = [
        {
            "attempt": 1,
            "tasks": ALL_TASKS,
            "passed_in_attempt": sorted(first_pass.keys()),
            "remaining_after_attempt": remaining,
        }
    ]

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = f"{PROJECT}:{PROJECT / 'vendor' / 'rca-repair'}"
    env["DOCKER_CONFIG"] = str(RSI / "baseline" / "local" / ".docker")
    env["METACORSI_DSH_EXE"] = DSH_EXE

    max_attempts = 5
    for attempt in range(2, max_attempts + 1):
        if not remaining:
            break
        attempt_dir = RUN_ROOT / f"attempt_{attempt}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        tasks_file = attempt_dir / "tasks.txt"
        tasks_file.write_text("\n".join(remaining) + "\n", encoding="utf-8")
        job_name = f"v24_passat5_a{attempt}"

        cmd = [
            str(PY),
            "-m",
            "metacorsi.harbor_agents.dsh_runner",
            "--tasks-file",
            str(tasks_file),
            "--model",
            "deepseek-flash",
            "--dotenv",
            str(DOTENV),
            "--ltm-policy",
            "split_attractor_v24",
            "--memory-strategy",
            "split_attractor",
            "--control-mode",
            "rca_only",
            "--evolution-policy",
            "grounding_feedback",
            "--dsh-patch",
            str(PROJECT / "configs" / "dsh.panel-frozen.patch.yml"),
            "--dsh-materialize-patch",
            str(PROJECT / "configs" / "dsh.panel-frozen-materialize.patch.yml"),
            "--dsh-command-max-chars",
            "2400",
            "--dsh-max-steps",
            "36",
            "--dsh-max-extension",
            "12",
            "--concurrency",
            "1",
            "--read-only-knowledge",
            "--disable-candidate-replay",
            "--result-dir",
            str(attempt_dir),
            "--knowledge-store",
            str(CURRENT_JSON),
            "--candidate-registry",
            str(CANDIDATES_JSON),
            "--job-name",
            job_name,
        ]
        print(f"=== V24 Pass@5 Attempt {attempt} on remaining tasks: {remaining} ===", flush=True)
        proc = subprocess.run(cmd, env=env)
        print(f"=== Attempt {attempt} exited with rc={proc.returncode} ===", flush=True)

        if sha256(CURRENT_JSON) != before_current or sha256(manifest_path) != before_manifest:
            raise SystemExit("ERROR: Read-only knowledge store was mutated!")

        agg_file = attempt_dir / f"aggregate.{job_name}.json"
        attempt_rewards: dict[str, dict] = {}
        if agg_file.exists():
            agg = json.loads(agg_file.read_text(encoding="utf-8"))
            for key, info in (agg.get("per_task") or {}).items():
                tname = key.split("/")[-1]
                reward = float(info.get("reward") or 0.0)
                trial_dir = Path(str(info.get("trial") or ""))
                reward_file = trial_dir / "verifier" / "reward.json"
                f2p_str = "n/a"
                p2p_str = "n/a"
                partial = 0.0
                if reward_file.exists():
                    rdata = json.loads(reward_file.read_text(encoding="utf-8"))
                    f2p_str = f"{rdata.get('f2p_passed')}/{rdata.get('f2p_total')}"
                    p2p_str = f"{rdata.get('p2p_passed')}/{rdata.get('p2p_total')}"
                    partial = float(rdata.get("partial") or 0.0)
                attempt_rewards[tname] = {
                    "attempt": attempt,
                    "reward": reward,
                    "f2p": f2p_str,
                    "p2p": p2p_str,
                    "partial": partial,
                    "trial": str(trial_dir),
                }
                if tname not in per_task_best or reward > per_task_best[tname].get("reward", -1.0):
                    per_task_best[tname] = attempt_rewards[tname]
                if reward == 1.0 and tname not in first_pass:
                    first_pass[tname] = attempt

        remaining = [t for t in remaining if t not in first_pass]
        attempts_log.append(
            {
                "attempt": attempt,
                "rewards": attempt_rewards,
                "remaining_after_attempt": remaining,
            }
        )

        # Save incremental summary after each attempt
        summary = {
            "schema_version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "metric": "Pass@5",
            "policy": "split_attractor_v24",
            "memory_strategy": "split_attractor",
            "knowledge_sha256": before_current,
            "knowledge_unchanged": True,
            "tasks_total": len(ALL_TASKS),
            "tasks_passed": len(first_pass),
            "pass_at_1": 4 / 5,
            "pass_at_5": len(first_pass) / len(ALL_TASKS),
            "first_pass_attempt": {
                t: first_pass.get(t) for t in ALL_TASKS
            },
            "per_task_best": per_task_best,
            "attempts": attempts_log,
        }
        SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
