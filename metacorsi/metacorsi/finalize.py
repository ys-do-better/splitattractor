from __future__ import annotations

import argparse
from pathlib import Path

from .harbor_agents.merge_results import merge
from .io import atomic_json


MODEL = "DeepSeek-V4.1-Flash-1017"


def _task_outcome(item: dict) -> str:
    if item["reward"] == 1.0:
        return "pass"
    f2p_passed, f2p_total = item.get("f2p_passed"), item.get("f2p_total")
    p2p_passed, p2p_total = item.get("p2p_passed"), item.get("p2p_total")
    if f2p_passed == f2p_total and p2p_passed != p2p_total:
        return "regression_after_feature_gain"
    if isinstance(f2p_passed, int) and isinstance(f2p_total, int) and 0 < f2p_passed < f2p_total:
        return "partial_contract_coverage"
    return "no_verified_feature_gain"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finalize the active DSH versus mini-swe-agent comparison."
    )
    parser.add_argument("--tasks-file", type=Path, required=True)
    parser.add_argument("--dsh-job-dir", action="append", type=Path, required=True)
    parser.add_argument("--mini-job-dir", action="append", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    outputs = {
        "dsh": merge(args.tasks_file, args.dsh_job_dir, "dsh-metaco-k7", MODEL),
        "mini-swe-agent": merge(
            args.tasks_file, args.mini_job_dir, "mini-swe-agent-metaco-k7", MODEL
        ),
    }
    if not args.allow_incomplete:
        incomplete = {
            name: value["valid_results"]
            for name, value in outputs.items()
            if not value["complete"]
        }
        if incomplete:
            raise SystemExit(f"cannot finalize incomplete runs: {incomplete}")

    for name, value in outputs.items():
        atomic_json(args.result_dir / name / "final_merged.json", value)

    dsh_passes = {
        task for task, item in outputs["dsh"]["per_task"].items()
        if item["reward"] == 1.0
    }
    mini_passes = {
        task for task, item in outputs["mini-swe-agent"]["per_task"].items()
        if item["reward"] == 1.0
    }
    comparison = {
        "schema_version": 1,
        "model": MODEL,
        "dsh": {
            "passes": outputs["dsh"]["passes"],
            "pass_rate": outputs["dsh"]["pass_rate"],
        },
        "mini-swe-agent": {
            "passes": outputs["mini-swe-agent"]["passes"],
            "pass_rate": outputs["mini-swe-agent"]["pass_rate"],
        },
        "common_passes": sorted(dsh_passes & mini_passes),
        "dsh_only_passes": sorted(dsh_passes - mini_passes),
        "mini_only_passes": sorted(mini_passes - dsh_passes),
    }
    atomic_json(args.result_dir / "comparison.json", comparison)


if __name__ == "__main__":
    main()
