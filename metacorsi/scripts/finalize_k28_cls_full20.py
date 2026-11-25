from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTROL = ROOT / "evidence" / "k28_cls_full20" / "CONTROL_K20_K28_FULL20.json"


def task_rows(run_root: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in run_root.rglob("result.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        task_name = payload.get("task_name")
        rewards = ((payload.get("verifier_result") or {}).get("rewards") or {})
        if not task_name or "reward" not in rewards:
            continue
        short = task_name.split("/", 1)[-1]
        rows[short] = {
            "task_name": task_name,
            "reward": int(rewards.get("reward") or 0),
            "partial": float(rewards.get("partial") or 0.0),
            "f2p_passed": int(rewards.get("f2p_passed") or 0),
            "f2p_total": int(rewards.get("f2p_total") or 0),
            "p2p_passed": int(rewards.get("p2p_passed") or 0),
            "p2p_total": int(rewards.get("p2p_total") or 0),
            "exception": bool(payload.get("exception_info")),
            "result": str(path),
        }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    rows = task_rows(run_root)
    control = json.loads(args.control.read_text(encoding="utf-8-sig"))
    k20 = control["arms"]["k20_215"]["tasks"]
    k28 = control["arms"]["k28_251"]["tasks"]

    passed = sorted(task for task, row in rows.items() if row["reward"] == 1)
    failed = sorted(task for task, row in rows.items() if row["reward"] == 0)
    exceptions = sorted(task for task, row in rows.items() if row["exception"])
    mean_partial = sum(row["partial"] for row in rows.values()) / max(1, len(rows))
    comparisons = []
    for task in sorted(rows):
        comparisons.append({
            "task": task,
            "k20_reward": int(k20[task]["reward"]),
            "k28_reward": int(k28[task]["reward"]),
            "cls_reward": rows[task]["reward"],
            "k20_partial": float(k20[task]["partial"]),
            "k28_partial": float(k28[task]["partial"]),
            "cls_partial": rows[task]["partial"],
            "cls_f2p": f"{rows[task]['f2p_passed']}/{rows[task]['f2p_total']}",
            "cls_p2p": f"{rows[task]['p2p_passed']}/{rows[task]['p2p_total']}",
        })
    payload = {
        "schema_version": 1,
        "status": "complete" if len(rows) == 20 and not exceptions else "incomplete",
        "profile": "cls-k20-slow-backbone-k28-fast-delta-v1",
        "valid": len(rows),
        "passes": len(passed),
        "mean_partial": mean_partial,
        "passed_tasks": passed,
        "failed_tasks": failed,
        "exceptions": exceptions,
        "controls": {
            "k20_passes": int(control["arms"]["k20_215"]["passes"]),
            "k28_passes": int(control["arms"]["k28_251"]["passes"]),
        },
        "comparisons": comparisons,
    }
    (run_root / "K28_CLS_FULL20_REPORT.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# K28-CLS 全量 DeepSWE-20 报告",
        "",
        f"- 状态：`{payload['status']}`；有效任务：{len(rows)}/20；异常：{len(exceptions)}。",
        f"- K20/215：**{payload['controls']['k20_passes']}/20**。",
        f"- 原始 K28/251：**{payload['controls']['k28_passes']}/20**。",
        f"- K28-CLS：**{len(passed)}/20**；mean partial={mean_partial:.6f}。",
        "- K28-CLS = K20 slow backbone + K20→K28 fast episodic delta + consolidated slow schema。",
        "",
        "| Task | K20 | 原始 K28 | K28-CLS | CLS F2P | CLS P2P |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['task']} | {row['k20_reward']} | {row['k28_reward']} | "
            f"{row['cls_reward']} | {row['cls_f2p']} | {row['cls_p2p']} |"
        )
    lines.extend(("", "## K28-CLS 严格通过", "", ", ".join(passed), ""))
    (run_root / "K28_CLS_FULL20_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    finished = {
        "status": payload["status"],
        "valid": len(rows),
        "passes": len(passed),
        "passed_tasks": passed,
        "report": str(run_root / "K28_CLS_FULL20_REPORT.json"),
    }
    (run_root / "FULL20_FINISHED.json").write_text(
        json.dumps(finished, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(finished, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
