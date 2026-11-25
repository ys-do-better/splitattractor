from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metacorsi.knowledge.grounding import task_matches
from metacorsi.knowledge.indexed_ltm import IndexedLTM


def tasks(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--knowledge", type=Path, default=ROOT / "knowledge" / "active")
    parser.add_argument("--tasks", type=Path, default=ROOT / "data" / "deepswe20_ids.txt")
    parser.add_argument("--output", type=Path, default=ROOT / "evidence" / "RETRIEVAL_AUDIT.json")
    parser.add_argument("--memory-strategy", default="cls_dual_store")
    args = parser.parse_args()
    knowledge = args.knowledge.resolve()
    current = json.loads((knowledge / "current.json").read_text(encoding="utf-8"))
    ltm = IndexedLTM.load(knowledge / "ltm", expected_version=int(current["version"]))
    layers = ltm.manifest.get("memory_layers") or {}
    delta_ids = set(layers.get("delta_atom_ids") or ())
    delta_tasks = set(layers.get("delta_tasks") or ())
    violations: list[dict] = []
    rows: list[dict] = []
    previous = os.environ.get("METACORSI_MEMORY_STRATEGY")
    os.environ["METACORSI_MEMORY_STRATEGY"] = args.memory_strategy
    try:
        for task in tasks(args.tasks):
            bundle, applications = ltm.ground(
                task_id=task,
                task_text=task.replace("-", " "),
                policy="modular_reverse_v23",
                limit=14,
            )
            buckets = bundle.audit.get("buckets") or {}
            direct_delta = set(buckets.get("delta_branch") or ())
            direct_backbone = set(buckets.get("backbone_capsule") or ())
            foreign_delta = {
                atom_id for atom_id in direct_delta
                if atom_id in ltm.records
                and not task_matches(ltm.records[atom_id].task, task)
            }
            if foreign_delta:
                violations.append({"task": task, "reason": "foreign fast-store atoms", "ids": sorted(foreign_delta)})
            if task not in delta_tasks and direct_delta:
                violations.append({"task": task, "reason": "delta activated for non-delta task", "ids": sorted(direct_delta)})
            if len(applications) > 1:
                violations.append({"task": task, "reason": "more than one slow schema activated"})
            for application in applications:
                for atom_id in application.source_atom_ids:
                    record = ltm.records.get(atom_id)
                    if record and record.task and task_matches(record.task, task):
                        violations.append({"task": task, "reason": "target episode leaked into slow schema", "id": atom_id})
            rows.append({
                "task": task,
                "backbone_atoms": sorted(direct_backbone),
                "fast_delta_atoms": sorted(direct_delta),
                "slow_schema_ids": [item.abstract_id for item in applications],
                "slow_schema_sources": [list(item.source_atom_ids) for item in applications],
            })
    finally:
        if previous is None:
            os.environ.pop("METACORSI_MEMORY_STRATEGY", None)
        else:
            os.environ["METACORSI_MEMORY_STRATEGY"] = previous
    payload = {
        "schema_version": 1,
        "valid": not violations,
        "profile": layers.get("profile"),
        "memory_strategy": args.memory_strategy,
        "backbone_atoms": layers.get("backbone_atom_count"),
        "delta_atoms": len(delta_ids),
        "delta_tasks": sorted(delta_tasks),
        "rows": rows,
        "violations": violations,
    }
    path = args.output.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "valid": not violations,
        "tasks": len(rows),
        "delta_tasks": len(delta_tasks),
        "violations": violations,
        "output": str(path),
    }, ensure_ascii=False, indent=2))
    if violations:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
