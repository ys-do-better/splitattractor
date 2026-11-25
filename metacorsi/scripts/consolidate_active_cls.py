from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metacorsi.knowledge.grounding import GroundedKnowledgeGraph
from metacorsi.knowledge.indexed_ltm import IndexedLTM


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild one evolved brain as K20 slow store + post-K20 fast store + consolidated schemas."
    )
    parser.add_argument("--knowledge", type=Path, required=True)
    parser.add_argument(
        "--backbone", type=Path,
        default=ROOT / "knowledge_sources" / "k20_215" / "current.json",
    )
    args = parser.parse_args()
    knowledge = args.knowledge.resolve()
    current_path = knowledge / "current.json"
    graph_path = knowledge / "grounding_graph.json"
    current = read(current_path)
    backbone = read(args.backbone.resolve())
    current_ids = {row["atom_id"] for row in current["atoms"]}
    backbone_ids = {row["atom_id"] for row in backbone["atoms"]}
    missing = backbone_ids - current_ids
    if missing:
        raise SystemExit(f"working brain lost {len(missing)} K20 backbone atoms")
    delta_ids = current_ids - backbone_ids

    source = IndexedLTM.load(knowledge / "ltm", expected_version=int(current["version"]))
    if not current_ids.issubset(source.records):
        raise SystemExit("working indexed LTM is missing active atoms")
    delta_tasks = sorted({source.records[atom_id].task for atom_id in delta_ids})
    by_task = Counter(source.records[atom_id].task for atom_id in delta_ids)
    by_kind = Counter(source.records[atom_id].atom.kind for atom_id in delta_ids)
    atoms = tuple(source.records[atom_id].atom for atom_id in sorted(current_ids))
    groundings = tuple(
        item for item in source.groundings
        if not item.source_atom_ids or set(item.source_atom_ids).issubset(current_ids)
    )
    indexed = IndexedLTM.build(
        atoms=atoms,
        version=int(current["version"]),
        parent=current.get("parent"),
        graph=GroundedKnowledgeGraph.from_file(graph_path),
        groundings=groundings,
    )
    indexed.manifest["memory_layers"] = {
        "profile": "cls-k20-slow-backbone-evolved-fast-store-v1",
        "theory": "complementary-learning-systems",
        "backbone_version": int(backbone["version"]),
        "backbone_atom_count": len(backbone_ids),
        "backbone_atom_ids": sorted(backbone_ids),
        "delta_atom_count": len(delta_ids),
        "delta_atom_ids": sorted(delta_ids),
        "delta_tasks": delta_tasks,
        "slow_store": {
            "version": int(backbone["version"]),
            "atom_count": len(backbone_ids),
            "sha256": sha(args.backbone.resolve()),
        },
        "fast_store": {
            "source_version": int(current["version"]),
            "atom_count": len(delta_ids),
            "tasks": delta_tasks,
            "by_task": dict(sorted(by_task.items())),
            "by_kind": dict(sorted(by_kind.items())),
        },
        "consolidation": {
            "abstractions_rebuilt_from": "stable-backbone-plus-all-verified-deltas",
            "interleaved_replay": True,
            "exact_task_delta_only": True,
            "cross_task_access": "consolidated-abstraction-only",
        },
        "answer_conditioned_routing": False,
    }
    path = indexed.commit(knowledge / "ltm")
    receipt = {
        "status": "consolidated",
        "version": int(current["version"]),
        "backbone_atoms": len(backbone_ids),
        "fast_atoms": len(delta_ids),
        "fast_tasks": delta_tasks,
        "abstractions": len(indexed.abstractions),
        "groundings": len(groundings),
        "indexed_ltm": str(path),
    }
    receipt_path = knowledge / "CLS_CONSOLIDATION.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
