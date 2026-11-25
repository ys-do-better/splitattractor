from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metacorsi.knowledge.grounding import GroundedKnowledgeGraph
from metacorsi.knowledge.indexed_ltm import IndexedLTM


DEFAULT_K20 = ROOT / "knowledge_sources" / "k20_215"
DEFAULT_K28 = ROOT / "knowledge_sources" / "k28_251"
DEFAULT_OUTPUT = ROOT / "knowledge" / "active"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k20", type=Path, default=DEFAULT_K20)
    parser.add_argument("--k28", type=Path, default=DEFAULT_K28)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    k20 = args.k20.resolve()
    k28 = args.k28.resolve()
    output = args.output.resolve()
    k20_current = k20 / "current.json"
    k28_current = k28 / "current.json"
    k20_raw = read_json(k20_current)
    k28_raw = read_json(k28_current)
    if int(k20_raw["version"]) != 20 or len(k20_raw["atoms"]) != 215:
        raise SystemExit("K20 source is not the expected v20/215 snapshot")
    if int(k28_raw["version"]) != 28 or len(k28_raw["atoms"]) != 251:
        raise SystemExit("K28 source is not the expected v28/251 snapshot")

    backbone_ids = {row["atom_id"] for row in k20_raw["atoms"]}
    combined_ids = {row["atom_id"] for row in k28_raw["atoms"]}
    missing = backbone_ids - combined_ids
    delta_ids = combined_ids - backbone_ids
    if missing or len(delta_ids) != 36:
        raise SystemExit(
            f"unexpected K20->K28 relation: missing_backbone={len(missing)}, delta={len(delta_ids)}"
        )

    source_ltm = IndexedLTM.load(k28 / "ltm", expected_version=28)
    if not combined_ids.issubset(source_ltm.records):
        raise SystemExit("K28 indexed LTM does not contain every K28 atom")
    delta_tasks = sorted({source_ltm.records[atom_id].task for atom_id in delta_ids})
    delta_by_task = Counter(source_ltm.records[atom_id].task for atom_id in delta_ids)
    delta_by_kind = Counter(source_ltm.records[atom_id].atom.kind for atom_id in delta_ids)

    output.mkdir(parents=True, exist_ok=True)
    combined = dict(k28_raw)
    combined["metadata"] = {
        **dict(combined.get("metadata") or {}),
        "profile": "cls-k20-slow-backbone-k28-fast-delta-v1",
        "backbone_version": 20,
        "backbone_atom_count": len(backbone_ids),
        "fast_delta_atom_count": len(delta_ids),
        "answer_conditioned_routing": False,
    }
    (output / "current.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    graph_path = k28 / "grounding_graph.json"
    shutil.copy2(graph_path, output / "grounding_graph.json")
    (output / "candidates.json").write_text(
        json.dumps({"schema_version": 1, "tasks": {}}, indent=2) + "\n",
        encoding="utf-8",
    )

    atoms = tuple(source_ltm.records[atom_id].atom for atom_id in sorted(combined_ids))
    retained_groundings = tuple(
        receipt
        for receipt in source_ltm.groundings
        if not receipt.source_atom_ids or set(receipt.source_atom_ids).issubset(combined_ids)
    )
    indexed = IndexedLTM.build(
        atoms=atoms,
        version=28,
        parent=27,
        graph=GroundedKnowledgeGraph.from_file(graph_path),
        groundings=retained_groundings,
    )
    # Rebuilding from all 251 records is the slow-store consolidation step.
    # The manifest keeps the newly learned 36 atoms in a separate fast store;
    # retrieval may expose them directly only to their exact task capsule.
    indexed.manifest["memory_layers"] = {
        "profile": "cls-k20-slow-backbone-k28-fast-delta-v1",
        "theory": "complementary-learning-systems",
        "slow_store": {
            "version": 20,
            "atom_count": len(backbone_ids),
            "atom_ids": sorted(backbone_ids),
            "sha256": digest(k20_current),
        },
        "fast_store": {
            "source_version": 28,
            "atom_count": len(delta_ids),
            "atom_ids": sorted(delta_ids),
            "tasks": delta_tasks,
            "by_task": dict(sorted(delta_by_task.items())),
            "by_kind": dict(sorted(delta_by_kind.items())),
            "sha256": digest(k28_current),
        },
        # Backward-compatible fields consumed by _layered_capsule/retrieve.
        "backbone_version": 20,
        "backbone_atom_count": len(backbone_ids),
        "backbone_atom_ids": sorted(backbone_ids),
        "delta_atom_count": len(delta_ids),
        "delta_atom_ids": sorted(delta_ids),
        "delta_tasks": delta_tasks,
        "consolidation": {
            "abstractions_rebuilt_from": "k20-backbone-plus-k28-delta",
            "interleaved_replay": True,
            "exact_task_delta_only": True,
            "cross_task_access": "consolidated-abstraction-only",
        },
        "answer_conditioned_routing": False,
    }
    indexed_path = indexed.commit(output / "ltm")

    receipt = {
        "status": "built",
        "profile": "cls-k20-slow-backbone-k28-fast-delta-v1",
        "knowledge": str(output / "current.json"),
        "indexed_ltm": str(indexed_path),
        "source_hashes": {
            "k20": digest(k20_current),
            "k28": digest(k28_current),
        },
        "slow_store": {"version": 20, "atoms": len(backbone_ids)},
        "fast_store": {
            "source_version": 28,
            "atoms": len(delta_ids),
            "tasks": delta_tasks,
            "by_task": dict(sorted(delta_by_task.items())),
            "by_kind": dict(sorted(delta_by_kind.items())),
        },
        "combined": {"version": 28, "atoms": len(combined_ids)},
        "consolidated_abstractions": len(indexed.abstractions),
        "retained_groundings": len(retained_groundings),
        "answer_access": False,
    }
    receipt_path = output.parent / "BUILD_RECEIPT.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
