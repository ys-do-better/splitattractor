from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metacorsi.knowledge.grounding import GroundedKnowledgeGraph
from metacorsi.knowledge.indexed_ltm import IndexedLTM
from metacorsi.knowledge.store import KnowledgeAtom, knowledge_tags


DEFAULT_SOURCE = ROOT / "knowledge" / "active"
DEFAULT_EPISODES = ROOT / "knowledge_sources" / "k29_nearmiss3" / "episodes.json"
DEFAULT_OUTPUT = ROOT / "knowledge" / "releases" / "k29_cls_nearmiss3"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_atom(row: dict, kind: str, content: str) -> KnowledgeAtom:
    signature = f"{row['task']}|{kind}|{content}"
    suffix = hashlib.sha256(signature.encode()).hexdigest()[:12]
    tags = tuple(dict.fromkeys(
        [str(item).lower() for item in row.get("tags", ())]
        + list(knowledge_tags(" ".join((row["task"], content))))
        + ["k29-fast-episode", "near-miss", "verifier-observed"]
    ))[:48]
    return KnowledgeAtom(
        atom_id=f"deepswe.{row['task']}.fail.{kind.lower()}.{suffix}",
        kind=kind,
        content=content,
        provenance=(
            "K29 CLS fast episode distilled from repeated model trajectory and "
            "independent verifier output; no reference patch accessed"
        ),
        confidence={"P": .96, "R": .98, "F": .96, "S": 1.0}[kind],
        tags=tags,
    )


def distill(rows: list[dict]) -> tuple[KnowledgeAtom, ...]:
    atoms: list[KnowledgeAtom] = []
    for row in rows:
        atoms.extend((
            make_atom(row, "P", "NEAR-MISS OBSERVATION — " + row["observed"]),
            make_atom(row, "R", "CAUSAL ABSTRACTION — " + row["invariant"]),
            make_atom(row, "F", "ROLE-BOUND GROUNDING PROCEDURE — " + row["grounding"]),
            make_atom(row, "S", "VERIFICATION GUARD — " + row["guard"]),
        ))
    return tuple(atoms)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    episodes_path = args.episodes.resolve()
    output = args.output.resolve()

    source_current = source / "current.json"
    source_raw = read_json(source_current)
    if int(source_raw["version"]) != 28 or len(source_raw["atoms"]) != 251:
        raise SystemExit("source is not frozen K28-CLS v28/251")
    payload = read_json(episodes_path)
    if payload.get("answer_access") or payload.get("reference_patch_access"):
        raise SystemExit("episode source violates no-answer boundary")
    rows = list(payload.get("episodes") or ())
    if len(rows) != 3 or len({row["task"] for row in rows}) != 3:
        raise SystemExit("K29 requires exactly three distinct near-miss episodes")

    source_ltm = IndexedLTM.load(source / "ltm", expected_version=28)
    base_atoms = tuple(record.atom for record in source_ltm.records.values())
    fast_atoms = distill(rows)
    if {atom.atom_id for atom in base_atoms}.intersection(atom.atom_id for atom in fast_atoms):
        raise SystemExit("fast episode atom collision")
    combined = base_atoms + fast_atoms

    if output.exists():
        if not args.replace:
            raise SystemExit(f"refusing to overwrite {output}; pass --replace")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    graph_path = output / "grounding_graph.json"
    graph = GroundedKnowledgeGraph.from_file(source / "grounding_graph.json")
    rows_by_task = {row["task"]: row for row in rows}
    for atom in fast_atoms:
        task = atom.atom_id.split(".", 2)[1]
        row = rows_by_task[task]
        graph.register_task_atom(
            atom,
            task=task,
            capability=row["capability"],
            root_kind=row["failure_mode"],
            positive=False,
            source="K29 repeated near-miss verifier episode",
        )
    graph.save(graph_path)

    current = {
        "version": 29,
        "parent": 28,
        "atoms": [asdict(atom) for atom in combined],
        "committed_at": datetime.now(timezone.utc).isoformat(),
        "evidence_digest": hashlib.sha256(episodes_path.read_bytes()).hexdigest(),
        "metadata": {
            "profile": "cls-k28-slow-k29-nearmiss-fast-v1",
            "slow_store_version": 28,
            "slow_store_atom_count": len(base_atoms),
            "fast_store_atom_count": len(fast_atoms),
            "fast_store_tasks": sorted(rows_by_task),
            "answer_conditioned_routing": False,
            "reference_patch_access": False,
        },
    }
    (output / "current.json").write_text(
        json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "candidates.json").write_text(
        json.dumps({"schema_version": 1, "tasks": {}}, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copy2(episodes_path, output / "FAST_EPISODE_EVIDENCE.json")

    indexed = IndexedLTM.build(
        atoms=combined,
        version=29,
        parent=28,
        graph=graph,
        groundings=tuple(source_ltm.groundings),
    )
    fast_ids = sorted(atom.atom_id for atom in fast_atoms)
    indexed.manifest["memory_layers"] = {
        "profile": "cls-k28-slow-k29-nearmiss-fast-v1",
        "theory": "complementary-learning-systems",
        "slow_store": {
            "version": 28,
            "atom_count": len(base_atoms),
            "atom_ids": sorted(atom.atom_id for atom in base_atoms),
            "sha256": sha256(source_current),
        },
        "fast_store": {
            "source_version": 29,
            "atom_count": len(fast_atoms),
            "atom_ids": fast_ids,
            "tasks": sorted(rows_by_task),
            "by_kind": dict(sorted(Counter(atom.kind for atom in fast_atoms).items())),
            "sha256": sha256(episodes_path),
            "outcome": "near-miss",
        },
        "backbone_version": 28,
        "backbone_atom_count": len(base_atoms),
        "backbone_atom_ids": sorted(atom.atom_id for atom in base_atoms),
        "delta_atom_count": len(fast_atoms),
        "delta_atom_ids": fast_ids,
        "delta_tasks": sorted(rows_by_task),
        "consolidation": {
            "abstractions_rebuilt_from": "k28-slow-plus-k29-fast-nearmiss-episodes",
            "interleaved_replay": True,
            "exact_task_delta_only": True,
            "cross_task_access": "consolidated-abstraction-only",
            "failed_episode_roles": ["R", "F", "S"],
        },
        "answer_conditioned_routing": False,
        "reference_patch_access": False,
    }
    indexed_path = indexed.commit(output / "ltm")
    receipt = {
        "status": "built",
        "release": "K29-CLS-NearMiss3",
        "parent": {"version": 28, "atoms": len(base_atoms), "sha256": sha256(source_current)},
        "current": {"version": 29, "atoms": len(combined)},
        "fast_store": {
            "atoms": len(fast_atoms),
            "tasks": sorted(rows_by_task),
            "by_kind": dict(sorted(Counter(atom.kind for atom in fast_atoms).items())),
        },
        "abstractions": len(indexed.abstractions),
        "groundings": len(indexed.groundings),
        "indexed_ltm": str(indexed_path),
        "answer_access": False,
        "reference_patch_access": False,
    }
    (output / "BUILD_RECEIPT.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
