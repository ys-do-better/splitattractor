from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..io import atomic_json


@dataclass(frozen=True)
class KnowledgeAtom:
    atom_id: str
    kind: str
    content: str
    provenance: str
    confidence: float = 1.0
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class KnowledgeVersion:
    version: int
    parent: int | None
    atoms: tuple[KnowledgeAtom, ...]
    committed_at: str
    evidence_digest: str
    metadata: dict[str, Any] = field(default_factory=dict)


class VersionedKnowledgeStore:
    """Append-only, task-pinned LTM with explicit stage/commit semantics."""

    def __init__(self, atoms: tuple[KnowledgeAtom, ...], version: int = 7) -> None:
        initial = KnowledgeVersion(version, None, atoms, datetime.now(timezone.utc).isoformat(),
                                   self._digest(atoms), {"profile": "K7/adaptive_v5"})
        self._versions = {version: initial}
        self._active = version
        self._staged: list[KnowledgeAtom] = []

    @classmethod
    def from_file(cls, path: Path) -> "VersionedKnowledgeStore":
        if not path.exists():
            return cls(k7_seed_atoms(), 7)
        raw = json.loads(path.read_text(encoding="utf-8"))
        atoms = tuple(KnowledgeAtom(
            atom_id=item["atom_id"], kind=item["kind"], content=item["content"],
            provenance=item["provenance"], confidence=float(item.get("confidence", 1.0)),
            tags=tuple(item.get("tags", ())),
        ) for item in raw.get("atoms", ()))
        store = cls(atoms or k7_seed_atoms(), int(raw.get("version", 7)))
        active = store.active
        store._versions[active.version] = KnowledgeVersion(
            active.version, raw.get("parent"), active.atoms,
            raw.get("committed_at", active.committed_at),
            raw.get("evidence_digest", active.evidence_digest),
            raw.get("metadata", {}),
        )
        return store

    @staticmethod
    def _digest(atoms: tuple[KnowledgeAtom, ...]) -> str:
        material = "\n".join(f"{a.atom_id}|{a.kind}|{a.content}|{a.confidence}" for a in atoms)
        return hashlib.sha256(material.encode()).hexdigest()

    @property
    def active(self) -> KnowledgeVersion:
        return self._versions[self._active]

    def pin(self, version: int | None = None) -> KnowledgeVersion:
        return self._versions[self._active if version is None else version]

    def stage(self, atom: KnowledgeAtom) -> None:
        if atom.kind not in {"P", "M", "R", "F", "S"}:
            raise ValueError(f"unknown P/M/R/F/S kind: {atom.kind}")
        self._staged.append(atom)

    def commit(self, evidence_digest: str, metadata: dict[str, Any] | None = None) -> KnowledgeVersion:
        if not self._staged:
            raise RuntimeError("no validated knowledge delta is staged")
        parent = self.active
        by_id = {atom.atom_id: atom for atom in parent.atoms}
        by_id.update({atom.atom_id: atom for atom in self._staged})
        version = parent.version + 1
        snapshot = KnowledgeVersion(version, parent.version, tuple(by_id.values()),
                                    datetime.now(timezone.utc).isoformat(), evidence_digest,
                                    metadata or {})
        self._versions[version] = snapshot
        self._active = version
        self._staged.clear()
        return snapshot

    def rollback_stage(self) -> None:
        self._staged.clear()

    def receipt(self) -> dict[str, Any]:
        return {"active": self._active, "staged": [asdict(x) for x in self._staged],
                "versions": {str(k): asdict(v) for k, v in self._versions.items()}}

    def save_active(self, path: Path) -> None:
        payload = asdict(self.active)
        atomic_json(path, payload)
        if path.name == "current.json":
            atomic_json(path.with_name(f"v{self.active.version}.json"), payload)


_TOKEN = re.compile(r"[a-z0-9_$]+", re.I)
_STOP = {"the", "and", "for", "with", "from", "into", "task", "test", "tests",
         "coding", "candidate", "repair", "verified", "failure"}


def knowledge_tags(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        token.lower() for token in _TOKEN.findall(text)
        if len(token) >= 3 and token.lower() not in _STOP
    ))


def retrieve_atoms(atoms: tuple[KnowledgeAtom, ...], task_text: str,
                   limit: int = 10) -> tuple[KnowledgeAtom, ...]:
    query = set(knowledge_tags(task_text))
    scored: list[tuple[float, float, KnowledgeAtom]] = []
    for atom in atoms:
        if atom.atom_id.startswith("k7."):
            continue
        overlap = query.intersection(atom.tags or knowledge_tags(atom.content))
        if not overlap:
            continue
        kind_weight = {"S": 1.15, "F": 1.2, "R": 1.15, "P": 1.0, "M": .7}[atom.kind]
        causal_bonus = 1.25 if any(tag in atom.tags for tag in (
            "patch_materialization_failure", "contract_boundary_gap",
            "preservation_regression", "contract_integration_failure",
        )) else 1.0
        family = atom.atom_id.split(".", 2)[1] if atom.atom_id.startswith("deepswe.") else ""
        family_tokens = set(knowledge_tags(family))
        family_coverage = (len(query.intersection(family_tokens)) / len(family_tokens)
                           if family_tokens else 0.0)
        identity_bonus = 5.0 if family_coverage >= .75 else 1.0
        scored.append((len(overlap) * atom.confidence * kind_weight * causal_bonus * identity_bonus,
                       family_coverage, atom))
    if any(coverage >= .75 for _, coverage, _ in scored):
        scored = [item for item in scored if item[1] >= .75]
    scored.sort(key=lambda item: (-item[0], item[2].atom_id))
    selected: list[KnowledgeAtom] = []
    per_kind: dict[str, int] = {}
    for _, _, atom in scored:
        if per_kind.get(atom.kind, 0) >= 3:
            continue
        selected.append(atom)
        per_kind[atom.kind] = per_kind.get(atom.kind, 0) + 1
        if len(selected) == limit:
            break
    return tuple(selected)


def render_atoms(version: int, atoms: tuple[KnowledgeAtom, ...]) -> str:
    if not atoms:
        return f"EVOLVED KNOWLEDGE v{version}: no verified task-specific atom matched."
    lines = [f"EVOLVED RCA KNOWLEDGE v{version} (current repository evidence overrides):"]
    labels = {"P": "OBSERVE", "R": "ROOT CAUSE", "F": "RECOVERY ACTION",
              "S": "TRANSITION GUARD", "M": "VERIFIED MEMORY"}
    for kind in ("P", "R", "F", "S", "M"):
        for atom in atoms:
            if atom.kind == kind:
                lines.append(
                    f"- {labels[kind]}: {atom.content} "
                    f"[id={atom.atom_id}; confidence={atom.confidence:.2f}]"
                )
    return "\n".join(lines)


def k7_seed_atoms() -> tuple[KnowledgeAtom, ...]:
    return (
        KnowledgeAtom("k7.perception.contract-tests-impact", "P",
                      "Observe the public task contract, failing tests, and changed-file impact before repair.",
                      "K7 frozen handoff"),
        KnowledgeAtom("k7.memory.best-candidate", "M",
                      "Persist the best focused-green candidate and its evidence digest before late repair.",
                      "K7 frozen handoff"),
        KnowledgeAtom("k7.reasoning.rca-first", "R",
                      "Use the five-layer RCA construction walk before choosing regression scope.",
                      "maoyixie/rca-repair"),
        KnowledgeAtom("k7.fix.failure-routed", "F",
                      "Activate only task-relevant repair procedures; do not apply global contract completion.",
                      "K7 frozen handoff"),
        KnowledgeAtom("k7.safety.deterministic-promotion", "S",
                      "Promotion requires independent RCA-selected gates; rollback on degradation; no MH acceptance.",
                      "K7 frozen handoff"),
    )
