from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any

from .knowledge.k7 import RoutedSkill
from .knowledge.store import KnowledgeVersion


@dataclass(frozen=True)
class PerceptionProbe:
    probe_id: str
    question: str
    supporting_atom_ids: tuple[str, ...]


@dataclass(frozen=True)
class PerceptionPolicy:
    policy_id: str
    knowledge_version: int
    probes: tuple[PerceptionProbe, ...]


@dataclass
class WorkingMemory:
    wm_id: str
    task_id: str
    knowledge_version: int
    facts: list[str] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    plans: list[str] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)

    def observe(self, kind: str, content: str, evidence_id: str) -> None:
        self.observations.append({"kind": kind, "content": content, "evidence_id": evidence_id})

    def receipt(self) -> dict[str, Any]:
        return asdict(self)


class PerceptionCompiler:
    def compile(self, task_id: str, knowledge: KnowledgeVersion,
                skills: tuple[RoutedSkill, ...]) -> PerceptionPolicy:
        atom_ids = tuple(atom.atom_id for atom in knowledge.atoms)
        probes = (
            PerceptionProbe("task-contract", "What public behavior and boundary cases are required?", atom_ids),
            PerceptionProbe("failing-tests", "Which focused assertion first demonstrates the gap?", atom_ids),
            PerceptionProbe("changed-impact", "Which old behavior can this source delta invalidate?", atom_ids),
            PerceptionProbe("skill-routing", "Which K7 procedures match the public task text?",
                            tuple(skill.skill_id for skill in skills)),
        )
        digest = hashlib.sha256(f"{task_id}|{knowledge.version}|{probes}".encode()).hexdigest()[:16]
        return PerceptionPolicy(f"policy-{digest}", knowledge.version, probes)
