from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..io import atomic_json
from .store import KnowledgeAtom, knowledge_tags


CAPABILITY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("parsing-schema", ("parse", "parser", "config", "html", "json", "schema", "format")),
    ("cache-state", ("cache", "memo", "provider", "result")),
    ("async-lifecycle", ("async", "task", "snapshot", "cancel", "lifecycle")),
    ("binding-type", ("alias", "mapping", "variable", "binding", "typed", "argument")),
    ("transformation-roundtrip", ("css", "shorthand", "expand", "compress", "roundtrip")),
    ("dependency-policy", ("action", "pin", "dependency", "workflow")),
    ("security-analysis", ("nosec", "taint", "security", "bandit", "directive")),
)


def repository_family(task: str) -> str:
    return re.split(r"[-_/]", task.strip().lower(), maxsplit=1)[0] or "unknown"


def capability_family(text: str) -> str:
    tokens = set(knowledge_tags(text))
    best = (0, "repository-contract")
    for name, cues in CAPABILITY_PATTERNS:
        score = len(tokens.intersection(cues))
        if score > best[0]:
            best = (score, name)
    return best[1]


def task_matches(left: str, right: str) -> bool:
    """Harbor truncates trial identifiers, so exact equality is too strict."""
    a = re.sub(r"[^a-z0-9]+", "-", left.lower()).strip("-")
    b = re.sub(r"[^a-z0-9]+", "-", right.lower()).strip("-")
    return bool(a and b and (a == b or a.startswith(b[:24]) or b.startswith(a[:24])))


@dataclass(frozen=True)
class RetrievalDecision:
    atom: KnowledgeAtom
    level: str
    status: str
    score: float
    grounding: tuple[str, ...]


class GroundedKnowledgeGraph:
    """Sidecar graph for scope, abstraction, grounding, and promotion evidence.

    The P/M/R/F/S atoms remain unchanged.  This graph controls where an atom may
    activate and records why an abstraction is (or is not) transferable.
    """

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload or {"schema_version": 1, "nodes": {}, "edges": []}
        self.payload.setdefault("schema_version", 1)
        self.payload.setdefault("nodes", {})
        self.payload.setdefault("edges", [])

    @classmethod
    def from_file(cls, path: Path | None) -> "GroundedKnowledgeGraph":
        if path is None or not path.exists():
            return cls()
        try:
            return cls(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            return cls()

    def save(self, path: Path) -> None:
        atomic_json(path, self.payload)

    @property
    def nodes(self) -> dict[str, dict[str, Any]]:
        return self.payload["nodes"]

    def metadata(self, atom: KnowledgeAtom) -> dict[str, Any]:
        explicit = self.nodes.get(atom.atom_id)
        if explicit is not None:
            return explicit
        if atom.atom_id.startswith("k7."):
            return {"level": "general", "status": "frozen", "scope": {}}
        if atom.atom_id.startswith("deepswe."):
            parts = atom.atom_id.split(".", 2)
            task = parts[1] if len(parts) > 1 else ""
            return {
                "level": "task",
                "status": "legacy-local",
                "scope": {"task": task, "repository": repository_family(task)},
            }
        return {"level": "general", "status": "legacy", "scope": {}}

    def register_task_atom(
        self,
        atom: KnowledgeAtom,
        *,
        task: str,
        capability: str,
        root_kind: str,
        positive: bool,
        source: str,
    ) -> None:
        node = self.nodes.setdefault(atom.atom_id, {})
        node.update({
            "level": "task",
            "status": "verified-local" if positive else "provisional-local",
            "scope": {
                "task": task,
                "repository": repository_family(task),
                "capability": capability,
                "failure_mode": root_kind,
            },
            "preconditions": [
                f"repository family is {repository_family(task)}",
                f"current evidence matches {root_kind}",
            ],
            "positive_groundings": [task] if positive else [],
            "negative_groundings": [] if positive else [task],
            "source": source,
        })

    def register_abstraction(
        self,
        atom: KnowledgeAtom,
        *,
        capability: str,
        root_kind: str,
        positive_tasks: Iterable[str],
        negative_tasks: Iterable[str],
        child_ids: Iterable[str],
    ) -> str:
        positives = sorted(set(positive_tasks))
        negatives = sorted(set(negative_tasks))
        if len(positives) >= 3 and not negatives:
            status = "transferable"
        elif len(positives) >= 2:
            status = "validated-abstraction"
        else:
            status = "provisional-abstraction"
        self.nodes[atom.atom_id] = {
            "level": "capability",
            "status": status,
            "scope": {"capability": capability, "failure_mode": root_kind},
            "preconditions": [
                f"task and repository evidence match capability family {capability}",
                f"a concrete failure or code path supports failure mode {root_kind}",
                "reproduce one focused counterexample before editing",
            ],
            "positive_groundings": positives,
            "negative_groundings": negatives,
        }
        known_edges = {
            (item.get("source"), item.get("target"), item.get("relation"))
            for item in self.payload["edges"]
        }
        for child in child_ids:
            edge = (atom.atom_id, child, "ABSTRACTS_FROM")
            if edge not in known_edges:
                self.payload["edges"].append({
                    "source": atom.atom_id,
                    "target": child,
                    "relation": "ABSTRACTS_FROM",
                })
        return status


def abstraction_atom(capability: str, root_kind: str) -> KnowledgeAtom:
    signature = f"{capability}|{root_kind}"
    digest = hashlib.sha256(signature.encode()).hexdigest()[:12]
    content = (
        f"Grounded {capability} hypothesis for {root_kind}: preserve already passing behavior; "
        "reproduce a named counterexample, locate the nearest shared causal decision point, "
        "and apply one minimal delta only after repository evidence confirms this pattern."
    )
    return KnowledgeAtom(
        f"abstract.{capability}.{root_kind}.{digest}",
        "R",
        content,
        "MetaCoRSI abstraction/grounding graph",
        0.72,
        tuple(knowledge_tags(f"{capability} {root_kind} counterexample causal boundary")),
    )


def retrieve_grounded_atoms(
    atoms: tuple[KnowledgeAtom, ...],
    graph: GroundedKnowledgeGraph,
    *,
    task_id: str,
    task_text: str,
    repository_evidence: str = "",
    limit: int = 10,
) -> tuple[RetrievalDecision, ...]:
    query_text = f"{task_id} {task_text} {repository_evidence}"
    query = set(knowledge_tags(query_text))
    current_capability = capability_family(query_text)
    current_repo = repository_family(task_id)
    buckets: dict[str, list[RetrievalDecision]] = {
        "task": [], "repository": [], "capability": [], "general": [], "explore": []
    }
    for atom in atoms:
        if atom.atom_id.startswith("k7."):
            continue
        meta = graph.metadata(atom)
        scope = meta.get("scope") or {}
        level = str(meta.get("level") or "general")
        status = str(meta.get("status") or "unknown")
        atom_tokens = set(atom.tags or knowledge_tags(atom.content))
        overlap = len(query.intersection(atom_tokens))
        grounding: list[str] = []
        bucket = "general"

        scoped_task = str(scope.get("task") or "")
        if level == "task":
            if not task_matches(scoped_task, task_id):
                continue
            bucket = "task"
            grounding.append("exact task branch")
        elif level == "repository":
            if str(scope.get("repository") or "") != current_repo:
                continue
            bucket = "repository"
            grounding.append("repository family matched")
        elif level == "capability":
            if str(scope.get("capability") or "") != current_capability:
                continue
            if status in {"validated-abstraction", "transferable"}:
                bucket = "capability"
                grounding.append("validated capability abstraction matched")
            else:
                bucket = "explore"
                grounding.append("provisional abstraction; verify before use")
        else:
            grounding.append("global policy")

        if overlap == 0 and bucket not in {"task", "general"}:
            continue
        kind_weight = {"S": 1.15, "F": 1.2, "R": 1.15, "P": 1.0, "M": .7}[atom.kind]
        scope_weight = {"task": 8.0, "repository": 4.0, "capability": 3.0,
                        "general": 1.0, "explore": .45}[bucket]
        score = (1 + overlap) * atom.confidence * kind_weight * scope_weight
        buckets[bucket].append(RetrievalDecision(atom, level, status, score, tuple(grounding)))

    for values in buckets.values():
        values.sort(key=lambda item: (-item.score, item.atom.atom_id))

    quotas = (("task", 5), ("repository", 2), ("capability", 2),
              ("general", 1), ("explore", 1))
    selected: list[RetrievalDecision] = []
    for name, quota in quotas:
        selected.extend(buckets[name][:quota])
    selected.sort(key=lambda item: (-item.score, item.atom.atom_id))
    return tuple(selected[:limit])


def render_grounded_atoms(version: int, decisions: tuple[RetrievalDecision, ...]) -> str:
    if not decisions:
        return (
            f"GROUNDED EVOLVED KNOWLEDGE v{version}: no scoped atom matched. "
            "Use repository evidence and the frozen K7 control policy."
        )
    lines = [
        f"GROUNDED EVOLVED KNOWLEDGE v{version}:",
        "Treat capability abstractions as hypotheses, never as answers. Current repository evidence overrides memory.",
    ]
    labels = {"P": "OBSERVE", "R": "ROOT CAUSE", "F": "RECOVERY ACTION",
              "S": "TRANSITION GUARD", "M": "VERIFIED MEMORY"}
    for item in decisions:
        lines.append(
            f"- {labels[item.atom.kind]} [{item.level}/{item.status}; "
            f"grounding={'; '.join(item.grounding)}]: {item.atom.content} "
            f"[id={item.atom.atom_id}; confidence={item.atom.confidence:.2f}]"
        )
    return "\n".join(lines)
