from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ..io import atomic_json
from .grounding import (CAPABILITY_PATTERNS, GroundedKnowledgeGraph, capability_family,
                        repository_family, task_matches)
from .store import KnowledgeAtom, knowledge_tags


_TASK_ATOM = re.compile(r"^deepswe\.(.+?)\.(pass|fail)\.[pmrfs]\.", re.I)
_CAUSAL_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("normalization", ("normalize", "normalization", "tokenize", "canonical")),
    ("dispatch-registration", ("dispatch", "register", "registration", "export", "public")),
    ("state-propagation", ("state", "context", "propagate", "recursive", "scope")),
    ("boundary-decision", ("boundary", "edge", "implicit", "precedence")),
    ("validation-preservation", ("validate", "validation", "regression", "preserve", "compatibility")),
    ("materialization", ("patch", "materialize", "checkpoint", "implementation")),
)
_FAILURE_MODES = (
    "patch_materialization_failure",
    "execution_protocol_failure",
    "preservation_regression",
    "contract_boundary_gap",
    "mixed_contract_and_preservation_failure",
    "contract_integration_failure",
    "verified_success",
)

# These tokens describe the repair-writing template rather than a reusable
# causal situation.  Letting them become relation triggers makes unrelated
# tasks look similar merely because both episodes mention commits or tests.
_RELATION_STOP_TAGS = {
    "added", "agent", "branch", "build", "change", "changed", "clean",
    "commit", "committed", "complete", "feature", "files", "focused",
    "full", "green", "implementation", "implemented", "main", "pass",
    "passed", "patch", "probes", "run", "sites", "suite", "test",
    "tests", "tree", "verified", "verify",
}

_MEMORY_STRATEGIES = {
    "v23_modular",
    "anthropic_search_notes",
    "anthropic_dreaming",
    "memtree_collapsed",
    "memtree_traversal",
    "cls_dual_store",
    "schema_assimilation",
    "art_resonance",
    "hippocampal_index",
    "event_segmentation",
    "split_attractor",
}

_V24_POLICIES = {"split_attractor_v24", "naive_latch_v24"}


def _extract_live_symbols(evidence: str, limit: int = 6) -> tuple[str, ...]:
    """Extract concrete repository files or code symbols observed in live evidence."""
    if not evidence.strip():
        return ()
    found: list[str] = []
    file_matches = re.findall(
        r"\b[\w@.-]+(?:/[\w@.{}-]+)+\.(?:py|go|rs|ts|tsx|js|jsx|java|c|cc|cpp|h|hpp)\b",
        evidence,
    )
    for item in file_matches:
        if item not in found:
            found.append(item)
        if len(found) >= limit:
            return tuple(found)
    symbol_matches = re.findall(
        r"\b(?:def|class|func|function|interface|type)\s+([A-Za-z_][\w$]{2,40})\b",
        evidence,
    )
    for sym in symbol_matches:
        if sym not in found:
            found.append(sym)
        if len(found) >= limit:
            break
    return tuple(found)


def _memory_strategy() -> str:
    """Select one auditable organization rule without changing the brain structure."""
    value = os.environ.get("METACORSI_MEMORY_STRATEGY", "v23_modular").strip().lower()
    return value if value in _MEMORY_STRATEGIES else "v23_modular"


def _strategy_score(strategy: str, *, capability: float, causal: float,
                    trigger: float, source: float, feedback: float,
                    episodic: float, specificity: float, status_ok: bool) -> float:
    """Comparable retrieval models over the same Situation Graph.

    These are organization/retrieval interventions, not answer-bearing recipes.
    Every arm still reverse-grounds at most one target-disjoint episode capsule.
    """
    if strategy == "anthropic_search_notes":
        score = .15 * capability + .10 * causal + .40 * trigger + .15 * source + .15 * feedback + .05 * episodic
    elif strategy == "anthropic_dreaming":
        score = .20 * capability + .20 * causal + .15 * trigger + .20 * source + .20 * feedback + .05 * episodic
    elif strategy == "memtree_collapsed":
        # Search every level and let the strongest node-level match win.
        score = .30 * max(capability, causal, trigger) + .18 * capability + .18 * causal + .18 * trigger + .10 * source + .06 * feedback
    elif strategy == "memtree_traversal":
        # Root(capability) -> relation(causal locus) -> leaf(trigger/episode).
        score = .38 * capability + .27 * causal + .18 * trigger + .10 * source + .07 * feedback
    elif strategy == "cls_dual_store":
        score = .18 * capability + .20 * causal + .12 * trigger + .22 * source + .13 * feedback + .15 * max(0.0, episodic)
    elif strategy == "schema_assimilation":
        congruence = capability * (.35 + .65 * causal) * (.35 + .65 * trigger)
        score = .55 * congruence + .15 * source + .15 * feedback + .10 * specificity + .05 * max(0.0, episodic)
    elif strategy == "art_resonance":
        resonance = .45 * capability + .30 * causal + .25 * trigger
        # Vigilance is continuous here: mismatches lose rank rather than being
        # discarded, allowing a novel category to remain representable.
        score = .62 * resonance + .13 * source + .13 * feedback + .12 * specificity
    elif strategy == "hippocampal_index":
        # Partial cues complete one sparse episode; common schema cues are weak.
        score = .24 * capability + .18 * causal + .32 * trigger + .10 * source + .08 * feedback + .08 * specificity
    elif strategy == "event_segmentation":
        # Prediction-error boundaries (causal locus) dominate whole-episode similarity.
        score = .20 * capability + .38 * causal + .22 * trigger + .08 * source + .07 * feedback + .05 * specificity
    elif strategy == "split_attractor":
        # Balance live causal/trigger resonance with slow verification reliability.
        score = .26 * capability + .30 * causal + .22 * trigger + .08 * source + .09 * feedback + .05 * specificity
    else:  # v23_modular
        score = .28 * capability + .27 * causal + .20 * trigger + .10 * source + .08 * feedback + .05 * episodic + .10 * specificity
    if not status_ok:
        score -= .04
    return max(.01, min(1.0, score))


def _atom_task(atom: KnowledgeAtom) -> tuple[str, str]:
    match = _TASK_ATOM.match(atom.atom_id)
    if not match:
        return "", "unknown"
    return match.group(1), match.group(2).lower()


def _causal_loci(text: str) -> tuple[str, ...]:
    tokens = set(knowledge_tags(text))
    found = [name for name, cues in _CAUSAL_CUES if tokens.intersection(cues)]
    return tuple(found or ("unspecified",))


def _capability_affinities(text: str) -> dict[str, float]:
    """Return a multi-label Situation-Graph profile instead of one hard class.

    Tasks commonly span parsing, lifecycle, and state propagation at once.  V22
    keeps every observed capability role and normalizes cue counts against the
    strongest role, so related abstractions can compete continuously.
    """
    tokens = set(knowledge_tags(text))
    raw = {
        name: sum(
            1 for cue in cues
            if any(
                token == cue
                or (len(cue) >= 4 and token.startswith(cue))
                or (len(token) >= 4 and cue.startswith(token))
                for token in tokens
            )
        )
        for name, cues in CAPABILITY_PATTERNS
    }
    strongest = max(raw.values(), default=0)
    if strongest == 0:
        return {name: 0.0 for name, _ in CAPABILITY_PATTERNS}
    return {name: count / strongest for name, count in raw.items()}


def _situation_capability_affinities(task_id: str, task_text: str) -> dict[str, float]:
    """Blend stable task identity with issue prose without collapsing to one label."""
    identity = _capability_affinities(task_id)
    description = _capability_affinities(task_text)
    if max(identity.values(), default=0.0) == 0.0:
        return description
    return {
        name: min(1.0, 0.70 * identity.get(name, 0.0) + 0.30 * description.get(name, 0.0))
        for name, _ in CAPABILITY_PATTERNS
    }


def _semantic_episode_projection(content: str, limit: int = 360) -> str:
    """Project a concrete episode atom into a source-agnostic role trace."""
    projected = content
    for marker in ("Semantic execution trace:", "Verified trajectory:"):
        if marker in projected:
            projected = projected.split(marker, 1)[1]
            break
    projected = re.sub(
        r"\b[\w@.-]+(?:/[\w@.{}-]+)+\.(?:py|go|rs|ts|tsx|js|jsx|java|c|cc|cpp|h|hpp)\b",
        "<source-file>", projected,
    )
    projected = re.sub(r"\s*(?:->|→)\s*", "; then ", projected)
    projected = re.sub(r"\s+", " ", projected).strip()
    return projected[:limit]


def _failure_mode(atom: KnowledgeAtom, graph_meta: dict[str, Any]) -> str:
    scoped = str((graph_meta.get("scope") or {}).get("failure_mode") or "")
    if scoped:
        return scoped
    joined = " ".join((atom.content, " ".join(atom.tags)))
    return next((mode for mode in _FAILURE_MODES if mode in joined), "unspecified")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    material = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    temp.write_text(material, encoding="utf-8")
    os.replace(temp, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class AtomRecord:
    atom: KnowledgeAtom
    task: str
    outcome: str
    repository: str
    capability: str
    failure_mode: str
    causal_loci: tuple[str, ...]
    status: str

    def json(self) -> dict[str, Any]:
        return {
            "atom": asdict(self.atom),
            "coordinates": {
                "task": self.task,
                "outcome": self.outcome,
                "repository": self.repository,
                "capability": self.capability,
                "failure_mode": self.failure_mode,
                "causal_loci": list(self.causal_loci),
                "status": self.status,
            },
        }


@dataclass(frozen=True)
class RetrievalBundle:
    version: int
    atoms: tuple[KnowledgeAtom, ...]
    nodes: tuple[dict[str, Any], ...]
    audit: dict[str, Any]


@dataclass(frozen=True)
class AbstractKnowledge:
    abstract_id: str
    capability: str
    failure_mode: str
    causal_locus: str
    trigger_tags: tuple[str, ...]
    invariant: str
    action_schema: tuple[str, ...]
    safety_boundary: tuple[str, ...]
    supporting_atom_ids: tuple[str, ...]
    positive_tasks: tuple[str, ...]
    negative_tasks: tuple[str, ...]
    status: str

    def json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GroundingRecord:
    grounding_id: str
    abstract_id: str
    task: str
    repository: str
    capability: str
    context_digest: str
    activation_evidence: tuple[str, ...]
    bindings: dict[str, str]
    instantiated_action: tuple[str, ...]
    status: str = "pending"
    reward: float | None = None
    source: str = "runtime"
    source_atom_ids: tuple[str, ...] = ()
    scope_guard: tuple[str, ...] = ()
    verification_plan: tuple[str, ...] = ()
    materialization_protocol: tuple[str, ...] = ()
    outcome_evidence: tuple[str, ...] = ()

    def json(self) -> dict[str, Any]:
        return asdict(self)


class IndexedLTM:
    """Versioned atom vault plus stable multi-axis indexes.

    Indexes reference immutable atom ids. Adding an unrelated branch cannot
    change a learned task's retention anchors or delete its retrieval path.
    """

    def __init__(self, *, version: int, records: dict[str, AtomRecord],
                 indexes: dict[str, dict[str, tuple[str, ...]]],
                 nodes: dict[str, dict[str, Any]], edges: tuple[dict[str, Any], ...],
                 anchors: dict[str, tuple[str, ...]],
                 abstractions: dict[str, AbstractKnowledge] | None = None,
                 groundings: tuple[GroundingRecord, ...] = (),
                 manifest: dict[str, Any] | None = None) -> None:
        self.version = version
        self.records = records
        self.indexes = indexes
        self.nodes = nodes
        self.edges = edges
        self.anchors = anchors
        self.abstractions = abstractions or {}
        self.groundings = groundings
        self.manifest = manifest or {}

    @classmethod
    def build(cls, *, atoms: tuple[KnowledgeAtom, ...], version: int,
              parent: int | None, graph: GroundedKnowledgeGraph,
              groundings: tuple[GroundingRecord, ...] = ()) -> "IndexedLTM":
        records: dict[str, AtomRecord] = {}
        mutable_indexes: dict[str, dict[str, set[str]]] = {
            name: {} for name in ("task", "repository", "capability", "failure_mode", "causal_locus")
        }
        for atom in atoms:
            task, outcome = _atom_task(atom)
            meta = graph.metadata(atom)
            scope = meta.get("scope") or {}
            repository = str(scope.get("repository") or (repository_family(task) if task else "general"))
            capability = str(scope.get("capability") or capability_family(
                " ".join((task, atom.content, " ".join(atom.tags)))
            ))
            failure = _failure_mode(atom, meta)
            loci = _causal_loci(" ".join((atom.content, " ".join(atom.tags))))
            record = AtomRecord(atom, task, outcome, repository, capability, failure, loci,
                                str(meta.get("status") or "legacy"))
            records[atom.atom_id] = record
            coordinates = {
                "task": (task,) if task else (),
                "repository": (repository,),
                "capability": (capability,),
                "failure_mode": (failure,),
                "causal_locus": loci,
            }
            for axis, values in coordinates.items():
                for value in values:
                    mutable_indexes[axis].setdefault(value, set()).add(atom.atom_id)

        indexes = {
            axis: {key: tuple(sorted(ids)) for key, ids in sorted(values.items())}
            for axis, values in mutable_indexes.items()
        }
        abstractions = cls._build_abstractions(records)
        abstractions = cls._consolidate_grounding_evidence(abstractions, groundings)
        nodes, edges = cls._build_nodes(records, indexes, abstractions)
        anchors = cls._build_anchors(records)
        manifest = {
            "schema_version": 1,
            "version": version,
            "parent": parent,
            "profile": "indexed-ltm-abstraction-grounding-v1",
            "built_at": datetime.now(timezone.utc).isoformat(),
        }
        return cls(version=version, records=records, indexes=indexes, nodes=nodes,
                   edges=edges, anchors=anchors, abstractions=abstractions,
                   groundings=groundings, manifest=manifest)

    @staticmethod
    def _consolidate_grounding_evidence(
        abstractions: dict[str, AbstractKnowledge],
        groundings: tuple[GroundingRecord, ...],
    ) -> dict[str, AbstractKnowledge]:
        consolidated = dict(abstractions)
        grouped: dict[str, list[GroundingRecord]] = {}
        for record in groundings:
            grouped.setdefault(record.abstract_id, []).append(record)
        for abstract_id, evidence in grouped.items():
            current = consolidated.get(abstract_id)
            if current is None:
                continue
            positive = set(current.positive_tasks)
            negative = set(current.negative_tasks)
            for record in evidence:
                if record.status == "verified":
                    positive.add(record.task)
                    negative.discard(record.task)
                elif record.status == "rejected" and record.task not in positive:
                    negative.add(record.task)
            if len(positive) >= 3 and not negative:
                status = "transferable"
            elif len(positive) >= 2:
                status = "validated"
            elif len(positive) == 1:
                status = "provisional"
            else:
                status = "unverified"
            consolidated[abstract_id] = AbstractKnowledge(
                abstract_id=current.abstract_id, capability=current.capability,
                failure_mode=current.failure_mode, causal_locus=current.causal_locus,
                trigger_tags=current.trigger_tags, invariant=current.invariant,
                action_schema=current.action_schema, safety_boundary=current.safety_boundary,
                supporting_atom_ids=current.supporting_atom_ids,
                positive_tasks=tuple(sorted(positive)), negative_tasks=tuple(sorted(negative)),
                status=status,
            )
        return consolidated

    @staticmethod
    def _build_anchors(records: dict[str, AtomRecord]) -> dict[str, tuple[str, ...]]:
        by_task: dict[str, list[AtomRecord]] = {}
        for record in records.values():
            if record.task and record.outcome == "pass":
                by_task.setdefault(record.task, []).append(record)
        anchors: dict[str, tuple[str, ...]] = {}
        for task, values in by_task.items():
            selected: list[str] = []
            for kind in ("P", "R", "F", "S", "M"):
                matches = sorted(
                    (item for item in values if item.atom.kind == kind),
                    key=lambda item: (-item.atom.confidence, item.atom.atom_id),
                )
                if matches:
                    selected.append(matches[0].atom.atom_id)
            anchors[task] = tuple(selected)
        return anchors

    @staticmethod
    def _build_abstractions(records: dict[str, AtomRecord]) -> dict[str, AbstractKnowledge]:
        groups: dict[tuple[str, str, str], list[AtomRecord]] = {}
        for record in records.values():
            if not record.task:
                continue
            locus = record.causal_loci[0] if record.causal_loci else "unspecified"
            groups.setdefault((record.capability, record.failure_mode, locus), []).append(record)
        abstractions: dict[str, AbstractKnowledge] = {}
        for (capability, failure, locus), values in sorted(groups.items()):
            positive = sorted({item.task for item in values if item.outcome == "pass"})
            negative = sorted({item.task for item in values if item.outcome == "fail"} - set(positive))
            if len(positive) >= 3 and not negative:
                status = "transferable"
            elif len(positive) >= 2:
                status = "validated"
            elif len(positive) == 1:
                status = "provisional"
            else:
                status = "unverified"
            task_tokens = {token for item in values for token in knowledge_tags(item.task)}
            tag_counts: dict[str, int] = {}
            tag_tasks: dict[str, set[str]] = {}
            for item in values:
                tags = (
                    set(item.atom.tags or knowledge_tags(item.atom.content))
                    - task_tokens
                    - _RELATION_STOP_TAGS
                )
                for tag in tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
                    tag_tasks.setdefault(tag, set()).add(item.task)
            trigger_tags = tuple(
                tag for tag, _ in sorted(
                    (
                        (tag, count) for tag, count in tag_counts.items()
                        if len(tag_tasks.get(tag, ())) >= 2
                    ),
                    key=lambda pair: (-len(tag_tasks[pair[0]]), -pair[1], pair[0]),
                )[:10]
            )
            ranked = lambda kind: sorted(
                (item for item in values if item.atom.kind == kind),
                key=lambda item: (-item.atom.confidence, item.atom.atom_id),
            )
            # Consolidation organizes evidence into a reusable schema. Concrete
            # files and symbols are bound later by ground(), never copied from a
            # source-task repair narrative and mislabeled as an abstraction.
            if capability == "cache-state" and locus == "state-propagation":
                invariant = (
                    "State that changes execution semantics must be propagated through every "
                    "nested, queued, or cached boundary; identity and lifecycle transitions "
                    "must not silently discard it."
                )
                actions = (
                    "Locate where the semantic state is created and bind it to the current {task} observation.",
                    "Trace every nested, queued, cached, or resumed path crossing {causal_locus} in {repository}.",
                    "Propagate one minimal state token or handle across the first missing boundary, then verify focused and retained behavior.",
                )
            elif capability == "parsing-schema" and locus == "validation-preservation":
                invariant = (
                    "A parser or schema extension is complete only when representation, "
                    "dispatch, validation, and preserved legacy behavior agree on the same contract."
                )
                actions = (
                    "Bind one failing syntax or schema example to its representation and parser/dispatcher path in {repository}.",
                    "Change the nearest shared validation decision for {task}, preserving the already-green grammar or schema cases.",
                    "Run the focused boundary example and retained parser/schema tests before broadening the implementation.",
                )
            elif failure == "patch_materialization_failure":
                invariant = (
                    "A repair is not an established capability until a minimal, "
                    "compilable vertical slice is materialized and exercised."
                )
                actions = (
                    "Bind one failing counterexample to the nearest {causal_locus} decision in {repository}.",
                    "Materialize the smallest compilable delta for {task}; run build or typecheck immediately.",
                    "Run the focused counterexample, then retained tests; expand only after both execute.",
                )
            elif failure == "contract_boundary_gap":
                invariant = (
                    "When a partial repair preserves old behavior, freeze its passing core and "
                    "change only the nearest decision shared by the remaining counterexamples."
                )
                actions = (
                    "Bind one remaining counterexample to the nearest {causal_locus} decision in {repository}.",
                    "Make one minimal boundary delta for {task} without rewriting the passing core.",
                    "Run build or typecheck, the focused counterexample, then retained tests in that order.",
                )
            else:
                invariant = f"Preserve the verified {capability} contract at {locus}."
                actions = (
                    "Reproduce one focused counterexample for {task}.",
                    "Bind its observed state to the nearest {causal_locus} decision and make one minimal delta.",
                    "Run build or typecheck, focused verification, and retained tests before expanding.",
                )
            boundaries = (
                "Do not copy source-task files or patches; bind only the abstract invariant to current evidence.",
                "Do not broaden implementation while compilation or the focused counterexample is red.",
                "Independent F2P and P2P results are the promotion boundary.",
            )
            signature = f"{capability}|{failure}|{locus}"
            abstract_id = "abstract." + hashlib.sha256(signature.encode()).hexdigest()[:16]
            abstractions[abstract_id] = AbstractKnowledge(
                abstract_id=abstract_id,
                capability=capability,
                failure_mode=failure,
                causal_locus=locus,
                trigger_tags=trigger_tags,
                invariant=invariant,
                action_schema=actions,
                safety_boundary=boundaries,
                supporting_atom_ids=tuple(sorted(item.atom.atom_id for item in values)),
                positive_tasks=tuple(positive),
                negative_tasks=tuple(negative),
                status=status,
            )
        return abstractions

    @staticmethod
    def _build_nodes(records: dict[str, AtomRecord],
                     indexes: dict[str, dict[str, tuple[str, ...]]],
                     abstractions: dict[str, AbstractKnowledge],
                     ) -> tuple[dict[str, dict[str, Any]], tuple[dict[str, Any], ...]]:
        nodes: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, Any]] = []
        for axis, values in indexes.items():
            for key, atom_ids in values.items():
                node_id = f"{axis}:{key}"
                task_outcomes: dict[str, set[str]] = {}
                for atom_id in atom_ids:
                    record = records[atom_id]
                    if record.task:
                        task_outcomes.setdefault(record.task, set()).add(record.outcome)
                    edges.append({"source": atom_id, "target": node_id, "relation": "INDEXED_BY"})
                positives = sorted(task for task, outcomes in task_outcomes.items() if "pass" in outcomes)
                negatives = sorted(task for task, outcomes in task_outcomes.items()
                                   if "fail" in outcomes and "pass" not in outcomes)
                if axis == "capability":
                    if len(positives) >= 3 and not negatives:
                        status = "transferable"
                    elif len(positives) >= 2:
                        status = "validated"
                    elif len(positives) == 1:
                        status = "provisional"
                    else:
                        status = "unverified"
                else:
                    status = "indexed"
                nodes[node_id] = {
                    "node_id": node_id,
                    "node_type": axis,
                    "key": key,
                    "status": status,
                    "atom_ids": list(atom_ids),
                    "positive_tasks": positives,
                    "negative_tasks": negatives,
                    "summary": (
                        f"{axis} knowledge branch '{key}' with {len(atom_ids)} atoms, "
                        f"{len(positives)} positive and {len(negatives)} unresolved task groundings."
                    ),
                }
        for task in indexes["task"]:
            repo = repository_family(task)
            if f"repository:{repo}" in nodes:
                edges.append({"source": f"task:{task}", "target": f"repository:{repo}",
                              "relation": "BELONGS_TO"})
        for capability, atom_ids in indexes["capability"].items():
            tasks = sorted({records[atom_id].task for atom_id in atom_ids if records[atom_id].task})
            for task in tasks:
                edges.append({"source": f"capability:{capability}", "target": f"task:{task}",
                              "relation": "ABSTRACTS_FROM"})
        for abstract in abstractions.values():
            nodes[abstract.abstract_id] = {
                "node_id": abstract.abstract_id,
                "node_type": "abstraction",
                "key": f"{abstract.capability}:{abstract.failure_mode}:{abstract.causal_locus}",
                "status": abstract.status,
                "atom_ids": list(abstract.supporting_atom_ids),
                "positive_tasks": list(abstract.positive_tasks),
                "negative_tasks": list(abstract.negative_tasks),
                "summary": abstract.invariant,
            }
            edges.append({"source": abstract.abstract_id,
                          "target": f"capability:{abstract.capability}",
                          "relation": "SPECIALIZES"})
            for atom_id in abstract.supporting_atom_ids:
                edges.append({"source": abstract.abstract_id, "target": atom_id,
                              "relation": "ABSTRACTS_FROM"})
        unique = {(edge["source"], edge["target"], edge["relation"]): edge for edge in edges}
        return nodes, tuple(unique[key] for key in sorted(unique))

    def commit(self, root: Path) -> Path:
        def activate(path: Path) -> None:
            atomic_json(root / "CURRENT.json", {
                "schema_version": 1,
                "version": self.version,
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "manifest_sha256": _sha256(path / "manifest.json"),
            })

        snapshot_material = json.dumps({
            "version": self.version,
            "atoms": sorted(self.records),
            "abstractions": {
                key: value.json() for key, value in sorted(self.abstractions.items())
            },
            "groundings": [item.json() for item in self.groundings],
            "anchors": self.anchors,
        }, ensure_ascii=False, sort_keys=True)
        snapshot_digest = hashlib.sha256(snapshot_material.encode()).hexdigest()
        version_dir = root / "versions" / f"v{self.version}"
        if version_dir.exists():
            existing = version_dir / "manifest.json"
            if existing.exists():
                existing_manifest = json.loads(existing.read_text(encoding="utf-8"))
                if existing_manifest.get("snapshot_digest") == snapshot_digest:
                    activate(version_dir)
                    return version_dir
                version_dir = root / "versions" / f"v{self.version}-g{snapshot_digest[:10]}"
                if version_dir.exists() and (version_dir / "manifest.json").exists():
                    activate(version_dir)
                    return version_dir
            else:
                raise RuntimeError(f"incomplete indexed LTM version exists: {version_dir}")
        version_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = version_dir.parent / f".{version_dir.name}.{os.getpid()}.staging"
        staging.mkdir()
        _write_jsonl(staging / "atoms.jsonl", (
            self.records[key].json() for key in sorted(self.records)
        ))
        atomic_json(staging / "nodes.json", self.nodes)
        _write_jsonl(staging / "edges.jsonl", self.edges)
        atomic_json(staging / "abstractions.json", {
            key: value.json() for key, value in sorted(self.abstractions.items())
        })
        _write_jsonl(staging / "groundings.jsonl", (item.json() for item in self.groundings))
        index_dir = staging / "indexes"
        for axis, values in self.indexes.items():
            atomic_json(index_dir / f"{axis}.json", values)
        atomic_json(staging / "retention_anchors.json", self.anchors)
        files = [staging / "atoms.jsonl", staging / "nodes.json",
                 staging / "edges.jsonl", staging / "abstractions.json",
                 staging / "groundings.jsonl", staging / "retention_anchors.json"]
        files.extend(index_dir / f"{axis}.json" for axis in self.indexes)
        manifest = dict(self.manifest)
        manifest.update({
            "snapshot_digest": snapshot_digest,
            "atom_count": len(self.records),
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "abstraction_count": len(self.abstractions),
            "grounding_count": len(self.groundings),
            "retention_task_count": len(self.anchors),
            "files": {str(path.relative_to(staging)).replace("\\", "/"): _sha256(path)
                      for path in files},
        })
        atomic_json(staging / "manifest.json", manifest)
        os.replace(staging, version_dir)
        activate(version_dir)
        self.manifest = manifest
        return version_dir

    @classmethod
    def load(cls, root: Path, expected_version: int | None = None) -> "IndexedLTM":
        pointer = json.loads((root / "CURRENT.json").read_text(encoding="utf-8"))
        version = int(pointer["version"])
        if expected_version is not None and version != expected_version:
            raise RuntimeError(f"indexed LTM version {version} != knowledge version {expected_version}")
        version_dir = root / pointer["path"]
        manifest_path = version_dir / "manifest.json"
        if _sha256(manifest_path) != pointer["manifest_sha256"]:
            raise RuntimeError("indexed LTM manifest digest mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for relative, digest in manifest["files"].items():
            if _sha256(version_dir / relative) != digest:
                raise RuntimeError(f"indexed LTM file digest mismatch: {relative}")
        records: dict[str, AtomRecord] = {}
        for row in _read_jsonl(version_dir / "atoms.jsonl"):
            raw = row["atom"]
            atom = KnowledgeAtom(raw["atom_id"], raw["kind"], raw["content"], raw["provenance"],
                                 float(raw.get("confidence", 1.0)), tuple(raw.get("tags") or ()))
            coord = row["coordinates"]
            records[atom.atom_id] = AtomRecord(
                atom, coord["task"], coord["outcome"], coord["repository"], coord["capability"],
                coord["failure_mode"], tuple(coord["causal_loci"]), coord["status"],
            )
        axes = ("task", "repository", "capability", "failure_mode", "causal_locus")
        indexes = {
            axis: {key: tuple(ids) for key, ids in json.loads(
                (version_dir / "indexes" / f"{axis}.json").read_text(encoding="utf-8")
            ).items()}
            for axis in axes
        }
        nodes = json.loads((version_dir / "nodes.json").read_text(encoding="utf-8"))
        edges = tuple(_read_jsonl(version_dir / "edges.jsonl"))
        abstractions_path = version_dir / "abstractions.json"
        abstractions = {
            key: AbstractKnowledge(
                abstract_id=value["abstract_id"], capability=value["capability"],
                failure_mode=value["failure_mode"], causal_locus=value["causal_locus"],
                trigger_tags=tuple(value.get("trigger_tags") or ()), invariant=value["invariant"],
                action_schema=tuple(value.get("action_schema") or ()),
                safety_boundary=tuple(value.get("safety_boundary") or ()),
                supporting_atom_ids=tuple(value.get("supporting_atom_ids") or ()),
                positive_tasks=tuple(value.get("positive_tasks") or ()),
                negative_tasks=tuple(value.get("negative_tasks") or ()), status=value["status"],
            )
            for key, value in (json.loads(abstractions_path.read_text(encoding="utf-8")).items()
                               if abstractions_path.exists() else ())
        }
        if not abstractions:
            abstractions = cls._build_abstractions(records)
        groundings_path = version_dir / "groundings.jsonl"
        groundings = tuple(
            GroundingRecord(
                grounding_id=row["grounding_id"], abstract_id=row["abstract_id"], task=row["task"],
                repository=row["repository"], capability=row["capability"],
                context_digest=row["context_digest"],
                activation_evidence=tuple(row.get("activation_evidence") or ()),
                bindings=dict(row.get("bindings") or {}),
                instantiated_action=tuple(row.get("instantiated_action") or ()),
                status=row.get("status", "pending"), reward=row.get("reward"),
                source=row.get("source", "runtime"),
                source_atom_ids=tuple(row.get("source_atom_ids") or ()),
                scope_guard=tuple(row.get("scope_guard") or ()),
                verification_plan=tuple(row.get("verification_plan") or ()),
                materialization_protocol=tuple(row.get("materialization_protocol") or ()),
                outcome_evidence=tuple(row.get("outcome_evidence") or ()),
            )
            for row in (_read_jsonl(groundings_path) if groundings_path.exists() else ())
        )
        anchors = {key: tuple(ids) for key, ids in json.loads(
            (version_dir / "retention_anchors.json").read_text(encoding="utf-8")
        ).items()}
        return cls(version=version, records=records, indexes=indexes, nodes=nodes,
                   edges=edges, anchors=anchors, abstractions=abstractions,
                   groundings=groundings, manifest=manifest)

    def _matching_task_key(self, task_id: str) -> str:
        return next((key for key in self.indexes["task"] if task_matches(key, task_id)), "")

    @staticmethod
    def _episode_signature(atom: KnowledgeAtom) -> tuple[str, ...]:
        """Recover the concrete situation signature without storing/replaying a patch.

        Successful M/F atoms contain the files touched by the verified episode.  The
        file set is a useful situation identifier: it lets retrieval keep observation,
        reasoning and action roles from one dominant solution family instead of
        independently mixing roles from unrelated successful attempts.
        """
        patterns = (r"changed_files\s*=\s*(\[[^\]]*\])", r"change\s+(\[[^\]]*\])")
        for pattern in patterns:
            match = re.search(pattern, atom.content)
            if not match:
                continue
            try:
                values = ast.literal_eval(match.group(1))
            except (SyntaxError, ValueError):
                continue
            if isinstance(values, list):
                return tuple(sorted(str(value).replace("\\", "/") for value in values))
        return ()

    def _situation_capsule(self, task_key: str, *, policy: str = "situation_v13") -> tuple[tuple[KnowledgeAtom, ...], dict[str, Any]]:
        """Bind P/R/F/S/M roles to one coherent verified situation family."""
        task_records = [
            self.records[atom_id] for atom_id in self.indexes["task"].get(task_key, ())
            if atom_id in self.records
        ]
        passing = [record for record in task_records if record.outcome == "pass"]
        signatures: dict[tuple[str, ...], list[AtomRecord]] = {}
        for record in passing:
            signature = self._episode_signature(record.atom)
            if signature:
                signatures.setdefault(signature, []).append(record)
        chosen_signature: tuple[str, ...] = ()
        if signatures:
            chosen_signature = sorted(
                signatures,
                key=lambda signature: (-len(signatures[signature]), -len(signature), signature),
            )[0]
        family = [
            record for record in passing
            if not self._episode_signature(record.atom)
            or self._episode_signature(record.atom) == chosen_signature
        ]

        selected: list[KnowledgeAtom] = []
        grounding_credit: dict[str, int] = {}
        profile_credit: dict[tuple[str, ...], int] = {}
        tried_ids: set[str] = set()
        for receipt in self.groundings:
            if not task_matches(receipt.task, task_key):
                continue
            tried_ids.update(receipt.source_atom_ids)
            delta = 1 if receipt.status == "verified" else (-1 if receipt.status == "rejected" else 0)
            if not delta:
                continue
            for atom_id in receipt.source_atom_ids:
                grounding_credit[atom_id] = grounding_credit.get(atom_id, 0) + delta
            if policy == "experience_v14" and receipt.source_atom_ids:
                profile = tuple(sorted(receipt.source_atom_ids))
                # A verified role binding is a reusable working-memory episode.
                # A rejected one remains evidence, but it should lose the next
                # competition instead of globally quarantining its atoms.
                profile_credit[profile] = profile_credit.get(profile, 0) + (3 if delta > 0 else -4)
        preferred_ids: set[str] = set()
        if profile_credit:
            best_profile, best_profile_score = sorted(
                profile_credit.items(), key=lambda item: (-item[1], len(item[0]), item[0])
            )[0]
            if best_profile_score > 0:
                preferred_ids.update(best_profile)
        action_candidates = [record.atom for record in family if record.atom.kind == "F"]
        action = sorted(
            action_candidates,
            key=lambda atom: (
                0 if atom.atom_id in preferred_ids else 1,
                -grounding_credit.get(atom.atom_id, 0),
                -atom.confidence, -len(atom.content), atom.atom_id,
            ),
        )[0] if action_candidates else None
        action_tokens = set(knowledge_tags(action.content)) if action else set()

        def experience_rank(atom: KnowledgeAtom) -> int:
            if policy != "experience_v14":
                return 1
            semantic_trace = (
                "semantic execution trace:" in atom.content.lower()
                or (atom.kind == "R" and " -> " in atom.content)
            )
            # A newly consolidated successful trace must be grounded once;
            # otherwise an older verified portfolio can permanently mask the
            # very experience produced by learning.  After that one trial the
            # normal verifier credit decides whether to keep or roll back.
            if semantic_trace and atom.atom_id not in tried_ids:
                return 0
            if atom.atom_id in preferred_ids:
                return 1
            return 2

        for kind in ("P", "R", "F", "S", "M"):
            candidates = [record.atom for record in family if record.atom.kind == kind]
            if not candidates:
                continue
            candidates.sort(key=lambda atom: (
                experience_rank(atom),
                -grounding_credit.get(atom.atom_id, 0),
                -len(action_tokens.intersection(knowledge_tags(atom.content)))
                if kind == "R" else 0,
                -atom.confidence,
                -len(atom.content),
                atom.atom_id,
            ))
            # A situation is a small constellation, not a single sentence.
            # Keep two mutually compatible causal/action facets for complex
            # tasks while binding them to the same concrete file signature.
            quota = 2 if kind in {"R", "F"} else 1
            selected.extend(candidates[:quota])
        digest = hashlib.sha256("|".join(chosen_signature).encode()).hexdigest()[:12] \
            if chosen_signature else "no-file-signature"
        return tuple(selected), {
            "task": task_key,
            "signature": digest,
            "files": list(chosen_signature),
            "roles": {
                kind: [atom.atom_id for atom in selected if atom.kind == kind]
                for kind in ("P", "R", "F", "S", "M")
                if any(atom.kind == kind for atom in selected)
            },
            "grounding_credit": {
                atom.atom_id: grounding_credit.get(atom.atom_id, 0) for atom in selected
            },
            "profile_credit": {"|".join(profile): score for profile, score in profile_credit.items()},
            "preferred_profile": sorted(preferred_ids),
            "untried_semantic_trace": [
                atom.atom_id for atom in selected
                if atom.atom_id not in tried_ids and (
                    "semantic execution trace:" in atom.content.lower()
                    or (atom.kind == "R" and " -> " in atom.content)
                )
            ],
        }

    def _layered_capsule(self, task_key: str) -> tuple[tuple[KnowledgeAtom, ...], dict[str, Any]]:
        """Keep the champion snapshot as a stable trunk and add only an exact-task delta branch.

        ``memory_layers.delta_atom_ids`` is written by the relational-memory builder.  Atom
        contents remain immutable: this method changes only role binding and retrieval order.
        Five slots are reserved for the old champion contract and at most three slots expose
        the new verified episode.  Consequently, adding a branch cannot evict the old task
        capsule from working memory.
        """
        layer = self.manifest.get("memory_layers") or {}
        delta_ids = set(layer.get("delta_atom_ids") or ())
        task_records = [
            self.records[atom_id] for atom_id in self.indexes["task"].get(task_key, ())
            if atom_id in self.records
        ]
        passing = [record for record in task_records if record.outcome == "pass"]
        if not delta_ids:
            return self._situation_capsule(task_key, policy="situation_v13")

        backbone = [record.atom for record in passing if record.atom.atom_id not in delta_ids]
        # The fast store is episodic rather than synonymous with success.  A recent
        # verified success contributes R/F/M; an unresolved near-miss contributes
        # R/F/S.  Both remain exact-task only and the stable slow-store capsule is
        # selected exclusively from verified episodes.
        delta_records = [
            record for record in task_records if record.atom.atom_id in delta_ids
        ]
        delta_pass = [record.atom for record in delta_records if record.outcome == "pass"]
        delta_fail = [record.atom for record in delta_records if record.outcome == "fail"]

        def rank(atom: KnowledgeAtom) -> tuple[float, int, str]:
            return (-atom.confidence, -len(atom.content), atom.atom_id)

        selected: list[KnowledgeAtom] = []
        backbone_selected: list[KnowledgeAtom] = []
        delta_selected: list[KnowledgeAtom] = []
        for kind in ("M", "F", "R", "S", "P"):
            candidates = sorted((atom for atom in backbone if atom.kind == kind), key=rank)
            if candidates:
                backbone_selected.append(candidates[0])
        # A verified fast episode carries its outcome memory.  A failed fast episode
        # carries a safety/verification boundary instead and is never mislabeled as M.
        fast_atoms = delta_pass if delta_pass else delta_fail
        fast_roles = ("R", "F", "M") if delta_pass else ("R", "F", "S")
        for kind in fast_roles:
            candidates = sorted((atom for atom in fast_atoms if atom.kind == kind), key=rank)
            if candidates:
                delta_selected.append(candidates[0])
        selected.extend(backbone_selected[:5])
        selected.extend(delta_selected[:3])
        return tuple(selected), {
            "task": task_key,
            "signature": f"backbone-v{layer.get('backbone_version', 'unknown')}+delta",
            "roles": {
                kind: [atom.atom_id for atom in selected if atom.kind == kind]
                for kind in ("P", "R", "F", "S", "M")
                if any(atom.kind == kind for atom in selected)
            },
            "layers": {
                "backbone": [atom.atom_id for atom in backbone_selected[:5]],
                "delta": [atom.atom_id for atom in delta_selected[:3]],
            },
        }

    def retrieve(self, *, task_id: str, task_text: str, repository_evidence: str = "",
                 limit: int = 14, policy: str = "full_v1") -> RetrievalBundle:
        material = f"{task_id} {task_text} {repository_evidence}"
        memory_strategy = _memory_strategy() if policy == "modular_reverse_v23" else "v23_modular"
        query = set(knowledge_tags(material))
        capability_affinities = _situation_capability_affinities(task_id, task_text)
        task_key = self._matching_task_key(task_id)
        repo = repository_family(task_id)
        if policy in {"soft_grounding_v22", "modular_reverse_v23", *_V24_POLICIES}:
            capability = max(
                capability_affinities,
                key=lambda name: capability_affinities[name],
                default="repository-contract",
            )
            if capability_affinities.get(capability, 0.0) == 0.0:
                capability = "repository-contract"
        elif policy == "failure_locked_v21":
            # The stable task identifier wins over verbose issue prose.  This
            # prevents repository/parser vocabulary from reclassifying an
            # Actionlint dependency task as parsing-schema mid-episode.
            task_capability = capability_family(task_id)
            capability = (
                task_capability if task_capability != "repository-contract"
                else capability_family(task_text)
            )
        else:
            capability = capability_family(material)
        loci = _causal_loci(material)
        buckets: list[tuple[str, tuple[str, ...], int]] = []
        anchors = self.anchors.get(task_key, ()) if task_key else ()
        buckets.append(("retention_anchor", anchors, 5))
        buckets.append(("task", self.indexes["task"].get(task_key, ()), 5))
        buckets.append(("repository", self.indexes["repository"].get(repo, ()), 2))
        buckets.append(("capability", self.indexes["capability"].get(capability, ()), 3))
        failure_ids: set[str] = set()
        for mode, ids in self.indexes["failure_mode"].items():
            if mode != "unspecified" and set(knowledge_tags(mode)).intersection(query):
                failure_ids.update(ids)
        buckets.append(("failure_mode", tuple(sorted(failure_ids)), 2))
        causal_ids = {atom_id for locus in loci
                      for atom_id in self.indexes["causal_locus"].get(locus, ())}
        buckets.append(("causal_locus", tuple(sorted(causal_ids)), 2))

        selected: list[KnowledgeAtom] = []
        selected_ids: set[str] = set()
        audit_buckets: dict[str, list[str]] = {}
        situation: dict[str, Any] | None = None
        relational_policies = {
            "isolated_v15", "layered_v15", "consensus_v15",
            "transfer_v15", "transfer_v16", "adaptive_v17", "competitive_v18", "bridge_v19",
            "evidence_gated_v20", "failure_locked_v21", "soft_grounding_v22", "modular_reverse_v23",
            *_V24_POLICIES,
        }
        if policy in {"situation_v13", "experience_v14", *relational_policies} and task_key:
            capsule, situation = (self._layered_capsule(task_key)
                                  if policy in relational_policies
                                  else self._situation_capsule(task_key, policy=policy))
            for atom in capsule[:limit]:
                selected.append(atom)
                selected_ids.add(atom.atom_id)
            capsule_bucket = "layered_capsule" if policy in relational_policies else "situation_capsule"
            audit_buckets[capsule_bucket] = [atom.atom_id for atom in selected]
            if policy in relational_policies:
                audit_buckets["backbone_capsule"] = list((situation.get("layers") or {}).get("backbone") or ())
                audit_buckets["delta_branch"] = list((situation.get("layers") or {}).get("delta") or ())
            # The capsule is the task branch.  Additional slots carry only
            # complementary repository/capability/causal context.
            buckets = [item for item in buckets if item[0] not in {"retention_anchor", "task"}]
            if policy == "isolated_v15":
                buckets = []
        for bucket, ids, quota in buckets:
            ranked = sorted(
                (self.records[atom_id] for atom_id in ids if atom_id in self.records),
                key=lambda record: (
                    -len(query.intersection(record.atom.tags or knowledge_tags(record.atom.content))),
                    -record.atom.confidence,
                    record.atom.atom_id,
                ),
            )
            accepted = []
            for record in ranked:
                if record.atom.atom_id in selected_ids:
                    continue
                if (policy in {"situation_v13", "experience_v14", *relational_policies} and task_key
                        and record.task and task_matches(record.task, task_key)):
                    # A second local episode would break the role-bound situation
                    # capsule.  Cross-task abstractions remain available through
                    # repository/capability nodes and ground().
                    continue
                if (policy in relational_policies
                        and record.atom.atom_id in set((self.manifest.get("memory_layers") or {}).get("delta_atom_ids") or ())):
                    # New experience may help its own situation, but must earn explicit
                    # abstraction/grounding evidence before crossing a task boundary.
                    continue
                selected.append(record.atom)
                selected_ids.add(record.atom.atom_id)
                accepted.append(record.atom.atom_id)
                if len(accepted) == quota or len(selected) == limit:
                    break
            audit_buckets[bucket] = accepted
            if len(selected) == limit:
                break

        node_ids = [f"capability:{capability}"]
        if task_key:
            node_ids.append(f"task:{task_key}")
        if f"repository:{repo}" in self.nodes:
            node_ids.append(f"repository:{repo}")
        nodes = tuple(self.nodes[node_id] for node_id in node_ids if node_id in self.nodes)
        return RetrievalBundle(
            self.version, tuple(selected), nodes,
            {"task_key": task_key, "repository": repo, "capability": capability,
             "causal_loci": list(loci), "buckets": audit_buckets,
             "situation": situation,
             "provisional_task": [
                 atom_id for atom_id in self.indexes["task"].get(task_key, ())
                 if atom_id in self.records
                 and self.records[atom_id].status == "provisional-local"
             ]},
        )

    def score_candidates(
        self,
        *,
        task_id: str,
        task_text: str,
        repository_evidence: str = "",
        limit: int = 14,
        policy: str = "split_attractor_v24",
    ) -> tuple[RetrievalBundle, tuple[Any, ...], dict[str, Any]]:
        """Compute PFG candidate signals u_i(t) and condition-matched late groundings for V24."""
        from .split_attractor import CandidateSignal

        bundle = self.retrieve(
            task_id=task_id,
            task_text=task_text,
            repository_evidence=repository_evidence,
            limit=limit,
            policy=policy,
        )
        live_text = repository_evidence.strip()
        task_material = f"{task_id} {task_text}"
        full_material = f"{task_material} {live_text}".strip()
        task_query = set(knowledge_tags(task_material))
        evidence_query = set(knowledge_tags(live_text)) if live_text else set()
        task_affinities = _situation_capability_affinities(task_id, task_text)
        live_affinities = _capability_affinities(live_text) if live_text else {}
        has_live_cap = max(live_affinities.values(), default=0.0) > 0.0
        if has_live_cap:
            blended_affinities = {
                name: min(
                    1.0,
                    0.65 * live_affinities.get(name, 0.0)
                    + 0.35 * task_affinities.get(name, 0.0),
                )
                for name, _ in CAPABILITY_PATTERNS
            }
        else:
            blended_affinities = dict(task_affinities)

        evidence_loci = set(_causal_loci(live_text)) if live_text else set()
        task_loci = set(_causal_loci(task_material))
        live_symbols = _extract_live_symbols(live_text)
        confirmed_failure = "METACORSI_CONFIRMED_FAILURE_EVIDENCE" in live_text
        has_rca_directive = "METACORSI RCA" in live_text

        task_key = self._matching_task_key(task_id)
        task_records = tuple(
            self.records[atom_id]
            for atom_id in self.indexes["task"].get(task_key, ())
            if atom_id in self.records
        )
        has_verified_task_memory = any(record.outcome == "pass" for record in task_records)
        verified_task_loci = {
            locus
            for record in task_records
            if record.outcome == "pass"
            for locus in record.causal_loci
        }
        grounding_evidence: dict[str, tuple[int, int]] = {}
        for record in self.groundings:
            verified, rejected = grounding_evidence.get(record.abstract_id, (0, 0))
            if record.status == "verified":
                verified += 1
            elif record.status == "rejected":
                rejected += 1
            grounding_evidence[record.abstract_id] = (verified, rejected)

        context_digest = hashlib.sha256(full_material.encode()).hexdigest()
        repository = repository_family(task_id)
        capability = bundle.audit["capability"]
        signals: list[CandidateSignal] = []

        for abstract in sorted(self.abstractions.values(), key=lambda a: a.abstract_id):
            support_tasks = {
                self.records[atom_id].task
                for atom_id in abstract.supporting_atom_ids
                if atom_id in self.records and self.records[atom_id].task
            }
            passing_support_tasks = {
                self.records[atom_id].task
                for atom_id in abstract.supporting_atom_ids
                if atom_id in self.records
                and self.records[atom_id].task
                and self.records[atom_id].outcome == "pass"
            }
            if not passing_support_tasks:
                continue
            if any(task_matches(task, task_id) for task in support_tasks):
                continue

            verified, rejected = grounding_evidence.get(abstract.abstract_id, (0, 0))
            cap_score = blended_affinities.get(abstract.capability, 0.0)
            live_cap_score = live_affinities.get(abstract.capability, 0.0) if live_text else 0.0

            # Causal and trigger scores are driven by live repository evidence so
            # Step 0 prose alone cannot prematurely open the disinhibitory gate.
            locus_in_live = (
                bool(live_text)
                and abstract.causal_locus != "unspecified"
                and abstract.causal_locus in evidence_loci
            )
            locus_in_task = (
                abstract.causal_locus != "unspecified"
                and abstract.causal_locus in task_loci
            )
            if locus_in_live:
                causal_score = 1.0
            elif bool(live_text) and locus_in_task:
                causal_score = 0.35
            elif not live_text and locus_in_task:
                causal_score = 0.15
            else:
                causal_score = 0.0

            matched_cues = tuple(sorted(evidence_query.intersection(abstract.trigger_tags)))
            trigger_denominator = max(1, min(3, len(abstract.trigger_tags)))
            if live_text:
                trigger_score = min(1.0, len(matched_cues) / trigger_denominator)
            else:
                trigger_score = 0.0

            source_score = min(1.0, len(support_tasks) / 2.0)
            feedback_score = (verified + 1.0) / (verified + rejected + 2.0)
            episodic_score = 0.0
            if has_verified_task_memory:
                episodic_score = (
                    1.0 if abstract.causal_locus in verified_task_loci else -0.5
                )
            specificity = min(1.0, len(abstract.trigger_tags) / 4.0)
            if abstract.invariant.startswith("Preserve the verified"):
                specificity *= 0.35

            u_score = _strategy_score(
                "v23_modular",
                capability=cap_score,
                causal=causal_score,
                trigger=trigger_score,
                source=source_score,
                feedback=feedback_score,
                episodic=episodic_score,
                specificity=specificity,
                status_ok=abstract.status in {"validated", "transferable"},
            )
            if not live_text:
                # Without live repository evidence, scale down static prior so
                # u_i(0) stays sub-threshold while preserving candidate ordering.
                u_score = min(0.35, u_score * 0.65)

            # Applicability condition matching: trigger signal in live evidence
            # must match the candidate schema's causal locus and/or trigger cues.
            cap_aligned = (
                live_cap_score >= 0.35
                or (not has_live_cap and task_affinities.get(abstract.capability, 0.0) >= 0.5)
            )
            condition_matched = bool(live_text) and cap_aligned and (
                (locus_in_live and (bool(matched_cues) or confirmed_failure or has_rca_directive or bool(live_symbols)))
                or (len(matched_cues) >= 1 and (locus_in_live or locus_in_task or confirmed_failure))
            )

            # Reverse-ground to Top-1 target-disjoint passing source episode
            by_task: dict[str, list[AtomRecord]] = {}
            for atom_id in abstract.supporting_atom_ids:
                source_record = self.records.get(atom_id)
                if source_record and source_record.task and source_record.outcome == "pass":
                    by_task.setdefault(source_record.task, []).append(source_record)

            combined_query = evidence_query | task_query

            def source_rank(item: tuple[str, list[AtomRecord]]) -> tuple[float, int, str]:
                source_task, source_records = item
                semantic = max(
                    (
                        len(
                            combined_query.intersection(
                                record.atom.tags or knowledge_tags(record.atom.content)
                            )
                        )
                        for record in source_records
                    ),
                    default=0,
                )
                role_coverage = len(
                    {record.atom.kind for record in source_records}.intersection({"R", "F", "S"})
                )
                leaf_score = 0.60 * semantic + 0.35 * role_coverage + 0.05 * len(source_records)
                return (-leaf_score, -role_coverage, source_task)

            source_task, source_records = sorted(by_task.items(), key=source_rank)[0]
            source_semantic_overlap = max(
                (
                    len(
                        combined_query.intersection(
                            record.atom.tags or knowledge_tags(record.atom.content)
                        )
                    )
                    for record in source_records
                ),
                default=0,
            )
            source_match_score = min(1.0, source_semantic_overlap / 3.0)
            role_order = {"R": 0, "F": 1, "S": 2, "P": 3, "M": 4}
            source_records = sorted(
                source_records,
                key=lambda record: (
                    role_order.get(record.atom.kind, 9),
                    -record.atom.confidence,
                    record.atom.atom_id,
                ),
            )
            chosen: list[AtomRecord] = []
            for kind in ("R", "F", "S"):
                match = next((record for record in source_records if record.atom.kind == kind), None)
                if match:
                    chosen.append(match)
            if not chosen:
                chosen = source_records[:2]
            selected_source_atoms = tuple(record.atom.atom_id for record in chosen)
            episode_roles = "; ".join(
                f"{record.atom.kind}={_semantic_episode_projection(record.atom.content)}"
                for record in chosen
            )
            symbol_clause = (
                f" Bind directly to observed repository objects: {', '.join(live_symbols)}."
                if live_symbols
                else ""
            )
            actions = (
                f"SPLIT-ATTRACTOR MODULE: {abstract.invariant}",
                f"SOURCE EPISODE {source_task}: {episode_roles}",
                (
                    "TARGET ROLE BINDING: bind state, boundary, and consumer roles to current evidence;"
                    f"{symbol_clause} never copy source files or symbols unless independently present here."
                ),
            )
            actions = tuple(re.sub(r"\s+", " ", item).strip()[:240] for item in actions)
            base_residual = 0.25 if has_verified_task_memory else 0.65
            residual_weight = base_residual * (0.25 + 0.75 * source_match_score)
            local_weight = 1.0 - residual_weight
            bindings = {
                "task": task_id,
                "repository": repository,
                "capability": abstract.capability,
                "causal_locus": abstract.causal_locus,
                "failure_mode": abstract.failure_mode,
                "grounding_weight": f"{u_score:.3f}",
                "capability_similarity": f"{cap_score:.3f}",
                "causal_similarity": f"{causal_score:.3f}",
                "trigger_similarity": f"{trigger_score:.3f}",
                "feedback_reliability": f"{feedback_score:.3f}",
                "target_roles": ",".join(matched_cues) or "structural-analogy",
                "live_symbols": ",".join(live_symbols) or "pending-observation",
                "source_situations": ",".join(sorted(support_tasks)),
                "counterevidence": str(rejected),
                "module": f"{abstract.capability}/{abstract.causal_locus}",
                "memory_strategy": "split_attractor",
                "source_episode": source_task,
                "source_episode_match": f"{source_match_score:.3f}",
                "local_episode_weight": f"{local_weight:.3f}",
                "abstract_residual_weight": f"{residual_weight:.3f}",
                "grounded_atom_ids": ",".join(selected_source_atoms),
                "condition_matched": "true" if condition_matched else "false",
            }
            scope_guard = tuple(
                item.format(**bindings)[:240] for item in abstract.safety_boundary[:2]
            )
            verification_plan = (
                "build/typecheck succeeds",
                "one focused counterexample turns green",
                "independent F2P and P2P verifier decides promotion",
            )
            materialization_protocol = (
                "Use a direct edit command; do not spend a turn planning shell quoting.",
                "A command may be up to 1600 characters; compact python -c, printf, sed, or git apply is allowed.",
                "After each edit run git diff --check plus repository build or typecheck before expanding.",
            )
            activation = matched_cues[:3] if matched_cues else (
                f"capability={abstract.capability}",
                f"causal_locus={abstract.causal_locus}",
            )
            signature = f"{abstract.abstract_id}|{task_id}|{context_digest}"
            grounding_id = "grounding." + hashlib.sha256(signature.encode()).hexdigest()[:20]
            grounding = GroundingRecord(
                grounding_id=grounding_id,
                abstract_id=abstract.abstract_id,
                task=task_id,
                repository=repository,
                capability=abstract.capability,
                context_digest=context_digest,
                activation_evidence=activation,
                bindings=bindings,
                instantiated_action=actions,
                source_atom_ids=selected_source_atoms,
                scope_guard=scope_guard,
                verification_plan=verification_plan,
                materialization_protocol=materialization_protocol,
            )
            signals.append(
                CandidateSignal(
                    abstract_id=abstract.abstract_id,
                    u_score=u_score,
                    capability_match=cap_score,
                    causal_match=causal_score,
                    trigger_match=trigger_score,
                    condition_matched=condition_matched,
                    grounding=grounding,
                    matched_cues=matched_cues,
                    causal_locus=abstract.causal_locus,
                    capability=abstract.capability,
                )
            )

        signals.sort(
            key=lambda s: (-1 if s.condition_matched else 0, -s.u_score, s.abstract_id)
        )
        pfg_summary = {
            "has_live_evidence": bool(live_text),
            "evidence_loci": sorted(evidence_loci),
            "live_symbols": list(live_symbols),
            "confirmed_failure": confirmed_failure,
            "candidate_count": len(signals),
        }
        return bundle, tuple(signals), pfg_summary

    def ground(self, *, task_id: str, task_text: str, repository_evidence: str = "",
               limit: int = 14, abstraction_limit: int = 3,
               policy: str = "full_v1",
               ) -> tuple[RetrievalBundle, tuple[GroundingRecord, ...]]:
        """Reverse-apply abstract knowledge to a concrete task context.

        The returned records are pending application receipts.  The harness
        persists them with the episode; post-verifier evolution marks each
        receipt verified or rejected and consolidates that evidence next turn.
        """
        if policy in _V24_POLICIES:
            from .split_attractor import SplitAttractorWM

            mode = "naive" if policy == "naive_latch_v24" else "sa"
            bundle, signals, pfg_summary = self.score_candidates(
                task_id=task_id,
                task_text=task_text,
                repository_evidence=repository_evidence,
                limit=limit,
                policy=policy,
            )
            wm = SplitAttractorWM(mode=mode)
            step_res = wm.step(
                signals,
                has_live_evidence=bool(repository_evidence.strip()),
                pfg_summary=pfg_summary,
            )
            groundings = (step_res.active_grounding,) if step_res.active_grounding else ()
            return bundle, groundings

        memory_strategy = _memory_strategy() if policy == "modular_reverse_v23" else "v23_modular"
        bundle = self.retrieve(
            task_id=task_id, task_text=task_text,
            repository_evidence=repository_evidence, limit=limit, policy=policy,
        )
        material = f"{task_id} {task_text} {repository_evidence}"
        query = set(knowledge_tags(material))
        evidence_query = set(knowledge_tags(repository_evidence))
        capability_affinities = _situation_capability_affinities(task_id, task_text)
        evidence_loci = set(_causal_loci(repository_evidence)) if repository_evidence.strip() else set()
        capability = bundle.audit["capability"]
        loci = set(bundle.audit["causal_loci"])
        status_weight = {"transferable": 4, "validated": 3, "provisional": 2,
                         "unverified": 1}
        ranked: list[tuple[int, int, str, AbstractKnowledge]] = []
        retention_policies = {
            "retention_v8", "hierarchical_v9", "feedback_weighted_v10", "champion_v11",
            "situation_v13", "experience_v14",
            "isolated_v15", "layered_v15", "consensus_v15", "transfer_v15", "transfer_v16", "adaptive_v17", "competitive_v18", "bridge_v19", "evidence_gated_v20", "failure_locked_v21", "soft_grounding_v22", "modular_reverse_v23",
        }
        task_key = self._matching_task_key(task_id)
        task_records = tuple(
            self.records[atom_id]
            for atom_id in self.indexes["task"].get(task_key, ())
            if atom_id in self.records
        )
        has_verified_task_memory = any(record.outcome == "pass" for record in task_records)
        verified_task_loci = {
            locus
            for record in task_records if record.outcome == "pass"
            for locus in record.causal_loci
        }
        grounding_evidence: dict[str, tuple[int, int]] = {}
        if policy in {"feedback_weighted_v10", "champion_v11", "situation_v13", "experience_v14",
                      "isolated_v15", "layered_v15", "consensus_v15", "transfer_v15", "transfer_v16", "adaptive_v17", "competitive_v18", "bridge_v19", "evidence_gated_v20", "failure_locked_v21", "soft_grounding_v22", "modular_reverse_v23"}:
            for record in self.groundings:
                verified, rejected = grounding_evidence.get(record.abstract_id, (0, 0))
                if record.status == "verified":
                    verified += 1
                elif record.status == "rejected":
                    rejected += 1
                grounding_evidence[record.abstract_id] = (verified, rejected)
        soft_scores: dict[str, dict[str, Any]] = {}
        for abstract in self.abstractions.values():
            exact_positive = any(task_matches(task, task_id) for task in abstract.positive_tasks)
            exact_negative = any(task_matches(task, task_id) for task in abstract.negative_tasks)
            # Perception's capability guess is fallible.  Once an episode has
            # task-local evidence, its exact abstraction path must remain
            # reachable even when the query is routed to another capability.
            if (policy != "soft_grounding_v22" and abstract.capability != capability) and not (
                    policy in retention_policies and (exact_positive or exact_negative)):
                continue
            if policy == "capsule_v2":
                exact_positive = any(task_matches(task, task_id) for task in abstract.positive_tasks)
                if abstract.status == "unverified":
                    continue
                if abstract.status == "provisional" and not exact_positive:
                    continue
                if any(task_matches(task, task_id) for task in abstract.negative_tasks):
                    continue
            overlap = len(query.intersection(abstract.trigger_tags))
            locus_match = 1 if abstract.causal_locus in loci else 0
            evidence_overlap = tuple(sorted(evidence_query.intersection(abstract.trigger_tags)))
            evidence_supports_cause = (
                bool(evidence_overlap)
                and abstract.causal_locus != "unspecified"
                and abstract.causal_locus in evidence_loci
            )
            if policy == "evidence_gated_v20" and not evidence_supports_cause:
                # Task wording can nominate a relation, but only observations
                # from the current checkout may ground it.  This prevents the
                # task statement itself from acting as an answer-shaped oracle.
                continue
            if policy == "failure_locked_v21":
                failure_capability = capability_family(repository_evidence)
                failure_confirmed = "METACORSI_CONFIRMED_FAILURE_EVIDENCE" in repository_evidence
                failure_supports_cause = (
                    failure_confirmed
                    and failure_capability == abstract.capability
                    and abstract.causal_locus != "unspecified"
                    and abstract.causal_locus in evidence_loci
                )
                if not failure_supports_cause:
                    continue
            if policy in {"boundary_v3", "executable_v4", *retention_policies}:
                # V20 deliberately removes capability-only activation.  A broad
                # task label is useful for indexing candidates, but it is not
                # evidence that the current repository exhibits the same cause.
                capability_fallback = policy in {"transfer_v16", "adaptive_v17", "competitive_v18", "bridge_v19", "soft_grounding_v22", "modular_reverse_v23"} and abstract.capability == capability
                transferable = (
                    abstract.status in {"validated", "transferable"}
                    and (overlap >= 1 or locus_match or capability_fallback)
                )
                if policy not in {"soft_grounding_v22", "modular_reverse_v23"} and not (exact_positive or exact_negative or transferable):
                    continue
                # A later verified episode supersedes stale failed-task advice;
                # the failure remains in LTM as evidence but is not activated
                # as the primary recipe for a solved task.
                if (policy in {"hierarchical_v9", "feedback_weighted_v10", "champion_v11",
                               "situation_v13", "experience_v14", "isolated_v15",
                               "layered_v15", "consensus_v15", "transfer_v15", "transfer_v16", "adaptive_v17", "competitive_v18", "bridge_v19", "evidence_gated_v20", "failure_locked_v21", "soft_grounding_v22", "modular_reverse_v23"}
                        and has_verified_task_memory and exact_negative and not exact_positive):
                    continue
                priority = 8 if exact_positive else (
                    (5 if abstract.failure_mode != "unspecified" else 4)
                    if exact_negative else status_weight.get(abstract.status, 0)
                )
                if policy in {"soft_grounding_v22", "modular_reverse_v23"}:
                    # V22 has no relevance gate.  Each abstraction competes in a
                    # continuous Situation-Graph score and the top-K relations are
                    # role-bound into working memory.  The only exclusion below is
                    # experimental integrity: a held-out target may not support the
                    # cross-task abstraction used to solve itself.
                    verified, rejected = grounding_evidence.get(abstract.abstract_id, (0, 0))
                    support_tasks = {
                        self.records[atom_id].task for atom_id in abstract.supporting_atom_ids
                        if atom_id in self.records and self.records[atom_id].task
                    }
                    if policy == "modular_reverse_v23":
                        passing_support_tasks = {
                            self.records[atom_id].task for atom_id in abstract.supporting_atom_ids
                            if atom_id in self.records
                            and self.records[atom_id].task
                            and self.records[atom_id].outcome == "pass"
                        }
                        # A negative-only abstraction may diagnose a failure but
                        # cannot supply the successful episode required by reverse
                        # grounding. Keep it in LTM; do not descend through it.
                        if not passing_support_tasks:
                            continue
                    if any(task_matches(task, task_id) for task in support_tasks):
                        continue
                    capability_score = capability_affinities.get(abstract.capability, 0.0)
                    if abstract.causal_locus in loci and abstract.causal_locus != "unspecified":
                        causal_score = 1.0
                    elif abstract.causal_locus == "unspecified" or "unspecified" in loci:
                        causal_score = 0.20
                    else:
                        causal_score = 0.0
                    trigger_denominator = max(1, min(3, len(abstract.trigger_tags)))
                    trigger_score = min(1.0, overlap / trigger_denominator)
                    source_score = min(1.0, len(support_tasks) / 2.0)
                    feedback_score = (verified + 1.0) / (verified + rejected + 2.0)
                    episodic_score = 0.0
                    if has_verified_task_memory:
                        episodic_score = 1.0 if abstract.causal_locus in verified_task_loci else -0.5
                    polarity_score = 1.0 if exact_positive else (-1.0 if exact_negative else 0.0)
                    score = (
                        0.28 * capability_score
                        + 0.27 * causal_score
                        + 0.20 * trigger_score
                        + 0.10 * source_score
                        + 0.08 * feedback_score
                        + 0.05 * episodic_score
                        + 0.02 * polarity_score
                    )
                    if policy == "modular_reverse_v23":
                        # Prefer mechanism-bearing modules over generic repair
                        # boilerplate.  This remains a continuous ranking term,
                        # not an activation gate.
                        specificity = min(1.0, len(abstract.trigger_tags) / 4.0)
                        if abstract.invariant.startswith("Preserve the verified"):
                            specificity *= 0.35
                        score = _strategy_score(
                            memory_strategy,
                            capability=capability_score,
                            causal=causal_score,
                            trigger=trigger_score,
                            source=source_score,
                            feedback=feedback_score,
                            episodic=episodic_score,
                            specificity=specificity,
                            status_ok=abstract.status in {"validated", "transferable"},
                        )
                    else:
                        if abstract.status not in {"validated", "transferable"}:
                            score -= 0.04
                        score = max(0.01, min(1.0, score))
                    score_milli = int(round(score * 1000))
                    shared_roles = tuple(sorted(query.intersection(abstract.trigger_tags)))
                    soft_scores[abstract.abstract_id] = {
                        "weight": score,
                        "capability_similarity": capability_score,
                        "causal_similarity": causal_score,
                        "trigger_similarity": trigger_score,
                        "source_diversity": source_score,
                        "feedback_reliability": feedback_score,
                        "episodic_compatibility": episodic_score,
                        "shared_roles": shared_roles,
                        "source_tasks": tuple(sorted(support_tasks)),
                        "counterevidence": rejected,
                        "situation_capabilities": capability_affinities,
                        "module": f"{abstract.capability}/{abstract.causal_locus}",
                        "memory_strategy": memory_strategy,
                    }
                    ranked.append((score_milli, locus_match + overlap, abstract.abstract_id, abstract))
                    continue
                if policy == "feedback_weighted_v10":
                    verified, rejected = grounding_evidence.get(abstract.abstract_id, (0, 0))
                    priority += min(3, verified) - min(3, rejected)
                elif policy in {"champion_v11", "situation_v13", "experience_v14",
                                "isolated_v15", "layered_v15", "consensus_v15", "transfer_v15", "transfer_v16", "adaptive_v17", "competitive_v18", "bridge_v19", "evidence_gated_v20", "failure_locked_v21"}:
                    verified, rejected = grounding_evidence.get(abstract.abstract_id, (0, 0))
                    if rejected and not verified and not exact_positive:
                        continue
                    if policy in {"transfer_v15", "transfer_v16", "adaptive_v17", "competitive_v18", "bridge_v19", "evidence_gated_v20", "failure_locked_v21"}:
                        delta_ids = set((self.manifest.get("memory_layers") or {}).get("delta_atom_ids") or ())
                        support_tasks = {
                            self.records[atom_id].task for atom_id in abstract.supporting_atom_ids
                            if atom_id in self.records and self.records[atom_id].task
                        }
                        delta_support_tasks = {
                            self.records[atom_id].task for atom_id in abstract.supporting_atom_ids
                            if atom_id in delta_ids and atom_id in self.records and self.records[atom_id].task
                        }
                        # Transfer is deliberately non-oracle: the target has no exact delta
                        # episode, and an abstraction must be independently supported by at
                        # least two source situations before it can cross the task boundary.
                        # Every supporting atom must also be target-disjoint.  Checking only
                        # delta supports is insufficient because an abstraction copied from a
                        # later graph can still encode a target's older/backbone episode.
                        minimum_support_met = (
                            len(support_tasks) >= 2 and len(delta_support_tasks) >= 1
                            if policy in {"bridge_v19", "evidence_gated_v20", "failure_locked_v21"}
                            else len(delta_support_tasks) >= 2
                        )
                        if (not minimum_support_met
                                or any(task_matches(task, task_id) for task in support_tasks)
                                or abstract.status not in {"validated", "transferable"}):
                            continue
                        if policy in {"competitive_v18", "bridge_v19", "evidence_gated_v20", "failure_locked_v21"} and has_verified_task_memory:
                            # The episodic capsule and an abstract relation compete
                            # instead of being separated by a hard old/new-task gate.
                            # Cross-task advice may reinforce a known task only when
                            # its capability, causal level, and observable trigger
                            # all agree with the task-local successful episode.
                            episodic_compatible = (
                                abstract.capability == capability
                                and abstract.causal_locus in verified_task_loci
                                and overlap >= 1
                            )
                            if not episodic_compatible:
                                continue
                    if policy in {"layered_v15", "consensus_v15"} and not exact_positive:
                        independent_support = len(set(abstract.positive_tasks))
                        minimum_support = 2
                        minimum_verified = 1 if policy == "layered_v15" else 2
                        if (independent_support < minimum_support
                                or verified < minimum_verified
                                or rejected >= verified):
                            continue
                    priority += min(2, verified) - min(4, rejected * 2)
                ranked.append((priority, locus_match + overlap, abstract.abstract_id, abstract))
                continue
            ranked.append((status_weight.get(abstract.status, 0), locus_match + overlap,
                           abstract.abstract_id, abstract))
        ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
        context_digest = hashlib.sha256(material.encode()).hexdigest()
        repository = repository_family(task_id)
        records: list[GroundingRecord] = []
        boundary_policy = policy in {"boundary_v3", "executable_v4", *retention_policies}
        if policy == "retention_v8":
            effective_limit = min(abstraction_limit, 3)
        elif policy in {"hierarchical_v9", "feedback_weighted_v10", "champion_v11"}:
            effective_limit = min(abstraction_limit, 2)
        elif policy in {"situation_v13", "experience_v14", "layered_v15", "consensus_v15", "transfer_v15", "transfer_v16", "adaptive_v17", "competitive_v18", "bridge_v19", "evidence_gated_v20", "failure_locked_v21", "soft_grounding_v22", "modular_reverse_v23"}:
            effective_limit = (min(abstraction_limit, 3) if policy == "soft_grounding_v22"
                               else (min(abstraction_limit, 1) if policy == "modular_reverse_v23"
                               else (0 if policy == "adaptive_v17" and has_verified_task_memory
                                     else min(abstraction_limit, 1))))
        elif policy == "isolated_v15":
            effective_limit = 0
        else:
            effective_limit = min(abstraction_limit, 2) if boundary_policy else abstraction_limit
        for _, _, _, abstract in ranked[:effective_limit]:
            activation_query = evidence_query if policy in {"evidence_gated_v20", "failure_locked_v21"} else query
            activation = tuple(sorted(activation_query.intersection(abstract.trigger_tags)))
            if not activation:
                activation = (f"capability={capability}", f"causal_locus={abstract.causal_locus}")
            actions = abstract.action_schema
            selected_source_atoms: tuple[str, ...] | None = None
            source_match_score = 0.0
            if policy == "modular_reverse_v23":
                # Reverse-ground through the chosen abstract module to one
                # concrete, target-disjoint source episode, then expose only its
                # causal/action/verification atoms to working memory.
                by_task: dict[str, list[AtomRecord]] = {}
                for atom_id in abstract.supporting_atom_ids:
                    source_record = self.records.get(atom_id)
                    if source_record and source_record.task and source_record.outcome == "pass":
                        by_task.setdefault(source_record.task, []).append(source_record)
                def source_rank(item: tuple[str, list[AtomRecord]]) -> tuple[float, int, str]:
                    source_task, source_records = item
                    semantic = max((
                        len(query.intersection(record.atom.tags or knowledge_tags(record.atom.content)))
                        for record in source_records
                    ), default=0)
                    role_coverage = len({record.atom.kind for record in source_records}.intersection({"R", "F", "S"}))
                    # Different theories select a leaf differently, but all are
                    # sparse and target-disjoint.  No strategy sees verifier answers.
                    if memory_strategy == "anthropic_dreaming":
                        leaf_score = .45 * role_coverage + .35 * semantic + .20 * len(source_records)
                    elif memory_strategy == "cls_dual_store":
                        leaf_score = .50 * role_coverage + .25 * semantic + .25 * len(source_records)
                    elif memory_strategy == "schema_assimilation":
                        leaf_score = .55 * semantic + .35 * role_coverage + .10 * len(source_records)
                    elif memory_strategy == "hippocampal_index":
                        # Sparse cue completion: reward discriminative overlap,
                        # mildly penalize a diffuse episode with many atoms.
                        leaf_score = .65 * semantic + .40 * role_coverage - .05 * len(source_records)
                    else:
                        leaf_score = .60 * semantic + .35 * role_coverage + .05 * len(source_records)
                    return (-leaf_score, -role_coverage, source_task)
                source_task, source_records = sorted(by_task.items(), key=source_rank)[0]
                source_semantic_overlap = max((
                    len(query.intersection(record.atom.tags or knowledge_tags(record.atom.content)))
                    for record in source_records
                ), default=0)
                source_match_score = min(1.0, source_semantic_overlap / 3.0)
                role_order = {"R": 0, "F": 1, "S": 2, "P": 3, "M": 4}
                source_records = sorted(
                    source_records,
                    key=lambda record: (role_order.get(record.atom.kind, 9), -record.atom.confidence, record.atom.atom_id),
                )
                chosen: list[AtomRecord] = []
                role_sequence = ("R", "F", "S")
                if memory_strategy == "event_segmentation":
                    # Retrieve the causal/action segment around the predicted
                    # boundary instead of replaying the complete source event.
                    role_sequence = ("R", "F")
                elif memory_strategy == "anthropic_search_notes":
                    role_sequence = ("R", "F")
                for kind in role_sequence:
                    match = next((record for record in source_records if record.atom.kind == kind), None)
                    if match:
                        chosen.append(match)
                if not chosen:
                    chosen = source_records[:2]
                selected_source_atoms = tuple(record.atom.atom_id for record in chosen)
                episode_roles = "; ".join(
                    f"{record.atom.kind}={_semantic_episode_projection(record.atom.content)}"
                    for record in chosen
                )
                strategy_prefix = {
                    "v23_modular": "MODULAR ABSTRACTION",
                    "anthropic_search_notes": "SEARCHED MEMORY NOTE",
                    "anthropic_dreaming": "DREAM-CONSOLIDATED NOTE",
                    "memtree_collapsed": "MEMTREE BEST-NODE SUMMARY",
                    "memtree_traversal": "MEMTREE ROOT-TO-LEAF PATH",
                    "cls_dual_store": "SLOW SCHEMA TRACE",
                    "schema_assimilation": "SCHEMA-CONGRUENT INVARIANT",
                    "art_resonance": "RESONANT CATEGORY PROTOTYPE",
                    "hippocampal_index": "SPARSE EPISODIC INDEX",
                    "event_segmentation": "EVENT-BOUNDARY SEGMENT",
                    "split_attractor": "SPLIT-ATTRACTOR MODULE",
                }[memory_strategy]
                target_instruction = (
                    "TARGET ROLE BINDING: bind state, boundary, and consumer roles to current evidence; "
                    "never copy source files or symbols unless independently present here."
                )
                if memory_strategy == "cls_dual_store":
                    target_instruction = (
                        "INTERLEAVED REPLAY: let the exact-task episode remain the fast store; use this "
                        "source-agnostic schema only as a slow residual and test both old and new behavior."
                    )
                elif memory_strategy == "art_resonance":
                    target_instruction = (
                        "MATCH OR RESET: map only resonant roles; treat mismatching roles as a new category "
                        "and reason from current repository evidence instead of forcing the prototype."
                    )
                elif memory_strategy == "event_segmentation":
                    target_instruction = (
                        "BOUNDARY UPDATE: apply this segment only at the matching prediction-error boundary; "
                        "rebuild the current event model from repository evidence on either side."
                    )
                actions = (
                    f"{strategy_prefix}: {abstract.invariant}",
                    f"SOURCE EPISODE {source_task}: {episode_roles}",
                    target_instruction,
                )
            if policy in {"evidence_gated_v20", "failure_locked_v21"}:
                # Cross-task knowledge is a diagnostic prior, never an edit
                # recipe.  Grounding may ask the agent to test a causal claim;
                # only current-code evidence can authorize a later repair.
                actions = (
                    "DIAGNOSTIC HYPOTHESIS: " + abstract.invariant,
                    "EVIDENCE TEST: trace the current failing value/state through the named boundary and identify one observable causal edge before editing.",
                )
            if policy in {"capsule_v2", "boundary_v3", "executable_v4", *retention_policies}:
                activation = activation[:3]
                action_limit = 3 if boundary_policy else 2
                actions = tuple(re.sub(r"\s+", " ", item).strip()[:240]
                                for item in actions[:action_limit])
            bindings = {
                "task": task_id,
                "repository": repository,
                "capability": capability,
                "causal_locus": abstract.causal_locus,
                "failure_mode": abstract.failure_mode,
            }
            if policy in {"soft_grounding_v22", "modular_reverse_v23"}:
                soft = soft_scores[abstract.abstract_id]
                bindings.update({
                    "grounding_weight": f"{soft['weight']:.3f}",
                    "capability_similarity": f"{soft['capability_similarity']:.3f}",
                    "causal_similarity": f"{soft['causal_similarity']:.3f}",
                    "trigger_similarity": f"{soft['trigger_similarity']:.3f}",
                    "feedback_reliability": f"{soft['feedback_reliability']:.3f}",
                    "target_roles": ",".join(soft["shared_roles"]) or "structural-analogy",
                    "source_situations": ",".join(soft["source_tasks"]),
                    "counterevidence": str(soft["counterevidence"]),
                    "module": soft["module"],
                    "situation_capabilities": json.dumps(
                        soft["situation_capabilities"], sort_keys=True, separators=(",", ":")
                    ),
                    "memory_strategy": memory_strategy,
                })
                if policy == "modular_reverse_v23":
                    base_residual = 0.25 if has_verified_task_memory else 0.65
                    residual_weight = base_residual * (0.25 + 0.75 * source_match_score)
                    local_weight = 1.0 - residual_weight
                    bindings.update({
                        "source_episode": source_task,
                        "source_episode_match": f"{source_match_score:.3f}",
                        "local_episode_weight": f"{local_weight:.3f}",
                        "abstract_residual_weight": f"{residual_weight:.3f}",
                        "grounded_atom_ids": ",".join(selected_source_atoms or ()),
                    })
            if boundary_policy:
                actions = tuple(action.format(**bindings) for action in actions)
                scope_guard = tuple(item.format(**bindings)[:240]
                                    for item in abstract.safety_boundary[:2])
                verification_plan = (
                    "build/typecheck succeeds",
                    "one focused counterexample turns green",
                    "independent F2P and P2P verifier decides promotion",
                )
                materialization_protocol = (
                    "Use a direct edit command; do not spend a turn planning shell quoting.",
                    "A command may be up to 1600 characters; compact python -c, printf, sed, or git apply is allowed.",
                    "After each edit run git diff --check plus repository build or typecheck before expanding.",
                ) if policy in {"executable_v4", *retention_policies} and policy not in {"evidence_gated_v20", "failure_locked_v21"} else ()
            else:
                scope_guard = ()
                verification_plan = ()
                materialization_protocol = ()
            source_atom_ids = abstract.supporting_atom_ids
            if selected_source_atoms is not None:
                source_atom_ids = selected_source_atoms
            if policy in {"situation_v13", "experience_v14", "isolated_v15", "layered_v15", "consensus_v15"}:
                roles = (bundle.audit.get("situation") or {}).get("roles") or {}
                source_atom_ids = tuple(
                    atom_id for kind in ("P", "R", "F", "S", "M")
                    for atom_id in roles.get(kind, ())
                )
            signature = f"{abstract.abstract_id}|{task_id}|{context_digest}"
            if policy == "experience_v14":
                # Different role bindings must retain separate verifier receipts;
                # otherwise the latest run overwrites the evidence needed for
                # portfolio credit assignment.
                signature += "|" + "|".join(sorted(source_atom_ids))
            grounding_id = "grounding." + hashlib.sha256(signature.encode()).hexdigest()[:20]
            records.append(GroundingRecord(
                grounding_id=grounding_id,
                abstract_id=abstract.abstract_id,
                task=task_id,
                repository=repository,
                capability=capability,
                context_digest=context_digest,
                activation_evidence=activation,
                bindings=bindings,
                instantiated_action=actions,
                source_atom_ids=source_atom_ids,
                scope_guard=scope_guard,
                verification_plan=verification_plan,
                materialization_protocol=materialization_protocol,
            ))
        return bundle, tuple(records)


def render_indexed_bundle(bundle: RetrievalBundle, policy: str = "full_v1") -> str:
    lines = [
        f"INDEXED LTM v{bundle.version} (atoms are evidence, not answers):",
        "Perception routed this task through stable LTM branches; current repository evidence overrides memory.",
    ]
    for node in bundle.nodes:
        lines.append(
            f"- INDEX NODE [{node['node_type']}/{node['status']}]: {node['summary']}"
        )
    labels = {"P": "OBSERVE", "R": "ROOT CAUSE", "F": "RECOVERY ACTION",
              "S": "TRANSITION GUARD", "M": "VERIFIED MEMORY"}
    anchor_ids = set(bundle.audit["buckets"].get("retention_anchor") or ())
    atoms = list(bundle.atoms)
    if policy == "capsule_v2":
        anchors = [atom for kind in ("F", "R", "S") for atom in atoms
                   if atom.atom_id in anchor_ids and atom.kind == kind][:3]
        transfer = [atom for atom in atoms if atom.atom_id not in anchor_ids][:2]
        atoms = anchors + transfer
    elif policy in {"boundary_v3", "executable_v4"}:
        task_ids = set(bundle.audit["buckets"].get("retention_anchor") or ())
        task_ids.update(bundle.audit["buckets"].get("task") or ())
        exact = [atom for kind in ("P", "R", "F", "S") for atom in atoms
                 if atom.atom_id in task_ids and atom.kind == kind]
        atoms = exact[:4]
    elif policy in {"retention_v8", "hierarchical_v9", "feedback_weighted_v10", "champion_v11"}:
        task_ids = set(bundle.audit["buckets"].get("retention_anchor") or ())
        task_ids.update(bundle.audit["buckets"].get("task") or ())
        exact = [atom for atom in atoms if atom.atom_id in task_ids]
        # Build a compact episode capsule rather than taking the first four
        # atoms.  One memory/reason/action/guard/observation from a successful
        # episode carries the whole learned contract without patch replay.
        def atom_rank(atom: KnowledgeAtom) -> tuple[int, float, int, str]:
            verified = 1 if ".pass." in atom.atom_id else 0
            return (-verified, -atom.confidence, -len(atom.content), atom.atom_id)
        grouped: list[KnowledgeAtom] = []
        for kind in ("M", "F", "R", "S", "P"):
            candidates = sorted((atom for atom in exact if atom.kind == kind), key=atom_rank)
            if candidates:
                grouped.append(candidates[0])
        if policy == "retention_v8":
            extras = sorted((atom for atom in exact if atom not in grouped), key=atom_rank)
            grouped.extend(extras[:1])
        elif policy == "feedback_weighted_v10":
            # A complete evolution iteration must expose the newest verifier
            # counterexample alongside the retained successful procedure.  It
            # is a residual boundary to ground, not a replacement recipe.
            provisional_ids = set(bundle.audit.get("provisional_task") or ())
            residual = sorted(
                (atom for atom in exact
                 if atom.atom_id in provisional_ids and ".fail." in atom.atom_id
                 and atom.kind in {"P", "R", "S"}),
                key=lambda atom: ({"P": 0, "R": 1, "S": 2}[atom.kind],
                                  -len(atom.content), atom.atom_id),
            )
            grouped.extend(residual[:2])
        atoms = grouped
    elif policy in {"situation_v13", "experience_v14", "isolated_v15", "layered_v15", "consensus_v15", "transfer_v15", "transfer_v16", "adaptive_v17", "competitive_v18", "bridge_v19", "evidence_gated_v20", "failure_locked_v21", "soft_grounding_v22", "modular_reverse_v23", *_V24_POLICIES}:
        capsule_ids = set(bundle.audit["buckets"].get("situation_capsule") or ())
        capsule_ids.update(bundle.audit["buckets"].get("layered_capsule") or ())
        capsule = [atom for atom in atoms if atom.atom_id in capsule_ids]
        transfer = [atom for atom in atoms if atom.atom_id not in capsule_ids]
        order = {kind: index for index, kind in enumerate(("P", "R", "F", "S", "M"))}
        atoms = sorted(capsule, key=lambda atom: (order[atom.kind], atom.atom_id))
        atoms.extend(transfer[:2])
        situation = bundle.audit.get("situation") or {}
        relation_line = ("- RELATIONAL MEMORY: the frozen champion is the stable trunk; "
                         "the exact-task delta is an adaptation branch and cannot evict it."
                         if policy in {"isolated_v15", "layered_v15", "consensus_v15", "transfer_v15", "transfer_v16", "adaptive_v17", "competitive_v18", "bridge_v19", "evidence_gated_v20", "failure_locked_v21", "soft_grounding_v22", "modular_reverse_v23", *_V24_POLICIES}
                         else "- SITUATION GRAPH: one verified episode family is bound into working memory.")
        lines.extend((
            relation_line,
            f"- SITUATION ID: {situation.get('signature', 'unknown')}; "
            f"task={situation.get('task', bundle.audit['task_key'])}; "
            f"repository={bundle.audit['repository']}; capability={bundle.audit['capability']}",
            "- ROLE BINDING: P=observation, R=causal model, F=action, "
            "S=verification boundary, M=verified outcome. Keep these roles coherent; "
            "use abstractions only to interpret current repository evidence.",
        ))
    for atom in atoms:
        role = "RETENTION ANCHOR" if atom.atom_id in anchor_ids else labels[atom.kind]
        if (policy in {"retention_v8", "hierarchical_v9", "feedback_weighted_v10", "champion_v11",
                       "situation_v13", "experience_v14", "isolated_v15", "layered_v15", "consensus_v15", "transfer_v15", "transfer_v16", "adaptive_v17", "competitive_v18", "bridge_v19", "evidence_gated_v20", "failure_locked_v21", "soft_grounding_v22", "modular_reverse_v23", *_V24_POLICIES}
                and atom.atom_id in anchor_ids):
            role = f"RETENTION ANCHOR/{labels[atom.kind]}"
        if (policy == "feedback_weighted_v10"
                and atom.atom_id in set(bundle.audit.get("provisional_task") or ())):
            role = f"RESIDUAL COUNTEREXAMPLE/{labels[atom.kind]}"
        if policy in {"isolated_v15", "layered_v15", "consensus_v15", "transfer_v15", "transfer_v16", "adaptive_v17", "competitive_v18", "bridge_v19", "evidence_gated_v20", "failure_locked_v21", "soft_grounding_v22", "modular_reverse_v23", *_V24_POLICIES}:
            delta_ids = set(bundle.audit["buckets"].get("delta_branch") or ())
            layer = "DELTA BRANCH" if atom.atom_id in delta_ids else "STABLE BACKBONE"
            role = f"{layer}/{labels[atom.kind]}"
        compact_policy = policy in {"capsule_v2", "boundary_v3", "executable_v4",
                                    "retention_v8", "hierarchical_v9", "feedback_weighted_v10", "champion_v11",
                                    "situation_v13", "experience_v14", "isolated_v15", "layered_v15", "consensus_v15", "transfer_v15", "transfer_v16", "adaptive_v17", "competitive_v18", "bridge_v19", "evidence_gated_v20", "failure_locked_v21", "soft_grounding_v22", "modular_reverse_v23", *_V24_POLICIES}
        content_limit = 1500 if policy == "experience_v14" else (
            520 if policy in {"retention_v8", "hierarchical_v9", "feedback_weighted_v10", "champion_v11", "situation_v13",
                              "isolated_v15", "layered_v15", "consensus_v15", "transfer_v15", "transfer_v16", "adaptive_v17", "competitive_v18", "bridge_v19", "evidence_gated_v20", "failure_locked_v21", "soft_grounding_v22", "modular_reverse_v23", *_V24_POLICIES} else 260
        )
        content = atom.content if not compact_policy else re.sub(r"\s+", " ", atom.content).strip()[:content_limit]
        lines.append(f"- {role}: {atom.content} [id={atom.atom_id}; confidence={atom.confidence:.2f}]")
        if compact_policy:
            lines[-1] = f"- {role}: {content} [id={atom.atom_id}]"
    if policy in {"capsule_v2", "boundary_v3", "executable_v4",
                  "retention_v8", "hierarchical_v9", "feedback_weighted_v10", "champion_v11",
                  "situation_v13", "experience_v14", "isolated_v15", "layered_v15", "consensus_v15", "transfer_v15", "transfer_v16", "adaptive_v17", "competitive_v18", "bridge_v19", "evidence_gated_v20", "failure_locked_v21", "soft_grounding_v22", "modular_reverse_v23", *_V24_POLICIES}:
        counts = {key: len(value) for key, value in bundle.audit["buckets"].items()}
        lines.append("- RETRIEVAL AUDIT: " + json.dumps({
            "task": bundle.audit["task_key"], "capability": bundle.audit["capability"],
            "causal_loci": bundle.audit["causal_loci"], "bucket_counts": counts,
        }, ensure_ascii=False, sort_keys=True))
    else:
        lines.append("- RETRIEVAL AUDIT: " + json.dumps(bundle.audit, ensure_ascii=False, sort_keys=True))
    return "\n".join(lines)


def render_grounded_application(bundle: RetrievalBundle,
                                groundings: tuple[GroundingRecord, ...],
                                policy: str = "full_v1") -> str:
    lines = [render_indexed_bundle(bundle, policy), "", "GROUNDED ABSTRACT APPLICATIONS:"]
    if not groundings:
        lines.append("- No abstraction was applicable; reason from repository evidence and task-local atoms.")
        return "\n".join(lines)
    for record in groundings:
        if policy in {"evidence_gated_v20", "failure_locked_v21"}:
            lines.extend((
                f"- EVIDENCE-GATED DIAGNOSTIC PRIOR {record.grounding_id}: abstract={record.abstract_id}",
                f"  observed_evidence={list(record.activation_evidence)}; bindings={record.bindings}",
                f"  hypotheses={list(record.instantiated_action)}",
                "  authority=DIAGNOSIS_ONLY: do not edit from this memory; first confirm one causal edge in current source, trace, or focused test.",
                f"  scope_guard={list(record.scope_guard)}",
                f"  verify={list(record.verification_plan)}",
            ))
        elif policy in {"modular_reverse_v23", *_V24_POLICIES}:
            header_tag = (
                "SPLIT-ATTRACTOR LATCHED GROUNDING"
                if policy in _V24_POLICIES
                else "MODULAR REVERSE GROUNDING"
            )
            lines.extend((
                f"- {header_tag} {record.grounding_id}: strategy={record.bindings.get('memory_strategy')}; module={record.bindings.get('module')}; abstract={record.abstract_id}",
                f"  mixture=local_episode:{record.bindings.get('local_episode_weight')} + abstract_residual:{record.bindings.get('abstract_residual_weight')}",
                f"  source_episode={record.bindings.get('source_episode')}; grounded_atoms={record.bindings.get('grounded_atom_ids')}",
                f"  target_roles={record.bindings.get('target_roles')}; live_symbols={record.bindings.get('live_symbols', 'n/a')}; reverse_grounded_knowledge={list(record.instantiated_action)}",
                "  authority=ROLE_BOUND_RESIDUAL: preserve the local episode; apply only the mapped mechanism, never source-specific files or symbols.",
                f"  verify={list(record.verification_plan)}",
            ))
        elif policy == "soft_grounding_v22":
            weight = record.bindings.get("grounding_weight", "0.000")
            lines.extend((
                f"- SOFT GROUNDED APPLICATION {record.grounding_id}: abstract={record.abstract_id}; weight={weight}",
                f"  role_binding={{'target_roles': '{record.bindings.get('target_roles', '')}', "
                f"'capability': '{record.bindings.get('capability_similarity', '')}', "
                f"'causal_locus': '{record.bindings.get('causal_similarity', '')}', "
                f"'trigger': '{record.bindings.get('trigger_similarity', '')}'}}",
                f"  source_situations={record.bindings.get('source_situations', '')}; "
                f"counterevidence={record.bindings.get('counterevidence', '0')}",
                f"  grounded_actions={list(record.instantiated_action)}",
                "  authority=WEIGHTED_HYPOTHESIS: combine with the other applications and the exact-task episode; verify against current source before editing.",
                f"  scope_guard={list(record.scope_guard)}",
                f"  verify={list(record.verification_plan)}",
            ))
        elif policy in {"boundary_v3", "executable_v4", "retention_v8",
                      "hierarchical_v9", "feedback_weighted_v10", "champion_v11",
                      "situation_v13", "experience_v14", "isolated_v15", "layered_v15", "consensus_v15", "transfer_v15", "transfer_v16", "adaptive_v17", "competitive_v18", "bridge_v19"}:
            lines.extend((
                f"- APPLICATION {record.grounding_id}: abstract={record.abstract_id}",
                f"  trigger={list(record.activation_evidence)}; bindings={record.bindings}",
                f"  minimal_actions={list(record.instantiated_action)}",
                f"  scope_guard={list(record.scope_guard)}",
                f"  verify={list(record.verification_plan)}",
            ))
            if record.materialization_protocol:
                lines.append(f"  materialize={list(record.materialization_protocol)}")
        else:
            lines.append(
                f"- GROUNDING {record.grounding_id}: abstract={record.abstract_id}; "
                f"activation={list(record.activation_evidence)}; bindings={record.bindings}; "
                f"instantiated_action={list(record.instantiated_action)}. "
                "Treat this as a hypothesis and verify it against the current repository before editing."
            )
    rendered = "\n".join(lines)
    if policy == "capsule_v2":
        return rendered[:5200]
    if policy in {"boundary_v3", "executable_v4"}:
        return rendered[:4600]
    if policy in {"retention_v8", "hierarchical_v9", "feedback_weighted_v10", "champion_v11",
                  "situation_v13", "experience_v14", "isolated_v15", "layered_v15", "consensus_v15", "transfer_v15", "transfer_v16", "adaptive_v17", "competitive_v18", "bridge_v19", "evidence_gated_v20", "failure_locked_v21", "soft_grounding_v22", "modular_reverse_v23", *_V24_POLICIES}:
        return rendered[:6800]
    return rendered


def render_split_attractor_directive(
    bundle: RetrievalBundle,
    step_result: Any,
    policy: str = "split_attractor_v24",
) -> str:
    """Render dynamic V24 SA-WM knowledge injection or re-inhibitory retraction directive."""
    if step_result.transition == "reinhibitory_erase":
        retracted = step_result.previous_abstract_id or "previous-schema"
        return (
            "\n\n[SPLIT-ATTRACTOR RE-INHIBITORY RESET]:\n"
            f"Gate g_t=0 (state={step_result.state}; reason={step_result.reason}). "
            f"The previously latched cross-task hypothesis ({retracted}) is refuted or stagnant "
            "under current repository evidence and has been retracted into short-term refractory inhibition. "
            "Do NOT continue editing along that premise; re-anchor on current task-local RCA and live code evidence."
        )
    if step_result.gate == 1 and step_result.active_grounding is not None:
        h_val = step_result.h_scores.get(step_result.active_abstract_id or "", 0.0)
        p_val = step_result.p_scores.get(step_result.active_abstract_id or "", 0.0)
        body = render_grounded_application(bundle, (step_result.active_grounding,), policy)
        return (
            f"\n\n[SPLIT-ATTRACTOR DISINHIBITORY WRITE (mode={step_result.mode}; "
            f"gate=1; transition={step_result.transition}; p={p_val:.3f}; h={h_val:.3f})]:\n"
            + body
        )
    return ""
