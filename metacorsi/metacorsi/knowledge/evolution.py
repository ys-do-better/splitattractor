from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal

from ..io import atomic_json
from ..rca.evolution import EvolutionDiagnosis, diagnose_verified_outcome
from ..reflection import reflect_outcome
from .grounding import (GroundedKnowledgeGraph, abstraction_atom,
                        capability_family)
from .indexed_ltm import GroundingRecord, IndexedLTM
from .store import KnowledgeAtom, VersionedKnowledgeStore, knowledge_tags


@dataclass(frozen=True)
class VerifiedEpisode:
    task: str
    reward: float
    f2p_passed: int
    f2p_total: int
    p2p_passed: int
    p2p_total: int
    final_analysis: str
    patch: str
    source: str
    failed_tests: tuple[str, ...] = ()
    steps: int = 0
    protocol_retries: int = 0
    observed_at: float = 0.0
    groundings: tuple[dict, ...] = ()
    execution_trace: str = ""
    implementation_blueprint: str = ""


def _failed_tests(trial: Path) -> tuple[str, ...]:
    path = trial / "verifier" / "ctrf.json"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    tests = (report.get("results") or {}).get("tests") or ()
    failures = []
    for item in tests:
        if item.get("status") in {"passed", "skipped"}:
            continue
        name = str(item.get("name") or "unnamed verifier failure")
        message = str(item.get("message") or "")
        failures.append(_compact(f"{name} :: {message}", 520))
    return tuple(failures[:24])


def _compact(text: str, limit: int = 700) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _execution_trace(history: list[dict]) -> str:
    """Keep a semantic recipe from a successful episode, never executable patch text.

    Final analyses are often only a generic completion sentence.  The useful
    self-generated experience lives in the intermediate observations: which
    contract was inspected, which implementation boundary was changed, and
    which focused checks became green.  Persist those decisions as an episodic
    trace while deliberately excluding shell commands and source hunks.
    """
    selected: list[str] = []
    seen: set[str] = set()
    administrative = (
        "git status", "worktree clean", "commit failed", "set repo-local",
        "create feature branch", "confirm worktree", "commit the feature",
    )
    for item in history:
        analysis = _compact(str(item.get("analysis") or ""), 320)
        if not analysis or analysis.lower() in seen:
            continue
        if any(marker in analysis.lower() for marker in administrative):
            continue
        # Retain decisions backed by an executed command or an explicit final
        # completion.  This is an experience summary, not a speculative chain.
        codes = item.get("return_codes") or ()
        if codes and not any(code == 0 for code in codes):
            continue
        if not codes and not item.get("completion"):
            continue
        seen.add(analysis.lower())
        selected.append(analysis)
    return _compact(" -> ".join(selected[-6:]), 1500)


def _patch_blueprint(patch: str) -> str:
    """Distil structural implementation knowledge without retaining patch text."""
    files: list[str] = []
    symbols: list[str] = []
    cases: list[str] = []
    sites: list[str] = []

    def add(target: list[str], value: str, limit: int) -> None:
        value = re.sub(r"\s+", " ", value).strip(" :{(")
        if value and value not in target and len(target) < limit:
            target.append(value)

    declaration_patterns = (
        r"^(?:export\s+)?(?:async\s+)?(?:function|class|interface|type)\s+([A-Za-z_$][\w$]*)",
        r"^(?:func\s+(?:\([^)]*\)\s*)?|def\s+|class\s+)([A-Za-z_][\w]*)",
        r"^(?:public\s+|private\s+|protected\s+|static\s+|async\s+)*([A-Za-z_$][\w$]*)\s*\([^)]*\)\s*\{?$",
    )
    case_pattern = re.compile(r"\b(?:box|comp|def|register|route)\(\s*['\"]([^'\"]+)['\"]")
    for raw in patch.splitlines():
        if raw.startswith("+++ b/"):
            add(files, raw[6:], 12)
            continue
        if raw.startswith("@@"):
            match = re.match(r"@@[^@]*@@\s*(.*)", raw)
            if match:
                add(sites, match.group(1), 12)
            continue
        if not raw.startswith("+") or raw.startswith("+++"):
            continue
        line = raw[1:].strip()
        for pattern in declaration_patterns:
            match = re.match(pattern, line)
            if match:
                add(symbols, match.group(1), 28)
                break
        match = case_pattern.search(line)
        if match:
            add(cases, match.group(1), 32)
        for route in re.findall(r"['\"](/[^'\"]{1,80})['\"]", line):
            add(cases, route, 32)
    parts = []
    if files:
        parts.append("files=" + ", ".join(files))
    if sites:
        parts.append("change-sites=" + ", ".join(sites))
    if symbols:
        parts.append("symbols=" + ", ".join(symbols))
    if cases:
        parts.append("contract-cases=" + ", ".join(cases))
    return _compact("; ".join(parts), 1800)


def _episode(result_path: Path) -> VerifiedEpisode | None:
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    task = str(result.get("task_name") or "")
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    reward = rewards.get("reward")
    if not task.startswith("datacurve/") or not isinstance(reward, (int, float)):
        return None
    trial = result_path.parent
    bridge_path = trial / "agent" / "dsh_bridge.json"
    bridge = {}
    if bridge_path.exists():
        try:
            bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    history = bridge.get("history") or []
    final_analysis = next((str(item.get("analysis") or "") for item in reversed(history)
                           if item.get("analysis")), "")
    patch_path = trial / "artifacts" / "logs" / "artifacts" / "model.patch"
    patch = patch_path.read_text(encoding="utf-8", errors="replace") if patch_path.exists() else ""
    raw_path = trial / "agent" / "dsh_raw.jsonl"
    raw_calls = 0
    if raw_path.exists():
        raw_calls = sum(1 for line in raw_path.read_text(encoding="utf-8", errors="replace").splitlines()
                        if line.strip())
    return VerifiedEpisode(
        task.removeprefix("datacurve/"), float(reward),
        int(rewards.get("f2p_passed") or 0), int(rewards.get("f2p_total") or 0),
        int(rewards.get("p2p_passed") or 0), int(rewards.get("p2p_total") or 0),
        _compact(final_analysis), patch, str(trial), _failed_tests(trial), len(history),
        max(0, raw_calls - len(history)), result_path.stat().st_mtime,
        tuple(item for item in (bridge.get("grounding") or ()) if isinstance(item, dict)),
        _execution_trace(history),
        _patch_blueprint(patch),
    )


def collect_episodes(roots: Iterable[Path]) -> tuple[VerifiedEpisode, ...]:
    by_task: dict[str, VerifiedEpisode] = {}
    for root in roots:
        for result_path in root.rglob("result.json"):
            episode = _episode(result_path)
            if episode is None:
                continue
            previous = by_task.get(episode.task)
            episode_key = (episode.reward,
                           episode.f2p_passed / max(1, episode.f2p_total),
                           episode.p2p_passed / max(1, episode.p2p_total),
                           bool(episode.patch), episode.observed_at)
            previous_key = ((previous.reward,
                             previous.f2p_passed / max(1, previous.f2p_total),
                             previous.p2p_passed / max(1, previous.p2p_total),
                             bool(previous.patch), previous.observed_at)
                            if previous is not None else None)
            if previous is None or episode_key > previous_key:
                by_task[episode.task] = episode
    return tuple(by_task[key] for key in sorted(by_task))


def _changed_files(patch: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(re.findall(r"^\+\+\+ b/(.+)$", patch, re.M)))[:12]


def _atom_id(task: str, kind: str, outcome: str, content: str) -> str:
    digest = hashlib.sha256(f"{task}|{kind}|{outcome}|{content}".encode()).hexdigest()[:12]
    return f"deepswe.{task}.{outcome}.{kind.lower()}.{digest}"


def diagnose_episode(ep: VerifiedEpisode) -> EvolutionDiagnosis:
    return diagnose_verified_outcome(
        reward=ep.reward, f2p_passed=ep.f2p_passed, f2p_total=ep.f2p_total,
        p2p_passed=ep.p2p_passed, p2p_total=ep.p2p_total, patch=ep.patch,
        steps=ep.steps, failed_tests=ep.failed_tests,
        protocol_retries=ep.protocol_retries,
    )


def distill_episode(ep: VerifiedEpisode) -> tuple[KnowledgeAtom, ...]:
    outcome = "pass" if ep.reward == 1 else "fail"
    files = _changed_files(ep.patch)
    tag_text = " ".join((ep.task, ep.final_analysis, ep.implementation_blueprint, " ".join(files)))
    tags = knowledge_tags(tag_text)[:32]
    provenance = f"DeepSWE verifier:{ep.task}"
    f2p = f"{ep.f2p_passed}/{ep.f2p_total}"
    p2p = f"{ep.p2p_passed}/{ep.p2p_total}"
    atoms: list[KnowledgeAtom] = []
    diagnosis = diagnose_episode(ep)
    reflection = reflect_outcome(
        root_kind=diagnosis.root_kind, task=ep.task,
        f2p_passed=ep.f2p_passed, f2p_total=ep.f2p_total,
        p2p_passed=ep.p2p_passed, p2p_total=ep.p2p_total,
        failures=ep.failed_tests,
    )

    failures = "; ".join(ep.failed_tests[:3]) or "no named verifier counterexample"
    perception = (f"For {ep.task}, observe patch materialization, step budget, public integration, "
                  f"and verifier counterexamples before editing. Last evidence: F2P={f2p}, P2P={p2p}, "
                  f"root={diagnosis.root_kind}, examples={failures}. Reflected invariant: {reflection.invariant}")
    atoms.append(KnowledgeAtom(_atom_id(ep.task, "P", outcome, perception), "P", perception,
                               provenance, 1.0 if ep.reward == 1 else .9, tags))

    if ep.reward == 1:
        patch_digest = hashlib.sha256(ep.patch.encode()).hexdigest()[:16]
        memory = (f"Verified episode {ep.task}: F2P={f2p}, P2P={p2p}, "
                  f"patch={patch_digest}, changed_files={list(files)}.")
        trace = ep.execution_trace or ep.final_analysis
        blueprint = ep.implementation_blueprint or f"files={', '.join(files)}"
        reasoning = (f"Verified reasoning pattern for {ep.task}: "
                     f"Implementation blueprint: {blueprint}. Verified trajectory: "
                     f"{trace or 'derive the causal contract, implement it, and verify both new and old behavior.'}")
        fix = (f"Reusable verified repair pattern for {ep.task}: change {list(files) or ['the causal implementation site']}; "
               f"preserve the behaviors summarized by P2P={p2p}. Implementation blueprint: {blueprint}. "
               f"Semantic execution trace: {trace}")
        safety = (f"Promotion evidence for {ep.task} requires both complete new capability F2P={f2p} "
                  f"and retained behavior P2P={p2p}; do not accept self-judgment alone.")
        for kind, content, confidence in (("M", memory, 1.0), ("R", reasoning, .95),
                                          ("F", fix, 1.0), ("S", safety, 1.0)):
            content = _compact(content, 1500 if kind in {"R", "F"} else 700)
            atoms.append(KnowledgeAtom(_atom_id(ep.task, kind, outcome, content), kind,
                                       content, provenance, confidence, tags))
    else:
        negative_tags = tags + ("negative-evidence", diagnosis.root_kind, diagnosis.root_stage)
        reasoning = _compact(
            f"RCA for {ep.task}: root={diagnosis.root_kind} at {diagnosis.root_stage} "
            f"(confidence={diagnosis.confidence:.2f}). {diagnosis.explanation} "
            f"Counterexample invariant: {reflection.invariant}"
        )
        recovery = _compact(
            f"RCA recovery policy for {ep.task}: {diagnosis.recovery} Reflection strategy: "
            f"{reflection.next_strategy} Alternative hypotheses: {list(reflection.alternative_hypotheses)}"
        )
        safety = _compact(f"RCA transition guard for {ep.task}: {diagnosis.safety}")
        for kind, content in (("R", reasoning), ("F", recovery), ("S", safety)):
            atoms.append(KnowledgeAtom(_atom_id(ep.task, kind, outcome, content), kind,
                                       content, provenance, diagnosis.confidence, negative_tags))
    return tuple(atoms)


def _merge_grounding_history(store_path: Path, store: VersionedKnowledgeStore,
                             episodes: tuple[VerifiedEpisode, ...]) -> tuple[GroundingRecord, ...]:
    root = store_path.parent / "ltm"
    existing: tuple[GroundingRecord, ...] = ()
    if (root / "CURRENT.json").exists():
        try:
            existing = IndexedLTM.load(root, expected_version=store.active.version).groundings
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
            existing = ()
    by_id = {item.grounding_id: item for item in existing}
    for ep in episodes:
        for raw in ep.groundings:
            try:
                grounding_id = str(raw["grounding_id"])
                item = GroundingRecord(
                    grounding_id=grounding_id,
                    abstract_id=str(raw["abstract_id"]),
                    task=ep.task,
                    repository=str(raw.get("repository") or ep.task.split("-", 1)[0]),
                    capability=str(raw.get("capability") or "repository-contract"),
                    context_digest=str(raw.get("context_digest") or ""),
                    activation_evidence=tuple(raw.get("activation_evidence") or ()),
                    bindings={str(k): str(v) for k, v in (raw.get("bindings") or {}).items()},
                    instantiated_action=tuple(raw.get("instantiated_action") or ()),
                    status="verified" if ep.reward == 1 else "rejected",
                    reward=ep.reward,
                    source=ep.source,
                    source_atom_ids=tuple(raw.get("source_atom_ids") or ()),
                    scope_guard=tuple(raw.get("scope_guard") or ()),
                    verification_plan=tuple(raw.get("verification_plan") or ()),
                    materialization_protocol=tuple(raw.get("materialization_protocol") or ()),
                    outcome_evidence=(
                        f"reward={ep.reward:g}; F2P={ep.f2p_passed}/{ep.f2p_total}; "
                        f"P2P={ep.p2p_passed}/{ep.p2p_total}",
                        *ep.failed_tests[:3],
                    ),
                )
            except (KeyError, TypeError, ValueError):
                continue
            by_id[grounding_id] = item
    return tuple(by_id[key] for key in sorted(by_id))


EvolutionPolicy = Literal[
    "standard", "positive_only", "quarantine_negative", "grounding_feedback",
    "grounding_observation",
]


def evolve(store_path: Path, result_roots: Iterable[Path],
           policy: EvolutionPolicy = "standard") -> dict:
    roots = tuple(result_roots)
    collected = collect_episodes(roots)
    if not collected:
        raise RuntimeError("no independently verified DeepSWE episodes found")
    if policy not in {
        "standard", "positive_only", "quarantine_negative", "grounding_feedback",
        "grounding_observation",
    }:
        raise ValueError(f"unknown evolution policy: {policy}")
    store = VersionedKnowledgeStore.from_file(store_path)
    if policy == "standard":
        episodes = collected
        candidate_episodes = collected
    elif policy == "positive_only":
        episodes = tuple(ep for ep in collected if ep.reward == 1)
        candidate_episodes = episodes
    elif policy == "grounding_observation":
        # Frozen evaluation is evidence about whether an existing grounding was
        # successfully re-instantiated.  It must not manufacture another copy
        # of task knowledge merely because the task passed again.
        episodes = ()
        candidate_episodes = collected
    else:
        episodes = tuple(ep for ep in collected if ep.reward == 1)
        candidate_episodes = collected
    candidate_updates = update_candidate_registry(
        store_path.with_name("candidates.json"), candidate_episodes
    )
    quarantined = sum(ep.reward != 1 for ep in collected)
    grounding_policies = {"grounding_feedback", "grounding_observation"}
    if not episodes and policy not in grounding_policies:
        return {
            "status": "quarantined-no-promotion" if policy == "quarantine_negative" else "no-positive-evidence",
            "policy": policy,
            "version": store.active.version,
            "episodes": len(collected),
            "passes": 0,
            "new_atoms": 0,
            "candidate_updates": candidate_updates,
            "quarantined_episodes": quarantined,
            "path": str(store_path),
        }
    grounding_episodes = collected if policy in grounding_policies else episodes
    grounding_history = _merge_grounding_history(store_path, store, grounding_episodes)
    graph_path = store_path.with_name("grounding_graph.json")
    graph = GroundedKnowledgeGraph.from_file(graph_path)
    if not episodes:
        indexed = IndexedLTM.build(
            atoms=store.active.atoms, version=store.active.version,
            parent=store.active.parent, graph=graph, groundings=grounding_history,
        )
        indexed_path = indexed.commit(store_path.parent / "ltm")
        return {
            "status": "grounding-consolidated-no-promotion", "policy": policy,
            "version": store.active.version, "episodes": len(collected), "passes": 0,
            "new_atoms": 0, "candidate_updates": candidate_updates,
            "quarantined_episodes": quarantined, "path": str(store_path),
            "grounding_graph": str(graph_path), "graph_nodes": len(graph.nodes),
            "indexed_ltm": str(indexed_path), "retention_tasks": len(indexed.anchors),
            "grounding_feedback": len(grounding_history),
        }
    existing = {atom.atom_id for atom in store.active.atoms}
    episode_atoms = tuple((ep, distill_episode(ep)) for ep in episodes)
    for ep, atoms in episode_atoms:
        diagnosis = diagnose_episode(ep)
        capability = capability_family(
            " ".join((ep.task, ep.final_analysis, " ".join(_changed_files(ep.patch))))
        )
        for atom in atoms:
            graph.register_task_atom(
                atom,
                task=ep.task,
                capability=capability,
                root_kind=diagnosis.root_kind,
                positive=ep.reward == 1,
                source=ep.source,
            )

    abstraction_atoms: list[KnowledgeAtom] = []
    groups: dict[tuple[str, str], dict[str, set[str]]] = {}
    for atom_id, node in graph.nodes.items():
        if node.get("level") != "task":
            continue
        scope = node.get("scope") or {}
        key = (str(scope.get("capability") or "repository-contract"),
               str(scope.get("failure_mode") or "unknown"))
        group = groups.setdefault(key, {"positive": set(), "negative": set(), "children": set()})
        group["positive"].update(node.get("positive_groundings") or ())
        group["negative"].update(node.get("negative_groundings") or ())
        group["children"].add(atom_id)
    for (capability, root_kind), group in groups.items():
        positives = group["positive"]
        if not positives:
            continue
        negatives = group["negative"] - positives
        atom = abstraction_atom(capability, root_kind)
        graph.register_abstraction(
            atom,
            capability=capability,
            root_kind=root_kind,
            positive_tasks=positives,
            negative_tasks=negatives,
            child_ids=group["children"],
        )
        abstraction_atoms.append(atom)

    distilled = tuple(atom for _, atoms in episode_atoms for atom in atoms) + tuple(abstraction_atoms)
    new_atoms = tuple(atom for atom in distilled if atom.atom_id not in existing)
    graph.save(graph_path)
    if not new_atoms:
        indexed = IndexedLTM.build(
            atoms=store.active.atoms, version=store.active.version,
            parent=store.active.parent, graph=graph, groundings=grounding_history,
        )
        indexed_path = indexed.commit(store_path.parent / "ltm")
        return {"status": "consolidated", "policy": policy,
                "version": store.active.version, "episodes": len(collected), "new_atoms": 0,
                "candidate_updates": candidate_updates, "path": str(store_path),
                "quarantined_episodes": quarantined,
                "grounding_graph": str(graph_path), "graph_nodes": len(graph.nodes),
                "indexed_ltm": str(indexed_path), "retention_tasks": len(indexed.anchors)}
    for atom in new_atoms:
        store.stage(atom)
    evidence = hashlib.sha256("\n".join(
        f"{ep.task}:{ep.reward}:{ep.f2p_passed}/{ep.f2p_total}:{ep.p2p_passed}/{ep.p2p_total}"
        for ep in episodes
    ).encode()).hexdigest()
    version = store.commit(evidence, {
        "profile": "RCA-distilled-self-evolution",
        "knowledge_policy": f"abstraction-grounding-v1:{policy}",
        "episodes": len(collected),
        "passes": sum(ep.reward == 1 for ep in episodes),
        "new_atoms": len(new_atoms),
        "sources": [str(root) for root in roots],
        "committed_at": datetime.now(timezone.utc).isoformat(),
    })
    store.save_active(store_path)
    indexed = IndexedLTM.build(
        atoms=version.atoms, version=version.version, parent=version.parent, graph=graph,
        groundings=grounding_history,
    )
    indexed_path = indexed.commit(store_path.parent / "ltm")
    return {"status": "promoted", "policy": policy,
            "version": version.version, "episodes": len(collected),
            "passes": sum(ep.reward == 1 for ep in episodes),
            "new_atoms": len(new_atoms), "candidate_updates": candidate_updates,
            "quarantined_episodes": quarantined,
            "path": str(store_path), "grounding_graph": str(graph_path),
            "graph_nodes": len(graph.nodes), "indexed_ltm": str(indexed_path),
            "retention_tasks": len(indexed.anchors)}


def update_candidate_registry(path: Path, episodes: Iterable[VerifiedEpisode]) -> int:
    """Quarantine partial patches for same-task continuation, never skill promotion."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {"schema_version": 1, "tasks": {}}
    tasks = payload.setdefault("tasks", {})
    updates = 0
    for ep in episodes:
        if ep.reward == 1:
            if ep.task in tasks:
                del tasks[ep.task]
                updates += 1
            continue
        if not ep.patch.strip() or ep.f2p_passed <= 0:
            continue
        diagnosis = diagnose_episode(ep)
        reflection = reflect_outcome(
            root_kind=diagnosis.root_kind, task=ep.task,
            f2p_passed=ep.f2p_passed, f2p_total=ep.f2p_total,
            p2p_passed=ep.p2p_passed, p2p_total=ep.p2p_total,
            failures=ep.failed_tests,
        )
        candidate = {
            "status": "quarantined-partial", "task": ep.task, "patch": ep.patch,
            "patch_digest": hashlib.sha256(ep.patch.encode()).hexdigest(),
            "f2p_passed": ep.f2p_passed, "f2p_total": ep.f2p_total,
            "p2p_passed": ep.p2p_passed, "p2p_total": ep.p2p_total,
            "root_kind": diagnosis.root_kind, "source": ep.source,
            "observed_at": ep.observed_at, "result_signature": reflection.result_signature,
            "reflection": reflection.json(),
        }
        old = tasks.get(ep.task) or {}
        if old.get("candidates"):
            candidates = list(old["candidates"])
        elif old.get("patch"):
            candidates = [dict(old)]
        else:
            candidates = []
        prior_active = old.get("active_patch_digest") or old.get("patch_digest")
        if prior_active:
            candidate["parent_patch_digest"] = prior_active
        existing_index = next((i for i, item in enumerate(candidates)
                               if item.get("patch_digest") == candidate["patch_digest"]), None)
        if existing_index is None:
            candidates.append(candidate)
        else:
            candidates[existing_index] = candidate

        def quality(item: dict) -> tuple[float, float]:
            return (item.get("f2p_passed", 0) / max(1, item.get("f2p_total", 1)),
                    item.get("p2p_passed", 0) / max(1, item.get("p2p_total", 1)))

        pareto = []
        for item in candidates:
            q = quality(item)
            dominated = any(
                quality(other)[0] >= q[0] and quality(other)[1] >= q[1] and
                quality(other) != q for other in candidates if other is not item
            )
            if not dominated:
                pareto.append(item)
        pareto.sort(key=lambda item: (*quality(item), item.get("observed_at", 0)), reverse=True)
        pareto = pareto[:3]
        active = pareto[0]
        previous_signature = old.get("last_signature")
        previous_observed = float(old.get("last_observed_at", 0))
        is_new_episode = ep.observed_at > previous_observed
        stagnation = int(old.get("stagnation_count", 0))
        if is_new_episode:
            stagnation = stagnation + 1 if previous_signature == reflection.result_signature else 0
        updated = {
            "status": "quarantined-portfolio", "task": ep.task,
            "active_patch_digest": active["patch_digest"], "candidates": pareto,
            "last_signature": reflection.result_signature,
            "last_observed_at": max(previous_observed, ep.observed_at),
            "stagnation_count": stagnation, "reflection": reflection.json(),
        }
        if updated != old:
            tasks[ep.task] = updated
            updates += 1
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(path, payload)
    return updates
