from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass
from typing import Any

from .grounding import CAPABILITY_PATTERNS, task_matches
from .indexed_ltm import (
    AbstractKnowledge,
    AtomRecord,
    GroundingRecord,
    IndexedLTM,
    RetrievalBundle,
    _CAUSAL_CUES,
    _RELATION_STOP_TAGS,
    _capability_affinities,
    _causal_loci,
    _extract_live_symbols,
    _semantic_episode_projection,
    _situation_capability_affinities,
)
from .store import KnowledgeAtom, knowledge_tags


@dataclass(frozen=True)
class SleepConsolidatedSchema:
    """NREM/REM sleep-consolidated schema with contrastive boundary guards and synaptic weight."""

    abstract_id: str
    capability: str
    failure_mode: str
    causal_locus: str
    discriminative_cues: tuple[str, ...]
    anti_pattern_cues: tuple[str, ...]
    synthesized_invariant: str
    contrastive_guard: str
    action_schema: tuple[str, ...]
    synaptic_weight: float
    vigilance_threshold: float
    pruned: bool
    positive_tasks: tuple[str, ...]
    negative_tasks: tuple[str, ...]
    best_source_task: str
    source_atom_ids: tuple[str, ...]

    def json(self) -> dict[str, Any]:
        return asdict(self)


class SleepConsolidationMemory:
    """Paradigm 2: NREM/REM Dual-Phase Sleep Consolidation & Contrastive Boundary Pruning.

    Inspired by Complementary Learning Systems (CLS) Sleep Replay (McClelland et al.)
    and Synaptic Homeostasis / Downscaling (Tononi & Cirelli):

    1. NREM Slow-Wave Contrastive Replay:
       - Replays positive (verified pass) episodes against negative / near-miss (fail)
         episodes within the same capability and causal locus.
       - Computes TF-IDF / log-odds discriminative cues for each schema so ubiquitous
         boilerplate words are downweighted while mechanism-specific cues are amplified.
       - Synthesizes an explicit Contrastive Boundary Guard from paired fail/near-miss
         atoms (`P/R/S`) so the agent knows what near-miss trap to avoid.
    2. REM Synaptic Downscaling & Pruning:
       - Computes a synaptic consolidation weight from source task diversity, verifier
         grounding reliability, and cue specificity.
       - Prunes weak, diffuse, or net-negative schemas (`pruned=True`) and calibrates
         a vigilance threshold that prevents cross-task negative transfer.
    """

    def __init__(self, ltm: IndexedLTM) -> None:
        self.ltm = ltm
        self.schemas: dict[str, SleepConsolidatedSchema] = {}
        self.task_near_miss_guards: dict[str, tuple[KnowledgeAtom, ...]] = {}
        self._consolidate()

    def _consolidate(self) -> None:
        records = self.ltm.records
        abstractions = self.ltm.abstractions

        # 1. Compute global tag document frequency across tasks for IDF weighting
        task_tag_sets: dict[str, set[str]] = {}
        for rec in records.values():
            if not rec.task:
                continue
            tags = set(rec.atom.tags or knowledge_tags(rec.atom.content)) - _RELATION_STOP_TAGS
            task_tag_sets.setdefault(rec.task, set()).update(tags)
        total_tasks = max(1, len(task_tag_sets))
        tag_df: dict[str, int] = {}
        for tags in task_tag_sets.values():
            for t in tags:
                tag_df[t] = tag_df.get(t, 0) + 1

        def idf(tag: str) -> float:
            df = tag_df.get(tag, 1)
            return math.log(1.0 + total_tasks / max(1, df))

        # 2. Collect task-local near-miss / negative R/F/S atoms for contrastive replay
        by_cap_locus_fail: dict[tuple[str, str], list[AtomRecord]] = {}
        by_task_fail: dict[str, list[AtomRecord]] = {}
        for rec in records.values():
            if rec.outcome == "fail" and rec.task:
                by_task_fail.setdefault(rec.task, []).append(rec)
                locus = rec.causal_loci[0] if rec.causal_loci else "unspecified"
                by_cap_locus_fail.setdefault((rec.capability, locus), []).append(rec)

        for task_name, fail_recs in by_task_fail.items():
            ordered = sorted(
                fail_recs,
                key=lambda r: (
                    {"R": 0, "F": 1, "S": 2, "P": 3}.get(r.atom.kind, 9),
                    -r.atom.confidence,
                    r.atom.atom_id,
                ),
            )
            chosen_guards: list[KnowledgeAtom] = []
            for kind in ("R", "F", "S"):
                m = next((r.atom for r in ordered if r.atom.kind == kind), None)
                if m is not None:
                    chosen_guards.append(m)
            if chosen_guards:
                self.task_near_miss_guards[task_name] = tuple(chosen_guards)

        # 3. Grounding verification history
        grounding_stats: dict[str, tuple[int, int]] = {}
        for gr in self.ltm.groundings:
            v, r = grounding_stats.get(gr.abstract_id, (0, 0))
            if gr.status == "verified":
                v += 1
            elif gr.status == "rejected":
                r += 1
            grounding_stats[gr.abstract_id] = (v, r)

        # 4. NREM Contrastive Replay & REM Synaptic Downscaling per abstraction
        for aid, abstract in sorted(abstractions.items()):
            support_recs = [
                records[atom_id]
                for atom_id in abstract.supporting_atom_ids
                if atom_id in records and records[atom_id].task
            ]
            pass_recs = [r for r in support_recs if r.outcome == "pass"]
            fail_recs = [r for r in support_recs if r.outcome == "fail"]
            if not fail_recs:
                fail_recs = by_cap_locus_fail.get(
                    (abstract.capability, abstract.causal_locus), []
                )

            pass_tasks = sorted({r.task for r in pass_recs})
            fail_tasks = sorted({r.task for r in fail_recs} - set(pass_tasks))

            # NREM: score positive cues by (task coverage within module) * IDF
            task_tokens = {tok for r in support_recs for tok in knowledge_tags(r.task)}
            pos_tag_tasks: dict[str, set[str]] = {}
            for r in pass_recs:
                tags = (
                    set(r.atom.tags or knowledge_tags(r.atom.content))
                    - task_tokens
                    - _RELATION_STOP_TAGS
                )
                for t in tags:
                    pos_tag_tasks.setdefault(t, set()).add(r.task)

            scored_cues = sorted(
                (
                    (len(tasks_set) * idf(tag), len(tasks_set), tag)
                    for tag, tasks_set in pos_tag_tasks.items()
                    if len(tasks_set) >= min(2, max(1, len(pass_tasks)))
                ),
                key=lambda item: (-item[0], -item[1], item[2]),
            )
            discriminative_cues = tuple(tag for _, _, tag in scored_cues[:8])
            if not discriminative_cues and abstract.trigger_tags:
                discriminative_cues = abstract.trigger_tags[:6]

            # Extract anti-pattern cues from negative episodes that never succeeded in this module
            neg_tag_counts: dict[str, int] = {}
            for r in fail_recs:
                if r.task in pass_tasks:
                    continue
                tags = (
                    set(r.atom.tags or knowledge_tags(r.atom.content))
                    - task_tokens
                    - _RELATION_STOP_TAGS
                    - set(discriminative_cues)
                )
                for t in tags:
                    neg_tag_counts[t] = neg_tag_counts.get(t, 0) + 1
            anti_cues = tuple(
                tag
                for tag, _ in sorted(
                    neg_tag_counts.items(), key=lambda kv: (-kv[1], kv[0])
                )[:5]
            )

            # Synthesize contrastive boundary guard from fail/near-miss S/R atoms
            guard_snippets: list[str] = []
            for r in sorted(
                fail_recs,
                key=lambda rec: (0 if rec.atom.kind == "S" else 1, -rec.atom.confidence),
            )[:2]:
                snippet = _semantic_episode_projection(r.atom.content, limit=180)
                if snippet and snippet not in guard_snippets:
                    guard_snippets.append(snippet)
            if guard_snippets:
                contrastive_guard = "CONTRASTIVE BOUNDARY GUARD: " + " | ".join(guard_snippets)
            else:
                contrastive_guard = (
                    f"CONTRASTIVE BOUNDARY GUARD: verify {abstract.causal_locus} closure "
                    "on both focused counterexample and retained P2P suite before broadening."
                )

            # Select best passing source task and its R/F/S trajectory
            by_source_task: dict[str, list[AtomRecord]] = {}
            for r in pass_recs:
                by_source_task.setdefault(r.task, []).append(r)

            best_source_task = ""
            source_atom_ids: tuple[str, ...] = ()
            synthesized_invariant = abstract.invariant
            if by_source_task:
                best_source_task, best_recs = sorted(
                    by_source_task.items(),
                    key=lambda kv: (
                        -len({r.atom.kind for r in kv[1]}.intersection({"R", "F", "S"})),
                        -len(kv[1]),
                        kv[0],
                    ),
                )[0]
                role_order = {"R": 0, "F": 1, "S": 2, "P": 3, "M": 4}
                best_recs = sorted(
                    best_recs,
                    key=lambda r: (role_order.get(r.atom.kind, 9), -r.atom.confidence, r.atom.atom_id),
                )
                chosen: list[AtomRecord] = []
                for kind in ("R", "F", "S"):
                    m = next((r for r in best_recs if r.atom.kind == kind), None)
                    if m is not None:
                        chosen.append(m)
                if not chosen:
                    chosen = best_recs[:2]
                source_atom_ids = tuple(r.atom.atom_id for r in chosen)
                r_atom = next((r for r in chosen if r.atom.kind == "R"), None)
                if r_atom and abstract.invariant.startswith("Preserve the verified"):
                    proj = _semantic_episode_projection(r_atom.atom.content, limit=200)
                    synthesized_invariant = (
                        f"Sleep-consolidated {abstract.capability}/{abstract.causal_locus} invariant: {proj}"
                    )

            # REM Synaptic Downscaling: compute synaptic weight and prune noisy schemas
            verified, rejected = grounding_stats.get(aid, (0, 0))
            diversity_factor = min(1.0, len(pass_tasks) / 2.0)
            cue_specificity = min(1.0, len(discriminative_cues) / 4.0)
            if abstract.invariant.startswith("Preserve the verified"):
                cue_specificity *= 0.65
            reliability = (verified + 1.0) / (verified + rejected + 2.0)

            synaptic_weight = round(
                0.35 * diversity_factor + 0.35 * cue_specificity + 0.30 * reliability,
                4,
            )
            # Prune if there are no passing tasks, or net-negative verification,
            # or diffuse unspecified causal locus without multi-task validation.
            pruned = (
                not pass_tasks
                or (rejected > verified and verified == 0)
                or (abstract.causal_locus == "unspecified" and len(pass_tasks) < 2)
                or synaptic_weight < 0.28
            )
            # Calibrate vigilance threshold: less-proven schemas require stronger cue resonance
            vigilance_threshold = round(
                0.52 - 0.12 * min(1.0, len(pass_tasks) / 3.0) + 0.08 * min(2, rejected),
                4,
            )

            episode_roles = "; ".join(
                f"{records[id_].atom.kind}={_semantic_episode_projection(records[id_].atom.content)}"
                for id_ in source_atom_ids
                if id_ in records
            )
            actions = (
                f"SLEEP-CONSOLIDATED SCHEMA ({abstract.capability}/{abstract.causal_locus}): {synthesized_invariant}",
                f"NREM REPLAYED SOURCE ({best_source_task}): {episode_roles}",
                contrastive_guard,
            )

            self.schemas[aid] = SleepConsolidatedSchema(
                abstract_id=aid,
                capability=abstract.capability,
                failure_mode=abstract.failure_mode,
                causal_locus=abstract.causal_locus,
                discriminative_cues=discriminative_cues,
                anti_pattern_cues=anti_cues,
                synthesized_invariant=synthesized_invariant,
                contrastive_guard=contrastive_guard,
                action_schema=tuple(re.sub(r"\s+", " ", a).strip()[:240] for a in actions),
                synaptic_weight=synaptic_weight,
                vigilance_threshold=vigilance_threshold,
                pruned=pruned,
                positive_tasks=tuple(pass_tasks),
                negative_tasks=tuple(fail_tasks),
                best_source_task=best_source_task,
                source_atom_ids=source_atom_ids,
            )

    def ground(
        self,
        *,
        task_id: str,
        task_text: str,
        repository_evidence: str = "",
        limit: int = 14,
    ) -> tuple[RetrievalBundle, tuple[GroundingRecord, ...], dict[str, Any]]:
        """Retrieve CLS layered bundle + REM-pruned, NREM-contrastive grounded schema."""
        bundle = self.ltm.retrieve(
            task_id=task_id,
            task_text=task_text,
            repository_evidence=repository_evidence,
            limit=limit,
            policy="sleep_consolidation_v24",
        )
        task_key = self.ltm._matching_task_key(task_id)
        task_records = tuple(
            self.ltm.records[atom_id]
            for atom_id in self.ltm.indexes["task"].get(task_key, ())
            if atom_id in self.ltm.records
        )
        has_verified_task_memory = any(r.outcome == "pass" for r in task_records)
        delta_ids = set(
            (self.ltm.manifest.get("memory_layers") or {}).get("delta_atom_ids") or ()
        )
        has_exact_fast_delta = any(
            r.atom.atom_id in delta_ids and r.outcome == "pass" for r in task_records
        )

        material = f"{task_id} {task_text} {repository_evidence}".strip()
        query_tokens = set(knowledge_tags(material))
        live_tokens = set(knowledge_tags(repository_evidence)) if repository_evidence.strip() else set()
        affinities = _situation_capability_affinities(task_id, task_text)
        if repository_evidence.strip():
            live_aff = _capability_affinities(repository_evidence)
            if max(live_aff.values(), default=0.0) > 0:
                affinities = {
                    k: min(1.0, 0.55 * affinities.get(k, 0.0) + 0.45 * live_aff.get(k, 0.0))
                    for k, _ in CAPABILITY_PATTERNS
                }
        observed_loci = set(_causal_loci(material))
        live_symbols = _extract_live_symbols(repository_evidence)

        # If the task already has a verified exact-task Fast Delta episode in CLS,
        # REM downscaling suppresses cross-task schemas unless concrete live counterevidence
        # demands external rescue. This protects exact fast episodes (e.g., cattrs/cliffy)
        # from cross-task interference!
        if has_exact_fast_delta and "METACORSI_CONFIRMED_FAILURE_EVIDENCE" not in repository_evidence:
            return bundle, (), {
                "paradigm": "sleep_consolidation_v24",
                "status": "rem_suppressed_for_exact_fast_delta",
                "pruned_count": sum(1 for s in self.schemas.values() if s.pruned),
                "active_schemas": len(self.schemas),
            }

        candidates: list[tuple[float, SleepConsolidatedSchema, tuple[str, ...]]] = []
        for schema in self.schemas.values():
            if schema.pruned:
                continue
            if any(task_matches(t, task_id) for t in schema.positive_tasks):
                continue

            cap_score = affinities.get(schema.capability, 0.0)
            locus_score = (
                1.0
                if (schema.causal_locus != "unspecified" and schema.causal_locus in observed_loci)
                else (0.25 if schema.causal_locus == "unspecified" else 0.0)
            )
            matched_pos = tuple(sorted(query_tokens.intersection(schema.discriminative_cues)))
            matched_neg = tuple(sorted(query_tokens.intersection(schema.anti_pattern_cues)))
            cue_denom = max(1, min(3, len(schema.discriminative_cues)))
            cue_score = min(1.0, len(matched_pos) / cue_denom)
            anti_penalty = 0.08 * len(matched_neg)

            resonance = (
                0.30 * cap_score
                + 0.28 * locus_score
                + 0.24 * cue_score
                + 0.18 * schema.synaptic_weight
                - anti_penalty
            )
            # Vigilance gating: reject schemas that do not cross their REM-calibrated threshold
            # or have zero discriminative cue overlap.
            effective_vigilance = (
                schema.vigilance_threshold + 0.08
                if has_verified_task_memory
                else schema.vigilance_threshold
            )
            if len(matched_pos) == 0 or resonance < effective_vigilance:
                continue
            candidates.append((resonance, schema, matched_pos))

        candidates.sort(key=lambda item: (-item[0], item[1].abstract_id))
        if not candidates:
            return bundle, (), {
                "paradigm": "sleep_consolidation_v24",
                "status": "below_rem_vigilance_threshold",
                "pruned_count": sum(1 for s in self.schemas.values() if s.pruned),
            }

        top_score, top_schema, matched_cues = candidates[0]
        context_digest = hashlib.sha256(material.encode()).hexdigest()
        repository = task_id.split("-", 1)[0] if "-" in task_id else task_id
        base_residual = 0.20 if has_verified_task_memory else 0.55
        local_weight = 1.0 - base_residual

        bindings = {
            "task": task_id,
            "repository": repository,
            "capability": top_schema.capability,
            "causal_locus": top_schema.causal_locus,
            "failure_mode": top_schema.failure_mode,
            "grounding_weight": f"{top_score:.3f}",
            "synaptic_weight": f"{top_schema.synaptic_weight:.3f}",
            "vigilance_threshold": f"{top_schema.vigilance_threshold:.3f}",
            "target_roles": ",".join(matched_cues),
            "live_symbols": ",".join(live_symbols) or "pending-observation",
            "source_episode": top_schema.best_source_task,
            "local_episode_weight": f"{local_weight:.3f}",
            "abstract_residual_weight": f"{base_residual:.3f}",
            "grounded_atom_ids": ",".join(top_schema.source_atom_ids),
            "memory_strategy": "sleep_consolidation",
            "module": f"{top_schema.capability}/{top_schema.causal_locus}",
        }
        signature = f"{top_schema.abstract_id}|{task_id}|{context_digest}|sleep"
        grounding_id = "grounding." + hashlib.sha256(signature.encode()).hexdigest()[:20]
        record = GroundingRecord(
            grounding_id=grounding_id,
            abstract_id=top_schema.abstract_id,
            task=task_id,
            repository=repository,
            capability=top_schema.capability,
            context_digest=context_digest,
            activation_evidence=matched_cues[:3],
            bindings=bindings,
            instantiated_action=top_schema.action_schema,
            source_atom_ids=top_schema.source_atom_ids,
            scope_guard=(top_schema.contrastive_guard[:240],),
            verification_plan=(
                "build/typecheck succeeds",
                "one focused counterexample turns green",
                "independent F2P and P2P verifier decides promotion",
            ),
            materialization_protocol=(
                "Use a direct edit command; do not spend a turn planning shell quoting.",
                "After each edit run git diff --check plus repository build or typecheck before expanding.",
            ),
        )
        return bundle, (record,), {
            "paradigm": "sleep_consolidation_v24",
            "status": "activated",
            "resonance": round(top_score, 4),
            "vigilance_threshold": top_schema.vigilance_threshold,
            "synaptic_weight": top_schema.synaptic_weight,
            "pruned_count": sum(1 for s in self.schemas.values() if s.pruned),
        }
