from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any

from .grounding import CAPABILITY_PATTERNS, task_matches
from .indexed_ltm import (
    AtomRecord,
    GroundingRecord,
    IndexedLTM,
    RetrievalBundle,
    _capability_affinities,
    _causal_loci,
    _extract_live_symbols,
    _semantic_episode_projection,
    _situation_capability_affinities,
)
from .store import knowledge_tags


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


# Complementary causal-locus pairs that frequently co-occur in multi-constraint repairs
# (e.g. AST normalization + boundary decision in Bandit; state propagation + validation
# preservation in Arktype; dispatch registration + state propagation in Aiomonitor).
_COMPLEMENTARY_LOCI_PAIRS: dict[tuple[str, str], float] = {
    ("state-propagation", "validation-preservation"): 0.85,
    ("validation-preservation", "state-propagation"): 0.85,
    ("normalization", "boundary-decision"): 0.85,
    ("boundary-decision", "normalization"): 0.85,
    ("dispatch-registration", "state-propagation"): 0.80,
    ("state-propagation", "dispatch-registration"): 0.80,
    ("boundary-decision", "validation-preservation"): 0.75,
    ("validation-preservation", "boundary-decision"): 0.75,
    ("normalization", "validation-preservation"): 0.75,
    ("validation-preservation", "normalization"): 0.75,
    ("dispatch-registration", "boundary-decision"): 0.70,
    ("boundary-decision", "dispatch-registration"): 0.70,
}


@dataclass(frozen=True)
class SynapticEdge:
    source_id: str
    target_id: str
    relation: str  # "CO_ACTIVATES" | "MUTUAL_INHIBITS"
    weight: float

    def json(self) -> dict[str, Any]:
        return asdict(self)


class EngramEnsembleMemory:
    """Paradigm 4: Hippocampal Engram Ensemble Competition & Spreading Activation.

    Inspired by Engram Ensemble Competition & Co-Allocation (Josselyn & Tonegawa, Science 2020):
    - Single Top-1 schema retrieval fails on compound bugs that span two orthogonal causal
      boundaries (such as Arktype recursive `$ref` requiring both `state-propagation` identity
      and `validation-preservation` SCC closure, or Bandit `# nosec` requiring both AST
      `normalization` statement range and `boundary-decision` directive propagation).
    - Builds an Associative Engram Graph over AbstractKnowledge nodes with:
      1. `CO_ACTIVATES` (excitatory synergy edges) between engrams with ORTHOGONAL causal
         loci that share complementary repair roles or bridge tags.
      2. `MUTUAL_INHIBITS` (lateral inhibitory edges) between engrams competing for the
         SAME causal locus, enforcing sharp Winner-Take-All within each locus.
    - Runs 2-step Spreading Activation:
          a_i^(k+1) = clip( d_i + lambda_co * sum_{j} W^co_{ij} a_j^(k)
                                - lambda_inh * sum_{m} W^inh_{im} a_m^(k) )
    - Selects either a single dominant engram OR a 2-engram Orthogonal Complementary
      Ensemble (Primary Engram E_1 + Synergistic Co-Engram E_2 at an orthogonal locus)
      and synthesizes a unified two-stage repair plan.
    """

    def __init__(
        self,
        ltm: IndexedLTM,
        *,
        lambda_co: float = 0.24,
        lambda_inh: float = 0.35,
        primary_threshold: float = 0.42,
        co_engram_threshold: float = 0.44,
    ) -> None:
        self.ltm = ltm
        self.lambda_co = float(lambda_co)
        self.lambda_inh = float(lambda_inh)
        self.primary_threshold = float(primary_threshold)
        self.co_engram_threshold = float(co_engram_threshold)
        self.co_edges: dict[str, list[SynapticEdge]] = {}
        self.inh_edges: dict[str, list[SynapticEdge]] = {}
        self.engram_cues: dict[str, tuple[str, ...]] = {}
        self._build_engram_graph()

    def _cues_for_abstract(self, abstract: Any) -> tuple[str, ...]:
        if abstract.trigger_tags:
            return abstract.trigger_tags
        from .indexed_ltm import _RELATION_STOP_TAGS

        task_tokens = {
            tok
            for atom_id in abstract.supporting_atom_ids
            if atom_id in self.ltm.records
            for tok in knowledge_tags(self.ltm.records[atom_id].task)
        }
        counts: dict[str, int] = {}
        for atom_id in abstract.supporting_atom_ids:
            rec = self.ltm.records.get(atom_id)
            if not rec or rec.outcome != "pass":
                continue
            tags = (
                set(rec.atom.tags or knowledge_tags(rec.atom.content))
                - task_tokens
                - _RELATION_STOP_TAGS
            )
            for t in tags:
                counts[t] = counts.get(t, 0) + 1
        return tuple(
            t for t, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
        )

    def _build_engram_graph(self) -> None:
        abstractions = list(self.ltm.abstractions.values())
        for a in abstractions:
            self.engram_cues[a.abstract_id] = self._cues_for_abstract(a)
        for i, a in enumerate(abstractions):
            if not a.positive_tasks:
                continue
            a_cues = set(self.engram_cues.get(a.abstract_id, ()))
            for b in abstractions[i + 1 :]:
                if not b.positive_tasks:
                    continue
                b_cues = set(self.engram_cues.get(b.abstract_id, ()))
                shared_tags = a_cues.intersection(b_cues)
                shared_tasks = set(a.positive_tasks).intersection(b.positive_tasks)

                if a.causal_locus == b.causal_locus and a.causal_locus != "unspecified":
                    # Same causal locus: competitive lateral inhibition (Winner-Take-All per locus)
                    same_cap_boost = 0.25 if a.capability == b.capability else 0.0
                    w_inh = _clip01(0.45 + same_cap_boost + 0.10 * min(3, len(shared_tags)))
                    e1 = SynapticEdge(a.abstract_id, b.abstract_id, "MUTUAL_INHIBITS", round(w_inh, 4))
                    e2 = SynapticEdge(b.abstract_id, a.abstract_id, "MUTUAL_INHIBITS", round(w_inh, 4))
                    self.inh_edges.setdefault(a.abstract_id, []).append(e1)
                    self.inh_edges.setdefault(b.abstract_id, []).append(e2)
                elif (
                    a.causal_locus != "unspecified"
                    and b.causal_locus != "unspecified"
                    and a.causal_locus != b.causal_locus
                ):
                    # Orthogonal causal loci: check complementary synergy
                    pair_synergy = _COMPLEMENTARY_LOCI_PAIRS.get(
                        (a.causal_locus, b.causal_locus), 0.0
                    )
                    if pair_synergy > 0 and (
                        a.capability == b.capability or len(shared_tags) >= 1 or len(shared_tasks) >= 1
                    ):
                        w_co = _clip01(
                            0.40 * pair_synergy
                            + (0.25 if a.capability == b.capability else 0.10)
                            + 0.15 * min(2, len(shared_tags))
                            + 0.15 * min(2, len(shared_tasks))
                        )
                        e1 = SynapticEdge(a.abstract_id, b.abstract_id, "CO_ACTIVATES", round(w_co, 4))
                        e2 = SynapticEdge(b.abstract_id, a.abstract_id, "CO_ACTIVATES", round(w_co, 4))
                        self.co_edges.setdefault(a.abstract_id, []).append(e1)
                        self.co_edges.setdefault(b.abstract_id, []).append(e2)

    def _select_source_episode(
        self, abstract_id: str, query_tokens: set[str]
    ) -> tuple[str, tuple[str, ...], str]:
        abstract = self.ltm.abstractions[abstract_id]
        by_task: dict[str, list[AtomRecord]] = {}
        for atom_id in abstract.supporting_atom_ids:
            rec = self.ltm.records.get(atom_id)
            if rec and rec.task and rec.outcome == "pass":
                by_task.setdefault(rec.task, []).append(rec)
        if not by_task:
            return "", (), ""
        source_task, source_records = sorted(
            by_task.items(),
            key=lambda kv: (
                -max(
                    (
                        len(
                            query_tokens.intersection(
                                r.atom.tags or knowledge_tags(r.atom.content)
                            )
                        )
                        for r in kv[1]
                    ),
                    default=0,
                ),
                -len({r.atom.kind for r in kv[1]}.intersection({"R", "F", "S"})),
                kv[0],
            ),
        )[0]
        role_order = {"R": 0, "F": 1, "S": 2}
        source_records = sorted(
            source_records,
            key=lambda r: (role_order.get(r.atom.kind, 9), -r.atom.confidence, r.atom.atom_id),
        )
        chosen = []
        for kind in ("R", "F", "S"):
            m = next((r for r in source_records if r.atom.kind == kind), None)
            if m is not None:
                chosen.append(m)
        if not chosen:
            chosen = source_records[:2]
        selected_ids = tuple(r.atom.atom_id for r in chosen)
        roles_str = "; ".join(
            f"{r.atom.kind}={_semantic_episode_projection(r.atom.content, limit=220)}"
            for r in chosen
        )
        return source_task, selected_ids, roles_str

    def ground(
        self,
        *,
        task_id: str,
        task_text: str,
        repository_evidence: str = "",
        limit: int = 14,
    ) -> tuple[RetrievalBundle, tuple[GroundingRecord, ...], dict[str, Any]]:
        """Run associative spreading activation and select an Orthogonal Engram Ensemble."""
        bundle = self.ltm.retrieve(
            task_id=task_id,
            task_text=task_text,
            repository_evidence=repository_evidence,
            limit=limit,
            policy="engram_ensemble_v24",
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
        if has_exact_fast_delta and "METACORSI_CONFIRMED_FAILURE_EVIDENCE" not in repository_evidence:
            return bundle, (), {
                "paradigm": "engram_ensemble_v24",
                "status": "local_fast_engram_dominant",
                "ensemble_size": 0,
            }

        material = f"{task_id} {task_text} {repository_evidence}".strip()
        query_tokens = set(knowledge_tags(material))
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

        # 1. Compute initial sensory drive d_i for all target-disjoint engrams
        drives: dict[str, float] = {}
        matched_cues_by_id: dict[str, tuple[str, ...]] = {}
        for aid, abstract in sorted(self.ltm.abstractions.items()):
            support_tasks = {
                self.ltm.records[atom_id].task
                for atom_id in abstract.supporting_atom_ids
                if atom_id in self.ltm.records and self.ltm.records[atom_id].task
            }
            passing_tasks = {
                self.ltm.records[atom_id].task
                for atom_id in abstract.supporting_atom_ids
                if atom_id in self.ltm.records
                and self.ltm.records[atom_id].task
                and self.ltm.records[atom_id].outcome == "pass"
            }
            if not passing_tasks or any(task_matches(t, task_id) for t in support_tasks):
                continue

            cap_score = affinities.get(abstract.capability, 0.0)
            locus_score = (
                1.0
                if (abstract.causal_locus != "unspecified" and abstract.causal_locus in observed_loci)
                else (0.20 if abstract.causal_locus == "unspecified" else 0.0)
            )
            engram_tags = self.engram_cues.get(aid, abstract.trigger_tags)
            cues = tuple(sorted(query_tokens.intersection(engram_tags)))
            matched_cues_by_id[aid] = cues
            cue_denom = max(1, min(3, len(engram_tags)))
            cue_score = min(1.0, len(cues) / cue_denom)
            source_div = min(1.0, len(passing_tasks) / 2.0)
            specificity = 0.35 if abstract.invariant.startswith("Preserve the verified") else 1.0

            drive = _clip01(
                0.30 * cap_score
                + 0.28 * locus_score
                + 0.24 * cue_score
                + 0.10 * source_div
                + 0.08 * specificity
            )
            if len(cues) == 0 and locus_score < 1.0:
                drive *= 0.5
            drives[aid] = drive

        if not drives:
            return bundle, (), {"paradigm": "engram_ensemble_v24", "status": "no_engrams", "ensemble_size": 0}

        # 2. Run 2-step Spreading Activation (CO_ACTIVATES synergy - MUTUAL_INHIBITS competition)
        act = dict(drives)
        for _ in range(2):
            next_act: dict[str, float] = {}
            for aid, d_val in drives.items():
                co_boost = sum(
                    edge.weight * act.get(edge.target_id, 0.0)
                    for edge in self.co_edges.get(aid, ())
                    if edge.target_id in act
                )
                # Take max lateral inhibitor in the same causal locus
                inh_penalty = max(
                    (
                        edge.weight * act.get(edge.target_id, 0.0)
                        for edge in self.inh_edges.get(aid, ())
                        if edge.target_id in act and act.get(edge.target_id, 0.0) >= act.get(aid, 0.0)
                    ),
                    default=0.0,
                )
                next_act[aid] = _clip01(
                    d_val + self.lambda_co * min(1.0, co_boost) - self.lambda_inh * inh_penalty
                )
            act = next_act

        # 3. Select Primary Engram E_1 and optional Orthogonal Synergistic Co-Engram E_2
        valid_primaries = [
            aid
            for aid, score in act.items()
            if score >= self.primary_threshold and len(matched_cues_by_id.get(aid, ())) >= 1
        ]
        if not valid_primaries:
            return bundle, (), {
                "paradigm": "engram_ensemble_v24",
                "status": "below_primary_threshold",
                "ensemble_size": 0,
            }

        primary_id = max(valid_primaries, key=lambda k: (round(act[k], 6), round(drives[k], 6), k))
        primary_abs = self.ltm.abstractions[primary_id]

        # Find best orthogonal co-engram connected via CO_ACTIVATES or complementary locus
        co_candidates: list[tuple[float, str]] = []
        co_neighbor_weights = {
            edge.target_id: edge.weight for edge in self.co_edges.get(primary_id, ())
        }
        for aid, score in act.items():
            if aid == primary_id:
                continue
            cand_abs = self.ltm.abstractions[aid]
            if (
                cand_abs.causal_locus == "unspecified"
                or cand_abs.causal_locus == primary_abs.causal_locus
            ):
                continue
            if len(matched_cues_by_id.get(aid, ())) == 0:
                continue
            synergy_w = co_neighbor_weights.get(
                aid,
                _COMPLEMENTARY_LOCI_PAIRS.get(
                    (primary_abs.causal_locus, cand_abs.causal_locus), 0.0
                )
                * 0.5,
            )
            if synergy_w <= 0.0:
                continue
            combined_co_score = score + 0.15 * synergy_w
            if combined_co_score >= self.co_engram_threshold:
                co_candidates.append((combined_co_score, aid))

        co_candidates.sort(key=lambda item: (-item[0], item[1]))
        secondary_id = co_candidates[0][1] if co_candidates else None

        # 4. Synthesize Co-Grounded Ensemble Record
        src1_task, src1_ids, src1_roles = self._select_source_episode(primary_id, query_tokens)
        cues1 = matched_cues_by_id.get(primary_id, ())
        context_digest = hashlib.sha256(material.encode()).hexdigest()
        repository = task_id.split("-", 1)[0] if "-" in task_id else task_id
        base_residual = 0.25 if has_verified_task_memory else 0.60

        if secondary_id is not None:
            secondary_abs = self.ltm.abstractions[secondary_id]
            src2_task, src2_ids, src2_roles = self._select_source_episode(secondary_id, query_tokens)
            cues2 = matched_cues_by_id.get(secondary_id, ())
            merged_cues = tuple(dict.fromkeys((*cues1, *cues2)))
            merged_atom_ids = tuple(dict.fromkeys((*src1_ids, *src2_ids)))
            actions = (
                f"STAGE 1 PRIMARY ENGRAM [{primary_abs.capability}/{primary_abs.causal_locus}]: {primary_abs.invariant} (Source {src1_task}: {src1_roles})",
                f"STAGE 2 SYNERGISTIC CO-ENGRAM [{secondary_abs.capability}/{secondary_abs.causal_locus}]: {secondary_abs.invariant} (Source {src2_task}: {src2_roles})",
                f"ENSEMBLE CO-GROUNDING: satisfy both [{primary_abs.causal_locus}] and [{secondary_abs.causal_locus}] in one coherent repair slice; symbols={','.join(live_symbols) or 'auto'}.",
            )
            module_label = (
                f"{primary_abs.capability}/{primary_abs.causal_locus}"
                f"+{secondary_abs.causal_locus}"
            )
            source_ep_label = f"{src1_task}+{src2_task}" if src2_task != src1_task else src1_task
            ensemble_ids = (primary_id, secondary_id)
        else:
            merged_cues = cues1
            merged_atom_ids = src1_ids
            actions = (
                f"PRIMARY ENGRAM [{primary_abs.capability}/{primary_abs.causal_locus}]: {primary_abs.invariant}",
                f"SOURCE EPISODE {src1_task}: {src1_roles}",
                "TARGET ROLE BINDING: bind state, boundary, and consumer roles to current evidence.",
            )
            module_label = f"{primary_abs.capability}/{primary_abs.causal_locus}"
            source_ep_label = src1_task
            ensemble_ids = (primary_id,)

        bindings = {
            "task": task_id,
            "repository": repository,
            "capability": primary_abs.capability,
            "causal_locus": primary_abs.causal_locus,
            "failure_mode": primary_abs.failure_mode,
            "grounding_weight": f"{act[primary_id]:.3f}",
            "ensemble_ids": ",".join(ensemble_ids),
            "ensemble_size": str(len(ensemble_ids)),
            "target_roles": ",".join(merged_cues) or "structural-analogy",
            "live_symbols": ",".join(live_symbols) or "pending-observation",
            "source_episode": source_ep_label,
            "local_episode_weight": f"{1.0 - base_residual:.3f}",
            "abstract_residual_weight": f"{base_residual:.3f}",
            "grounded_atom_ids": ",".join(merged_atom_ids),
            "memory_strategy": "engram_ensemble",
            "module": module_label,
        }
        sig = f"{'|'.join(ensemble_ids)}|{task_id}|{context_digest}|engram"
        gid = "grounding." + hashlib.sha256(sig.encode()).hexdigest()[:20]
        record = GroundingRecord(
            grounding_id=gid,
            abstract_id=primary_id,
            task=task_id,
            repository=repository,
            capability=primary_abs.capability,
            context_digest=context_digest,
            activation_evidence=merged_cues[:4],
            bindings=bindings,
            instantiated_action=tuple(
                re.sub(r"\s+", " ", a).strip()[:240] for a in actions
            ),
            source_atom_ids=merged_atom_ids,
            scope_guard=tuple(primary_abs.safety_boundary[:2]),
            verification_plan=(
                "build/typecheck succeeds",
                "both primary and co-activated boundary counterexamples turn green",
                "independent F2P and P2P verifier decides promotion",
            ),
        )
        return bundle, (record,), {
            "paradigm": "engram_ensemble_v24",
            "status": "activated",
            "ensemble_ids": list(ensemble_ids),
            "ensemble_size": len(ensemble_ids),
            "primary_activation": round(act[primary_id], 4),
            "secondary_activation": round(act[secondary_id], 4) if secondary_id else 0.0,
        }
