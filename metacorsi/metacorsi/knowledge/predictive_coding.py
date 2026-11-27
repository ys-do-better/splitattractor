from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .grounding import CAPABILITY_PATTERNS, task_matches
from .indexed_ltm import (
    AtomRecord,
    GroundingRecord,
    IndexedLTM,
    RetrievalBundle,
    _RELATION_STOP_TAGS,
    _capability_affinities,
    _causal_loci,
    _extract_live_symbols,
    _semantic_episode_projection,
    _situation_capability_affinities,
)
from .store import knowledge_tags


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class PredictionErrorSignal:
    """Bottom-up prediction error (Surprise) between expected contract and live observation."""

    step: int
    error_type: str
    surprise_magnitude: float
    precision: float
    error_tokens: tuple[str, ...]
    error_loci: tuple[str, ...]
    live_symbols: tuple[str, ...]

    def json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PredictiveStepResult:
    """Snapshot of predictive coding belief state and error-driven retrieval at step t."""

    step: int
    error_signal: PredictionErrorSignal
    posterior_beliefs: dict[str, float]
    active_abstract_id: str | None
    active_grounding: GroundingRecord | None
    transition: str
    reason: str
    falsified_ids: tuple[str, ...] = ()

    def json(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "paradigm": "predictive_coding_v24",
            "error_signal": self.error_signal.json(),
            "posterior_beliefs": {
                k: round(v, 4) for k, v in self.posterior_beliefs.items() if v > 1e-3
            },
            "active_abstract_id": self.active_abstract_id,
            "transition": self.transition,
            "reason": self.reason,
            "falsified_ids": list(self.falsified_ids),
            "grounding": (
                [self.active_grounding.json()] if self.active_grounding is not None else []
            ),
        }


class PredictiveCodingMemory:
    """Paradigm 3: Hierarchical Predictive Coding & Surprise-Driven Residual Memory.

    Inspired by Karl Friston (Free-Energy Principle) and Rao & Ballard (1999):
    - Working memory holds a top-down expectation model (the task description + local CLS
      backbone capsule).
    - Cross-task LTM is NOT queried by static task similarity at Step 0 (where prediction
      error is zero).
    - Instead, at each step t, the bottom-up observation O_t is compared against the
      top-down expectation to extract the Prediction Error vector:
          epsilon_t = O_t (-) O_hat_t  (unexplained failure tokens, unexpected causal loci,
                                        test failures, or RCA surprise)
    - Posterior beliefs mu_i(t) over candidate error-correcting schemas are updated via
      precision-weighted Kalman gain:
          mu_i(t) = (1 - kappa_t) * mu_i(t-1) + kappa_t * ErrorExplanation_i(epsilon_t)
          where kappa_t = clip(precision_t * surprise_t, 0.15, 0.85)
    - If an edit under active belief i* fails to reduce prediction error (explicit
      counterevidence), falsification surprise suppresses mu_{i*}(t) -> 0 and shifts
      to the next highest error-explaining hypothesis.
    """

    def __init__(
        self,
        ltm: IndexedLTM,
        *,
        surprise_threshold: float = 0.36,
        belief_threshold: float = 0.42,
    ) -> None:
        self.ltm = ltm
        self.surprise_threshold = float(surprise_threshold)
        self.belief_threshold = float(belief_threshold)
        self.step_count: int = 0
        self.beliefs: dict[str, float] = {}
        self.falsified: set[str] = set()
        self.active_abstract_id: str | None = None
        self.active_grounding: GroundingRecord | None = None
        self.history: list[PredictiveStepResult] = []

    def reset(self) -> None:
        self.step_count = 0
        self.beliefs.clear()
        self.falsified.clear()
        self.active_abstract_id = None
        self.active_grounding = None
        self.history.clear()

    def compute_prediction_error(
        self,
        *,
        task_id: str,
        task_text: str,
        bundle: RetrievalBundle,
        repository_evidence: str,
        explicit_counterevidence: bool = False,
    ) -> PredictionErrorSignal:
        """Compute bottom-up prediction error epsilon_t from live observation minus top-down prior."""
        live = repository_evidence.strip()
        if not live:
            return PredictionErrorSignal(
                step=self.step_count,
                error_type="none",
                surprise_magnitude=0.0,
                precision=0.0,
                error_tokens=(),
                error_loci=(),
                live_symbols=(),
            )

        # Top-down expected vocabulary from task prose + local backbone atoms
        expected_tokens = set(knowledge_tags(f"{task_id} {task_text}"))
        for atom in bundle.atoms[:3]:
            expected_tokens.update(atom.tags or knowledge_tags(atom.content[:200]))

        live_tokens = set(knowledge_tags(live)) - _RELATION_STOP_TAGS
        # Prediction error tokens: novel diagnostic tokens in the live trace/output,
        # plus failure-bearing keywords
        failure_cues = {
            tok
            for tok in live_tokens
            if tok
            in {
                "error", "failed", "failure", "assertion", "exception", "traceback",
                "mismatch", "null", "undefined", "recursive", "cycle", "nosec",
                "directive", "ast", "statement", "boundary", "cache", "state",
                "snapshot", "cancel", "schema", "parse", "parser", "alias",
                "route", "dispatch", "register", "shorthand", "roundtrip",
            }
        }
        novel_tokens = (live_tokens - expected_tokens) | failure_cues
        error_loci = tuple(
            loc for loc in _causal_loci(live) if loc != "unspecified"
        )
        live_symbols = _extract_live_symbols(live)

        confirmed_failure = "METACORSI_CONFIRMED_FAILURE_EVIDENCE" in live
        has_rca = "METACORSI RCA" in live
        has_error_trace = bool(
            re.search(
                r"(?:AssertionError|Traceback|FAIL|FAILED|Error:|Exception|panic:|rc=[1-9])",
                live,
                re.I,
            )
        )

        if explicit_counterevidence:
            error_type = "falsification_surprise"
            base_surprise = 0.90
        elif confirmed_failure:
            error_type = "diagnostic_test_surprise"
            base_surprise = 0.78
        elif has_rca or has_error_trace:
            error_type = "execution_trace_surprise"
            base_surprise = 0.65
        elif error_loci and novel_tokens:
            error_type = "structural_boundary_surprise"
            base_surprise = 0.45
        else:
            error_type = "low_surprise_observation"
            base_surprise = 0.15

        locus_specificity = min(1.0, len(error_loci) * 0.5 + (0.3 if live_symbols else 0.0))
        token_specificity = min(1.0, len(novel_tokens) / 6.0)
        precision = _clip01(0.55 * locus_specificity + 0.45 * token_specificity + 0.15)
        surprise_magnitude = _clip01(base_surprise * (0.65 + 0.35 * precision))

        return PredictionErrorSignal(
            step=self.step_count,
            error_type=error_type,
            surprise_magnitude=round(surprise_magnitude, 4),
            precision=round(precision, 4),
            error_tokens=tuple(sorted(novel_tokens)[:16]),
            error_loci=error_loci,
            live_symbols=live_symbols,
        )

    def step(
        self,
        *,
        task_id: str,
        task_text: str,
        repository_evidence: str = "",
        explicit_counterevidence: bool = False,
        stagnation_steps: int = 0,
        limit: int = 14,
    ) -> tuple[RetrievalBundle, PredictiveStepResult]:
        """Update predictive coding beliefs from prediction error and select residual memory."""
        self.step_count += 1
        bundle = self.ltm.retrieve(
            task_id=task_id,
            task_text=task_text,
            repository_evidence=repository_evidence,
            limit=limit,
            policy="predictive_coding_v24",
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

        err = self.compute_prediction_error(
            task_id=task_id,
            task_text=task_text,
            bundle=bundle,
            repository_evidence=repository_evidence,
            explicit_counterevidence=explicit_counterevidence,
        )

        # Falsification surprise: if an active hypothesis failed under edit/counterevidence
        # or persistent stagnation, suppress its belief and record falsification.
        prev_active = self.active_abstract_id
        if (explicit_counterevidence or stagnation_steps >= 2) and prev_active is not None:
            self.falsified.add(prev_active)
            self.beliefs[prev_active] = 0.0
            self.active_abstract_id = None
            self.active_grounding = None

        # If prediction error (surprise) is below threshold at Step 0 or routine step,
        # no cross-task error-correcting memory is triggered.
        if err.surprise_magnitude < self.surprise_threshold:
            # Decay existing beliefs gently if no active lock
            for aid in list(self.beliefs):
                if aid != self.active_abstract_id:
                    self.beliefs[aid] = round(self.beliefs[aid] * 0.85, 4)
            transition = (
                "sustained_belief"
                if self.active_abstract_id is not None
                else "zero_surprise_top_down_only"
            )
            res = PredictiveStepResult(
                step=self.step_count,
                error_signal=err,
                posterior_beliefs=dict(self.beliefs),
                active_abstract_id=self.active_abstract_id,
                active_grounding=self.active_grounding,
                transition=transition,
                reason=(
                    f"surprise={err.surprise_magnitude:.3f} < threshold={self.surprise_threshold:.3f}; "
                    "top-down task-local contract suffices"
                ),
                falsified_ids=tuple(sorted(self.falsified)),
            )
            self.history.append(res)
            return bundle, res

        # If task has an exact Fast Delta recipe, only allow cross-task error memory on
        # high-surprise confirmed failure (surprise >= 0.65).
        if has_exact_fast_delta and err.surprise_magnitude < 0.65:
            res = PredictiveStepResult(
                step=self.step_count,
                error_signal=err,
                posterior_beliefs=dict(self.beliefs),
                active_abstract_id=None,
                active_grounding=None,
                transition="exact_fast_delta_predicts_outcome",
                reason="exact-task Fast Delta handles current expectation; cross-task residual suppressed",
                falsified_ids=tuple(sorted(self.falsified)),
            )
            self.history.append(res)
            return bundle, res

        # Precision-weighted Kalman belief update over candidate abstractions
        error_token_set = set(err.error_tokens)
        error_loci_set = set(err.error_loci)
        live_cap_affinities = _capability_affinities(repository_evidence)
        task_cap_affinities = _situation_capability_affinities(task_id, task_text)
        kappa = _clip01(max(0.25, min(0.85, err.precision * err.surprise_magnitude)))

        grounding_by_id: dict[str, GroundingRecord] = {}
        context_digest = hashlib.sha256(
            f"{task_id}|{repository_evidence}".encode()
        ).hexdigest()
        repository = task_id.split("-", 1)[0] if "-" in task_id else task_id

        for aid, abstract in sorted(self.ltm.abstractions.items()):
            if aid in self.falsified:
                self.beliefs[aid] = 0.0
                continue
            support_tasks = {
                self.ltm.records[atom_id].task
                for atom_id in abstract.supporting_atom_ids
                if atom_id in self.ltm.records and self.ltm.records[atom_id].task
            }
            passing_support = {
                self.ltm.records[atom_id].task
                for atom_id in abstract.supporting_atom_ids
                if atom_id in self.ltm.records
                and self.ltm.records[atom_id].task
                and self.ltm.records[atom_id].outcome == "pass"
            }
            if not passing_support or any(task_matches(t, task_id) for t in support_tasks):
                continue

            # How well does abstract explain the PREDICTION ERROR epsilon_t?
            locus_explains = (
                1.0
                if (abstract.causal_locus != "unspecified" and abstract.causal_locus in error_loci_set)
                else 0.0
            )
            matched_error_cues = tuple(sorted(error_token_set.intersection(abstract.trigger_tags)))
            cue_denom = max(1, min(3, len(abstract.trigger_tags)))
            cue_explains = (
                min(1.0, len(matched_error_cues) / cue_denom)
                if abstract.trigger_tags
                else (0.5 if locus_explains > 0 else 0.0)
            )
            live_cap = live_cap_affinities.get(abstract.capability, 0.0)
            task_cap = task_cap_affinities.get(abstract.capability, 0.0)
            cap_explains = max(live_cap, 0.65 * task_cap)

            if len(matched_error_cues) == 0 and locus_explains == 0.0:
                error_explanation = 0.0
            else:
                error_explanation = _clip01(
                    0.38 * locus_explains
                    + 0.34 * cue_explains
                    + 0.28 * cap_explains
                )

            prior_mu = self.beliefs.get(aid, 0.0)
            new_mu = _clip01((1.0 - kappa) * prior_mu + kappa * error_explanation)
            self.beliefs[aid] = new_mu

            if new_mu >= self.belief_threshold and (matched_error_cues or locus_explains > 0):
                # Build error-indexed reverse grounding record
                by_task: dict[str, list[AtomRecord]] = {}
                for atom_id in abstract.supporting_atom_ids:
                    rec = self.ltm.records.get(atom_id)
                    if rec and rec.task and rec.outcome == "pass":
                        by_task.setdefault(rec.task, []).append(rec)
                source_task, source_records = sorted(
                    by_task.items(),
                    key=lambda kv: (
                        -max(
                            (
                                len(
                                    error_token_set.intersection(
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
                chosen = []
                for kind in ("R", "F", "S"):
                    m = next((r for r in source_records if r.atom.kind == kind), None)
                    if m is not None:
                        chosen.append(m)
                if not chosen:
                    chosen = source_records[:2]
                selected_ids = tuple(r.atom.atom_id for r in chosen)
                roles_str = "; ".join(
                    f"{r.atom.kind}={_semantic_episode_projection(r.atom.content)}"
                    for r in chosen
                )
                sym_str = (
                    f" Target symbols at error site: {', '.join(err.live_symbols)}."
                    if err.live_symbols
                    else ""
                )
                actions = (
                    f"PREDICTION-ERROR HYPOTHESIS ({err.error_type}; surprise={err.surprise_magnitude:.2f}): {abstract.invariant}",
                    f"ERROR-CORRECTING SOURCE ({source_task}): {roles_str}",
                    f"RESIDUAL CORRECTION: eliminate prediction error on loci={list(err.error_loci) or [abstract.causal_locus]};{sym_str} preserve passing P2P expectations.",
                )
                base_residual = 0.25 if has_verified_task_memory else 0.60
                bindings = {
                    "task": task_id,
                    "repository": repository,
                    "capability": abstract.capability,
                    "causal_locus": abstract.causal_locus,
                    "failure_mode": abstract.failure_mode,
                    "grounding_weight": f"{new_mu:.3f}",
                    "surprise_magnitude": f"{err.surprise_magnitude:.3f}",
                    "error_precision": f"{err.precision:.3f}",
                    "error_type": err.error_type,
                    "target_roles": ",".join(matched_error_cues) or abstract.causal_locus,
                    "live_symbols": ",".join(err.live_symbols) or "pending-observation",
                    "source_episode": source_task,
                    "local_episode_weight": f"{1.0 - base_residual:.3f}",
                    "abstract_residual_weight": f"{base_residual:.3f}",
                    "grounded_atom_ids": ",".join(selected_ids),
                    "memory_strategy": "predictive_coding",
                    "module": f"{abstract.capability}/{abstract.causal_locus}",
                }
                sig = f"{aid}|{task_id}|{context_digest}|pred"
                gid = "grounding." + hashlib.sha256(sig.encode()).hexdigest()[:20]
                grounding_by_id[aid] = GroundingRecord(
                    grounding_id=gid,
                    abstract_id=aid,
                    task=task_id,
                    repository=repository,
                    capability=abstract.capability,
                    context_digest=context_digest,
                    activation_evidence=matched_error_cues[:3] or err.error_loci[:2],
                    bindings=bindings,
                    instantiated_action=tuple(
                        re.sub(r"\s+", " ", a).strip()[:240] for a in actions
                    ),
                    source_atom_ids=selected_ids,
                    scope_guard=tuple(abstract.safety_boundary[:2]),
                    verification_plan=(
                        "prediction error on focused counterexample vanishes (turns green)",
                        "top-down P2P expectations remain unviolated",
                    ),
                )

        if not grounding_by_id:
            if explicit_counterevidence and prev_active is not None:
                transition = "falsification_reset"
                reason = f"falsified {prev_active}; awaiting next diagnostic prediction error"
            else:
                transition = "below_belief_threshold"
                reason = "no candidate schema reached posterior belief threshold for current error"
            res = PredictiveStepResult(
                step=self.step_count,
                error_signal=err,
                posterior_beliefs=dict(self.beliefs),
                active_abstract_id=None,
                active_grounding=None,
                transition=transition,
                reason=reason,
                falsified_ids=tuple(sorted(self.falsified)),
            )
            self.history.append(res)
            return bundle, res

        winner_id = max(grounding_by_id, key=lambda k: (self.beliefs.get(k, 0.0), k))
        self.active_abstract_id = winner_id
        self.active_grounding = grounding_by_id[winner_id]
        if prev_active is None:
            transition = "surprise_driven_activation"
        elif prev_active == winner_id:
            transition = "sustained_belief"
        else:
            transition = "prediction_error_shift"

        res = PredictiveStepResult(
            step=self.step_count,
            error_signal=err,
            posterior_beliefs=dict(self.beliefs),
            active_abstract_id=winner_id,
            active_grounding=self.active_grounding,
            transition=transition,
            reason=(
                f"surprise={err.surprise_magnitude:.3f} ({err.error_type}) activated "
                f"{winner_id} with belief={self.beliefs[winner_id]:.3f}"
            ),
            falsified_ids=tuple(sorted(self.falsified)),
        )
        self.history.append(res)
        return bundle, res
