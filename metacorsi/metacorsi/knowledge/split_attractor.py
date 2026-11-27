from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .indexed_ltm import GroundingRecord


SplitAttractorMode = Literal["sa", "naive"]


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class CandidateSignal:
    """Evidence evaluation u_i(t) and condition-matching metadata for one schema."""

    abstract_id: str
    u_score: float
    capability_match: float
    causal_match: float
    trigger_match: float
    condition_matched: bool
    grounding: GroundingRecord
    matched_cues: tuple[str, ...] = ()
    causal_locus: str = "unspecified"
    capability: str = "repository-contract"

    def json(self) -> dict[str, Any]:
        return {
            "abstract_id": self.abstract_id,
            "u_score": round(self.u_score, 4),
            "capability_match": round(self.capability_match, 4),
            "causal_match": round(self.causal_match, 4),
            "trigger_match": round(self.trigger_match, 4),
            "condition_matched": self.condition_matched,
            "matched_cues": list(self.matched_cues),
            "causal_locus": self.causal_locus,
            "capability": self.capability,
        }


@dataclass(frozen=True)
class SplitAttractorStepResult:
    """Complete snapshot of PFG, hDeltaK, and gate g_t after one discrete step."""

    step: int
    mode: SplitAttractorMode
    state: str
    gate: int
    p_scores: dict[str, float]
    h_scores: dict[str, float]
    u_scores: dict[str, float]
    refractory: dict[str, float]
    active_abstract_id: str | None
    previous_abstract_id: str | None
    active_grounding: GroundingRecord | None
    transition: str
    reason: str
    pfg_summary: dict[str, Any] = field(default_factory=dict)

    def to_wm_snapshot(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "mode": self.mode,
            "state": self.state,
            "gate": self.gate,
            "active_abstract_id": self.active_abstract_id,
            "previous_abstract_id": self.previous_abstract_id,
            "transition": self.transition,
            "reason": self.reason,
            "pfg": {
                "p_scores": {k: round(v, 4) for k, v in self.p_scores.items()},
                "u_scores": {k: round(v, 4) for k, v in self.u_scores.items()},
                "refractory": {k: round(v, 4) for k, v in self.refractory.items() if v > 1e-4},
                **self.pfg_summary,
            },
            "hdk": {
                "h_scores": {k: round(v, 4) for k, v in self.h_scores.items() if v > 1e-4},
                "latched_abstract_id": self.active_abstract_id,
                "grounding_id": (
                    self.active_grounding.grounding_id if self.active_grounding else None
                ),
            },
        }

    def json(self) -> dict[str, Any]:
        payload = self.to_wm_snapshot()
        payload["grounding"] = (
            [self.active_grounding.json()] if self.active_grounding is not None else []
        )
        return payload


class SplitAttractorWM:
    """MetaCoRSI V24 Split-Attractor Working Memory (SA-WM).

    Implements the minimal two-state recurrent model inspired by Lanz et al. (Nature, 2026):
        p_i(t) = sigma( u_eff_i(t) + g_{t-1} * w1 * h_i(t-1) )
        h_i(t) = g_t * sigma( (1 - alpha) * h_i(t-1) + alpha * w2 * p_i(t) )   [for Top-1 i*]

    Modes:
    - "sa" (V24-SA): bidirectional feedback (w1 > 0) and slow recurrent integration (0 < alpha < 1).
    - "naive" (V24-Naive): w1 = 0, alpha = 1.0 (no recurrent feedback from hDeltaK to PFG).
    """

    def __init__(
        self,
        *,
        mode: SplitAttractorMode = "sa",
        w1: float = 0.35,
        w2: float = 1.0,
        alpha: float = 0.45,
        open_threshold: float = 0.42,
        maintain_threshold: float = 0.22,
        refractory_penalty: float = 0.35,
        refractory_decay: float = 0.60,
        stagnation_threshold: int = 2,
    ) -> None:
        if mode not in {"sa", "naive"}:
            raise ValueError(f"unsupported SplitAttractorWM mode: {mode}")
        self.mode: SplitAttractorMode = mode
        self.w1 = float(w1) if mode == "sa" else 0.0
        self.w2 = float(w2)
        self.alpha = float(alpha) if mode == "sa" else 1.0
        self.open_threshold = float(open_threshold)
        self.maintain_threshold = float(maintain_threshold)
        self.refractory_penalty = float(refractory_penalty)
        self.refractory_decay = float(refractory_decay)
        self.stagnation_threshold = max(1, int(stagnation_threshold))

        self.step_count: int = 0
        self.gate: int = 0
        self.state: str = "TRACKING"
        self.p_state: dict[str, float] = {}
        self.h_state: dict[str, float] = {}
        self.refractory: dict[str, float] = {}
        self.active_abstract_id: str | None = None
        self.active_grounding: GroundingRecord | None = None
        self.history: list[SplitAttractorStepResult] = []
        self._counterevidence_onset_step: int | None = None
        self._retraction_latencies: list[int] = []

    def reset(self) -> None:
        self.step_count = 0
        self.gate = 0
        self.state = "TRACKING"
        self.p_state.clear()
        self.h_state.clear()
        self.refractory.clear()
        self.active_abstract_id = None
        self.active_grounding = None
        self.history.clear()
        self._counterevidence_onset_step = None
        self._retraction_latencies.clear()

    def _decay_refractory(self) -> None:
        updated: dict[str, float] = {}
        for key, value in self.refractory.items():
            decayed = value * self.refractory_decay
            if decayed >= 0.02:
                updated[key] = decayed
        self.refractory = updated

    def step(
        self,
        candidates: tuple[CandidateSignal, ...],
        *,
        has_live_evidence: bool = False,
        explicit_counterevidence: bool = False,
        stagnation_steps: int = 0,
        ordinary_command_failure: bool = False,
        pfg_summary: dict[str, Any] | None = None,
    ) -> SplitAttractorStepResult:
        """Advance PFG and hDeltaK by one step under disinhibitory/re-inhibitory gating."""
        self.step_count += 1
        step_idx = self.step_count
        prev_abstract_id = self.active_abstract_id
        prev_gate = self.gate

        # Decay short-term refractory penalties from earlier retractions.
        self._decay_refractory()

        by_id: dict[str, CandidateSignal] = {c.abstract_id: c for c in candidates}
        all_ids = sorted(set(by_id) | set(self.h_state))

        if explicit_counterevidence and self._counterevidence_onset_step is None:
            self._counterevidence_onset_step = step_idx

        # ------------------------------------------------------------------
        # 1. Check Re-inhibitory Erasing (g_t: 1 -> 0) before recurrent hold
        # ------------------------------------------------------------------
        should_erase = False
        erase_reason = ""
        if prev_gate == 1 and prev_abstract_id is not None:
            current_signal = by_id.get(prev_abstract_id)
            if explicit_counterevidence:
                should_erase = True
                erase_reason = "explicit counterevidence refuted latched hypothesis"
            elif stagnation_steps >= self.stagnation_threshold:
                # Re-verify whether current evidence still matches the latched schema's conditions
                still_strongly_supported = (
                    current_signal is not None
                    and current_signal.condition_matched
                    and current_signal.u_score >= self.open_threshold
                )
                if not still_strongly_supported:
                    should_erase = True
                    erase_reason = (
                        f"persistent stagnation ({stagnation_steps} steps) without "
                        "confirming condition support"
                    )
                elif stagnation_steps >= self.stagnation_threshold + 1:
                    should_erase = True
                    erase_reason = (
                        f"persistent stagnation ({stagnation_steps} steps) exceeded hard limit"
                    )
            # Note: ordinary_command_failure alone deliberately does NOT erase!

        if should_erase and prev_abstract_id is not None:
            # Apply short-term refractory penalty to the retracted schema, clear hDeltaK
            self.refractory[prev_abstract_id] = self.refractory_penalty
            self.gate = 0
            self.h_state = {aid: 0.0 for aid in all_ids}
            self.active_abstract_id = None
            self.active_grounding = None
            self.state = "REINHIBITED_RESET"
            if self._counterevidence_onset_step is not None:
                self._retraction_latencies.append(step_idx - self._counterevidence_onset_step)
                self._counterevidence_onset_step = None

            u_scores: dict[str, float] = {}
            p_scores: dict[str, float] = {}
            for aid in all_ids:
                raw_u = by_id[aid].u_score if aid in by_id else 0.0
                u_scores[aid] = raw_u
                u_eff = _clip01(raw_u - self.refractory.get(aid, 0.0))
                p_scores[aid] = _clip01(u_eff)
            self.p_state = p_scores

            result = SplitAttractorStepResult(
                step=step_idx,
                mode=self.mode,
                state=self.state,
                gate=0,
                p_scores=dict(p_scores),
                h_scores={},
                u_scores=u_scores,
                refractory=dict(self.refractory),
                active_abstract_id=None,
                previous_abstract_id=prev_abstract_id,
                active_grounding=None,
                transition="reinhibitory_erase",
                reason=erase_reason,
                pfg_summary=dict(pfg_summary or {}),
            )
            self.history.append(result)
            return result

        # ------------------------------------------------------------------
        # 2. Compute PFG activation p_i(t) = sigma(u_eff_i(t) + g_{t-1} * w1 * h_i(t-1))
        # ------------------------------------------------------------------
        u_scores = {}
        p_scores = {}
        for aid in all_ids:
            sig = by_id.get(aid)
            raw_u = sig.u_score if sig is not None else 0.0
            u_scores[aid] = raw_u
            penalty = self.refractory.get(aid, 0.0)
            # If fresh strong condition-matched evidence supports a previously retracted
            # schema, allow it to overcome the residual refractory penalty.
            if sig is not None and sig.condition_matched and raw_u >= self.open_threshold + 0.12:
                penalty *= 0.35
            u_eff = _clip01(raw_u - penalty)
            recurrent_boost = prev_gate * self.w1 * self.h_state.get(aid, 0.0)
            p_scores[aid] = _clip01(u_eff + recurrent_boost)
        self.p_state = p_scores

        if not by_id:
            self.gate = 0
            self.state = "TRACKING"
            self.h_state = {}
            self.active_abstract_id = None
            self.active_grounding = None
            result = SplitAttractorStepResult(
                step=step_idx,
                mode=self.mode,
                state=self.state,
                gate=0,
                p_scores={},
                h_scores={},
                u_scores={},
                refractory=dict(self.refractory),
                active_abstract_id=None,
                previous_abstract_id=prev_abstract_id,
                active_grounding=None,
                transition="stay_tracking",
                reason="no candidate abstractions available",
                pfg_summary=dict(pfg_summary or {}),
            )
            self.history.append(result)
            return result

        # Rank candidates by PFG activation p_i(t), breaking ties by condition_matched & raw u_i(t)
        winner_id = max(
            by_id,
            key=lambda aid: (
                round(p_scores.get(aid, 0.0), 6),
                1 if by_id[aid].condition_matched else 0,
                round(u_scores.get(aid, 0.0), 6),
                aid,
            ),
        )
        winner_sig = by_id[winner_id]
        winner_p = p_scores.get(winner_id, 0.0)

        # ------------------------------------------------------------------
        # 3. Disinhibitory Gating Decision (g_t in {0, 1})
        # ------------------------------------------------------------------
        transition = "stay_tracking"
        reason = ""

        if prev_gate == 0:
            # Gate is closed: require live evidence + condition match + threshold
            if (
                has_live_evidence
                and winner_sig.condition_matched
                and winner_p >= self.open_threshold
            ):
                self.gate = 1
                self.state = "LATCHED"
                self.active_abstract_id = winner_id
                self.active_grounding = winner_sig.grounding
                transition = "disinhibitory_write"
                reason = (
                    f"disinhibition triggered: {winner_id} matched conditions "
                    f"(cues={list(winner_sig.matched_cues)}, p={winner_p:.3f} >= {self.open_threshold:.3f})"
                )
            else:
                self.gate = 0
                self.state = "TRACKING"
                self.active_abstract_id = None
                self.active_grounding = None
                transition = "stay_tracking"
                if not has_live_evidence:
                    reason = "gate closed: awaiting live repository/RCA evidence (Step 0 inhibition)"
                elif not winner_sig.condition_matched:
                    reason = (
                        f"gate closed: top candidate {winner_id} (p={winner_p:.3f}) "
                        "does not match causal/trigger preconditions in current evidence"
                    )
                else:
                    reason = (
                        f"gate closed: top candidate {winner_id} p={winner_p:.3f} "
                        f"< open_threshold={self.open_threshold:.3f}"
                    )
        else:
            # Gate was already open (prev_gate == 1) and no explicit erase triggered.
            latched_id = prev_abstract_id or winner_id
            latched_p = p_scores.get(latched_id, 0.0)
            if self.mode == "sa" and latched_p >= self.maintain_threshold and latched_id in by_id:
                # In a true Split Attractor (V24-SA), the closed PFG-hDeltaK recurrent
                # loop and fast lateral inhibition lock the active bump in place until
                # re-inhibition (counterevidence or stagnation) breaks the loop.
                self.gate = 1
                self.state = "LATCHED"
                self.active_abstract_id = latched_id
                self.active_grounding = by_id[latched_id].grounding
                transition = "sustained_latch"
                reason = (
                    f"sustained attractor lock on {latched_id} (p={latched_p:.3f}, "
                    f"ordinary_failure_ignored={ordinary_command_failure})"
                )
            elif winner_id == latched_id and latched_p >= self.maintain_threshold:
                self.gate = 1
                self.state = "LATCHED"
                self.active_abstract_id = latched_id
                self.active_grounding = by_id[latched_id].grounding
                transition = "sustained_latch"
                reason = (
                    f"sustained latch on {latched_id} (p={latched_p:.3f}, "
                    f"ordinary_failure_ignored={ordinary_command_failure})"
                )
            elif (
                winner_id != latched_id
                and winner_sig.condition_matched
                and winner_p >= self.open_threshold
                and winner_p > latched_p
            ):
                # In V24-Naive (w1=0), lack of attractor locking allows mid-run switching
                self.gate = 1
                self.state = "LATCHED"
                self.active_abstract_id = winner_id
                self.active_grounding = winner_sig.grounding
                transition = "attractor_switch"
                reason = (
                    f"competing candidate {winner_id} (p={winner_p:.3f}) overcame "
                    f"latched {latched_id} (p={latched_p:.3f})"
                )
            elif latched_p >= self.maintain_threshold and latched_id in by_id:
                # Winner did not meet condition_matched to take over; keep latched_id
                self.gate = 1
                self.state = "LATCHED"
                self.active_abstract_id = latched_id
                self.active_grounding = by_id[latched_id].grounding
                transition = "sustained_latch"
                reason = (
                    f"retained latch on {latched_id} (p={latched_p:.3f}); "
                    f"challenger {winner_id} lacked condition match"
                )
            else:
                # Signal decayed below maintenance threshold (commonly in V24-Naive where w1=0)
                self.gate = 0
                self.state = "TRACKING"
                self.active_abstract_id = None
                self.active_grounding = None
                transition = "reinhibitory_erase"
                reason = (
                    f"latched activation p={latched_p:.3f} fell below "
                    f"maintain_threshold={self.maintain_threshold:.3f}"
                )

        # ------------------------------------------------------------------
        # 4. Update hDeltaK maintenance layer h_i(t) with Top-1 Fast Inhibition
        # ------------------------------------------------------------------
        h_scores: dict[str, float] = {}
        if self.gate == 1 and self.active_abstract_id is not None:
            active_id = self.active_abstract_id
            prev_h = self.h_state.get(active_id, 0.0)
            active_p = p_scores.get(active_id, 0.0)
            if transition in {"disinhibitory_write", "attractor_switch"} or prev_h <= 1e-6:
                # Disinhibition immediately closes the PFG-hDeltaK loop at full latch potential
                new_h = _clip01(self.w2 * active_p)
            else:
                # Subsequent steps evolve via slow recurrent integration; fast inhibition zeros j != i*
                new_h = _clip01((1.0 - self.alpha) * prev_h + self.alpha * self.w2 * active_p)
            for aid in all_ids:
                h_scores[aid] = new_h if aid == active_id else 0.0
        else:
            for aid in all_ids:
                h_scores[aid] = 0.0
        self.h_state = h_scores

        result = SplitAttractorStepResult(
            step=step_idx,
            mode=self.mode,
            state=self.state,
            gate=self.gate,
            p_scores=dict(p_scores),
            h_scores=dict(h_scores),
            u_scores=u_scores,
            refractory=dict(self.refractory),
            active_abstract_id=self.active_abstract_id,
            previous_abstract_id=prev_abstract_id,
            active_grounding=self.active_grounding,
            transition=transition,
            reason=reason,
            pfg_summary=dict(pfg_summary or {}),
        )
        self.history.append(result)
        return result

    def metrics(self) -> dict[str, Any]:
        """Compute V24 evaluation metrics over the episode trajectory."""
        writes = sum(1 for item in self.history if item.transition == "disinhibitory_write")
        erases = sum(1 for item in self.history if item.transition == "reinhibitory_erase")
        unjustified_switches = sum(
            1 for item in self.history if item.transition == "attractor_switch"
        )
        tracking_steps = sum(1 for item in self.history if item.gate == 0)
        latched_steps = sum(1 for item in self.history if item.gate == 1)
        return {
            "mode": self.mode,
            "total_steps": len(self.history),
            "disinhibitory_writes": writes,
            "reinhibitory_erases": erases,
            "unjustified_switches": unjustified_switches,
            "retraction_latencies": list(self._retraction_latencies),
            "mean_retraction_latency": (
                sum(self._retraction_latencies) / len(self._retraction_latencies)
                if self._retraction_latencies
                else 0.0
            ),
            "tracking_steps": tracking_steps,
            "latched_steps": latched_steps,
        }
