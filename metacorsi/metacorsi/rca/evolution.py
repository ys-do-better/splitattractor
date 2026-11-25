from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvolutionDiagnosis:
    """Post-verifier RCA for deciding what may become reusable knowledge."""

    root_kind: str
    root_stage: str
    confidence: float
    explanation: str
    recovery: str
    safety: str


def diagnose_verified_outcome(*, reward: float, f2p_passed: int, f2p_total: int,
                              p2p_passed: int, p2p_total: int, patch: str,
                              steps: int, failed_tests: tuple[str, ...],
                              protocol_retries: int = 0,
                              max_steps: int = 24) -> EvolutionDiagnosis:
    """Classify the earliest causal stage contradicted by independent evidence.

    This deliberately distinguishes a bad implementation from a trajectory that
    never materialized an implementation.  Treating both as a generic test
    failure caused the first self-evolution prototype to repeat long exploration.
    """
    missing = max(0, f2p_total - f2p_passed)
    lost = max(0, p2p_total - p2p_passed)
    examples = "; ".join(failed_tests[:3]) or "verifier did not expose test names"

    if reward == 1:
        return EvolutionDiagnosis(
            "verified_success", "promotion", 1.0,
            "Independent verifier confirmed the new contract and preserved behavior.",
            "Reuse the causal patch shape only when the same contract and repository signals match.",
            "Require independent F2P and P2P confirmation before promotion.",
        )

    if not patch.strip():
        exhausted = steps >= max_steps
        root = "patch_materialization_failure" if exhausted else "execution_protocol_failure"
        detail = (
            f"The trajectory used {steps}/{max_steps} steps but produced no transferable patch. "
            "The verifier therefore evaluated the unchanged repository, not the explored implementation."
            if exhausted else
            f"The trajectory stopped after {steps} steps without a transferable patch."
        )
        if protocol_retries:
            detail += f" It also needed {protocol_retries} command-plan protocol retries."
        return EvolutionDiagnosis(
            root, "execution", .99 if exhausted else .94, detail,
            "By step 8 create a minimal vertical slice; by step 12 wire the public API; "
            "by step 18 run one focused test; reserve the final steps for integration and "
            "materialize a git checkpoint even when the implementation is incomplete.",
            "A zero-byte model.patch is never evidence of a capability attempt and must not "
            "be distilled as implementation knowledge.",
        )

    if missing == 0 and lost > 0:
        return EvolutionDiagnosis(
            "preservation_regression", "validation", .99,
            f"The new contract is complete, but {lost} pre-existing tests regressed. Example: {examples}.",
            "Keep the implemented capability fixed; localize the compatibility delta from the exact "
            "P2P failure and make the smallest counterfactual correction.",
            "Do not broaden the feature implementation while only a preservation counterexample remains.",
        )

    if missing > 0 and lost == 0 and f2p_passed > 0:
        return EvolutionDiagnosis(
            "contract_boundary_gap", "contract", .98,
            f"The implementation is reachable and preserves old behavior, but {missing} new-contract "
            f"tests remain. Example: {examples}.",
            "Freeze the working core and derive a boundary matrix from the remaining named tests; "
            "change only the narrow decision point shared by those counterexamples.",
            "Do not rewrite the working core or trade retained P2P behavior for broader speculative coverage.",
        )

    if missing > 0 and lost > 0:
        return EvolutionDiagnosis(
            "mixed_contract_and_preservation_failure", "validation", .97,
            f"The candidate is incomplete ({missing} F2P missing) and regresses {lost} P2P tests. "
            f"Example: {examples}.",
            "Return to the smallest causal implementation boundary; separate feature semantics from "
            "compatibility behavior before adding more cases.",
            "Reject the candidate shape; do not distill its code changes as a reusable fix.",
        )

    return EvolutionDiagnosis(
        "contract_integration_failure", "integration", .98,
        f"A patch exists but no new-contract test can observe the capability ({missing} missing). "
        f"Example: {examples}.",
        "Check public export, registration, dispatch, naming, and build inclusion before extending semantics.",
        "Do not spend the remaining budget on deeper implementation until one end-to-end F2P probe is green.",
    )
