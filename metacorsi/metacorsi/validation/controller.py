from __future__ import annotations

from ..contracts import FailureEvidence, RCADiagnosis, RegressionPlan, TestResult
from ..rca.adapter import diagnosis_digest


class RCARegressionController:
    """RCA chooses test scope; regression evidence decides promotion."""

    def plan(self, evidence: FailureEvidence, d: RCADiagnosis) -> RegressionPlan:
        focused = tuple(dict.fromkeys((*evidence.failed_tests, *d.impacted_tests)))
        boundary = tuple(dict.fromkeys(d.boundary_tests))
        impacted = tuple(dict.fromkeys(evidence.metadata.get("impacted_tests", ())))
        broad = len(evidence.changed_files) >= 5 or len(impacted) >= 8
        uncertain = d.confidence < .8
        if broad or d.root_step in {"fde", "domain"}:
            tier, full = "T3", True
        elif d.root_step in {"schema", "boundary"} or uncertain:
            tier, full = "T2", False
        elif impacted:
            tier, full = "T1", False
        else:
            tier, full = "T0", False
        return RegressionPlan(tier=tier, focused=focused, boundary=boundary, impacted=impacted,
                              full=full, reason=f"RCA root={d.root_step}, confidence={d.confidence:.2f}",
                              diagnosis_digest=diagnosis_digest(d))

    def accept(self, plan: RegressionPlan, results: list[TestResult]) -> bool:
        by_group = {x.group: x for x in results}
        required = {"focused"}
        if plan.boundary or plan.tier in {"T2", "T3"}:
            required.add("boundary")
        if plan.impacted or plan.tier in {"T1", "T2", "T3"}:
            required.add("impacted")
        if plan.full:
            required.add("full")
        return required.issubset(by_group) and all(by_group[g].ok for g in required)
