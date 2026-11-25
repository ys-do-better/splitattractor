from __future__ import annotations

import hashlib
import re
from typing import Any

from rcarepair.core import walk as upstream_walk

from ..contracts import FailureEvidence, RCADiagnosis


KEYWORDS = {
    "fde": ("requirement", "expected", "spec", "contract", "unsupported"),
    "domain": ("semantic", "protocol", "standard", "domain", "rule"),
    "schema": ("schema", "$ref", "$defs", "recursive", "nested", "state", "shape", "type"),
    "boundary": ("boundary", "edge", "null", "empty", "overflow", "external", "compat"),
    "logic": ("assert", "wrong", "mismatch", "off-by-one", "exception", "traceback"),
}


class _WalkTarget:
    """Evidence-backed target for the generic upstream five-layer walk."""
    deterministic = True

    def __init__(self, evidence: FailureEvidence, inferred: str,
                 scores: dict[str, int] | None = None) -> None:
        self.e, self.inferred = evidence, inferred
        self.scores = scores or {}

    def _entry_layer(self) -> str:
        """Select between boundary (④) and schema (③) when climbing above logic (⑤)."""
        if self.inferred in {"boundary", "schema"}:
            return self.inferred
        if self.e.metadata.get("outside_value") is not None or self.e.metadata.get("boundary_signal"):
            return "boundary"
        if self.e.metadata.get("schema_signal"):
            return "schema"
        return "boundary" if self.scores.get("boundary", 0) > self.scores.get("schema", 0) else "schema"

    def outside_value(self, _hops: list) -> dict[str, Any] | None:
        v = self.e.metadata.get("outside_value")
        if v is not None:
            return {"value": v, "who_zh": "外部输入"}
        if self.inferred == "boundary" or (
            self.inferred in {"domain", "fde"} and self._entry_layer() == "boundary"
        ):
            observed = self.e.metadata.get("observed_outside_value", 1)
            return {"value": observed, "who_zh": "外部边界输入"}
        return None

    def state_fits(self, _iota: dict, _hops: list) -> dict:
        needs_schema = self.inferred == "schema" or (
            self.inferred in {"domain", "fde"} and self._entry_layer() == "schema"
        )
        ok = not needs_schema
        return {"ok": ok, "zh": "状态符合 schema" if ok else "失败状态无法由现有 schema 表达"}

    def logic_under(self, _iota: dict, _logic: dict) -> dict:
        ok = self.inferred != "logic"
        return {"ok": ok, "zh": "规则在有效输入下成立" if ok else "输入有效但实现逻辑仍产生错误"}

    def rule_known(self, _iota: dict) -> dict:
        ok = self.inferred != "domain"
        return {"ok": ok, "zh": "领域规则已知" if ok else "知识库缺少此领域规则"}

    def case_expected(self, _iota: dict) -> dict:
        ok = self.inferred != "fde"
        return {"ok": ok, "zh": "需求覆盖该场景" if ok else "需求模型未覆盖该场景"}

    def replay_as_is(self, *_args) -> dict:
        fails = bool(self.e.metadata.get("as_is_fails", True))
        return {"fails": fails, "ok": not fails,
                "zh": "真实测试执行复现失败" if fails else "真实测试执行未复现失败"}

    def replay_with_root_right(self, *_args) -> dict:
        succeeds = self.e.metadata.get("counterfactual_succeeds") is True
        return {"fails": not succeeds, "ok": succeeds,
                "pending": not succeeds,
                "zh": ("源码变化后同类测试真实转绿" if succeeds else
                       "尚未观察到源码变化后的同类测试转绿；反事实验证待完成")}

    def class_sites(self, _root: dict, _iota: dict, _logic: dict) -> list[dict]:
        locations = list(self.e.metadata.get("code_locations", ()))
        if locations:
            changed = set(self.e.changed_files)
            locations.sort(key=lambda item: (
                str(item.get("file")) not in changed,
                item.get("source") != "git-diff",
                str(item.get("file", "")).startswith("test"),
                int(item.get("line", 0)),
            ))
            return [dict(item, this=i == 0) for i, item in enumerate(locations[:8])]
        return [{"file": f, "line": 1, "source": "git", "this": i == 0}
                for i, f in enumerate(self.e.changed_files)] or [
            {"file": "unknown", "line": 1, "source": "insufficient-evidence", "this": True}]

    def class_test(self, site: dict) -> tuple[str | None, dict]:
        actual = self.e.metadata.get("site_tests", {}).get(site["file"])
        if actual:
            return actual, {"source": "observed site-specific test"}
        return None, {"source": "not executed; no synthetic class evidence"}


class DeepSWEAgentRCA:
    """Maps DeepSWE task/trajectory/verifier evidence into rca-repair's construction walk."""

    def _infer(self, evidence: FailureEvidence) -> tuple[str, float, dict[str, int]]:
        observed = " ".join((evidence.assertion, evidence.trace,
                             " ".join(evidence.failed_tests))).lower()
        task = evidence.task_text.lower()
        scores = {step: 2 * sum(observed.count(k) for k in keys) +
                        sum(task.count(k) for k in keys)
                  for step, keys in KEYWORDS.items()}
        if evidence.metadata.get("code_locations"):
            scores["logic"] += 2
        if evidence.metadata.get("schema_signal"):
            scores["schema"] += 4
        if evidence.metadata.get("boundary_signal"):
            scores["boundary"] += 4
        root = max(scores, key=lambda k: (scores[k], k == "logic"))
        top = scores[root]
        quality = float(evidence.metadata.get("evidence_quality", .35))
        confidence = min(.98, .55 + .03 * top + .25 * quality) if top else .55 + .2 * quality
        return root, confidence, scores

    def diagnose(self, evidence: FailureEvidence) -> RCADiagnosis:
        root, confidence, scores = self._infer(evidence)
        target = _WalkTarget(evidence, root, scores)
        bound_value = evidence.metadata.get("bound_value")
        if bound_value is None and target.outside_value([]) is not None and evidence.metadata.get("outside_value") is None:
            bound_value = 0
        iota = {"bound_value": bound_value}
        steps, upstream_root = upstream_walk.walk(target, None, [], iota, {"file": "", "line": 0})
        if upstream_root is None:
            raise RuntimeError("RCA construction walk was inconclusive")
        back = upstream_walk.walk_back(target, None, upstream_root, iota, {})
        impacted = tuple(dict.fromkeys(evidence.metadata.get("impacted_symbols", ())))
        boundary = tuple(dict.fromkeys(evidence.metadata.get("boundary_tests", ())))
        raw = {"scores": scores, "walk": steps, "root": upstream_root, "back": back,
               "evidence_quality": evidence.metadata.get("evidence_quality", .35),
               "localization": evidence.metadata.get("code_locations", ()),
               "test_command": evidence.metadata.get("test_command", ""),
               "upstream": "maoyixie/rca-repair@9d4df4a1dcde02ad5f7bdba8f0e697cfc9df4410"}
        return RCADiagnosis(root_step=upstream_root["step"], root_kind=upstream_root["kind"],
                            confidence=confidence, explanation=upstream_root["text_zh"],
                            impacted_symbols=impacted, impacted_tests=tuple(evidence.failed_tests),
                            boundary_tests=boundary, raw=raw)


def diagnosis_digest(d: RCADiagnosis) -> str:
    material = f"{d.root_step}|{d.root_kind}|{d.confidence}|{d.impacted_tests}|{d.boundary_tests}"
    return hashlib.sha256(material.encode()).hexdigest()
