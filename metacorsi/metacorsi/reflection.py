from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Counterexample:
    test: str
    expected: str
    actual: str
    evidence: str
    invariant_hint: str


@dataclass(frozen=True)
class ReflectionRecord:
    root_kind: str
    result_signature: str
    counterexamples: tuple[Counterexample, ...]
    invariant: str
    next_strategy: str
    alternative_hypotheses: tuple[str, ...]

    def json(self) -> dict:
        # Return the same JSON-native shape before and after persistence.  In
        # particular, tuples must not compare unequal to the lists produced by
        # json.loads(), otherwise replaying identical evidence looks like a new
        # reflection and falsely advances the evolution state.
        return json.loads(json.dumps(asdict(self), ensure_ascii=False))


_MISMATCH = re.compile(r"MismatchError:\s*(.+?)\s*!=\s*(.+?)(?:\s*$|\n)", re.S)
_JS_EXPECTED = re.compile(r"expected\s+(.+?)\s+to\s+(?:be|equal)\s+(.+?)(?:\s*$|\n)", re.I | re.S)


def parse_counterexample(text: str) -> Counterexample:
    name, _, message = text.partition(" :: ")
    expected = "unspecified"
    actual = "unspecified"
    match = _MISMATCH.search(message)
    if match:
        expected, actual = match.group(1).strip(), match.group(2).strip()
    else:
        match = _JS_EXPECTED.search(message)
        if match:
            actual, expected = match.group(1).strip(), match.group(2).strip()
        elif "unexpected keyword argument" in message:
            expected, actual = "public API accepts the requested argument", "argument rejected by API"
        elif "is not a function" in message:
            expected, actual = "public callable is exported", "callable is absent"
        elif "missing from report" in message:
            expected, actual = "test is discovered and executed", "test/capability path is absent"

    lower = name.lower()
    if "region_unioned_across_statement_lines" in lower:
        hint = ("For one logical AST statement spanning physical lines, suppression selectors "
                "from every covered line must be unioned before issue filtering.")
    elif "alias_type_error_non_mapping" in lower:
        hint = ("Alias presence checks require a mapping-type guard first; non-mapping input must "
                "flow through the framework's TypeLoadError/AggregateLoadError path, never Python membership.")
    elif "alias_conflict_required_all_mode_no_spurious_not_found" in lower:
        hint = ("In DebugTrail.ALL, an alias collision is accumulated as the conflict error only; "
                "the conflicted required field must be marked handled so no spurious not-found error is added.")
    elif "cacheresults" in lower or "result caching" in lower:
        hint = ("Equivalent logical searches must share one successful provider result across "
                "the core and wrapper layers; cache lookup must precede provider invocation.")
    elif "recursive" in lower or "$ref" in lower:
        hint = "Recursive references must preserve resolution context through every nested boundary."
    elif expected != "unspecified":
        hint = f"Preserve the invariant that expected {expected}, rather than observed {actual}."
    else:
        hint = "Derive the smallest state invariant contradicted by this verifier counterexample."
    return Counterexample(name.strip(), expected[:240], actual[:240], message[:520], hint)


def reflect_outcome(*, root_kind: str, task: str, f2p_passed: int, f2p_total: int,
                    p2p_passed: int, p2p_total: int,
                    failures: tuple[str, ...]) -> ReflectionRecord:
    counterexamples = tuple(parse_counterexample(item) for item in failures[:8])
    signature_material = "|".join((
        task, f"{f2p_passed}/{f2p_total}", f"{p2p_passed}/{p2p_total}",
        *(f"{x.test}:{x.expected}:{x.actual}" for x in counterexamples),
    ))
    signature = hashlib.sha256(signature_material.encode()).hexdigest()[:16]
    invariant = " ".join(dict.fromkeys(x.invariant_hint for x in counterexamples)) or (
        "Materialize one callable vertical slice and obtain an executable counterexample before expanding."
    )
    if root_kind == "preservation_regression":
        strategy = ("Freeze the complete F2P implementation. Reproduce only the named P2P counterexample, "
                    "identify the changed state relation, and apply one counterfactual compatibility delta.")
        hypotheses = ("range/state aggregation is narrower than the logical object",
                      "new precedence changed a legacy default", "compatibility adapter bypasses the new core")
    elif root_kind == "contract_boundary_gap":
        strategy = ("Freeze the passing core. Cluster remaining F2P counterexamples by invariant and change "
                    "only their nearest shared decision point.")
        hypotheses = ("boundary normalization occurs after dispatch", "wrapper and core use different state",
                      "cache/recursion identity is scoped too narrowly")
    elif root_kind == "contract_integration_failure":
        strategy = ("Stop semantic expansion. Prove registration, export, dispatch, build inclusion, and one "
                    "end-to-end call before implementing more cases.")
        hypotheses = ("public export missing", "registration path missing", "implementation excluded from build")
    else:
        strategy = ("Create and checkpoint a minimal vertical slice, then run one decisive focused test before "
                    "continuing implementation.")
        hypotheses = ("trajectory never materialized", "test path did not execute", "protocol budget displaced delivery")
    return ReflectionRecord(root_kind, signature, counterexamples, invariant, strategy, hypotheses)
