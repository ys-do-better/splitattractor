from metacorsi.reflection import parse_counterexample, reflect_outcome


def test_bandit_assertion_becomes_statement_union_invariant():
    text = (
        "[p2p] test_058_region_unioned_across_statement_lines :: "
        "MismatchError: [] != ['B602']"
    )
    counterexample = parse_counterexample(text)
    assert counterexample.expected == "[]"
    assert counterexample.actual == "['B602']"
    assert "logical AST statement" in counterexample.invariant_hint


def test_reflection_proposes_alternative_hypotheses_for_stagnation():
    record = reflect_outcome(
        root_kind="preservation_regression", task="bandit-structured",
        f2p_passed=69, f2p_total=69, p2p_passed=281, p2p_total=282,
        failures=("test_058 :: MismatchError: [] != ['B602']",),
    )
    assert record.result_signature
    assert len(record.alternative_hypotheses) >= 2
    assert "Freeze" in record.next_strategy
