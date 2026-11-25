"""Unit checks for the parts that the w1_ds28e17 run does not exercise: the other method families, the table, the
memories. Run: python3 -m pytest -q tests"""
from rcarepair.core import table
from rcarepair.core.memory import LTM1, LTM2
from rcarepair.methods.families import (PRIORITY, ExperimentalIsolation, ModelBasedDiagnosis,
                                        StatisticalAssociation, choose)


def test_ochiai_ranks_the_faulty_statement_first():
    # unit: (executed in failing runs, executed in passing runs, failing runs that did not execute it)
    spectra = {"s1": (2, 8, 0), "s2": (2, 0, 0), "s3": (1, 5, 1)}
    rank = StatisticalAssociation.ochiai(spectra)
    assert rank[0][0] == "s2" and rank[0][1] == 1.0


def test_ddmin_finds_the_single_failure_inducing_change():
    deltas = list(range(16))
    assert ExperimentalIsolation.ddmin(deltas, lambda sub: 11 in sub) == [11]


def test_ddmin_finds_an_interacting_pair():
    deltas = list(range(8))
    assert sorted(ExperimentalIsolation.ddmin(deltas, lambda sub: 2 in sub and 6 in sub)) == [2, 6]


def test_minimal_hitting_sets():
    # the classic full-adder style: two conflicts sharing one component
    conflicts = [{"A1", "X1"}, {"A1", "A2", "X2"}]
    diags = ModelBasedDiagnosis.minimal_hitting_sets(conflicts)
    assert {"A1"} in diags and all(len(d) <= 2 for d in diags)


def test_method_choice_follows_what_is_in_hand():
    assert choose({"failing_trace"}).key == "causal"
    assert choose({"spectra"}).key == "statistical"
    assert choose({"component_model", "failing_trace"}).key == "model"
    assert choose({"controllable_rerun"}).key == "experimental"
    assert choose(set()) is None or choose(set()).key == "hypothesis"
    assert [m.key for m in PRIORITY][0] == "model"


def test_table_matches_page3():
    assert [e.sym for e in table.cell("Factual", "black")] == ["Bl", "Tn", "Fb", "Rs", "Sw"]
    assert [e.sym for e in table.cell("Conceptual", "white")] == ["Cd", "Lg"]
    assert [e.sym for e in table.cell("Conceptual", "spec")] == ["Sp", "Rw"]
    assert table.ELEMENTS["Rw"].solo and table.ELEMENTS["By"].solo
    assert table.rows_within("white") == ["black", "white"]


def test_memories(tmp_path):
    l1 = LTM1(str(tmp_path / "l1.json"))
    sig = {"bug": "heap-buffer-overflow", "access": "WRITE", "sink": "f"}
    assert l1.match(sig)[0] is None
    l1.add({"signature": sig, "diagnosis": {}})
    assert LTM1(str(tmp_path / "l1.json")).match(sig)[1] == 1.0
    l2 = LTM2(str(tmp_path / "l2.json"))
    l2.add_or_update({"k": "Conceptual", "signature": {"bug": "heap-buffer-overflow"}, "compound": ["Cd"],
                      "tries": 1, "success": 1})
    assert l2.lookup("Conceptual", dict(sig, origin="device"))[0]["compound"] == ["Cd"]
    assert l2.lookup("Factual", sig) == []
