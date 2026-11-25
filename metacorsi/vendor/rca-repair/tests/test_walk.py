"""The walk up the build steps (core/walk.py) on a fake target: one question per step, the root is the first step
whose inputs are right. Run: python3 -m pytest -q --assert=plain -p no:anyio tests"""
from rcarepair.core import walk


class Fake:
    """answers the walk's questions from a dict of switches"""

    def __init__(self, outside=255, fits=True, under=True, known=True, expected=True, replay_ok=True, others_fail=True,
                 as_is_fails=True):
        self.o, self.f, self.u, self.k, self.e = outside, fits, under, known, expected
        self.r, self.of, self.af = replay_ok, others_fail, as_is_fails

    def outside_value(self, hops):
        return None if self.o is None else {"value": self.o, "who": "device", "who_zh": "设备"}

    def state_fits(self, iota, hops):
        return {"ok": self.f, "zh": "fits" if self.f else "does not fit"}

    def logic_under(self, iota, L):
        return {"ok": self.u, "zh": "ok" if self.u else "fails"}

    def rule_known(self, iota):
        return {"ok": self.k, "zh": "", "not_at_boundary_zh": "rule elsewhere"}

    def case_expected(self, iota):
        return {"ok": self.e, "zh": ""}

    def replay_as_is(self, root, iota, L):
        return {"fails": self.af, "zh": ""}

    def replay_with_root_right(self, root, iota, L):
        return {"ok": self.r, "zh": ""}

    def class_sites(self, root, iota, L):
        return [{"line": 1, "this": True}, {"line": 2, "this": False}]

    def class_test(self, site):
        return ("t", {}) if self.of else (None, None)


IOTA = {"bound_value": 32}


def run(**kw):
    steps, root = walk.walk(Fake(**kw), None, [], IOTA, {})
    return [s["step"] for s in steps], root["step"]


def test_outside_breaks_a_known_rule_root_is_the_boundary():
    assert run() == (["logic", "boundary"], "boundary")


def test_rule_nowhere_in_the_system_root_is_domain_knowledge():
    assert run(known=False) == (["logic", "boundary", "domain"], "domain")


def test_case_not_expected_root_is_the_requirement():
    assert run(expected=False) == (["logic", "boundary", "fde"], "fde")


def test_inputs_right_root_is_the_logic():
    assert run(outside=10) == (["logic"], "logic")
    assert run(outside=None, under=False) == (["logic"], "logic")


def test_state_does_not_fit_root_is_the_schema():
    assert run(outside=None, fits=False) == (["logic", "schema"], "schema")


def test_walk_back_needs_this_case_and_the_class():
    t = Fake()
    _, root = walk.walk(t, None, [], IOTA, {})
    assert walk.walk_back(t, None, root, IOTA, {})["ok"]
    assert not walk.walk_back(Fake(replay_ok=False), None, root, IOTA, {})["ok"]
    assert not walk.walk_back(Fake(others_fail=False), None, root, IOTA, {})["ok"]
    assert not walk.walk_back(Fake(as_is_fails=False), None, root, IOTA, {})["ok"]  # the control must fail


class Noisy(Fake):
    """an agent-like target: answers differ from run to run (deck v17 page 6), scripted so the tests are exact"""
    deterministic, runs = False, 5

    def __init__(self, outsides, as_is, fixed, found=(True,) * 5, **kw):
        super().__init__(**kw)
        self.seq = {"o": list(outsides), "a": list(as_is), "f": list(fixed), "c": list(found)}

    def _next(self, k):
        s = self.seq[k]
        s.append(s.pop(0))
        return s[-1]

    def outside_value(self, hops):
        return {"value": self._next("o"), "who": "user", "who_zh": "用户"}

    def replay_as_is(self, root, iota, L):
        return {"fails": self._next("a"), "zh": ""}

    def replay_with_root_right(self, root, iota, L):
        return {"ok": self._next("f"), "zh": ""}

    def class_test(self, site):
        return ("t", {}) if self._next("c") else (None, None)


ALL, NONE = (True,) * 5, (False,) * 5


def test_agent_answers_that_agree_in_most_runs_stand():
    steps, root = walk.walk(Noisy([255, 255, 10, 255, 255], ALL, ALL), None, [], IOTA, {})
    assert root["step"] == "boundary" and steps[0]["checks"][0]["agree"] == 4


def test_agent_answers_that_split_are_not_taken():
    steps, root = walk.walk(Noisy([255, 10, 255, 10, 255], ALL, ALL), None, [], IOTA, {})
    assert root is None and steps[-1]["verdict"] == "unsure"


def test_agent_walk_back_compares_with_replaying_as_it_is():
    t = Noisy([255] * 5, ALL, ALL)
    _, root = walk.walk(t, None, [], IOTA, {})
    back = walk.walk_back(t, None, root, IOTA, {})
    assert back["ok"] and back["case"]["as_is_fails"] == 5 and back["case"]["fixed_fails"] == 0
    assert back["case"]["p"] < walk.ALPHA
    # fails 3 of 5 as it is and 2 of 5 with the step done right: not clearly rarer, so not explained
    t = Noisy([255] * 5, (True, True, True, False, False), (True, True, True, False, False))
    assert not walk.walk_back(t, None, root, IOTA, {})["ok"]
    # this case is explained, but the other place the same step made fails in only 3 of 5 runs
    t = Noisy([255] * 5, ALL, ALL, found=(True, True, True, False, False))
    assert not walk.walk_back(t, None, root, IOTA, {})["ok"]


def test_fisher_numbers():
    assert abs(walk.rarer(5, 0, 5) - 1 / 252) < 1e-12
    assert walk.rarer(3, 0, 5) > walk.ALPHA > walk.rarer(4, 0, 5)
    assert walk.rarer(0, 0, 5) == 1
