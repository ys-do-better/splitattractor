from metacorsi.brain import MetaCoRSIBrain
from metacorsi.candidate import CandidateStateManager, Phase
from metacorsi.config import REQUIRED_MODEL, load
from metacorsi.contracts import FailureEvidence, RCADiagnosis, TestResult as GateResult
from metacorsi.targets import registry
from metacorsi.knowledge import K7SkillRouter
from metacorsi_dsh_agent import (_answer_search_violation,
                                 _confirmed_focused_failure, _focused_probe,
                                 _json_object, _lock_grounding_receipt,
                                 _runtime_home)
from metacorsi.rca.evidence import (
    RepositoryEvidence, code_locations, failed_tests, is_test_command, observed_failure,
    strong_failure, test_signature as normalize_test_signature
)


def evidence(**kw):
    base = dict(task_id="t", assertion="deep recursive $defs nested property assertions",
                changed_files=("schema.py",), failed_tests=("test_recursive",),
                metadata={"boundary_tests": ["test_nested"], "impacted_tests": ["test_schema"]})
    base.update(kw)
    return FailureEvidence(**base)


def test_v21_unlocks_only_on_executed_test_failure_and_locks_one_relation():
    assert not _confirmed_focused_failure(["grep -R parser src"], [1])
    assert not _confirmed_focused_failure(["pytest -q tests/test_x.py"], [0])
    assert _confirmed_focused_failure(["pytest -q tests/test_x.py"], [1])

    first = [
        {"abstract_id": "abstract.a"},
        {"abstract_id": "abstract.b"},
    ]
    selected, locked, suppressed = _lock_grounding_receipt(first, None)
    assert [item["abstract_id"] for item in selected] == ["abstract.a"]
    assert locked == "abstract.a"
    assert suppressed == ["abstract.b"]

    selected, locked, suppressed = _lock_grounding_receipt(
        [{"abstract_id": "abstract.b"}], locked
    )
    assert not selected
    assert locked == "abstract.a"
    assert suppressed == ["abstract.b"]


def test_model_and_agent_registry():
    cfg = load("configs/default.json")
    assert cfg.model == REQUIRED_MODEL == "DeepSeek-V4.1-Flash-1017"
    assert set(registry) == {"dsh", "mini-swe-agent", "poa"}


def test_rca_only_config_selects_target_and_disables_gate():
    cfg = load("configs/rca_only.json")
    assert cfg.target_agent == "dsh"
    assert cfg.control_mode == "rca_only"
    brain = MetaCoRSIBrain(control_mode=cfg.control_mode)
    generated = brain.generate_agent("mini-swe-agent", "task-1", "recursive JSON Schema $defs")
    assert generated.control_mode == "rca_only"
    assert generated.context()["control_mode"] == "rca_only"
    assert not brain.snapshot()["regression_gate_enabled"]
    assert "K7 VALIDATION" not in brain.skill_router.prompt(
        "recursive JSON Schema $defs", include_validation=False
    )
    diagnosis = brain.analyze(evidence(task_id="task-1"))
    assert diagnosis.root_step == "schema"
    try:
        brain.analyze_and_plan(evidence(task_id="task-1"))
        assert False
    except RuntimeError as exc:
        assert "disabled" in str(exc)


def test_upstream_rca_walk_drives_scope():
    brain = MetaCoRSIBrain()
    d, plan = brain.analyze_and_plan(evidence())
    assert d.root_step == "schema"
    assert d.raw["upstream"].startswith("maoyixie/rca-repair@")
    assert plan.tier == "T2"
    assert plan.focused == ("test_recursive",)
    assert plan.boundary == ("test_nested",)


def test_deep_rca_uses_real_locations_and_keeps_counterfactual_pending():
    ev = evidence(
        assertion="FAILED tests/test_schema.py::test_nested\napp/schema.py:42: AssertionError",
        trace="app/schema.py:42: AssertionError",
        changed_files=("app/schema.py",),
        metadata={
            "as_is_fails": True,
            "counterfactual_succeeds": None,
            "code_locations": ({"file": "app/schema.py", "line": 42, "source": "trace"},),
            "evidence_quality": 1.0,
            "schema_signal": True,
        },
    )
    d = MetaCoRSIBrain(control_mode="rca_only").analyze(ev)
    assert d.raw["localization"][0]["line"] == 42
    assert d.raw["back"]["case"]["pending"]
    assert not d.raw["back"]["ok"]


def test_five_layer_rca_walk_reaches_all_five_layers_and_closes_walk_back():
    brain = MetaCoRSIBrain(control_mode="rca_only")
    layer_cases = (
        ("logic", dict(assertion="assert mismatch off-by-one traceback exception", task_text="fix logic")),
        ("boundary", dict(assertion="null empty overflow boundary edge external compat", task_text="fix boundary")),
        ("schema", dict(assertion="recursive $ref $defs nested schema state shape", task_text="fix schema")),
        ("domain", dict(assertion="semantic protocol standard domain rule violation", task_text="fix domain")),
        ("fde", dict(assertion="unsupported requirement expected spec contract case", task_text="fix fde")),
    )
    for expected_step, kwargs in layer_cases:
        ev = FailureEvidence(
            task_id=f"task-{expected_step}",
            changed_files=("src/core.py",),
            failed_tests=(f"test_{expected_step}",),
            metadata={"as_is_fails": True, "counterfactual_succeeds": True},
            **kwargs,
        )
        diag = brain.analyze(ev)
        assert diag.root_step == expected_step, f"expected {expected_step}, got {diag.root_step}"
        assert diag.raw["back"]["case"]["ok"] is True
        assert diag.raw["back"]["ok"] is True


def test_deep_rca_evidence_extracts_tests_diff_and_locations():
    output = "FAILED tests/test_x.py::test_edge\nsrc/core.py:17: AssertionError"
    diff = "+++ b/src/core.py\n@@ -10,0 +18,2 @@\n+fix"
    assert is_test_command("cd /app && pytest -q tests/test_x.py")
    assert not is_test_command("grep -n pytest README.md")
    assert observed_failure(output)
    assert strong_failure(output)
    assert not strong_failure("src/errors.py:20: ERROR = 'display only'")
    assert normalize_test_signature("  pytest   -q tests/test_x.py  ") == "pytest -q tests/test_x.py"
    assert failed_tests(output) == ("tests/test_x.py::test_edge",)
    locations = code_locations(output, diff)
    assert {x["source"] for x in locations} == {"trace", "git-diff"}
    repo = RepositoryEvidence(status=" M src/core.py", diff_names="src/core.py", diff=diff)
    assert repo.files == ("src/core.py",)
    assert len(repo.digest) == 64


def test_freeze_repair_lease_and_rollback():
    c = CandidateStateManager(max_repair_mutations=1)
    c.explore("v1")
    c.mutate("v2")
    c.checkpoint(); c.freeze()
    try:
        c.mutate("bad")
        assert False
    except PermissionError:
        pass
    d = RCADiagnosis("logic", "wrong_rule", .9, "logic")
    c.authorize_repair(d); c.mutate("v3")
    try:
        c.mutate("v4")
        assert False
    except PermissionError:
        pass
    c.rollback()
    assert c.source == "v2" and c.phase == Phase.ROLLED_BACK


def test_source_mutation_invalidates_evidence():
    c = CandidateStateManager(); c.explore("a")
    digest = c.digest("a")
    assert c.evidence_valid(digest)
    c.mutate("b")
    assert not c.evidence_valid(digest)


def test_promotion_requires_rca_selected_groups():
    brain = MetaCoRSIBrain()
    _, p = brain.begin_candidate("candidate", evidence())
    assert not brain.validate(p, [GateResult("focused", 1, 0)])
    _, p = brain.begin_candidate("candidate2", evidence())
    ok = brain.validate(p, [GateResult("focused", 1, 0), GateResult("boundary", 1, 0),
                            GateResult("impacted", 1, 0)])
    assert ok


def test_dsh_bridge_accepts_only_structured_command_plan():
    got = _json_object('{"analysis":"inspect", "commands":["pwd"], "done":false}')
    assert got["commands"] == ["pwd"]
    noisy = 'dsh: reasoning: use {not json} maybe {"example":1}\n' + \
            '{"analysis":"real", "commands":["ls"], "done":false}'
    assert _json_object(noisy)["commands"] == ["ls"]


def test_dsh_runtime_uses_short_hashed_path_when_configured(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    monkeypatch.setenv("METACORSI_DSH_RUNTIME_ROOT", str(root))
    got = _runtime_home(tmp_path / ("very-long-trial-name-" * 12) / "agent")
    assert got.parent == root
    assert len(got.name) == 16


def test_dsh_rca_gate_replays_tests_not_tool_failures():
    assert _focused_probe("cd /app && git add -A && git commit -m x") is None
    assert _focused_probe("cd /app && pnpm test && git commit -m x") == "cd /app && pnpm test"


def test_answer_access_policy_blocks_gold_search_but_allows_repo_inspection():
    assert _answer_search_violation("cd /app && git fsck --unreachable")
    assert _answer_search_violation("find / -name html_test.go")
    assert _answer_search_violation("cat /verifier/tests.py")
    assert _answer_search_violation("docker exec hidden cat /solution/patch")
    assert _answer_search_violation("cd /app && find . -name '*.go'") is None
    assert _answer_search_violation("cd /app && git log -5 --oneline") is None


def test_k7_router_is_semantic_and_always_preserves_candidate():
    routed = K7SkillRouter().route("Resolve deep recursive $defs and nested JSON Schema properties")
    ids = {item.skill_id for item in routed}
    assert "coding.propagate_recursive_schema_context" in ids
    assert "coding.preserve_best_candidate_before_targeted_repair" in ids
    assert "coding.model_html_implicit_table_closure" not in ids


def test_full_brain_generates_pinned_target_and_commits_validated_delta():
    brain = MetaCoRSIBrain()
    generated = brain.generate_agent("dsh", "task-1", "recursive JSON Schema $defs")
    assert generated.target == "dsh"
    assert generated.knowledge_version == 7
    assert generated.perception_policy.knowledge_version == 7
    assert {x.kind for x in brain.ltm.active.atoms} == {"P", "M", "R", "F", "S"}

    _, plan = brain.begin_candidate("candidate", evidence(task_id="task-1"))
    results = [GateResult("focused", 1, 0), GateResult("boundary", 1, 0),
               GateResult("impacted", 1, 0)]
    assert brain.validate(plan, results)
    assert brain.ltm.active.version == 8
    assert brain.ltm.active.parent == 7
    assert any(item["kind"] == "knowledge-commit" for item in brain.snapshot()["monitor"])
