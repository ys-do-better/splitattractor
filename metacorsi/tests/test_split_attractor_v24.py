from __future__ import annotations

from pathlib import Path

from metacorsi.brain import MetaCoRSIBrain
from metacorsi.knowledge.grounding import GroundedKnowledgeGraph
from metacorsi.knowledge.indexed_ltm import IndexedLTM
from metacorsi.knowledge.split_attractor import SplitAttractorWM
from metacorsi.knowledge.store import KnowledgeAtom, VersionedKnowledgeStore
from metacorsi_dsh_agent import _classify_step_feedback


def _build_multi_schema_ltm() -> IndexedLTM:
    """Create an LTM with local backbone + two distinct cross-task schemas (cache vs parser)."""
    atoms = (
        # Target task local backbone atom
        KnowledgeAtom(
            "deepswe.target-app.pass.m.base",
            "M",
            "verified target baseline contract",
            "test",
            1.0,
            ("baseline", "contract"),
        ),
        # Cross-task Schema A: cache-state / state-propagation
        KnowledgeAtom(
            "deepswe.source-a-cache.pass.r.base",
            "R",
            "cache state created by producer must propagate across recursive scope boundary",
            "test",
            1.0,
            ("cache", "state", "propagate", "recursive", "scope"),
        ),
        KnowledgeAtom(
            "deepswe.source-a-cache.pass.f.base",
            "F",
            "pass cache state handle to nested recursive consumer",
            "test",
            1.0,
            ("cache", "state", "propagate", "recursive", "consumer"),
        ),
        KnowledgeAtom(
            "deepswe.source-a-cache.pass.s.base",
            "S",
            "verify nested recursive cache and retained behavior",
            "test",
            1.0,
            ("cache", "recursive", "verify"),
        ),
        KnowledgeAtom(
            "deepswe.source-b-cache.pass.r.delta",
            "R",
            "propagate cache state token through recursive evaluation scope",
            "test",
            1.0,
            ("cache", "state", "propagate", "recursive", "scope"),
        ),
        # Cross-task Schema B: parsing-schema / validation-preservation
        KnowledgeAtom(
            "deepswe.source-a-parser.pass.r.base",
            "R",
            "parser schema extension must validate syntax and preserve grammar compatibility",
            "test",
            1.0,
            ("parser", "schema", "validate", "preserve", "syntax"),
        ),
        KnowledgeAtom(
            "deepswe.source-a-parser.pass.f.base",
            "F",
            "update shared parser validation decision while preserving green grammar cases",
            "test",
            1.0,
            ("parser", "schema", "validate", "preserve", "grammar"),
        ),
        KnowledgeAtom(
            "deepswe.source-b-parser.pass.r.delta",
            "R",
            "parser schema validation must preserve compatibility on legacy syntax",
            "test",
            1.0,
            ("parser", "schema", "validate", "preserve", "syntax"),
        ),
    )
    ltm = IndexedLTM.build(
        atoms=atoms,
        version=30,
        parent=29,
        graph=GroundedKnowledgeGraph(),
    )
    ltm.manifest["memory_layers"] = {
        "backbone_version": 28,
        "delta_atom_ids": [atoms[4].atom_id, atoms[7].atom_id],
    }
    # Build cross-task abstractions from target-disjoint source atoms
    source_records = {a.atom_id: ltm.records[a.atom_id] for a in atoms[1:]}
    ltm.abstractions = IndexedLTM._build_abstractions(source_records)
    return ltm


def test_v24_step0_inhibits_cross_task_schema_while_preserving_local_backbone():
    ltm = _build_multi_schema_ltm()
    bundle, grounded = ltm.ground(
        task_id="target-app__trial",
        task_text="repair recursive cache state propagation in target-app",
        repository_evidence="",
        policy="split_attractor_v24",
    )
    # Local backbone atom is preserved in bundle, but cross-task schema is gated off at Step 0
    assert any("target-app" in a.atom_id for a in bundle.atoms)
    assert grounded == ()


def test_v24_condition_matched_disinhibition_and_live_symbol_binding():
    ltm = _build_multi_schema_ltm()
    # Unrelated test failure that does NOT match cache or parser schema conditions
    _, mismatch_Signals, _ = ltm.score_candidates(
        task_id="target-app__trial",
        task_text="repair target-app",
        repository_evidence="METACORSI_CONFIRMED_FAILURE_EVIDENCE\nnetwork timeout in socket",
        policy="split_attractor_v24",
    )
    wm = SplitAttractorWM(mode="sa")
    step_mismatch = wm.step(mismatch_Signals, has_live_evidence=True)
    assert step_mismatch.gate == 0
    assert step_mismatch.active_grounding is None
    assert step_mismatch.transition == "stay_tracking"

    # Now provide condition-matched live evidence with concrete file and symbol
    live_evidence = (
        "METACORSI_CONFIRMED_FAILURE_EVIDENCE\n"
        "File src/cache/engine.py, in def propagate_recursive_scope\n"
        "AssertionError: cache state failed to propagate across recursive scope"
    )
    _, matched_signals, pfg_summary = ltm.score_candidates(
        task_id="target-app__trial",
        task_text="repair target-app",
        repository_evidence=live_evidence,
        policy="split_attractor_v24",
    )
    step_write = wm.step(matched_signals, has_live_evidence=True, pfg_summary=pfg_summary)
    assert step_write.gate == 1
    assert step_write.state == "LATCHED"
    assert step_write.transition == "disinhibitory_write"
    assert step_write.active_grounding is not None
    assert step_write.active_grounding.capability == "cache-state"
    assert "src/cache/engine.py" in step_write.active_grounding.bindings["live_symbols"]
    assert "propagate_recursive_scope" in step_write.active_grounding.bindings["live_symbols"]


def test_v24_sa_bidirectional_coupling_outperforms_v24_naive_under_sensory_dip():
    """Key V24 ablation: V24-SA (w1 > 0) bridges uninformative intermediate steps while V24-Naive (w1 = 0) drops."""
    ltm = _build_multi_schema_ltm()
    strong_evidence = (
        "METACORSI_CONFIRMED_FAILURE_EVIDENCE\n"
        "src/cache/engine.py: cache state failed to propagate across recursive scope"
    )
    _, step1_signals, _ = ltm.score_candidates(
        task_id="target-app__trial",
        task_text="repair target-app",
        repository_evidence=strong_evidence,
        policy="split_attractor_v24",
    )

    sa_wm = SplitAttractorWM(mode="sa", w1=0.35, alpha=0.5, maintain_threshold=0.22)
    naive_wm = SplitAttractorWM(mode="naive", maintain_threshold=0.22)

    res_sa_1 = sa_wm.step(step1_signals, has_live_evidence=True)
    res_naive_1 = naive_wm.step(step1_signals, has_live_evidence=True)
    assert res_sa_1.gate == 1 and res_naive_1.gate == 1
    latched_id = res_sa_1.active_abstract_id
    assert latched_id == res_naive_1.active_abstract_id

    # Step 2: Agent runs a routine inspection command ("git status --short") with weak/neutral output
    # so u_i(2) drops below maintain_threshold (0.22).
    neutral_evidence = "M src/main.py\n?? tmp.log"
    _, step2_signals, _ = ltm.score_candidates(
        task_id="target-app__trial",
        task_text="repair target-app",
        repository_evidence=neutral_evidence,
        policy="split_attractor_v24",
    )
    # Force low u_score on the neutral step to test the w1 * h(t-1) recurrent bridge directly
    from dataclasses import replace

    low_signals = tuple(replace(s, u_score=0.12, condition_matched=False) for s in step2_signals)

    res_sa_2 = sa_wm.step(low_signals, has_live_evidence=True, ordinary_command_failure=True)
    res_naive_2 = naive_wm.step(low_signals, has_live_evidence=True, ordinary_command_failure=True)

    # V24-SA maintains the attractor via w1 * h_i(1) feedback; V24-Naive loses it!
    assert res_sa_2.gate == 1
    assert res_sa_2.active_abstract_id == latched_id
    assert res_sa_2.transition == "sustained_latch"

    assert res_naive_2.gate == 0
    assert res_naive_2.active_abstract_id is None
    assert res_naive_2.transition == "reinhibitory_erase"


def test_v24_reinhibitory_erasing_and_recovery_to_alternative_schema():
    ltm = _build_multi_schema_ltm()
    wm = SplitAttractorWM(mode="sa")

    # 1. Initially latch onto cache-state schema
    cache_evidence = (
        "METACORSI_CONFIRMED_FAILURE_EVIDENCE\n"
        "src/cache/engine.py: cache state failed to propagate across recursive scope"
    )
    _, sig_cache, _ = ltm.score_candidates(
        task_id="target-app__trial",
        task_text="repair target-app",
        repository_evidence=cache_evidence,
        policy="split_attractor_v24",
    )
    s1 = wm.step(sig_cache, has_live_evidence=True)
    assert s1.gate == 1
    cache_id = s1.active_abstract_id
    assert s1.active_grounding.capability == "cache-state"

    # 2. Ordinary command failure (e.g. grep rc=1) does NOT retract the memory
    s2 = wm.step(sig_cache, has_live_evidence=True, ordinary_command_failure=True)
    assert s2.gate == 1
    assert s2.active_abstract_id == cache_id
    assert s2.transition == "sustained_latch"

    # 3. Explicit counterevidence arrives (e.g. edit failed focused test or RCA gate blocked)
    s3 = wm.step(sig_cache, has_live_evidence=True, explicit_counterevidence=True)
    assert s3.gate == 0
    assert s3.state == "REINHIBITED_RESET"
    assert s3.active_abstract_id is None
    assert s3.transition == "reinhibitory_erase"
    assert wm.refractory.get(cache_id, 0.0) > 0.0

    # 4. New evidence supports the alternative parser-schema hypothesis; system recovers and latches it
    parser_evidence = (
        "METACORSI_CONFIRMED_FAILURE_EVIDENCE\n"
        "src/parser/grammar.py: parser schema validation failed to preserve syntax compatibility"
    )
    _, sig_parser, _ = ltm.score_candidates(
        task_id="target-app__trial",
        task_text="repair target-app",
        repository_evidence=parser_evidence,
        policy="split_attractor_v24",
    )
    s4 = wm.step(sig_parser, has_live_evidence=True)
    assert s4.gate == 1
    assert s4.state == "LATCHED"
    assert s4.transition == "disinhibitory_write"
    assert s4.active_grounding.capability == "parsing-schema"
    assert s4.active_abstract_id != cache_id

    metrics = wm.metrics()
    assert metrics["disinhibitory_writes"] == 2
    assert metrics["reinhibitory_erases"] == 1
    assert metrics["retraction_latencies"] == [0]


def test_brain_step_working_memory_and_dsh_feedback_classifier(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("METACORSI_LTM_POLICY", "split_attractor_v24")
    ltm = _build_multi_schema_ltm()
    store_path = tmp_path / "current.json"
    store = VersionedKnowledgeStore(
        tuple(r.atom for r in ltm.records.values()),
        version=30,
    )
    store.save_active(store_path)
    ltm.commit(tmp_path / "ltm")

    brain = MetaCoRSIBrain(knowledge_path=store_path)
    brain.generate_agent("dsh", "target-app__trial", "repair target-app")

    # Step 0: Inhibited tracking
    prompt0, step0 = brain.step_working_memory(
        "repair target-app",
        task_id="target-app__trial",
        repository_evidence="",
        has_live_evidence=False,
    )
    assert step0.gate == 0
    assert "No abstraction was applicable" in prompt0
    assert brain.working_memory.gate == 0

    # Step 1: Condition-matched disinhibitory write
    prompt1, step1 = brain.step_working_memory(
        "repair target-app",
        task_id="target-app__trial",
        repository_evidence=(
            "METACORSI_CONFIRMED_FAILURE_EVIDENCE\n"
            "src/cache/engine.py: cache state failed to propagate across recursive scope"
        ),
        has_live_evidence=True,
    )
    assert step1.gate == 1
    assert "SPLIT-ATTRACTOR DISINHIBITORY WRITE" in prompt1
    assert brain.working_memory.gate == 1
    assert brain.working_memory.active_abstract_id == step1.active_abstract_id

    # Classify ordinary command error vs explicit test counterevidence after edit
    ordinary = _classify_step_feedback(
        ["grep -n missing_symbol src/cache/engine.py"],
        [1],
        gate_Active=True,
        edited_since_latch=False,
    )
    assert ordinary["ordinary_failure"] is True
    assert ordinary["explicit_counterevidence"] is False

    counter = _classify_step_feedback(
        ["sed -i 's/old/new/' src/cache/engine.py", "pytest tests/test_cache.py"],
        [0, 1],
        gate_Active=True,
        edited_since_latch=True,
    )
    assert counter["edited"] is True
    assert counter["explicit_counterevidence"] is True

    # Step 2: Re-inhibitory reset on counterevidence
    prompt2, step2 = brain.step_working_memory(
        "repair target-app",
        task_id="target-app__trial",
        repository_evidence="pytest tests/test_cache.py failed after edit",
        has_live_evidence=True,
        explicit_counterevidence=counter["explicit_counterevidence"],
    )
    assert step2.gate == 0
    assert "[SPLIT-ATTRACTOR RE-INHIBITORY RESET]" in prompt2
    assert brain.working_memory.gate == 0
    assert len(brain.working_memory.sa_transitions) == 3


def test_v24_on_frozen_k29_prevents_step0_negative_transfer_and_latches_on_evidence():
    k29_root = (
        Path(__file__).resolve().parents[1]
        / "knowledge"
        / "releases"
        / "k29_cls_nearmiss3"
        / "ltm"
    )
    if not (k29_root / "CURRENT.json").exists():
        return
    ltm = IndexedLTM.load(k29_root, expected_version=29)

    # On a held-out task without exact-task positive memory, Step 0 keeps gate closed (no negative transfer)
    _, step0_groundings = ltm.ground(
        task_id="heldout-clack-autocomplete__trial",
        task_text="fix async autocomplete prompt lifecycle cancellation",
        repository_evidence="",
        policy="split_attractor_v24",
    )
    assert step0_groundings == ()

    # Once live repository evidence confirms a matching causal boundary, gate opens and latches Top-1 schema
    _, step1_groundings = ltm.ground(
        task_id="heldout-clack-autocomplete__trial",
        task_text="fix async autocomplete prompt lifecycle cancellation",
        repository_evidence=(
            "METACORSI_CONFIRMED_FAILURE_EVIDENCE\n"
            "src/prompts/autocomplete.ts: async task lifecycle snapshot state propagation failed on cancel"
        ),
        policy="split_attractor_v24",
    )
    assert len(step1_groundings) == 1
    assert "src/prompts/autocomplete.ts" in step1_groundings[0].bindings["live_symbols"]

