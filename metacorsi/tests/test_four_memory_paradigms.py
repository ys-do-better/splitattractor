from __future__ import annotations

from pathlib import Path

from metacorsi.knowledge.engram_ensemble import EngramEnsembleMemory
from metacorsi.knowledge.grounding import GroundedKnowledgeGraph
from metacorsi.knowledge.indexed_ltm import IndexedLTM, render_grounded_application
from metacorsi.knowledge.predictive_coding import PredictiveCodingMemory
from metacorsi.knowledge.sleep_consolidation import SleepConsolidationMemory
from metacorsi.knowledge.store import KnowledgeAtom
from test_split_attractor_v24 import _build_multi_schema_ltm


def test_sleep_consolidation_nrem_contrastive_guards_and_rem_pruning():
    ltm = _build_multi_schema_ltm()
    # Add a near-miss fail atom and a noisy unverified atom to test NREM & REM phases
    near_miss_s = KnowledgeAtom(
        "deepswe.source-a-cache.fail.s.nearmiss",
        "S",
        "Do not discard recursive scope token when cache hit bypasses child evaluation",
        "verifier",
        0.98,
        ("cache", "state", "recursive", "scope"),
    )
    noisy_fail = KnowledgeAtom(
        "deepswe.noisy-task.fail.r.noise",
        "R",
        "unverified random failure without positive evidence",
        "verifier",
        0.5,
        ("random",),
    )
    all_atoms = tuple(r.atom for r in ltm.records.values()) + (near_miss_s, noisy_fail)
    ltm2 = IndexedLTM.build(
        atoms=all_atoms,
        version=31,
        parent=30,
        graph=GroundedKnowledgeGraph(),
    )
    sleep_mem = SleepConsolidationMemory(ltm2)

    # Unverified noisy schema must be pruned by REM synaptic downscaling
    pruned_schemas = [s for s in sleep_mem.schemas.values() if s.pruned]
    active_schemas = [s for s in sleep_mem.schemas.values() if not s.pruned]
    assert pruned_schemas
    assert active_schemas

    # Contrastive guard from near-miss S atom must be woven into the cache schema
    cache_schemas = [s for s in active_schemas if s.capability == "cache-state"]
    assert cache_schemas
    assert "CONTRASTIVE BOUNDARY GUARD" in cache_schemas[0].contrastive_guard

    # Grounding on a matching cache task succeeds above REM vigilance threshold
    bundle, grounded = ltm2.ground(
        task_id="novel-cache-app__trial",
        task_text="propagate cache state across recursive scope boundary",
        repository_evidence="src/cache.py recursive scope state missing",
        policy="sleep_consolidation_v24",
    )
    assert len(grounded) == 1
    assert grounded[0].bindings["memory_strategy"] == "sleep_consolidation"
    rendered = render_grounded_application(bundle, grounded, "sleep_consolidation_v24")
    assert "SLEEP-CONSOLIDATED SCHEMA" in rendered


def test_predictive_coding_zero_step0_surprise_and_error_driven_shift():
    ltm = _build_multi_schema_ltm()
    pc = PredictiveCodingMemory(ltm)

    # Step 0: No live observation -> zero prediction error -> zero cross-task activation
    _, step0 = pc.step(
        task_id="target-app__trial",
        task_text="repair target-app",
        repository_evidence="",
    )
    assert step0.error_signal.surprise_magnitude == 0.0
    assert step0.active_grounding is None
    assert step0.transition == "zero_surprise_top_down_only"

    # Step 1: Routine directory check -> low surprise -> still top-down only
    _, step1 = pc.step(
        task_id="target-app__trial",
        task_text="repair target-app",
        repository_evidence="README.md\nsrc/\ntests/",
    )
    assert step1.active_grounding is None

    # Step 2: Diagnostic test failure on cache recursive scope -> high surprise activates cache schema
    _, step2 = pc.step(
        task_id="target-app__trial",
        task_text="repair target-app",
        repository_evidence=(
            "METACORSI_CONFIRMED_FAILURE_EVIDENCE\n"
            "AssertionError in src/cache/scope.py: recursive cache state failed to propagate"
        ),
    )
    assert step2.error_signal.surprise_magnitude >= 0.5
    assert step2.active_grounding is not None
    assert step2.active_grounding.capability == "cache-state"
    first_id = step2.active_abstract_id

    # Step 3: Edit under cache hypothesis fails with parser validation syntax error (falsification surprise!)
    _, step3 = pc.step(
        task_id="target-app__trial",
        task_text="repair target-app",
        repository_evidence=(
            "METACORSI_CONFIRMED_FAILURE_EVIDENCE\n"
            "AssertionError in src/parser/grammar.py: parser schema failed to validate syntax and preserve grammar"
        ),
        explicit_counterevidence=True,
    )
    assert first_id in step3.falsified_ids
    assert step3.active_grounding is not None
    assert step3.active_grounding.capability == "parsing-schema"
    assert step3.active_abstract_id != first_id


def test_engram_ensemble_co_activates_orthogonal_complementary_loci():
    # Build two orthogonal causal-locus schemas within parsing-schema:
    # 1) state-propagation (recursive $ref node identity)
    # 2) validation-preservation (schema grammar validation compatibility)
    atoms = (
        KnowledgeAtom(
            "deepswe.src-ref-a.pass.r.1",
            "R",
            "recursive schema state context must propagate across nested ref scope",
            "test",
            1.0,
            ("schema", "recursive", "state", "propagate", "scope"),
        ),
        KnowledgeAtom(
            "deepswe.src-ref-b.pass.f.1",
            "F",
            "propagate recursive state handle across nested schema scope",
            "test",
            1.0,
            ("schema", "recursive", "state", "propagate", "scope"),
        ),
        KnowledgeAtom(
            "deepswe.src-val-a.pass.r.1",
            "R",
            "schema parser extension must validate syntax and preserve compatibility",
            "test",
            1.0,
            ("schema", "parser", "validate", "preserve", "compatibility"),
        ),
        KnowledgeAtom(
            "deepswe.src-val-b.pass.f.1",
            "F",
            "validate schema boundary and preserve legacy grammar compatibility",
            "test",
            1.0,
            ("schema", "parser", "validate", "preserve", "compatibility"),
        ),
    )
    ltm = IndexedLTM.build(
        atoms=atoms,
        version=32,
        parent=31,
        graph=GroundedKnowledgeGraph(),
    )
    engram_mem = EngramEnsembleMemory(ltm)
    assert engram_mem.co_edges  # Orthogonal complementary loci form CO_ACTIVATES edges

    bundle, grounded, audit = engram_mem.ground(
        task_id="arktype-json-schema-refs__trial",
        task_text="recursive schema state propagate scope and validate parser preserve compatibility",
        repository_evidence="src/schema.ts recursive state propagate and validate preserve compatibility",
    )
    assert len(grounded) == 1
    assert audit["ensemble_size"] == 2
    assert grounded[0].bindings["ensemble_size"] == "2"
    assert "STAGE 1 PRIMARY ENGRAM" in grounded[0].instantiated_action[0]
    assert "STAGE 2 SYNERGISTIC CO-ENGRAM" in grounded[0].instantiated_action[1]
