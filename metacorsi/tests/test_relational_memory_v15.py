from metacorsi.knowledge.grounding import GroundedKnowledgeGraph
from dataclasses import replace

from metacorsi.knowledge.indexed_ltm import IndexedLTM, render_grounded_application
from metacorsi.knowledge.store import KnowledgeAtom


def atom(task: str, layer: str, kind: str, text: str) -> KnowledgeAtom:
    return KnowledgeAtom(
        f"deepswe.{task}.pass.{kind.lower()}.{layer}", kind, text, "test", 1.0,
        ("parser", "boundary"),
    )


def layered_ltm() -> tuple[IndexedLTM, set[str], set[str]]:
    task = "alpha-parser"
    backbone = {
        atom(task, "base-p", "P", "observe old parser contract"),
        atom(task, "base-m", "M", "verified old parser episode"),
        atom(task, "base-r", "R", "old parser causal model"),
        atom(task, "base-f", "F", "old parser recovery action"),
        atom(task, "base-s", "S", "old parser verification guard"),
    }
    delta = {
        atom(task, "new-r", "R", "new parser causal refinement"),
        atom(task, "new-f", "F", "new parser recovery refinement"),
        atom(task, "new-m", "M", "verified new parser episode"),
    }
    unrelated_delta = atom("beta-cache", "new-f", "F", "new cache recovery action")
    atoms = tuple(backbone | delta | {unrelated_delta})
    ltm = IndexedLTM.build(atoms=atoms, version=30, parent=29,
                           graph=GroundedKnowledgeGraph())
    delta_ids = {item.atom_id for item in delta | {unrelated_delta}}
    ltm.manifest["memory_layers"] = {
        "backbone_version": 22,
        "delta_atom_ids": sorted(delta_ids),
    }
    return ltm, {item.atom_id for item in backbone}, delta_ids


def test_layered_policy_reserves_backbone_and_only_exact_delta_branch():
    ltm, backbone_ids, delta_ids = layered_ltm()
    bundle, _ = ltm.ground(task_id="alpha-parser__trial", task_text="repair parser boundary",
                           policy="layered_v15")
    selected = {item.atom_id for item in bundle.atoms}
    assert backbone_ids.issubset(selected)
    assert len(selected & delta_ids) == 3
    assert "deepswe.beta-cache.pass.f.new-f" not in selected
    assert bundle.audit["buckets"]["backbone_capsule"]
    assert bundle.audit["buckets"]["delta_branch"]


def test_layered_policy_uses_failed_fast_episode_as_rfs_without_promoting_memory():
    task = "near-miss-cache"
    backbone = {
        atom(task, "base-m", "M", "verified stable cache memory"),
        atom(task, "base-f", "F", "verified stable cache action"),
        atom(task, "base-r", "R", "verified stable cache reason"),
        atom(task, "base-s", "S", "verified stable cache guard"),
        atom(task, "base-p", "P", "verified stable cache observation"),
    }
    fast = {
        KnowledgeAtom(f"deepswe.{task}.fail.r.fast-r", "R", "recent cache state transition cause", "verifier", .98, ("cache", "state")),
        KnowledgeAtom(f"deepswe.{task}.fail.f.fast-f", "F", "ground the recent cache transition", "verifier", .96, ("cache", "state")),
        KnowledgeAtom(f"deepswe.{task}.fail.s.fast-s", "S", "verify focused and retained behavior", "verifier", 1.0, ("cache", "state")),
        KnowledgeAtom(f"deepswe.{task}.fail.p.fast-p", "P", "observe the recent cache transition", "verifier", .96, ("cache", "state")),
    }
    ltm = IndexedLTM.build(
        atoms=tuple(backbone | fast), version=29, parent=28,
        graph=GroundedKnowledgeGraph(),
    )
    ltm.manifest["memory_layers"] = {
        "backbone_version": 28,
        "delta_atom_ids": sorted(item.atom_id for item in fast),
    }
    bundle, _ = ltm.ground(
        task_id=task, task_text="repair cache state transition",
        policy="modular_reverse_v23",
    )
    selected = {item.atom_id for item in bundle.atoms}
    assert {item.atom_id for item in backbone}.issubset(selected)
    assert {item.atom_id for item in fast if item.kind in {"R", "F", "S"}}.issubset(selected)
    assert next(item.atom_id for item in fast if item.kind == "P") not in selected
    assert not any(item.kind == "M" and ".fail." in item.atom_id for item in bundle.atoms)


def test_isolated_policy_disables_cross_task_grounding():
    ltm, _, _ = layered_ltm()
    bundle, applications = ltm.ground(
        task_id="alpha-parser__trial", task_text="repair parser boundary",
        policy="isolated_v15",
    )
    rendered = render_grounded_application(bundle, applications, "isolated_v15")
    assert not applications
    assert "STABLE BACKBONE" in rendered
    assert "DELTA BRANCH" in rendered
    assert "RELATIONAL MEMORY" in rendered


def test_transfer_policy_requires_two_target_disjoint_source_tasks_and_hides_direct_delta():
    target = atom("target-parser", "base-f", "F", "parser schema boundary recovery")
    source_a = atom("source-a-parser", "new-f", "F", "parser schema boundary recovery")
    source_b = atom("source-b-parser", "new-f", "F", "parser schema boundary recovery")
    ltm = IndexedLTM.build(
        atoms=(target, source_a, source_b), version=30, parent=29,
        graph=GroundedKnowledgeGraph(),
    )
    ltm.manifest["memory_layers"] = {
        "backbone_version": 22,
        "delta_atom_ids": [source_a.atom_id, source_b.atom_id],
    }
    bundle, applications = ltm.ground(
        task_id="target-parser__trial", task_text="repair parser schema boundary",
        policy="transfer_v15",
    )
    assert not bundle.audit["buckets"]["delta_branch"]
    assert all(item.atom_id not in {source_a.atom_id, source_b.atom_id}
               for item in bundle.atoms)
    # The automatically consolidated abstraction also used the target's backbone
    # episode.  It is therefore not held-out evidence and must not activate.
    assert not applications

    abstract_id = next(iter(ltm.abstractions))
    ltm.abstractions[abstract_id] = replace(
        ltm.abstractions[abstract_id],
        supporting_atom_ids=(source_a.atom_id, source_b.atom_id),
        positive_tasks=("source-a-parser", "source-b-parser"),
        negative_tasks=(),
        status="transferable",
    )
    _, applications = ltm.ground(
        task_id="target-parser__trial", task_text="repair parser schema boundary",
        policy="transfer_v15",
    )
    assert applications
    assert set(applications[0].source_atom_ids) == {source_a.atom_id, source_b.atom_id}


def test_transfer_v16_can_fall_back_to_target_disjoint_capability_node():
    target = KnowledgeAtom(
        "deepswe.target-json.pass.f.base", "F", "existing json schema behavior",
        "test", 1.0, ("targetonly",),
    )
    source_a = KnowledgeAtom(
        "deepswe.source-a-parser.pass.f.delta", "F", "verified patch implementation",
        "test", 1.0, ("sourceonly",),
    )
    source_b = KnowledgeAtom(
        "deepswe.source-b-parser.pass.f.delta", "F", "verified patch implementation",
        "test", 1.0, ("sourceonly",),
    )
    ltm = IndexedLTM.build(
        atoms=(target, source_a, source_b), version=30, parent=29,
        graph=GroundedKnowledgeGraph(),
    )
    ltm.manifest["memory_layers"] = {
        "backbone_version": 22,
        "delta_atom_ids": [source_a.atom_id, source_b.atom_id],
    }
    # Replace the automatically mixed abstraction with the source-only relation
    # that the held-out builder persists.
    source_records = {
        source_a.atom_id: ltm.records[source_a.atom_id],
        source_b.atom_id: ltm.records[source_b.atom_id],
    }
    ltm.abstractions = IndexedLTM._build_abstractions(source_records)

    _, strict = ltm.ground(
        task_id="target-json__trial", task_text="repair json schema recursion",
        policy="transfer_v15",
    )
    _, hierarchical = ltm.ground(
        task_id="target-json__trial", task_text="repair json schema recursion",
        policy="transfer_v16",
    )
    _, adaptive_known = ltm.ground(
        task_id="target-json__trial", task_text="repair json schema recursion",
        policy="adaptive_v17",
    )
    _, adaptive_novel = ltm.ground(
        task_id="novel-json__trial", task_text="repair json schema recursion",
        policy="adaptive_v17",
    )
    assert not strict
    assert hierarchical
    assert set(hierarchical[0].source_atom_ids) == {source_a.atom_id, source_b.atom_id}
    assert not adaptive_known
    assert adaptive_novel
    assert set(adaptive_novel[0].source_atom_ids) == {source_a.atom_id, source_b.atom_id}


def test_competitive_v18_uses_relation_compatibility_instead_of_hard_task_gate():
    target = KnowledgeAtom(
        "deepswe.target-json.pass.f.base", "F", "existing json schema behavior",
        "test", 1.0, ("targetonly",),
    )
    source_a = KnowledgeAtom(
        "deepswe.source-a-parser.pass.f.delta", "F", "verified patch implementation",
        "test", 1.0, ("sourceonly",),
    )
    source_b = KnowledgeAtom(
        "deepswe.source-b-parser.pass.f.delta", "F", "verified patch implementation",
        "test", 1.0, ("sourceonly",),
    )
    ltm = IndexedLTM.build(
        atoms=(target, source_a, source_b), version=30, parent=29,
        graph=GroundedKnowledgeGraph(),
    )
    ltm.manifest["memory_layers"] = {
        "backbone_version": 22,
        "delta_atom_ids": [source_a.atom_id, source_b.atom_id],
    }
    source_records = {
        source_a.atom_id: ltm.records[source_a.atom_id],
        source_b.atom_id: ltm.records[source_b.atom_id],
    }
    ltm.abstractions = IndexedLTM._build_abstractions(source_records)

    _, incompatible_known = ltm.ground(
        task_id="target-json__trial", task_text="repair json schema recursion",
        policy="competitive_v18",
    )
    _, novel = ltm.ground(
        task_id="novel-json__trial", task_text="repair json schema recursion",
        policy="competitive_v18",
    )
    assert not incompatible_known
    assert novel

    # Rebind the existing successful episode to the same causal relation and
    # expose a matching observation. The schema may now reinforce the episode
    # without target support or an exact-task allowlist.
    abstract = next(iter(ltm.abstractions.values()))
    ltm.records[target.atom_id] = replace(
        ltm.records[target.atom_id], causal_loci=(abstract.causal_locus,),
    )
    ltm.abstractions[abstract.abstract_id] = replace(
        abstract, trigger_tags=("schema",),
    )
    _, compatible_known = ltm.ground(
        task_id="target-json__trial", task_text="repair json schema recursion",
        policy="competitive_v18",
    )
    assert compatible_known
    assert set(compatible_known[0].source_atom_ids) == {source_a.atom_id, source_b.atom_id}


def test_abstraction_triggers_require_independent_episode_support():
    repeated_a = KnowledgeAtom(
        "deepswe.source-a-parser.pass.r.one", "R", "first source evidence",
        "test", 1.0, ("sharedsemantic", "templateonly", "full"),
    )
    repeated_a_2 = KnowledgeAtom(
        "deepswe.source-a-parser.pass.f.two", "F", "more first source evidence",
        "test", 1.0, ("sharedsemantic", "templateonly", "files"),
    )
    source_b = KnowledgeAtom(
        "deepswe.source-b-parser.pass.r.one", "R", "second source evidence",
        "test", 1.0, ("sharedsemantic", "different"),
    )
    ltm = IndexedLTM.build(
        atoms=(repeated_a, repeated_a_2, source_b), version=30, parent=29,
        graph=GroundedKnowledgeGraph(),
    )
    abstraction = next(iter(ltm.abstractions.values()))
    assert "sharedsemantic" in abstraction.trigger_tags
    assert "templateonly" not in abstraction.trigger_tags
    assert "full" not in abstraction.trigger_tags
    assert "files" not in abstraction.trigger_tags


def test_bridge_v19_can_join_one_stable_and_one_delta_source_episode():
    source_stable = KnowledgeAtom(
        "deepswe.source-stable-cache.pass.r.base", "R",
        "cache state propagation contract", "test", 1.0, ("cache", "state"),
    )
    source_delta = KnowledgeAtom(
        "deepswe.source-delta-cache.pass.f.delta", "F",
        "cache state propagation repair", "test", 1.0, ("cache", "state"),
    )
    ltm = IndexedLTM.build(
        atoms=(source_stable, source_delta), version=30, parent=29,
        graph=GroundedKnowledgeGraph(),
    )
    ltm.manifest["memory_layers"] = {
        "backbone_version": 22,
        "delta_atom_ids": [source_delta.atom_id],
    }
    _, strict_delta_only = ltm.ground(
        task_id="heldout-cache__trial", task_text="cache state propagation",
        policy="competitive_v18",
    )
    _, bridge = ltm.ground(
        task_id="heldout-cache__trial", task_text="cache state propagation",
        policy="bridge_v19",
    )
    assert not strict_delta_only
    assert bridge
    assert set(bridge[0].source_atom_ids) == {
        source_stable.atom_id, source_delta.atom_id,
    }
    abstraction = ltm.abstractions[bridge[0].abstract_id]
    assert "nested, queued, or cached boundary" in abstraction.invariant
    assert any("state token or handle" in action for action in bridge[0].instantiated_action)


def test_evidence_gated_v20_needs_current_repository_evidence_and_renders_diagnostic_only():
    target_memory = KnowledgeAtom(
        "deepswe.heldout-cache.pass.m.base", "M",
        "verified schema validation contract", "test", 1.0, ("schema", "validation"),
    )
    source_stable = KnowledgeAtom(
        "deepswe.source-stable-cache.pass.r.base", "R",
        "cache state propagation contract", "test", 1.0, ("cache", "state"),
    )
    source_delta = KnowledgeAtom(
        "deepswe.source-delta-cache.pass.f.delta", "F",
        "cache state propagation repair", "test", 1.0, ("cache", "state"),
    )
    ltm = IndexedLTM.build(
        atoms=(target_memory, source_stable, source_delta), version=30, parent=29,
        graph=GroundedKnowledgeGraph(),
    )
    ltm.manifest["memory_layers"] = {
        "backbone_version": 22,
        "delta_atom_ids": [source_delta.atom_id],
    }
    # Model the held-out construction correctly: target memory remains in the
    # local capsule but may not support the cross-task abstraction itself.
    ltm.abstractions = IndexedLTM._build_abstractions({
        source_stable.atom_id: ltm.records[source_stable.atom_id],
        source_delta.atom_id: ltm.records[source_delta.atom_id],
    })

    # The task statement may nominate the right semantic family, but it is not
    # allowed to ground cross-task memory by itself.
    _, task_only = ltm.ground(
        task_id="heldout-cache__trial",
        task_text="cache state propagation",
        repository_evidence="",
        policy="evidence_gated_v20",
    )
    assert not task_only

    _, incompatible = ltm.ground(
        task_id="heldout-cache__trial",
        task_text="cache state propagation",
        repository_evidence="trace: cached child state is not propagated into recursive evaluation",
        policy="evidence_gated_v20",
    )
    assert not incompatible

    abstract = next(iter(ltm.abstractions.values()))
    ltm.records[target_memory.atom_id] = replace(
        ltm.records[target_memory.atom_id], causal_loci=(abstract.causal_locus,),
    )

    # A current-checkout observation must independently expose both the trigger
    # and causal level before the relation becomes a diagnostic prior.
    bundle, grounded = ltm.ground(
        task_id="heldout-cache__trial",
        task_text="cache state propagation",
        repository_evidence="trace: cached child state is not propagated into recursive evaluation",
        policy="evidence_gated_v20",
    )
    assert len(grounded) == 1
    assert set(grounded[0].source_atom_ids) == {
        source_stable.atom_id, source_delta.atom_id,
    }
    assert grounded[0].activation_evidence
    assert all("DIAGNOSTIC" in item or "EVIDENCE TEST" in item
               for item in grounded[0].instantiated_action)

    rendered = render_grounded_application(bundle, grounded, "evidence_gated_v20")
    assert "EVIDENCE-GATED DIAGNOSTIC PRIOR" in rendered
    assert "authority=DIAGNOSIS_ONLY" in rendered
    assert "minimal_actions=" not in rendered
    assert "materialize=" not in rendered


def test_failure_locked_v21_requires_confirmed_failure_and_keeps_task_capability_stable():
    source_stable = KnowledgeAtom(
        "deepswe.source-stable-cache.pass.r.base", "R",
        "cache state propagation contract", "test", 1.0, ("cache", "state"),
    )
    source_delta = KnowledgeAtom(
        "deepswe.source-delta-cache.pass.f.delta", "F",
        "cache state propagation repair", "test", 1.0, ("cache", "state"),
    )
    ltm = IndexedLTM.build(
        atoms=(source_stable, source_delta), version=30, parent=29,
        graph=GroundedKnowledgeGraph(),
    )
    ltm.manifest["memory_layers"] = {
        "backbone_version": 22,
        "delta_atom_ids": [source_delta.atom_id],
    }

    failure = (
        "METACORSI_CONFIRMED_FAILURE_EVIDENCE\n"
        "focused test: cache child state is not propagated into recursive evaluation"
    )
    _, unconfirmed = ltm.ground(
        task_id="heldout-cache__trial", task_text="cache state propagation",
        repository_evidence=failure.replace("METACORSI_CONFIRMED_FAILURE_EVIDENCE", ""),
        policy="failure_locked_v21",
    )
    assert not unconfirmed

    bundle, confirmed = ltm.ground(
        task_id="heldout-cache__trial", task_text="cache state propagation",
        repository_evidence=failure,
        policy="failure_locked_v21",
    )
    assert len(confirmed) == 1
    assert bundle.audit["capability"] == "cache-state"
    rendered = render_grounded_application(bundle, confirmed, "failure_locked_v21")
    assert "DIAGNOSIS_ONLY" in rendered
    assert "minimal_actions=" not in rendered

    # Repository output containing parser vocabulary must not rewrite an
    # Actionlint-like task from dependency-policy into parsing-schema.
    other_bundle, wrong_capability = ltm.ground(
        task_id="heldout-action__trial",
        task_text="pin workflow action dependencies",
        repository_evidence=(
            "METACORSI_CONFIRMED_FAILURE_EVIDENCE\n"
            "parser validation failed at recursive state boundary"
        ),
        policy="failure_locked_v21",
    )
    assert other_bundle.audit["capability"] == "dependency-policy"
    assert not wrong_capability


def test_soft_grounding_v22_composes_ranked_role_bindings_without_failure_gate():
    target_memory = KnowledgeAtom(
        "deepswe.heldout-cache.pass.m.base", "M",
        "verified cache state propagation contract", "test", 1.0,
        ("cache", "state", "propagate"),
    )
    cache_a = KnowledgeAtom(
        "deepswe.source-a-cache.pass.r.base", "R",
        "cache state propagation invariant", "test", 1.0,
        ("cache", "state", "propagate"),
    )
    cache_b = KnowledgeAtom(
        "deepswe.source-b-cache.pass.f.delta", "F",
        "propagate cache state through recursive boundary", "test", 1.0,
        ("cache", "state", "propagate"),
    )
    parser_a = KnowledgeAtom(
        "deepswe.source-a-parser.pass.r.base", "R",
        "parser schema validation contract", "test", 1.0,
        ("parser", "schema", "validation"),
    )
    parser_b = KnowledgeAtom(
        "deepswe.source-b-parser.pass.f.delta", "F",
        "preserve parser validation boundary", "test", 1.0,
        ("parser", "schema", "validation"),
    )
    dependency_a = KnowledgeAtom(
        "deepswe.source-a-action.pass.r.base", "R",
        "workflow dependency pinning contract", "test", 1.0,
        ("workflow", "dependency", "pin"),
    )
    dependency_b = KnowledgeAtom(
        "deepswe.source-b-action.pass.f.delta", "F",
        "preserve pinned action dependency boundary", "test", 1.0,
        ("workflow", "dependency", "pin"),
    )
    atoms = (
        target_memory, cache_a, cache_b, parser_a, parser_b,
        dependency_a, dependency_b,
    )
    ltm = IndexedLTM.build(
        atoms=atoms, version=30, parent=29, graph=GroundedKnowledgeGraph(),
    )
    ltm.manifest["memory_layers"] = {
        "backbone_version": 22,
        "delta_atom_ids": [cache_b.atom_id, parser_b.atom_id, dependency_b.atom_id],
    }
    # Held-out target memory remains a local episode but may not support the
    # cross-task abstractions being evaluated.
    sources = {atom.atom_id: ltm.records[atom.atom_id] for atom in atoms[1:]}
    ltm.abstractions = IndexedLTM._build_abstractions(sources)

    bundle, grounded = ltm.ground(
        task_id="heldout-cache__trial",
        task_text="repair recursive cache state propagation",
        repository_evidence="",
        policy="soft_grounding_v22",
    )

    assert len(grounded) == 3
    weights = [float(item.bindings["grounding_weight"]) for item in grounded]
    assert weights == sorted(weights, reverse=True)
    assert all(weight > 0 for weight in weights)
    assert grounded[0].capability == "cache-state"
    assert "source-a-cache" in grounded[0].bindings["source_situations"]
    assert bundle.audit["buckets"]["layered_capsule"]
    rendered = render_grounded_application(bundle, grounded, "soft_grounding_v22")
    assert rendered.count("SOFT GROUNDED APPLICATION") == 3
    assert "WEIGHTED_HYPOTHESIS" in rendered
    assert "EVIDENCE-GATED" not in rendered


def test_modular_reverse_v23_descends_to_one_episode_and_small_atom_capsule():
    atoms = (
        KnowledgeAtom(
            "deepswe.heldout-cache.pass.m.base", "M",
            "verified cache state propagation contract", "test", 1.0,
            ("cache", "state", "propagate"),
        ),
        KnowledgeAtom(
            "deepswe.source-a-cache.pass.r.base", "R",
            "state created by producer must cross recursive boundary", "test", 1.0,
            ("cache", "state", "propagate", "producer"),
        ),
        KnowledgeAtom(
            "deepswe.source-a-cache.pass.f.base", "F",
            "pass the state handle to the nested consumer", "test", 1.0,
            ("cache", "state", "consumer"),
        ),
        KnowledgeAtom(
            "deepswe.source-a-cache.pass.s.base", "S",
            "verify nested and retained cache behavior", "test", 1.0,
            ("cache", "nested", "verify"),
        ),
        KnowledgeAtom(
            "deepswe.source-b-cache.pass.r.delta", "R",
            "queued cache state propagation invariant", "test", 1.0,
            ("cache", "queue", "state"),
        ),
    )
    ltm = IndexedLTM.build(
        atoms=atoms, version=30, parent=29, graph=GroundedKnowledgeGraph(),
    )
    ltm.manifest["memory_layers"] = {
        "backbone_version": 22,
        "delta_atom_ids": [atoms[-1].atom_id],
    }
    sources = {atom.atom_id: ltm.records[atom.atom_id] for atom in atoms[1:]}
    ltm.abstractions = IndexedLTM._build_abstractions(sources)

    bundle, grounded = ltm.ground(
        task_id="heldout-cache__trial",
        task_text="propagate cache state from producer to nested consumer",
        policy="modular_reverse_v23",
    )

    assert len(grounded) == 1
    item = grounded[0]
    assert item.bindings["module"].startswith("cache-state/")
    assert item.bindings["source_episode"] == "source-a-cache"
    assert 1 <= len(item.source_atom_ids) <= 3
    assert all("source-a-cache" in atom_id for atom_id in item.source_atom_ids)
    local_weight = float(item.bindings["local_episode_weight"])
    residual_weight = float(item.bindings["abstract_residual_weight"])
    assert abs(local_weight + residual_weight - 1.0) < 0.002
    assert local_weight > residual_weight
    assert bundle.audit["buckets"]["layered_capsule"]
    rendered = render_grounded_application(bundle, grounded, "modular_reverse_v23")
    assert "MODULAR REVERSE GROUNDING" in rendered
    assert "ROLE_BOUND_RESIDUAL" in rendered
    assert "SOFT GROUNDED APPLICATION" not in rendered


def test_research_matrix_keeps_every_strategy_sparse_and_target_disjoint(monkeypatch):
    strategies = (
        "v23_modular", "anthropic_search_notes", "anthropic_dreaming",
        "memtree_collapsed", "memtree_traversal", "cls_dual_store",
        "schema_assimilation", "art_resonance", "hippocampal_index",
        "event_segmentation",
    )
    atoms = (
        KnowledgeAtom("deepswe.target-cache.pass.m.base", "M", "local retained behavior", "test", 1.0, ("cache", "state")),
        KnowledgeAtom("deepswe.source-cache.pass.r.base", "R", "propagate state across recursive boundary", "test", 1.0, ("cache", "state", "propagate")),
        KnowledgeAtom("deepswe.source-cache.pass.f.base", "F", "bind producer state to consumer", "test", 1.0, ("cache", "state", "consumer")),
        KnowledgeAtom("deepswe.source-cache.pass.s.base", "S", "verify nested behavior", "test", 1.0, ("cache", "nested", "verify")),
        KnowledgeAtom("deepswe.other-cache.pass.r.delta", "R", "preserve queued state boundary", "test", 1.0, ("cache", "queue", "state")),
    )
    ltm = IndexedLTM.build(atoms=atoms, version=30, parent=29, graph=GroundedKnowledgeGraph())
    ltm.manifest["memory_layers"] = {"backbone_version": 22, "delta_atom_ids": [atoms[-1].atom_id]}
    ltm.abstractions = IndexedLTM._build_abstractions({
        item.atom_id: ltm.records[item.atom_id] for item in atoms[1:]
    })
    for strategy in strategies:
        monkeypatch.setenv("METACORSI_MEMORY_STRATEGY", strategy)
        _, grounded = ltm.ground(
            task_id="target-cache__trial",
            task_text="propagate cache state to nested consumer",
            policy="modular_reverse_v23",
        )
        assert len(grounded) <= 1
        if grounded:
            item = grounded[0]
            assert item.bindings["memory_strategy"] == strategy
            assert 1 <= len(item.source_atom_ids) <= 3
            assert all("target-cache" not in atom_id for atom_id in item.source_atom_ids)
