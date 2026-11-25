import json

from metacorsi.knowledge.grounding import GroundedKnowledgeGraph
from metacorsi.knowledge.indexed_ltm import (GroundingRecord, IndexedLTM,
                                             render_grounded_application)
from metacorsi.knowledge.store import KnowledgeAtom


def task_atoms(task: str, outcome: str = "pass") -> tuple[KnowledgeAtom, ...]:
    return tuple(
        KnowledgeAtom(
            f"deepswe.{task}.{outcome}.{kind.lower()}.{kind}", kind,
            f"{task} parser boundary {kind} verified procedure", "test", 1.0,
            ("parser", "boundary", task),
        )
        for kind in ("P", "M", "R", "F", "S")
    )


def test_retention_anchors_are_stable_when_unrelated_atoms_grow():
    learned = task_atoms("alpha-parser")
    base = IndexedLTM.build(
        atoms=learned, version=11, parent=10, graph=GroundedKnowledgeGraph()
    )
    before = base.retrieve(
        task_id="alpha-parser__trial", task_text="repair alpha parser boundary"
    )
    noise = tuple(
        KnowledgeAtom(
            f"deepswe.noise-{i}.fail.r.x", "R", f"unrelated cache event {i}",
            "noise", .9, ("cache", f"noise{i}"),
        )
        for i in range(200)
    )
    grown = IndexedLTM.build(
        atoms=learned + noise, version=12, parent=11, graph=GroundedKnowledgeGraph()
    )
    after = grown.retrieve(
        task_id="alpha-parser__trial", task_text="repair alpha parser boundary"
    )
    expected = set(base.anchors["alpha-parser"])
    assert set(before.audit["buckets"]["retention_anchor"]) == expected
    assert set(after.audit["buckets"]["retention_anchor"]) == expected
    assert expected.issubset({atom.atom_id for atom in after.atoms})


def test_two_positive_tasks_validate_capability_node():
    atoms = task_atoms("alpha-parser") + task_atoms("beta-html")
    ltm = IndexedLTM.build(
        atoms=atoms, version=12, parent=11, graph=GroundedKnowledgeGraph()
    )
    assert ltm.nodes["capability:parsing-schema"]["status"] == "validated"
    assert ltm.nodes["capability:parsing-schema"]["positive_tasks"] == [
        "alpha-parser", "beta-html"
    ]


def test_version_commit_is_hash_verified_and_reloadable(tmp_path):
    ltm = IndexedLTM.build(
        atoms=task_atoms("alpha-parser"), version=11, parent=10,
        graph=GroundedKnowledgeGraph(),
    )
    version_dir = ltm.commit(tmp_path / "ltm")
    loaded = IndexedLTM.load(tmp_path / "ltm", expected_version=11)
    assert loaded.anchors == ltm.anchors
    assert loaded.indexes == ltm.indexes
    manifest = json.loads((version_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["atom_count"] == 5
    assert manifest["retention_task_count"] == 1


def test_unverified_failure_branch_has_no_retention_anchor():
    ltm = IndexedLTM.build(
        atoms=task_atoms("alpha-parser", "fail"), version=11, parent=10,
        graph=GroundedKnowledgeGraph(),
    )
    assert "alpha-parser" not in ltm.anchors
    assert ltm.nodes["capability:parsing-schema"]["status"] == "unverified"


def test_abstraction_is_reverse_applied_as_a_grounding_record():
    ltm = IndexedLTM.build(
        atoms=task_atoms("alpha-parser") + task_atoms("beta-html"),
        version=12, parent=11, graph=GroundedKnowledgeGraph(),
    )
    bundle, groundings = ltm.ground(
        task_id="gamma-json__trial",
        task_text="parse nested JSON schema boundary",
        repository_evidence="src/parser/schema.py recursive context",
    )
    assert bundle.audit["capability"] == "parsing-schema"
    assert groundings
    assert groundings[0].status == "pending"
    assert groundings[0].bindings["task"] == "gamma-json__trial"
    assert groundings[0].instantiated_action


def test_grounding_only_consolidation_creates_same_version_generation(tmp_path):
    root = tmp_path / "ltm"
    base = IndexedLTM.build(
        atoms=task_atoms("alpha-parser"), version=11, parent=10,
        graph=GroundedKnowledgeGraph(),
    )
    first = base.commit(root)
    abstract_id = next(iter(base.abstractions))
    receipt = GroundingRecord(
        "grounding-1", abstract_id, "beta-parser", "beta", "parsing-schema", "ctx",
        ("parser",), {"task": "beta-parser"}, ("apply parser boundary rule",),
        "verified", 1.0, "trial",
    )
    consolidated = IndexedLTM.build(
        atoms=task_atoms("alpha-parser"), version=11, parent=10,
        graph=GroundedKnowledgeGraph(), groundings=(receipt,),
    )
    second = consolidated.commit(root)
    assert first != second
    assert "-g" in second.name
    loaded = IndexedLTM.load(root, expected_version=11)
    assert loaded.groundings == (receipt,)


def test_capsule_policy_is_compact_and_skips_unverified_abstractions():
    ltm = IndexedLTM.build(
        atoms=task_atoms("alpha-parser") + task_atoms("beta-html"),
        version=12, parent=11, graph=GroundedKnowledgeGraph(),
    )
    bundle, groundings = ltm.ground(
        task_id="gamma-json__trial", task_text="parse nested JSON schema boundary",
        repository_evidence="src/parser.py recursive context", policy="capsule_v2",
    )
    rendered = render_grounded_application(bundle, groundings, "capsule_v2")
    assert groundings
    assert all(len(action) <= 240 for item in groundings for action in item.instantiated_action)
    assert len(rendered) <= 5200


def test_rejected_grounding_is_consolidated_as_negative_evidence():
    atoms = task_atoms("alpha-parser")
    base = IndexedLTM.build(
        atoms=atoms, version=11, parent=10, graph=GroundedKnowledgeGraph(),
    )
    abstract_id = next(iter(base.abstractions))
    rejected = GroundingRecord(
        "g-rejected", abstract_id, "beta-parser", "beta", "parsing-schema", "ctx",
        ("parser",), {"task": "beta-parser"}, ("apply",), "rejected", 0.0, "trial",
    )
    consolidated = IndexedLTM.build(
        atoms=atoms, version=11, parent=10, graph=GroundedKnowledgeGraph(),
        groundings=(rejected,),
    )
    assert "beta-parser" in consolidated.abstractions[abstract_id].negative_tasks


def test_champion_policy_suppresses_rejected_transfer_without_negative_recipe():
    atoms = task_atoms("alpha-parser")
    base = IndexedLTM.build(
        atoms=atoms, version=11, parent=10, graph=GroundedKnowledgeGraph(),
    )
    abstract_id = next(iter(base.abstractions))
    rejected = GroundingRecord(
        "g-rejected", abstract_id, "beta-parser", "beta", "parsing-schema", "ctx",
        ("parser",), {"task": "beta-parser"}, ("apply",), "rejected", 0.0, "trial",
    )
    learned = IndexedLTM.build(
        atoms=atoms, version=11, parent=10, graph=GroundedKnowledgeGraph(),
        groundings=(rejected,),
    )
    _, groundings = learned.ground(
        task_id="beta-parser__retry", task_text="parser boundary",
        policy="champion_v11",
    )
    assert not groundings


def test_boundary_policy_reverse_applies_schema_without_copying_source_patch():
    atoms = task_atoms("alpha-parser") + task_atoms("beta-html")
    ltm = IndexedLTM.build(
        atoms=atoms, version=12, parent=11, graph=GroundedKnowledgeGraph(),
    )
    bundle, groundings = ltm.ground(
        task_id="gamma-json__trial", task_text="parse nested JSON schema boundary parser",
        repository_evidence="src/parser.py schema parser boundary", policy="boundary_v3",
    )
    rendered = render_grounded_application(bundle, groundings, "boundary_v3")
    assert groundings
    assert all(item.scope_guard and item.verification_plan for item in groundings)
    assert all("gamma-json__trial" in " ".join(item.instantiated_action) for item in groundings)
    assert "APPLICATION" in rendered
    assert len(rendered) <= 4600


def test_executable_grounding_records_harness_materialization_protocol():
    ltm = IndexedLTM.build(
        atoms=task_atoms("alpha-parser") + task_atoms("beta-html"),
        version=12, parent=11, graph=GroundedKnowledgeGraph(),
    )
    bundle, groundings = ltm.ground(
        task_id="gamma-json__trial", task_text="parse nested JSON schema boundary parser",
        repository_evidence="src/parser.py schema parser boundary", policy="executable_v4",
    )
    rendered = render_grounded_application(bundle, groundings, "executable_v4")
    assert groundings
    assert all(item.materialization_protocol for item in groundings)
    assert "1600 characters" in rendered


def test_retention_policy_keeps_exact_task_path_when_capability_routing_is_wrong():
    ltm = IndexedLTM.build(
        atoms=task_atoms("alpha-parser") + task_atoms("beta-html"),
        version=12, parent=11, graph=GroundedKnowledgeGraph(),
    )
    bundle, groundings = ltm.ground(
        task_id="alpha-parser__trial",
        task_text="cache invalidation state lifecycle",  # deliberately misroutes
        repository_evidence="cache key generation",
        policy="retention_v8",
    )
    assert bundle.audit["capability"] == "cache-state"
    assert groundings
    assert any("alpha-parser" in item.bindings["task"] for item in groundings)


def test_retention_policy_renders_one_atom_from_each_brain_role():
    ltm = IndexedLTM.build(
        atoms=task_atoms("alpha-parser"), version=12, parent=11,
        graph=GroundedKnowledgeGraph(),
    )
    bundle, groundings = ltm.ground(
        task_id="alpha-parser__trial", task_text="repair parser boundary",
        policy="retention_v8",
    )
    rendered = render_grounded_application(bundle, groundings, "retention_v8")
    for role in ("VERIFIED MEMORY", "RECOVERY ACTION", "ROOT CAUSE",
                 "TRANSITION GUARD", "OBSERVE"):
        assert role in rendered
    assert len(rendered) <= 6800


def test_situation_policy_binds_roles_to_dominant_verified_family():
    task = "alpha-parser"
    common = (
        KnowledgeAtom(f"deepswe.{task}.pass.p.obs", "P", "observe parser contract", "test", 1.0),
        KnowledgeAtom(f"deepswe.{task}.pass.r.cause", "R", "parser boundary causal decision", "test", .95),
        KnowledgeAtom(f"deepswe.{task}.pass.s.guard", "S", "focused and retained tests", "test", 1.0),
    )
    dominant = (
        KnowledgeAtom(f"deepswe.{task}.pass.m.a", "M",
                      "Verified episode changed_files=['src/a.py', 'src/b.py']", "test", 1.0),
        KnowledgeAtom(f"deepswe.{task}.pass.f.a1", "F",
                      "Reusable repair: change ['src/a.py', 'src/b.py']; parser boundary", "test", 1.0),
        KnowledgeAtom(f"deepswe.{task}.pass.f.a2", "F",
                      "Repair family: change ['src/a.py', 'src/b.py']; normalize parser boundary", "test", 1.0),
    )
    competing = (
        KnowledgeAtom(f"deepswe.{task}.pass.m.z", "M",
                      "Verified episode changed_files=['src/z.py']", "test", 1.0),
        KnowledgeAtom(f"deepswe.{task}.pass.f.z", "F",
                      "Competing repair: change ['src/z.py']", "test", 1.0),
    )
    ltm = IndexedLTM.build(
        atoms=common + dominant + competing, version=13, parent=12,
        graph=GroundedKnowledgeGraph(),
    )
    bundle, groundings = ltm.ground(
        task_id=f"{task}__trial", task_text="repair parser boundary",
        policy="situation_v13",
    )
    rendered = render_grounded_application(bundle, groundings, "situation_v13")
    ids = {atom.atom_id for atom in bundle.atoms}
    assert f"deepswe.{task}.pass.m.a" in ids
    assert f"deepswe.{task}.pass.f.z" not in ids
    assert bundle.audit["situation"]["files"] == ["src/a.py", "src/b.py"]
    assert "SITUATION GRAPH" in rendered
    assert "ROLE BINDING" in rendered
    assert len(rendered) <= 6800


def test_situation_policy_assigns_grounding_credit_to_selected_facets():
    task = "alpha-parser"
    atoms = task_atoms(task) + (
        KnowledgeAtom(f"deepswe.{task}.pass.f.alt1", "F",
                      "change ['src/a.py']; alternate one", "test", 1.0),
        KnowledgeAtom(f"deepswe.{task}.pass.f.alt2", "F",
                      "change ['src/a.py']; alternate two", "test", 1.0),
        KnowledgeAtom(f"deepswe.{task}.pass.m.files", "M",
                      "changed_files=['src/a.py']", "test", 1.0),
    )
    base = IndexedLTM.build(
        atoms=atoms, version=13, parent=12, graph=GroundedKnowledgeGraph(),
    )
    preferred = f"deepswe.{task}.pass.f.alt2"
    rejected = f"deepswe.{task}.pass.f.alt1"
    abstract_id = next(iter(base.abstractions))
    receipts = (
        GroundingRecord("g-ok", abstract_id, task, "alpha", "parsing-schema", "ctx1",
                        ("parser",), {"task": task}, ("apply",), "verified", 1.0,
                        "runtime", (preferred,)),
        GroundingRecord("g-bad", abstract_id, task, "alpha", "parsing-schema", "ctx2",
                        ("parser",), {"task": task}, ("apply",), "rejected", 0.0,
                        "runtime", (rejected,)),
    )
    learned = IndexedLTM.build(
        atoms=atoms, version=13, parent=12, graph=GroundedKnowledgeGraph(),
        groundings=receipts,
    )
    bundle, groundings = learned.ground(
        task_id=f"{task}__trial", task_text="repair parser boundary",
        policy="situation_v13",
    )
    selected = bundle.audit["situation"]["roles"]["F"]
    assert preferred in selected
    assert bundle.audit["situation"]["grounding_credit"][preferred] == 1
    assert all(set(item.source_atom_ids).issuperset(selected) for item in groundings)


def test_experience_policy_reuses_verified_role_binding_and_separates_receipts():
    task = "alpha-parser"
    atoms = task_atoms(task) + (
        KnowledgeAtom(f"deepswe.{task}.pass.f.alt1", "F",
                      "change ['src/a.py']; alternate one", "test", 1.0),
        KnowledgeAtom(f"deepswe.{task}.pass.f.alt2", "F",
                      "change ['src/a.py']; alternate two", "test", 1.0),
        KnowledgeAtom(f"deepswe.{task}.pass.m.files", "M",
                      "changed_files=['src/a.py']", "test", 1.0),
    )
    base = IndexedLTM.build(atoms=atoms, version=14, parent=13,
                            graph=GroundedKnowledgeGraph())
    abstract_id = next(iter(base.abstractions))
    preferred = f"deepswe.{task}.pass.f.alt2"
    rejected = f"deepswe.{task}.pass.f.alt1"
    receipts = (
        GroundingRecord("g-ok", abstract_id, task, "alpha", "parsing-schema", "ctx",
                        ("parser",), {"task": task}, ("apply",), "verified", 1.0,
                        "runtime", (preferred,)),
        GroundingRecord("g-bad", abstract_id, task, "alpha", "parsing-schema", "ctx",
                        ("parser",), {"task": task}, ("apply",), "rejected", 0.0,
                        "runtime", (rejected,)),
    )
    learned = IndexedLTM.build(atoms=atoms, version=14, parent=13,
                               graph=GroundedKnowledgeGraph(), groundings=receipts)
    bundle, applications = learned.ground(
        task_id=f"{task}__trial", task_text="repair parser boundary",
        policy="experience_v14",
    )
    assert preferred in bundle.audit["situation"]["roles"]["F"]
    assert bundle.audit["situation"]["preferred_profile"] == [preferred]
    assert applications
    # The selected role portfolio participates in the receipt identity, so a
    # different binding cannot overwrite this verifier observation.
    expected = learned.ground(
        task_id=f"{task}__trial", task_text="repair parser boundary",
        policy="experience_v14",
    )[1][0]
    assert applications[0].grounding_id == expected.grounding_id


def test_experience_policy_explores_new_semantic_trace_before_old_profile():
    task = "alpha-parser"
    old_fix = f"deepswe.{task}.pass.f.old"
    new_fix = f"deepswe.{task}.pass.f.trace"
    atoms = task_atoms(task) + (
        KnowledgeAtom(old_fix, "F", "change ['src/a.py']; old generic recipe", "test", 1.0),
        KnowledgeAtom(new_fix, "F",
                      "change ['src/a.py']; Semantic execution trace: inspect -> bind -> verify",
                      "test", 1.0),
        KnowledgeAtom(f"deepswe.{task}.pass.m.files", "M",
                      "changed_files=['src/a.py']", "test", 1.0),
    )
    base = IndexedLTM.build(atoms=atoms, version=14, parent=13,
                            graph=GroundedKnowledgeGraph())
    abstract_id = next(iter(base.abstractions))
    old_success = GroundingRecord(
        "g-old", abstract_id, task, "alpha", "parsing-schema", "ctx",
        ("parser",), {"task": task}, ("apply",), "verified", 1.0,
        "runtime", (old_fix,),
    )
    learned = IndexedLTM.build(atoms=atoms, version=14, parent=13,
                               graph=GroundedKnowledgeGraph(), groundings=(old_success,))
    bundle, _ = learned.ground(
        task_id=f"{task}__trial", task_text="repair parser boundary",
        policy="experience_v14",
    )
    selected = bundle.audit["situation"]["roles"]["F"]
    assert new_fix in selected
    assert new_fix in bundle.audit["situation"]["untried_semantic_trace"]
