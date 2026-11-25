from metacorsi.knowledge.grounding import (
    GroundedKnowledgeGraph,
    abstraction_atom,
    retrieve_grounded_atoms,
)
from metacorsi.knowledge.store import KnowledgeAtom


def atom(atom_id: str, content: str = "normalize parser boundary") -> KnowledgeAtom:
    return KnowledgeAtom(atom_id, "R", content, "test", 1.0,
                         ("parser", "boundary", "normalize"))


def test_task_leaf_does_not_leak_to_unrelated_task():
    graph = GroundedKnowledgeGraph()
    leaf = atom("deepswe.alpha-config.pass.r.1")
    graph.register_task_atom(
        leaf, task="alpha-config", capability="parsing-schema",
        root_kind="verified_success", positive=True, source="trial",
    )
    selected = retrieve_grounded_atoms(
        (leaf,), graph, task_id="beta-cache", task_text="cache provider results",
    )
    assert not selected


def test_provisional_abstraction_is_soft_hypothesis_not_hard_isolation():
    graph = GroundedKnowledgeGraph()
    abstract = abstraction_atom("parsing-schema", "verified_success")
    status = graph.register_abstraction(
        abstract, capability="parsing-schema", root_kind="verified_success",
        positive_tasks=("alpha-config",), negative_tasks=(), child_ids=("leaf-a",),
    )
    selected = retrieve_grounded_atoms(
        (abstract,), graph, task_id="beta-html",
        task_text="parse html document format and schema boundary",
        repository_evidence="src/parser.py tests/test_html.py",
    )
    assert status == "provisional-abstraction"
    assert selected and selected[0].status == "provisional-abstraction"
    assert "verify before use" in selected[0].grounding[0]


def test_two_independent_groundings_promote_capability_node():
    graph = GroundedKnowledgeGraph()
    abstract = abstraction_atom("parsing-schema", "verified_success")
    status = graph.register_abstraction(
        abstract, capability="parsing-schema", root_kind="verified_success",
        positive_tasks=("alpha-config", "beta-html"), negative_tasks=(),
        child_ids=("leaf-a", "leaf-b"),
    )
    assert status == "validated-abstraction"
    selected = retrieve_grounded_atoms(
        (abstract,), graph, task_id="gamma-json",
        task_text="JSON schema parser boundary",
        repository_evidence="src/schema/parser.py",
    )
    assert selected and selected[0].status == "validated-abstraction"
    assert graph.payload["edges"] == [
        {"source": abstract.atom_id, "target": "leaf-a", "relation": "ABSTRACTS_FROM"},
        {"source": abstract.atom_id, "target": "leaf-b", "relation": "ABSTRACTS_FROM"},
    ]


def test_negative_grounding_prevents_transferable_status():
    graph = GroundedKnowledgeGraph()
    abstract = abstraction_atom("binding-type", "verified_success")
    status = graph.register_abstraction(
        abstract, capability="binding-type", root_kind="verified_success",
        positive_tasks=("a", "b", "c"), negative_tasks=("d",), child_ids=(),
    )
    assert status == "validated-abstraction"
