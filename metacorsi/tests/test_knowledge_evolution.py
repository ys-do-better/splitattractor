import json
from pathlib import Path

from metacorsi.brain import MetaCoRSIBrain
from metacorsi.knowledge.evolution import (VerifiedEpisode, collect_episodes,
                                           diagnose_episode, distill_episode, evolve)
from metacorsi.knowledge.store import VersionedKnowledgeStore, retrieve_atoms


def _trial(root: Path, task: str, reward: int, f2p: tuple[int, int], p2p: tuple[int, int]):
    trial = root / task
    (trial / "agent").mkdir(parents=True)
    (trial / "artifacts" / "logs" / "artifacts").mkdir(parents=True)
    (trial / "result.json").write_text(json.dumps({
        "task_name": f"datacurve/{task}",
        "verifier_result": {"rewards": {"reward": reward,
            "f2p_passed": f2p[0], "f2p_total": f2p[1],
            "p2p_passed": p2p[0], "p2p_total": p2p[1]}},
    }), encoding="utf-8")
    (trial / "agent" / "dsh_bridge.json").write_text(json.dumps({
        "history": [{"analysis": "Propagate recursive schema context through nested properties."}]
    }), encoding="utf-8")
    (trial / "artifacts" / "logs" / "artifacts" / "model.patch").write_text(
        "+++ b/src/schema.py\n@@ -1 +1 @@\n+fixed\n", encoding="utf-8")
    return trial


def test_distiller_commits_only_verified_positive_fixes(tmp_path):
    root = tmp_path / "results"
    _trial(root, "json-schema-recursive", 1, (5, 5), (9, 9))
    _trial(root, "html-boundary", 0, (3, 5), (9, 9))
    episodes = collect_episodes((root,))
    atoms = {ep.task: distill_episode(ep) for ep in episodes}
    assert {atom.kind for atom in atoms["json-schema-recursive"]} == {"P", "M", "R", "F", "S"}
    assert {atom.kind for atom in atoms["html-boundary"]} == {"P", "R", "F", "S"}
    assert "boundary matrix" in next(
        atom.content for atom in atoms["html-boundary"] if atom.kind == "F"
    )


def test_rca_distinguishes_materialization_boundary_and_regression():
    empty = VerifiedEpisode("css-api", 0, 0, 10, 100, 100, "exploring", "", "trial",
                            steps=24, protocol_retries=3)
    boundary = VerifiedEpisode("cache-api", 0, 8, 10, 100, 100, "almost", "+++ b/a\n+x", "trial",
                               failed_tests=("cache key case",), steps=12)
    regression = VerifiedEpisode("directive-api", 0, 10, 10, 99, 100, "almost", "+++ b/a\n+x", "trial",
                                 failed_tests=("legacy line union",), steps=14)
    assert diagnose_episode(empty).root_kind == "patch_materialization_failure"
    assert diagnose_episode(boundary).root_kind == "contract_boundary_gap"
    assert diagnose_episode(regression).root_kind == "preservation_regression"
    empty_atoms = distill_episode(empty)
    assert any("step 8" in atom.content for atom in empty_atoms if atom.kind == "F")
    assert all("negative-evidence" in atom.tags for atom in empty_atoms if atom.kind in {"R", "F", "S"})


def test_success_distillation_keeps_semantic_execution_trace_without_commands():
    episode = VerifiedEpisode(
        "parser-api", 1, 4, 4, 9, 9, "all tests pass",
        "+++ b/src/parser.py\n+def parse_config(): pass\n", "trial",
        execution_trace=(
            "Inspect option precedence at the parser boundary -> "
            "Propagate nested config values before environment overrides -> "
            "Focused precedence and retained parser suites pass"
        ),
        implementation_blueprint=(
            "files=src/parser.py; symbols=parse_config, merge_values; "
            "contract-cases=environment, nested"
        ),
    )
    atoms = distill_episode(episode)
    repair = next(atom.content for atom in atoms if atom.kind == "F")
    assert "Propagate nested config values" in repair
    assert "symbols=parse_config, merge_values" in repair
    assert "+++ b/" not in repair


def test_episode_extracts_structural_blueprint_without_patch_body(tmp_path):
    trial = _trial(tmp_path, "parser-blueprint", 1, (2, 2), (3, 3))
    (trial / "artifacts" / "logs" / "artifacts" / "model.patch").write_text(
        "diff --git a/src/parser.py b/src/parser.py\n"
        "+++ b/src/parser.py\n"
        "@@ -1,2 +1,6 @@ class Parser:\n"
        "+def parse_config(value):\n"
        "+    register('nested', value)\n"
        "+    return value\n",
        encoding="utf-8",
    )
    episode = collect_episodes((tmp_path,))[0]
    assert "symbols=parse_config" in episode.implementation_blueprint
    assert "contract-cases=nested" in episode.implementation_blueprint
    assert "return value" not in episode.implementation_blueprint


def test_evolved_ltm_persists_versions_and_retrieves(tmp_path):
    root = tmp_path / "results"
    _trial(root, "json-schema-recursive", 1, (5, 5), (9, 9))
    store_path = tmp_path / "knowledge" / "versions" / "current.json"
    receipt = evolve(store_path, (root,))
    assert receipt["version"] == 8
    assert store_path.exists() and store_path.with_name("v8.json").exists()
    store = VersionedKnowledgeStore.from_file(store_path)
    selected = retrieve_atoms(store.active.atoms, "deep recursive JSON schema nested properties")
    assert selected and any(atom.kind == "F" for atom in selected)
    brain = MetaCoRSIBrain(control_mode="rca_only", knowledge_path=store_path)
    prompt = brain.evolved_knowledge_prompt(
        "recursive JSON schema properties", task_id="json-schema-recursive"
    )
    assert "INDEXED LTM v8" in prompt
    assert "json-schema-recursive" in prompt


def test_evolution_is_idempotent_for_same_evidence(tmp_path):
    root = tmp_path / "results"
    _trial(root, "cache-control", 1, (2, 2), (3, 3))
    store_path = tmp_path / "current.json"
    assert evolve(store_path, (root,))["new_atoms"] > 0
    second = evolve(store_path, (root,))
    assert second["version"] == 8 and second["new_atoms"] == 0
    assert second["candidate_updates"] == 0


def test_partial_patch_is_quarantined_for_same_task(tmp_path):
    root = tmp_path / "results"
    _trial(root, "cache-boundary", 0, (9, 10), (20, 20))
    store_path = tmp_path / "versions" / "current.json"
    receipt = evolve(store_path, (root,))
    registry = json.loads(store_path.with_name("candidates.json").read_text(encoding="utf-8"))
    portfolio = registry["tasks"]["cache-boundary"]
    candidate = portfolio["candidates"][0]
    assert receipt["candidate_updates"] == 1
    assert portfolio["status"] == "quarantined-portfolio"
    assert candidate["f2p_passed"] == 9
    assert "+++ b/src/schema.py" in candidate["patch"]


def test_verified_success_removes_quarantined_candidate(tmp_path):
    failed = tmp_path / "failed"
    passed = tmp_path / "passed"
    _trial(failed, "cache-boundary", 0, (9, 10), (20, 20))
    _trial(passed, "cache-boundary", 1, (10, 10), (20, 20))
    store_path = tmp_path / "versions" / "current.json"
    evolve(store_path, (failed,))
    assert "cache-boundary" in json.loads(
        store_path.with_name("candidates.json").read_text(encoding="utf-8")
    )["tasks"]
    evolve(store_path, (passed,))
    assert "cache-boundary" not in json.loads(
        store_path.with_name("candidates.json").read_text(encoding="utf-8")
    )["tasks"]


def test_exact_task_family_suppresses_semantically_noisy_atoms(tmp_path):
    root = tmp_path / "results"
    _trial(root, "clack-async-cache", 0, (8, 10), (20, 20))
    _trial(root, "awilix-async-container", 1, (10, 10), (20, 20))
    store_path = tmp_path / "current.json"
    evolve(store_path, (root,))
    atoms = VersionedKnowledgeStore.from_file(store_path).active.atoms
    selected = retrieve_atoms(atoms, "clack-async-cache__trial cache behavior")
    assert selected
    assert all("clack-async-cache" in atom.atom_id for atom in selected)


def test_quarantine_negative_does_not_promote_failed_episode(tmp_path):
    root = tmp_path / "results"
    _trial(root, "css-boundary", 0, (9, 10), (20, 20))
    store_path = tmp_path / "knowledge" / "current.json"
    before = VersionedKnowledgeStore.from_file(store_path).active
    receipt = evolve(store_path, (root,), policy="quarantine_negative")
    after = VersionedKnowledgeStore.from_file(store_path).active
    registry = json.loads(store_path.with_name("candidates.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "quarantined-no-promotion"
    assert receipt["new_atoms"] == 0 and receipt["quarantined_episodes"] == 1
    assert after.version == before.version
    assert tuple(atom.atom_id for atom in after.atoms) == tuple(atom.atom_id for atom in before.atoms)
    assert "css-boundary" in registry["tasks"]


def test_positive_only_bootstrap_ignores_failed_episode(tmp_path):
    root = tmp_path / "results"
    _trial(root, "css-retention", 1, (10, 10), (20, 20))
    _trial(root, "unresolved-boundary", 0, (9, 10), (20, 20))
    store_path = tmp_path / "knowledge" / "current.json"
    receipt = evolve(store_path, (root,), policy="positive_only")
    atoms = VersionedKnowledgeStore.from_file(store_path).active.atoms
    assert receipt["status"] == "promoted"
    assert receipt["passes"] == 1 and receipt["quarantined_episodes"] == 1
    assert any("css-retention" in atom.atom_id for atom in atoms)
    assert not any("unresolved-boundary" in atom.atom_id for atom in atoms)


def test_grounding_feedback_rejects_application_without_promoting_failure_atoms(tmp_path):
    root = tmp_path / "results"
    trial = _trial(root, "beta-parser", 0, (9, 10), (20, 20))
    bridge = json.loads((trial / "agent" / "dsh_bridge.json").read_text(encoding="utf-8"))
    bridge["grounding"] = [{
        "grounding_id": "grounding-rejected-1",
        "abstract_id": "abstract.parsing-schema.contract-boundary-gap",
        "repository": "beta",
        "capability": "parsing-schema",
        "context_digest": "ctx",
        "activation_evidence": ["parser"],
        "bindings": {"task": "beta-parser"},
        "instantiated_action": ["apply parser boundary rule"],
    }]
    (trial / "agent" / "dsh_bridge.json").write_text(json.dumps(bridge), encoding="utf-8")
    store_path = tmp_path / "knowledge" / "current.json"
    before = VersionedKnowledgeStore.from_file(store_path).active
    receipt = evolve(store_path, (root,), policy="grounding_feedback")
    after = VersionedKnowledgeStore.from_file(store_path).active
    assert receipt["status"] == "grounding-consolidated-no-promotion"
    assert receipt["new_atoms"] == 0 and receipt["quarantined_episodes"] == 1
    assert after.version == before.version
    assert not any("beta-parser" in atom.atom_id for atom in after.atoms)
    groundings = json.loads(
        (Path(receipt["indexed_ltm"]) / "groundings.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert groundings["status"] == "rejected"


def test_grounding_observation_records_pass_without_duplicating_atoms(tmp_path):
    root = tmp_path / "results"
    _trial(root, "already-known-task", 1, (10, 10), (20, 20))
    store_path = tmp_path / "knowledge" / "current.json"
    before = VersionedKnowledgeStore.from_file(store_path).active

    receipt = evolve(store_path, (root,), policy="grounding_observation")
    after = VersionedKnowledgeStore.from_file(store_path).active

    assert receipt["status"] == "grounding-consolidated-no-promotion"
    assert receipt["new_atoms"] == 0
    assert receipt["passes"] == 0
    assert after.version == before.version
    assert tuple(atom.atom_id for atom in after.atoms) == tuple(
        atom.atom_id for atom in before.atoms
    )
