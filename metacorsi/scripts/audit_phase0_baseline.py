from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "vendor" / "rca-repair")):
    if p not in sys.path:
        sys.path.insert(0, p)

from metacorsi.contracts import FailureEvidence
from metacorsi.knowledge.indexed_ltm import IndexedLTM
from metacorsi.rca.adapter import DeepSWEAgentRCA


def load_tasks(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def audit_coordinates(ltm: IndexedLTM) -> dict[str, Any]:
    cap_counts: Counter[str] = Counter()
    locus_counts: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    default_cap_atoms: list[str] = []
    default_locus_atoms: list[str] = []

    for atom_id, record in sorted(ltm.records.items()):
        cap_counts[record.capability] += 1
        failure_counts[record.failure_mode] += 1
        kind_counts[record.atom.kind] += 1
        outcome_counts[record.outcome] += 1
        for locus in record.causal_loci:
            locus_counts[locus] += 1
        if record.capability == "repository-contract":
            default_cap_atoms.append(atom_id)
        if record.causal_loci == ("unspecified",):
            default_locus_atoms.append(atom_id)

    total = max(1, len(ltm.records))
    return {
        "total_atoms": len(ltm.records),
        "by_kind": dict(sorted(kind_counts.items())),
        "by_outcome": dict(sorted(outcome_counts.items())),
        "by_capability": dict(sorted(cap_counts.items(), key=lambda x: (-x[1], x[0]))),
        "by_causal_locus": dict(sorted(locus_counts.items(), key=lambda x: (-x[1], x[0]))),
        "by_failure_mode": dict(sorted(failure_counts.items(), key=lambda x: (-x[1], x[0]))),
        "default_capability_fallback": {
            "count": len(default_cap_atoms),
            "rate": round(len(default_cap_atoms) / total, 4),
            "sample_atom_ids": default_cap_atoms[:10],
        },
        "default_causal_locus_fallback": {
            "count": len(default_locus_atoms),
            "rate": round(len(default_locus_atoms) / total, 4),
            "sample_atom_ids": default_locus_atoms[:10],
        },
    }


def audit_abstractions(ltm: IndexedLTM) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    template_types: Counter[str] = Counter()
    details: list[dict[str, Any]] = []
    for abs_id, ab in sorted(ltm.abstractions.items()):
        status_counts[ab.status] += 1
        if ab.invariant.startswith("Preserve the verified"):
            tpl = "generic_fallback_template"
        elif "State that changes execution semantics" in ab.invariant:
            tpl = "cache_state_propagation_template"
        elif "A parser or schema extension is complete" in ab.invariant:
            tpl = "parsing_schema_validation_template"
        elif "minimal, compilable vertical slice" in ab.invariant:
            tpl = "patch_materialization_template"
        elif "When a partial repair preserves old behavior" in ab.invariant:
            tpl = "contract_boundary_gap_template"
        else:
            tpl = "custom"
        template_types[tpl] += 1
        details.append({
            "abstract_id": abs_id,
            "key": f"{ab.capability}:{ab.failure_mode}:{ab.causal_locus}",
            "status": ab.status,
            "template": tpl,
            "positive_tasks_count": len(ab.positive_tasks),
            "negative_tasks_count": len(ab.negative_tasks),
            "supporting_atoms_count": len(ab.supporting_atom_ids),
            "trigger_tags": list(ab.trigger_tags),
        })
    total = max(1, len(ltm.abstractions))
    generic_count = template_types["generic_fallback_template"]
    return {
        "total_abstractions": len(ltm.abstractions),
        "by_status": dict(sorted(status_counts.items())),
        "by_template_type": dict(sorted(template_types.items(), key=lambda x: (-x[1], x[0]))),
        "generic_fallback_rate": round(generic_count / total, 4),
        "abstractions": details,
    }


def audit_graph_edge_usage(ltm: IndexedLTM) -> dict[str, Any]:
    relation_counts = Counter(edge.get("relation", "unknown") for edge in ltm.edges)
    ltm_src = (ROOT / "metacorsi" / "knowledge" / "indexed_ltm.py").read_text(encoding="utf-8")
    tree = ast.parse(ltm_src)
    methods_reading_edges: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for child in ast.walk(node):
                if isinstance(child, ast.Attribute) and child.attr == "edges":
                    methods_reading_edges.append(node.name)
                    break
    retrieval_reads_edges = any(m in {"retrieve", "ground"} for m in methods_reading_edges)
    return {
        "total_edges": len(ltm.edges),
        "by_relation": dict(sorted(relation_counts.items())),
        "methods_referencing_self_edges": sorted(set(methods_reading_edges)),
        "used_in_retrieve_or_ground": retrieval_reads_edges,
    }


def audit_retrieval_and_grounding(ltm: IndexedLTM, task_ids: list[str], strategy: str = "cls_dual_store") -> dict[str, Any]:
    prev = os.environ.get("METACORSI_MEMORY_STRATEGY")
    os.environ["METACORSI_MEMORY_STRATEGY"] = strategy
    rows: list[dict[str, Any]] = []
    p_atom_selected_as_source: list[str] = []
    no_backbone_tasks: list[str] = []
    no_slow_schema_tasks: list[str] = []
    low_similarity_forced_groundings: list[str] = []

    try:
        for task in task_ids:
            bundle, groundings = ltm.ground(
                task_id=task,
                task_text=task.replace("-", " "),
                policy="modular_reverse_v23",
                limit=14,
            )
            buckets = bundle.audit.get("buckets") or {}
            backbone = list(buckets.get("backbone_capsule") or ())
            delta = list(buckets.get("delta_branch") or ())
            if not backbone:
                no_backbone_tasks.append(task)
            if not groundings:
                no_slow_schema_tasks.append(task)

            g_info = []
            for g in groundings:
                source_kinds = [
                    ltm.records[aid].atom.kind for aid in g.source_atom_ids if aid in ltm.records
                ]
                if "P" in source_kinds or not set(source_kinds).intersection({"R", "F"}):
                    p_atom_selected_as_source.append(task)
                cap_sim = float(g.bindings.get("capability_similarity", "0") or 0)
                causal_sim = float(g.bindings.get("causal_similarity", "0") or 0)
                trig_sim = float(g.bindings.get("trigger_similarity", "0") or 0)
                weight = float(g.bindings.get("grounding_weight", "0") or 0)
                if cap_sim == 0.0 and causal_sim <= 0.2:
                    low_similarity_forced_groundings.append(task)
                g_info.append({
                    "abstract_id": g.abstract_id,
                    "module": g.bindings.get("module"),
                    "source_episode": g.bindings.get("source_episode"),
                    "source_atom_ids": list(g.source_atom_ids),
                    "source_atom_kinds": source_kinds,
                    "grounding_weight": weight,
                    "capability_similarity": cap_sim,
                    "causal_similarity": causal_sim,
                    "trigger_similarity": trig_sim,
                })
            rows.append({
                "task": task,
                "routed_capability": bundle.audit.get("capability"),
                "routed_causal_loci": bundle.audit.get("causal_loci"),
                "backbone_count": len(backbone),
                "fast_delta_count": len(delta),
                "groundings": g_info,
            })
    finally:
        if prev is None:
            os.environ.pop("METACORSI_MEMORY_STRATEGY", None)
        else:
            os.environ["METACORSI_MEMORY_STRATEGY"] = prev

    return {
        "tasks_audited": len(task_ids),
        "tasks_without_backbone_memory": no_backbone_tasks,
        "tasks_without_slow_schema": no_slow_schema_tasks,
        "tasks_with_fallback_P_or_S_only_source_atoms": sorted(set(p_atom_selected_as_source)),
        "tasks_with_low_similarity_grounding": sorted(set(low_similarity_forced_groundings)),
        "per_task": rows,
    }


def audit_rca_walk() -> dict[str, Any]:
    rca = DeepSWEAgentRCA()
    probes = {
        "logic": FailureEvidence(
            task_id="probe-logic",
            assertion="assert mismatch off-by-one traceback exception",
            task_text="fix logic",
        ),
        "boundary": FailureEvidence(
            task_id="probe-boundary",
            assertion="null empty overflow boundary edge external compat",
            task_text="fix boundary",
        ),
        "schema": FailureEvidence(
            task_id="probe-schema",
            assertion="recursive $ref $defs nested schema state shape",
            task_text="fix schema",
        ),
        "domain": FailureEvidence(
            task_id="probe-domain",
            assertion="semantic protocol standard domain rule violation",
            task_text="fix domain",
        ),
        "fde": FailureEvidence(
            task_id="probe-fde",
            assertion="unsupported requirement expected spec contract case",
            task_text="fix fde",
        ),
    }
    results = {}
    all_matched = True
    for expected, ev in probes.items():
        inferred, conf, scores = rca._infer(ev)
        diag = rca.diagnose(ev)
        matched = diag.root_step == expected
        all_matched = all_matched and matched
        results[expected] = {
            "inferred_by_keywords": inferred,
            "walk_root_step": diag.root_step,
            "walk_root_kind": diag.root_kind,
            "confidence": round(conf, 4),
            "matched": matched,
        }
    return {
        "all_five_layers_reachable": all_matched,
        "probes": results,
    }


def audit_runtime_switches() -> dict[str, Any]:
    agent_src = (ROOT / "metacorsi_dsh_agent" / "__init__.py").read_text(encoding="utf-8")
    return {
        "v23_enables_step_gt_1_dynamic_retrieval": "modular_reverse_v23" in [
            "evidence_gated_v20", "failure_locked_v21"
        ],
        "evidence_gated_policies_in_harness": ["evidence_gated_v20", "failure_locked_v21"],
        "candidate_state_manager_called_in_harness": "brain.begin_candidate" in agent_src or "brain.candidates." in agent_src,
        "phase_directive_hardcoded_steps": {
            "DIAGNOSE": "1..4",
            "MATERIALIZE": "5..12",
            "VERIFY": "13..18",
            "CLOSE": "19..max_steps",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 baseline audit for MetaCoRSI V23/K29.")
    parser.add_argument(
        "--knowledge",
        type=Path,
        default=ROOT / "knowledge" / "releases" / "k29_cls_nearmiss3",
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=ROOT / "data" / "deepswe20_ids.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "PHASE0_BASELINE_AUDIT.json",
    )
    args = parser.parse_args()

    knowledge_dir = args.knowledge.resolve()
    current = json.loads((knowledge_dir / "current.json").read_text(encoding="utf-8"))
    ltm = IndexedLTM.load(knowledge_dir / "ltm", expected_version=int(current["version"]))
    task_ids = load_tasks(args.tasks.resolve())

    report = {
        "schema_version": 1,
        "knowledge_path": str(knowledge_dir.relative_to(ROOT)),
        "knowledge_version": ltm.version,
        "coordinate_audit": audit_coordinates(ltm),
        "abstraction_audit": audit_abstractions(ltm),
        "graph_edge_audit": audit_graph_edge_usage(ltm),
        "retrieval_and_grounding_audit": audit_retrieval_and_grounding(ltm, task_ids),
        "rca_five_layer_walk_audit": audit_rca_walk(),
        "runtime_switch_audit": audit_runtime_switches(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "knowledge_version": report["knowledge_version"],
        "total_atoms": report["coordinate_audit"]["total_atoms"],
        "default_capability_fallback": report["coordinate_audit"]["default_capability_fallback"],
        "default_causal_locus_fallback": report["coordinate_audit"]["default_causal_locus_fallback"],
        "total_abstractions": report["abstraction_audit"]["total_abstractions"],
        "generic_fallback_rate": report["abstraction_audit"]["generic_fallback_rate"],
        "graph_edges_used_in_retrieval": report["graph_edge_audit"]["used_in_retrieve_or_ground"],
        "tasks_with_fallback_P_or_S_only_source_atoms": report["retrieval_and_grounding_audit"]["tasks_with_fallback_P_or_S_only_source_atoms"],
        "all_five_rca_layers_reachable": report["rca_five_layer_walk_audit"]["all_five_layers_reachable"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
