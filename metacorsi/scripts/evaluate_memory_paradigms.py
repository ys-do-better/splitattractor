#!/usr/bin/env python3
"""Comprehensive 20-Task Offline Audit & Real Trajectory Replay across the 4 Memory Paradigms.

Compares:
0. Baseline V23 (`modular_reverse_v23`): Static Step-0 Reverse Grounding
1. Paradigm 1 (`split_attractor_v24` & `naive_latch_v24`): Split-Attractor Working Memory (PFG <-> hDeltaK + Disinhibitory Gate)
2. Paradigm 2 (`sleep_consolidation_v24`): NREM Contrastive Replay + REM Synaptic Downscaling & Vigilance Pruning
3. Paradigm 3 (`predictive_coding_v24`): Hierarchical Predictive Coding & Surprise-Driven Residual Retrieval
4. Paradigm 4 (`engram_ensemble_v24`): Hippocampal Engram Ensemble Competition & Orthogonal Co-Activation
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metacorsi.knowledge.engram_ensemble import EngramEnsembleMemory
from metacorsi.knowledge.indexed_ltm import IndexedLTM
from metacorsi.knowledge.predictive_coding import PredictiveCodingMemory
from metacorsi.knowledge.sleep_consolidation import SleepConsolidationMemory
from metacorsi.knowledge.split_attractor import SplitAttractorWM


# Expected ground-truth primary and secondary causal mechanisms for the 20 DeepSWE tasks
TASK_PROFILES: dict[str, dict[str, Any]] = {
    "abs-module-cache": {
        "prose": "Cache loaded modules in ABS interpreter to prevent duplicate execution and preserve module state",
        "live_failure": "METACORSI_CONFIRMED_FAILURE_EVIDENCE\nevaluator/module.go: module cache state not preserved across nested require calls",
        "expected_cap": "cache-state",
        "expected_loci": ("state-propagation",),
    },
    "abs-stepped-slices": {
        "prose": "Add stepped slice parsing, slicing, and assignment for arrays and strings with rune-correct indexing",
        "live_failure": "METACORSI_CONFIRMED_FAILURE_EVIDENCE\nparser/parser.go: stepped slice syntax [a:b:c] boundary validation failed on negative step",
        "expected_cap": "parsing-schema",
        "expected_loci": ("validation-preservation", "boundary-decision"),
    },
    "actionlint-action-pinning-lint": {
        "prose": "Add action pinning policy rule to actionlint workflow linter",
        "live_failure": "METACORSI_CONFIRMED_FAILURE_EVIDENCE\nrule_action.go: workflow dependency action pin validation failed",
        "expected_cap": "dependency-policy",
        "expected_loci": ("validation-preservation", "boundary-decision"),
    },
    "adaptix-field-alias-mapping": {
        "prose": "Support field alias mapping and bidirectional conversion in Adaptix model loader/dumper",
        "live_failure": "METACORSI_CONFIRMED_FAILURE_EVIDENCE\nsrc/adaptix/load.py: field alias binding mapping failed on nested schema boundary",
        "expected_cap": "binding-type",
        "expected_loci": ("boundary-decision", "normalization"),
    },
    "aiomonitor-task-snapshots-diff": {
        "prose": "Add asyncio task snapshot capture, diffing, and Web API route registration to aiomonitor",
        "live_failure": "METACORSI_CONFIRMED_FAILURE_EVIDENCE\naiomonitor/webui.py: async snapshot Web API route registration dispatch returned 404",
        "expected_cap": "async-lifecycle",
        "expected_loci": ("dispatch-registration", "state-propagation"),
    },
    "anko-default-function-arguments": {
        "prose": "Support default function arguments in Anko script VM and AST invocation",
        "live_failure": "METACORSI_CONFIRMED_FAILURE_EVIDENCE\nvm/vm.go: function default argument binding failed during variadic call",
        "expected_cap": "binding-type",
        "expected_loci": ("boundary-decision", "state-propagation"),
    },
    "anko-typed-variable-bindings": {
        "prose": "Enforce typed variable bindings and type conversion checks in Anko VM scopes",
        "live_failure": "METACORSI_CONFIRMED_FAILURE_EVIDENCE\nvm/env.go: typed variable binding failed to propagate across block scope",
        "expected_cap": "binding-type",
        "expected_loci": ("state-propagation", "validation-preservation"),
    },
    "arcane-drift-baseline-snapshots": {
        "prose": "Track drift baseline snapshots and state comparison in Arcane configuration engine",
        "live_failure": "METACORSI_CONFIRMED_FAILURE_EVIDENCE\nsrc/drift.py: snapshot cache state comparison lost baseline context",
        "expected_cap": "async-lifecycle",
        "expected_loci": ("state-propagation", "validation-preservation"),
    },
    "arktype-json-schema-refs-dependencies": {
        "prose": "Support recursive JSON Schema $ref, $defs, and dependentSchemas in ArkType compiler",
        "live_failure": "METACORSI_CONFIRMED_FAILURE_EVIDENCE\nsrc/jsonSchema.ts: recursive schema state failed to propagate across $ref cycle and validate SCC boundary",
        "expected_cap": "parsing-schema",
        "expected_loci": ("state-propagation", "validation-preservation"),
    },
    "awilix-async-container-initialization": {
        "prose": "Add async container initialization lifecycle and dependency disposal ordering in Awilix",
        "live_failure": "METACORSI_CONFIRMED_FAILURE_EVIDENCE\nsrc/container.ts: async lifecycle init state failed to propagate to scoped child container",
        "expected_cap": "async-lifecycle",
        "expected_loci": ("state-propagation", "boundary-decision"),
    },
    "bandit-incremental-cache-control": {
        "prose": "Implement incremental scan cache control and force-rescan writeback lifecycle in Bandit CLI",
        "live_failure": "METACORSI_CONFIRMED_FAILURE_EVIDENCE\nbandit/core/cache.py: cache state writeback lost on force-rescan transition boundary",
        "expected_cap": "cache-state",
        "expected_loci": ("state-propagation", "boundary-decision"),
    },
    "bandit-interprocedural-taint-analysis": {
        "prose": "Track interprocedural taint propagation across function calls and return values in Bandit",
        "live_failure": "METACORSI_CONFIRMED_FAILURE_EVIDENCE\nbandit/core/taint.py: security taint state failed to propagate across recursive call scope",
        "expected_cap": "security-analysis",
        "expected_loci": ("state-propagation", "boundary-decision"),
    },
    "bandit-structured-nosec-directives": {
        "prose": "Parse structured # nosec directives and bind suppression scope to complete multiline AST statements in Bandit",
        "live_failure": "METACORSI_CONFIRMED_FAILURE_EVIDENCE\nbandit/core/node_visitor.py: nosec directive suppression boundary failed to normalize multiline AST statement",
        "expected_cap": "security-analysis",
        "expected_loci": ("boundary-decision", "normalization"),
    },
    "boa-async-job-cancellation": {
        "prose": "Implement async job queue cancellation and lifecycle cleanup in Boa JS engine",
        "live_failure": "METACORSI_CONFIRMED_FAILURE_EVIDENCE\ncore/engine/src/job.rs: async task cancellation state failed to propagate to pending promise queue",
        "expected_cap": "async-lifecycle",
        "expected_loci": ("state-propagation", "boundary-decision"),
    },
    "cattrs-partial-structuring-recovery": {
        "prose": "Support partial structuring error recovery and path-annotated validation errors in cattrs",
        "live_failure": "METACORSI_CONFIRMED_FAILURE_EVIDENCE\nsrc/cattrs/preconf.py: partial structuring boundary failed to preserve valid fields during validation",
        "expected_cap": "parsing-schema",
        "expected_loci": ("validation-preservation", "boundary-decision"),
    },
    "clack-async-autocomplete-options": {
        "prose": "Support async dynamic autocomplete options, debounce cancellation, and state updates in Clack prompts",
        "live_failure": "METACORSI_CONFIRMED_FAILURE_EVIDENCE\npackages/prompts/src/autocomplete.ts: async lifecycle cancel failed to discard stale option state",
        "expected_cap": "async-lifecycle",
        "expected_loci": ("state-propagation", "boundary-decision"),
    },
    "claude-code-subagent-delegation-policy": {
        "prose": "Enforce subagent delegation depth policy and permission scope propagation",
        "live_failure": "METACORSI_CONFIRMED_FAILURE_EVIDENCE\nsrc/delegation.ts: dependency policy scope state failed to propagate to delegated child agent",
        "expected_cap": "dependency-policy",
        "expected_loci": ("state-propagation", "boundary-decision"),
    },
    "cliffy-config-file-parsing": {
        "prose": "Add hierarchical configuration file parsing, option merging, and validation preservation in Cliffy CLI",
        "live_failure": "METACORSI_CONFIRMED_FAILURE_EVIDENCE\ncommand/command.ts: config parser schema validation failed to preserve CLI flag precedence boundary",
        "expected_cap": "parsing-schema",
        "expected_loci": ("validation-preservation", "boundary-decision"),
    },
    "csstree-shorthand-expansion-compression": {
        "prose": "Implement CSS shorthand expansion and compression roundtrip with canonical background-size normalization",
        "live_failure": "METACORSI_CONFIRMED_FAILURE_EVIDENCE\nlib/shorthand.js: css shorthand expand compress roundtrip failed canonical normalization boundary",
        "expected_cap": "transformation-roundtrip",
        "expected_loci": ("normalization", "boundary-decision"),
    },
    "dasel-html-selector-roundtrip": {
        "prose": "Add HTML selector query and modification roundtrip to Dasel document parser",
        "live_failure": "METACORSI_CONFIRMED_FAILURE_EVIDENCE\ninternal/storage/html.go: html parser roundtrip failed to preserve implicit table boundary",
        "expected_cap": "parsing-schema",
        "expected_loci": ("validation-preservation", "boundary-decision"),
    },
}


def evaluate_20_tasks(ltm: IndexedLTM) -> dict[str, Any]:
    policies = (
        "modular_reverse_v23",
        "naive_latch_v24",
        "split_attractor_v24",
        "sleep_consolidation_v24",
        "predictive_coding_v24",
        "engram_ensemble_v24",
    )
    delta_ids = set((ltm.manifest.get("memory_layers") or {}).get("delta_atom_ids") or ())
    results: dict[str, dict[str, Any]] = {}

    for policy in policies:
        step0_unjustified_cross_task = 0
        fast_delta_uninterfered = 0
        fast_delta_total = 0
        novel_task_total = 0
        live_evidence_causal_hit = 0
        multi_locus_coverage_on_hard = 0
        hard_tasks = {
            "arktype-json-schema-refs-dependencies",
            "bandit-structured-nosec-directives",
            "csstree-shorthand-expansion-compression",
            "aiomonitor-task-snapshots-diff",
        }

        for task_id, prof in sorted(TASK_PROFILES.items()):
            task_key = ltm._matching_task_key(task_id)
            task_recs = [
                ltm.records[aid]
                for aid in ltm.indexes["task"].get(task_key, ())
                if aid in ltm.records
            ]
            has_fast_pass = any(
                r.atom.atom_id in delta_ids and r.outcome == "pass" for r in task_recs
            )
            has_any_pass = any(r.outcome == "pass" for r in task_recs)
            if has_fast_pass:
                fast_delta_total += 1
            if not has_any_pass:
                novel_task_total += 1

            # 1. Step 0 evaluation (before any container command or error trace exists)
            _, g_step0 = ltm.ground(
                task_id=task_id,
                task_text=prof["prose"],
                repository_evidence="",
                policy=policy,
            )
            if not has_any_pass and len(g_step0) > 0:
                # Firing cross-task edit recipe at Step 0 on a novel task without live evidence
                step0_unjustified_cross_task += 1
            if has_fast_pass and len(g_step0) == 0:
                fast_delta_uninterfered += 1

            # 2. Step 1+ evaluation (once live container failure evidence arrives)
            _, g_live = ltm.ground(
                task_id=task_id,
                task_text=prof["prose"],
                repository_evidence=prof["live_failure"],
                policy=policy,
            )
            if g_live:
                rec = g_live[0]
                mod = rec.bindings.get("module", "")
                if any(loc in mod or rec.bindings.get("causal_locus") == loc for loc in prof["expected_loci"]):
                    live_evidence_causal_hit += 1
                if task_id in hard_tasks:
                    # Check if the grounded guidance covers multiple causal facets or contrastive guards
                    action_text = " ".join(rec.instantiated_action).lower()
                    covers_multi = (
                        rec.bindings.get("ensemble_size") == "2"
                        or "contrastive boundary guard" in action_text
                        or "live_symbols" in rec.bindings and rec.bindings["live_symbols"] != "pending-observation"
                    )
                    if covers_multi:
                        multi_locus_coverage_on_hard += 1
            elif has_fast_pass:
                # For tasks with exact fast delta, keeping cross-task interference off is a hit
                live_evidence_causal_hit += 1

        results[policy] = {
            "step0_premature_lock_on_novel_tasks": f"{step0_unjustified_cross_task}/{novel_task_total}",
            "step0_negative_transfer_risk_rate": round(
                step0_unjustified_cross_task / max(1, novel_task_total), 4
            ),
            "fast_delta_non_interference_rate": f"{fast_delta_uninterfered}/{fast_delta_total}",
            "live_evidence_causal_alignment": f"{live_evidence_causal_hit}/20",
            "hard4_multi_boundary_or_symbol_grounding": f"{multi_locus_coverage_on_hard}/4",
        }
    return results


def replay_real_dev5_trajectories(ltm: IndexedLTM) -> dict[str, Any]:
    """Replay actual step-by-step container observations from dev5 trials through dynamic paradigms."""
    bridge_paths = sorted(
        glob.glob(
            str(
                ROOT.parent
                / "runs/dev5_v24_sa/jobs/panel_dev5_v24_sa/*/agent/dsh_bridge.json"
            )
        )
    )
    if not bridge_paths:
        return {}

    replay_summary: dict[str, Any] = {}
    for bp in bridge_paths:
        data = json.load(open(bp, encoding="utf-8"))
        trial_name = Path(bp).parents[1].name
        task_id = trial_name.split("__", 1)[0]
        hist = data.get("history") or []
        prof = next(
            (v for k, v in TASK_PROFILES.items() if k.startswith(task_id[:18])),
            {"prose": task_id},
        )

        # Replay through V24-SA (refined), V24-Naive, and Predictive Coding
        sa_wm = SplitAttractorWM(mode="sa")
        naive_wm = SplitAttractorWM(mode="naive")
        pc_mem = PredictiveCodingMemory(ltm)

        # Step 0
        _, sig0, pfg0 = ltm.score_candidates(
            task_id=task_id, task_text=prof["prose"], repository_evidence=""
        )
        sa_wm.step(sig0, has_live_evidence=False, pfg_summary=pfg0)
        naive_wm.step(sig0, has_live_evidence=False, pfg_summary=pfg0)
        pc_mem.step(task_id=task_id, task_text=prof["prose"], repository_evidence="")

        for item in hist:
            cmds = item.get("commands") or []
            rcs = item.get("return_codes") or []
            rca = item.get("rca") or {}
            analysis = item.get("analysis") or ""
            has_fail = any(c != 0 for c in rcs)
            is_test = any("test" in c or "pytest" in c or "deno" in c or "go test" in c for c in cmds)
            edited = any("apply" in c or "sed" in c or "python" in c or "printf" in c for c in cmds)

            obs_parts = [analysis, " ".join(cmds)]
            if rca:
                obs_parts.append("METACORSI RCA " + json.dumps(rca))
            if has_fail and is_test:
                obs_parts.insert(0, "METACORSI_CONFIRMED_FAILURE_EVIDENCE")
            obs_text = "\n".join(obs_parts)

            _, sig_t, pfg_t = ltm.score_candidates(
                task_id=task_id, task_text=prof["prose"], repository_evidence=obs_text
            )
            counter = bool(has_fail and is_test and edited)
            ordinary_fail = bool(has_fail and not is_test)

            sa_wm.step(
                sig_t,
                has_live_evidence=True,
                explicit_counterevidence=counter,
                ordinary_command_failure=ordinary_fail,
                pfg_summary=pfg_t,
            )
            naive_wm.step(
                sig_t,
                has_live_evidence=True,
                explicit_counterevidence=counter,
                ordinary_command_failure=ordinary_fail,
                pfg_summary=pfg_t,
            )
            pc_mem.step(
                task_id=task_id,
                task_text=prof["prose"],
                repository_evidence=obs_text,
                explicit_counterevidence=counter,
            )

        replay_summary[task_id] = {
            "steps": len(hist),
            "refined_split_attractor_sa": sa_wm.metrics(),
            "naive_latch": naive_wm.metrics(),
            "predictive_coding": {
                "activations": sum(
                    1
                    for x in pc_mem.history
                    if x.transition in {"surprise_driven_activation", "prediction_error_shift"}
                ),
                "falsifications": len(pc_mem.falsified),
                "active_steps": sum(1 for x in pc_mem.history if x.active_abstract_id is not None),
            },
        }
    return replay_summary


def main() -> None:
    k28 = IndexedLTM.load(ROOT / "knowledge/active/ltm", expected_version=28)
    k29 = IndexedLTM.load(
        ROOT / "knowledge/releases/k29_cls_nearmiss3/ltm", expected_version=29
    )
    report = {
        "k28_20task_audit": evaluate_20_tasks(k28),
        "k29_20task_audit": evaluate_20_tasks(k29),
        "dev5_trajectory_replay_on_k28": replay_real_dev5_trajectories(k28),
    }
    out_path = ROOT.parent / "MEMORY_PARADIGMS_COMPARISON.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
