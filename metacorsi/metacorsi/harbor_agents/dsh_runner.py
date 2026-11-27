from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from metacorsi.harbor_agents.common import dsh_executable, harbor_executable
from metacorsi.io import atomic_json, read_dotenv
from metacorsi.knowledge.evolution import evolve


def reward(result: dict) -> float | None:
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    if isinstance(rewards, dict):
        value = rewards.get("reward")
        return float(value) if isinstance(value, (int, float)) else None
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks-file", type=Path, required=True)
    p.add_argument("--result-dir", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--dsh-exe", help="DSH CLI; auto-discovered on Windows and Linux")
    p.add_argument("--dsh-call-timeout", type=int, default=360)
    p.add_argument("--dsh-max-steps", type=int, default=24)
    p.add_argument("--dsh-max-extension", type=int, default=6,
                   help="evidence-gated extra CLOSE steps after a successful focused test")
    p.add_argument("--knowledge-store", type=Path,
                   help="persistent versioned LTM; updated only after independent verification")
    p.add_argument("--candidate-registry", type=Path,
                   help="same-task quarantined partial candidates; never promoted as skills")
    p.add_argument("--disable-candidate-replay", action="store_true",
                   help="frozen evaluation: do not inject a stored same-task patch")
    p.add_argument("--read-only-knowledge", action="store_true",
                   help="evaluate without committing verifier outcomes to LTM or candidates")
    p.add_argument("--evolution-policy", choices=(
                       "standard", "positive_only", "quarantine_negative", "grounding_feedback",
                   ),
                   default="standard",
                   help="control whether failed episodes enter active LTM or task-local quarantine")
    p.add_argument("--ltm-policy", choices=(
                       "full_v1", "capsule_v2", "boundary_v3", "executable_v4",
                       "retention_v8", "hierarchical_v9", "feedback_weighted_v10", "champion_v11",
                       "situation_v13", "experience_v14", "isolated_v15", "layered_v15",
                       "consensus_v15", "transfer_v15", "transfer_v16", "adaptive_v17",
                       "competitive_v18", "bridge_v19", "evidence_gated_v20",
                       "failure_locked_v21", "soft_grounding_v22", "modular_reverse_v23",
                       "split_attractor_v24", "naive_latch_v24",
                   ), default="full_v1",
                   help="abstract/grounding rendering and activation policy")
    p.add_argument("--memory-strategy", choices=(
                       "v23_modular", "anthropic_search_notes", "anthropic_dreaming",
                       "memtree_collapsed", "memtree_traversal", "cls_dual_store",
                       "schema_assimilation", "art_resonance", "hippocampal_index",
                       "event_segmentation", "split_attractor",
                   ), default="v23_modular",
                   help="research-matrix memory organization used by modular_reverse_v23")
    p.add_argument("--dsh-patch", type=Path,
                   help="DSH runtime patch config; defaults to configs/dsh.cordis.patch.yml")
    p.add_argument("--dsh-materialize-patch", type=Path,
                   help="optional higher-budget DSH patch used only in MATERIALIZE phase")
    p.add_argument("--dsh-command-max-chars", type=int, default=240)
    p.add_argument("--dsh-stateless-compiler", action="store_true")
    p.add_argument("--dotenv", type=Path, required=True,
                   help="required provider key/base_url file; DSH has no account/quota fallback")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--job-name", default="deepswe_dsh_metaco_k7_rca_1017_fixed20_20261105")
    p.add_argument("--control-mode", choices=("rca_only", "rca_with_regression_gates"),
                   default="rca_only")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    dsh_patch = args.dsh_patch or root / "configs" / "dsh.cordis.patch.yml"
    if not dsh_patch.is_absolute():
        dsh_patch = (root / dsh_patch).resolve()
    materialize_patch = args.dsh_materialize_patch
    if materialize_patch and not materialize_patch.is_absolute():
        materialize_patch = (root / materialize_patch).resolve()
    knowledge_store = args.knowledge_store or root / "knowledge" / "versions" / "current.json"
    if not knowledge_store.is_absolute():
        knowledge_store = (root / knowledge_store).resolve()
    candidate_registry = args.candidate_registry or knowledge_store.with_name("candidates.json")
    if not candidate_registry.is_absolute():
        candidate_registry = (root / candidate_registry).resolve()
    ids = [x.strip() for x in args.tasks_file.read_text(encoding="utf-8").splitlines()
           if x.strip() and not x.lstrip().startswith("#")]
    if not ids:
        raise SystemExit("task file is empty")
    jobs = args.result_dir / "jobs"
    job = args.job_name
    jobs.mkdir(parents=True, exist_ok=True)
    harbor = harbor_executable()
    dsh = args.dsh_exe or dsh_executable()
    cmd = [harbor, "run", "-d", "datacurve/deep-swe-1-1",
           "--agent", "metacorsi_dsh_agent:AgentHarness", "-m", f"openai/{args.model}",
           "-k", "1", "-n", str(args.concurrency), "--jobs-dir", str(jobs), "--job-name", job]
    for task in ids:
        cmd += ["--include-task-name", f"datacurve/{task}"]
    env = os.environ.copy()
    secret = read_dotenv(args.dotenv)
    # Official DeepSeek examples use ``model`` while older MetaCoRSI provider
    # files use ``default_model``.  Accept both spellings but keep the strict
    # requested-model check so an experiment cannot silently hit another model.
    provider_model = secret.get("model") or secret.get("default_model")
    missing = [key for key in ("key", "base_url") if not secret.get(key)]
    if not provider_model:
        missing.append("model (or default_model)")
    if missing:
        raise SystemExit(f"provider dotenv is missing required fields: {', '.join(missing)}")
    if provider_model != args.model:
        raise SystemExit(
            f"provider model mismatch: dotenv={provider_model!r}, requested={args.model!r}"
        )
    parsed_base = urlparse(secret["base_url"])
    if parsed_base.scheme not in {"http", "https"} or not parsed_base.hostname:
        raise SystemExit("provider base_url must be an absolute HTTP(S) endpoint")
    env.update({"OPENAI_API_KEY": secret["key"],
                "OPENAI_API_BASE": secret["base_url"],
                "OPENAI_API_BASE_URL": secret["base_url"]})
    env.update({"METACORSI_DSH_EXE": dsh,
                "METACORSI_DSH_PATCH": str(dsh_patch),
                "METACORSI_DSH_COMMAND_MAX_CHARS": str(args.dsh_command_max_chars),
                "METACORSI_DSH_RUNTIME_ROOT": str(root / ".dsh-runtime"),
                "METACORSI_DSH_CALL_TIMEOUT": str(args.dsh_call_timeout),
                "METACORSI_DSH_MAX_STEPS": str(args.dsh_max_steps),
                "METACORSI_DSH_MAX_EXTENSION": str(args.dsh_max_extension),
                "METACORSI_KNOWLEDGE_STORE": str(knowledge_store),
                "METACORSI_CANDIDATE_REGISTRY": str(candidate_registry),
                "METACORSI_DSH_API_KEY": env["OPENAI_API_KEY"],
                "METACORSI_DSH_BASE_URL": env["OPENAI_API_BASE"],
                "DSH_TELEMETRY_MODE": "DISABLED",
                "METACORSI_CONTROL_MODE": args.control_mode,
                "METACORSI_LTM_POLICY": args.ltm_policy,
                "METACORSI_MEMORY_STRATEGY": args.memory_strategy,
                "PYTHONPATH": os.pathsep.join((str(root), str(root / "vendor" / "rca-repair"),
                                                env.get("PYTHONPATH", ""))),
                "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
    if materialize_patch:
        env["METACORSI_DSH_MATERIALIZE_PATCH"] = str(materialize_patch)
    if args.disable_candidate_replay:
        env["METACORSI_DISABLE_CANDIDATE_REPLAY"] = "1"
    if args.dsh_stateless_compiler:
        env["METACORSI_DSH_STATELESS_COMPILER"] = "1"
    log = args.result_dir / "harbor.log"
    with log.open("a", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, cwd=root, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    evolution_receipt = None
    if args.read_only_knowledge:
        evolution_receipt = {"status": "read-only", "path": str(knowledge_store)}
    else:
        try:
            evolution_receipt = evolve(knowledge_store, (jobs / job,), policy=args.evolution_policy)
        except RuntimeError as exc:
            evolution_receipt = {"error": str(exc)}
    per_task = {}
    for result_path in (jobs / job).glob("*/result.json"):
        data = json.loads(result_path.read_text(encoding="utf-8"))
        task_id = str(data.get("task_name") or data.get("config", {}).get("task", {}).get("name") or result_path.parent.name)
        per_task[task_id] = {"reward": reward(data), "trial": str(result_path.parent),
                             "exception": bool(data.get("exception_info"))}
    passes = sum(1 for x in per_task.values() if x["reward"] == 1.0)
    aggregate_name = "aggregate.json" if job == "deepswe_dsh_metaco_k7_rca_1017_fixed20_20261105" else f"aggregate.{job}.json"
    atomic_json(args.result_dir / aggregate_name, {"schema_version": 1,
        "finished_at": datetime.now(timezone.utc).isoformat(), "framework": "dsh",
        "model": args.model, "provider_host": parsed_base.hostname,
        "provider_source": str(args.dotenv.resolve()), "dsh_profile": "headless",
        "control_mode": args.control_mode,
        "ltm_policy": args.ltm_policy, "memory_strategy": args.memory_strategy,
        "evolution_policy": args.evolution_policy,
        "tasks_requested": len(ids), "tasks_found": len(per_task),
        "passes": passes, "pass_rate": passes / len(ids), "exit_code": rc, "per_task": per_task})
    if evolution_receipt:
        atomic_json(args.result_dir / f"knowledge-evolution.{job}.json", evolution_receipt)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
