from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .candidate import CandidateStateManager
from .cognition import PerceptionCompiler, WorkingMemory
from .contracts import FailureEvidence, RCADiagnosis, RegressionPlan, TestResult
from .generator import AgentGenerator, GeneratedAgent
from .knowledge import K7SkillRouter
from .knowledge.engram_ensemble import EngramEnsembleMemory
from .knowledge.grounding import GroundedKnowledgeGraph
from .knowledge.indexed_ltm import (GroundingRecord, IndexedLTM,
                                    render_grounded_application,
                                    render_split_attractor_directive)
from .knowledge.predictive_coding import PredictiveCodingMemory
from .knowledge.sleep_consolidation import SleepConsolidationMemory
from .knowledge.split_attractor import SplitAttractorStepResult, SplitAttractorWM
from .knowledge.store import (KnowledgeAtom, VersionedKnowledgeStore, k7_seed_atoms,
                              render_atoms, retrieve_atoms)
from .monitor import Monitor
from .rca.adapter import DeepSWEAgentRCA, diagnosis_digest
from .validation.controller import RCARegressionController


@dataclass
class KnowledgeSystem:
    version: int = 7
    perception: list[dict[str, Any]] | None = None
    memory: list[dict[str, Any]] | None = None
    reasoning: list[dict[str, Any]] | None = None
    repair: list[dict[str, Any]] | None = None
    safety: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        for name in ("perception", "memory", "reasoning", "repair", "safety"):
            if getattr(self, name) is None:
                setattr(self, name, [])


class MetaCoRSIBrain:
    """A Monitor -> B RCA -> C Plan -> D Execute/Validate -> knowledge commit."""

    def __init__(self, knowledge: KnowledgeSystem | None = None,
                 control_mode: str = "rca_with_regression_gates",
                 knowledge_path: Path | None = None) -> None:
        if control_mode not in {"rca_only", "rca_with_regression_gates"}:
            raise ValueError(f"unsupported control_mode: {control_mode}")
        self.control_mode = control_mode
        self.knowledge = knowledge or KnowledgeSystem()
        self.ltm = (VersionedKnowledgeStore.from_file(knowledge_path)
                    if knowledge_path and knowledge_path.exists()
                    else VersionedKnowledgeStore(k7_seed_atoms(), self.knowledge.version))
        graph_path = knowledge_path.with_name("grounding_graph.json") if knowledge_path else None
        self.grounding_graph = GroundedKnowledgeGraph.from_file(graph_path)
        indexed_root = knowledge_path.parent / "ltm" if knowledge_path else None
        if indexed_root and (indexed_root / "CURRENT.json").exists():
            self.indexed_ltm = IndexedLTM.load(
                indexed_root, expected_version=self.ltm.active.version
            )
        else:
            self.indexed_ltm = IndexedLTM.build(
                atoms=self.ltm.active.atoms,
                version=self.ltm.active.version,
                parent=self.ltm.active.parent,
                graph=self.grounding_graph,
            )
        self.last_groundings: tuple[GroundingRecord, ...] = ()
        self.ltm_policy = os.environ.get("METACORSI_LTM_POLICY", "full_v1")
        sa_mode = "naive" if self.ltm_policy == "naive_latch_v24" else "sa"
        self.split_attractor = SplitAttractorWM(mode=sa_mode)
        self.predictive_coding = PredictiveCodingMemory(self.indexed_ltm)
        self.working_memory: WorkingMemory | None = None
        self.knowledge.version = self.ltm.active.version
        self.skill_router = K7SkillRouter()
        self.perception = PerceptionCompiler()
        self.generator = AgentGenerator()
        self.monitor = Monitor()
        self.rca = DeepSWEAgentRCA()
        self.regression = RCARegressionController()
        self.candidates = CandidateStateManager()

    def evolved_knowledge_prompt(self, task_text: str, limit: int = 8,
                                 task_id: str = "",
                                 repository_evidence: str = "") -> str:
        bundle, groundings = self.indexed_ltm.ground(
            task_id=task_id,
            task_text=task_text,
            repository_evidence=repository_evidence,
            limit=limit,
            policy=self.ltm_policy,
        )
        self.last_groundings = groundings
        return render_grounded_application(bundle, groundings, self.ltm_policy)

    def grounding_receipt(self) -> list[dict[str, Any]]:
        return [item.json() for item in self.last_groundings]

    def step_working_memory(
        self,
        task_text: str,
        *,
        task_id: str = "",
        repository_evidence: str = "",
        has_live_evidence: bool | None = None,
        explicit_counterevidence: bool = False,
        stagnation_steps: int = 0,
        ordinary_command_failure: bool = False,
        limit: int = 14,
    ) -> tuple[str, SplitAttractorStepResult]:
        """Advance V24 Split-Attractor Working Memory by one step and render context."""
        bundle, signals, pfg_summary = self.indexed_ltm.score_candidates(
            task_id=task_id,
            task_text=task_text,
            repository_evidence=repository_evidence,
            limit=limit,
            policy=self.ltm_policy,
        )
        live_flag = (
            bool(repository_evidence.strip())
            if has_live_evidence is None
            else bool(has_live_evidence)
        )
        step_result = self.split_attractor.step(
            signals,
            has_live_evidence=live_flag,
            explicit_counterevidence=explicit_counterevidence,
            stagnation_steps=stagnation_steps,
            ordinary_command_failure=ordinary_command_failure,
            pfg_summary=pfg_summary,
        )
        self.last_groundings = (
            (step_result.active_grounding,)
            if step_result.active_grounding is not None
            else ()
        )
        if self.working_memory is not None:
            self.working_memory.update_split_attractor(step_result.to_wm_snapshot())
        if step_result.step == 1:
            rendered = render_grounded_application(
                bundle, self.last_groundings, self.ltm_policy
            )
        else:
            rendered = render_split_attractor_directive(
                bundle, step_result, self.ltm_policy
            )
        return rendered, step_result

    def generate_agent(self, target: str, task_id: str, task_text: str) -> GeneratedAgent:
        """Steps 1-3: pin LTM, compile perception, and generate a target-specific cognitive agent."""
        snapshot = self.ltm.pin()
        skills = self.skill_router.route(task_text)
        policy = self.perception.compile(task_id, snapshot, skills)
        wm_digest = hashlib.sha256(f"{task_id}|{snapshot.version}".encode()).hexdigest()[:16]
        plans = (["Monitor", "RCA walk", "RCA walk-back", "minimal causal delta"]
                 if self.control_mode == "rca_only" else
                 ["Monitor", "RCA", "minimal delta", "RCA regression", "promote/rollback"])
        memory = WorkingMemory(f"wm-{wm_digest}", task_id, snapshot.version, plans=plans)
        self.working_memory = memory
        return self.generator.generate(target, snapshot.version, policy, memory, skills,
                                       self.control_mode)

    def analyze(self, evidence: FailureEvidence) -> RCADiagnosis:
        """Run upstream RCA without constructing or applying a regression gate."""
        observation = self.monitor.record("failure", evidence.assertion,
                                          {"task_id": evidence.task_id,
                                           "source_digest": evidence.source_digest})
        diagnosis = self.rca.diagnose(evidence)
        self.monitor.record("rca", diagnosis.explanation,
                            {"evidence_id": observation.evidence_id,
                             "root_step": diagnosis.root_step,
                             "confidence": diagnosis.confidence,
                             "walk_back_ok": diagnosis.raw.get("back", {}).get("ok")})
        return diagnosis

    def analyze_and_plan(self, evidence: FailureEvidence) -> tuple[RCADiagnosis, RegressionPlan]:
        if self.control_mode == "rca_only":
            raise RuntimeError("regression planning is disabled in rca_only mode; call analyze()")
        observation = self.monitor.record("failure", evidence.assertion,
                                          {"task_id": evidence.task_id,
                                           "source_digest": evidence.source_digest})
        diagnosis = self.rca.diagnose(evidence)
        plan = self.regression.plan(evidence, diagnosis)
        self.monitor.record("rca", diagnosis.explanation,
                            {"evidence_id": observation.evidence_id, "root_step": diagnosis.root_step,
                             "confidence": diagnosis.confidence, "regression_tier": plan.tier})
        return diagnosis, plan

    def begin_candidate(self, source: str, evidence: FailureEvidence) -> tuple[RCADiagnosis, RegressionPlan]:
        diagnosis, plan = self.analyze_and_plan(evidence)
        self.candidates.explore(source)
        self.candidates.checkpoint()
        self.candidates.freeze()
        self.candidates.authorize_repair(diagnosis)
        self.ltm.stage(KnowledgeAtom(
            f"candidate.{diagnosis_digest(diagnosis)[:16]}", "F",
            f"Candidate repair hypothesis: {diagnosis.root_step} / {diagnosis.explanation}",
            f"task:{evidence.task_id}", diagnosis.confidence,
            ("staged", plan.tier),
        ))
        return diagnosis, plan

    def validate(self, plan: RegressionPlan, results: list[TestResult]) -> bool:
        accepted = self.regression.accept(plan, results)
        if accepted:
            self.candidates.promote()
            receipt_digest = hashlib.sha256(
                "|".join(f"{x.group}:{x.passed}:{x.failed}:{x.command}" for x in results).encode()
            ).hexdigest()
            version = self.ltm.commit(receipt_digest, {"regression_tier": plan.tier,
                                                       "diagnosis_digest": plan.diagnosis_digest})
            self.knowledge.version = version.version
            self.monitor.record("knowledge-commit", f"Knowledge v{version.version}",
                                {"evidence_digest": receipt_digest})
        else:
            self.candidates.rollback()
            self.ltm.rollback_stage()
            self.monitor.record("rollback", "candidate rejected; restored checkpoint",
                                {"diagnosis_digest": plan.diagnosis_digest})
        return accepted

    def snapshot(self) -> dict[str, Any]:
        return {"control_mode": self.control_mode,
                "regression_gate_enabled": self.control_mode != "rca_only",
                "knowledge": asdict(self.knowledge), "ltm": self.ltm.receipt(),
                "indexed_ltm": {"version": self.indexed_ltm.version,
                                "atoms": len(self.indexed_ltm.records),
                                "nodes": len(self.indexed_ltm.nodes),
                                "abstractions": len(self.indexed_ltm.abstractions),
                                "groundings": len(self.indexed_ltm.groundings),
                                "retention_tasks": len(self.indexed_ltm.anchors)},
                "split_attractor": self.split_attractor.metrics(),
                "working_memory": self.working_memory.receipt() if self.working_memory else None,
                "candidate": self.candidates.receipt(), "monitor": self.monitor.receipt()}
