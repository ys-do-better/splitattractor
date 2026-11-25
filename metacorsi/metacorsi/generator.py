from __future__ import annotations

from dataclasses import asdict, dataclass

from .cognition import PerceptionPolicy, WorkingMemory
from .knowledge.k7 import RoutedSkill
from .targets import registry


@dataclass(frozen=True)
class GeneratedAgent:
    target: str
    knowledge_version: int
    perception_policy: PerceptionPolicy
    working_memory: WorkingMemory
    active_skills: tuple[RoutedSkill, ...]
    processor_policy: tuple[str, ...]
    control_mode: str

    def context(self) -> dict:
        return {
            "target": self.target,
            "knowledge_version": self.knowledge_version,
            "perception_policy": asdict(self.perception_policy),
            "working_memory": self.working_memory.receipt(),
            "active_skills": [asdict(x) for x in self.active_skills],
            "processor_policy": list(self.processor_policy),
            "control_mode": self.control_mode,
        }


class AgentGenerator:
    def generate(self, target: str, knowledge_version: int, policy: PerceptionPolicy,
                 memory: WorkingMemory, skills: tuple[RoutedSkill, ...],
                 control_mode: str = "rca_with_regression_gates") -> GeneratedAgent:
        if target not in registry:
            raise KeyError(f"unknown Target Agent: {target}")
        if control_mode == "rca_only":
            processor_policy = (
                "Monitor concrete failure evidence before modification",
                "After the first source mutation, construct the smallest task-contract reproducer and execute it",
                "Use the upstream five-step RCA walk to identify the causal root",
                "Rerun the exact same reproducer after repair so RCA walk-back has a real counterfactual",
                "Implement the smallest causal repair; no regression promotion gate",
                "Treat the external DeepSWE verifier as evaluation, not a state-transition gate",
            )
        else:
            processor_policy = (
                "Monitor evidence before modification",
                "RCA before repair and regression selection",
                "propose the smallest causal delta",
                "promote only on independent gate evidence; otherwise rollback",
            )
        return GeneratedAgent(target, knowledge_version, policy, memory, skills,
                              processor_policy, control_mode)
