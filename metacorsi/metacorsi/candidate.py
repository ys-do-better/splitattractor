from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from enum import Enum

from .contracts import RCADiagnosis


class Phase(str, Enum):
    IDLE = "idle"
    EXPLORE = "explore"
    CHECKPOINT = "checkpoint"
    FROZEN = "frozen"
    REPAIR = "repair"
    VERIFY = "verify"
    PROMOTED = "promoted"
    ROLLED_BACK = "rolled_back"


@dataclass
class Transition:
    before: str
    after: str
    reason: str


class CandidateStateManager:
    def __init__(self, max_repair_mutations: int = 1) -> None:
        self.phase = Phase.IDLE
        self.source = ""
        self.best = ""
        self.source_digest = ""
        self.best_digest = ""
        self.max_repair_mutations = max_repair_mutations
        self.repair_mutations = 0
        self.log: list[Transition] = []

    @staticmethod
    def digest(source: str) -> str:
        return hashlib.sha256(source.encode()).hexdigest()

    def _move(self, after: Phase, reason: str) -> None:
        self.log.append(Transition(self.phase.value, after.value, reason))
        self.phase = after

    def explore(self, source: str) -> None:
        if self.phase not in {Phase.IDLE, Phase.ROLLED_BACK, Phase.PROMOTED}:
            raise RuntimeError("candidate already active")
        self.source = source
        self.source_digest = self.digest(source)
        self._move(Phase.EXPLORE, "new candidate")

    def mutate(self, source: str) -> None:
        if self.phase == Phase.FROZEN:
            raise PermissionError("scope is frozen; RCA repair lease required")
        if self.phase == Phase.REPAIR:
            if self.repair_mutations >= self.max_repair_mutations:
                raise PermissionError("bounded RCA repair lease exhausted")
            self.repair_mutations += 1
        elif self.phase != Phase.EXPLORE:
            raise PermissionError(f"mutation forbidden in {self.phase.value}")
        self.source = source

    def checkpoint(self) -> None:
        if self.phase != Phase.EXPLORE:
            raise RuntimeError("checkpoint requires explore")
        self.best, self.best_digest = self.source, self.digest(self.source)
        self._move(Phase.CHECKPOINT, "best candidate saved")

    def freeze(self) -> None:
        if self.phase != Phase.CHECKPOINT:
            raise RuntimeError("freeze requires checkpoint")
        self._move(Phase.FROZEN, "scope frozen")

    def authorize_repair(self, diagnosis: RCADiagnosis) -> None:
        if self.phase != Phase.FROZEN or diagnosis.confidence < .7:
            raise PermissionError("repair requires frozen candidate and confident RCA")
        self.repair_mutations = 0
        self._move(Phase.REPAIR, f"RCA lease: {diagnosis.root_step}")

    def verify(self) -> None:
        if self.phase not in {Phase.FROZEN, Phase.REPAIR}:
            raise RuntimeError("verify requires frozen or repaired candidate")
        self._move(Phase.VERIFY, "independent verification")

    def promote(self) -> None:
        if self.phase != Phase.VERIFY:
            self.verify()
        self._move(Phase.PROMOTED, "RCA-selected regression passed")

    def rollback(self) -> None:
        if not self.best:
            raise RuntimeError("no checkpoint")
        self.source = self.best
        self._move(Phase.ROLLED_BACK, "validation failed; restored checkpoint")

    def evidence_valid(self, digest: str) -> bool:
        return digest == self.digest(self.source)

    def receipt(self) -> dict:
        return {"phase": self.phase.value, "source_digest": self.digest(self.source),
                "best_digest": self.best_digest, "repair_mutations": self.repair_mutations,
                "transitions": [asdict(x) for x in self.log]}
