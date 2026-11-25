from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class FailureEvidence:
    task_id: str
    assertion: str
    trace: str = ""
    changed_files: tuple[str, ...] = ()
    failed_tests: tuple[str, ...] = ()
    passed_tests: tuple[str, ...] = ()
    task_text: str = ""
    source_digest: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RCADiagnosis:
    root_step: str
    root_kind: str
    confidence: float
    explanation: str
    impacted_symbols: tuple[str, ...] = ()
    impacted_tests: tuple[str, ...] = ()
    boundary_tests: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RegressionPlan:
    tier: str
    focused: tuple[str, ...]
    boundary: tuple[str, ...]
    impacted: tuple[str, ...]
    full: bool
    reason: str
    diagnosis_digest: str


@dataclass(frozen=True)
class TestResult:
    group: str
    passed: int
    failed: int
    command: str = ""
    failures: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.failed == 0


@dataclass
class TrialReceipt:
    task_id: str
    agent: str
    model: str
    status: str
    reward: float | None = None
    diagnosis: dict[str, Any] | None = None
    regression_plan: dict[str, Any] | None = None
    transitions: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    def json(self) -> dict[str, Any]:
        return asdict(self)


class TargetAgent(Protocol):
    name: str

    def command(self, task_ids: Path, result_dir: Path, config: dict[str, Any]) -> list[str]: ...
