from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class TargetAgentAdapter(ABC):
    name: str

    @abstractmethod
    def command(self, task_ids: Path, result_dir: Path, config: dict[str, Any]) -> list[str]:
        raise NotImplementedError
