from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import TargetAgentAdapter


class POAAdapter(TargetAgentAdapter):
    """Reserved stable plugin boundary; no POA implementation is silently substituted."""
    name = "poa"

    def command(self, task_ids: Path, result_dir: Path, config: dict[str, Any]) -> list[str]:
        raise NotImplementedError("POA adapter contract exists, but a POA runtime has not been configured")
