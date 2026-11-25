from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import TargetAgentAdapter


class MiniSWEAgentAdapter(TargetAgentAdapter):
    name = "mini-swe-agent"

    def command(self, task_ids: Path, result_dir: Path, config: dict[str, Any]) -> list[str]:
        return [
            str(Path(config["paths"]["python"])),
            "-m", "metacorsi.harbor_agents.mini_swe_runner",
            "--tasks-file", str(task_ids),
            "--result-dir", str(result_dir / "mini-swe-agent"),
            "--model", config["llm"]["model"],
            "--concurrency", str(config["benchmark"]["concurrency"]),
            "--control-mode", config.get("experiment", {}).get("control_mode", "rca_only"),
        ]
