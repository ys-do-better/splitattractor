from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import TargetAgentAdapter


class DSHAdapter(TargetAgentAdapter):
    name = "dsh"

    def command(self, task_ids: Path, result_dir: Path, config: dict[str, Any]) -> list[str]:
        return [str(Path(config["paths"]["python"])), "-m", "metacorsi.harbor_agents.dsh_runner",
                "--tasks-file", str(task_ids), "--result-dir", str(result_dir / "dsh"),
                "--model", config["llm"]["model"], "--concurrency", str(config["benchmark"]["concurrency"]),
                "--dsh-exe", str(config["paths"]["dsh"]),
                "--dotenv", str(Path(config["llm"]["dotenv"])),
                "--control-mode", config.get("experiment", {}).get(
                    "control_mode", "rca_only")]
