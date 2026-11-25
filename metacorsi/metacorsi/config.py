from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_MODEL = "DeepSeek-V4.1-Flash-1017"


@dataclass(frozen=True)
class Settings:
    root: Path
    raw: dict[str, Any]

    @property
    def model(self) -> str:
        return self.raw["llm"]["model"]

    @property
    def result_dir(self) -> Path:
        p = Path(self.raw["result_dir"])
        return p if p.is_absolute() else self.root / p

    @property
    def target_agent(self) -> str:
        return self.raw.get("experiment", {}).get("target_agent", "dsh")

    @property
    def control_mode(self) -> str:
        return self.raw.get("experiment", {}).get(
            "control_mode", "rca_with_regression_gates"
        )


def load(path: str | Path) -> Settings:
    p = Path(path).resolve()
    raw = json.loads(p.read_text(encoding="utf-8"))
    if raw.get("llm", {}).get("model") != REQUIRED_MODEL:
        raise ValueError(f"model must be exactly {REQUIRED_MODEL}")
    if raw.get("benchmark", {}).get("limit") != 20:
        raise ValueError("the formal protocol requires exactly 20 tasks")
    target = raw.get("experiment", {}).get("target_agent", "dsh")
    if target not in {"dsh", "mini-swe-agent", "poa"}:
        raise ValueError(f"unsupported target_agent: {target}")
    mode = raw.get("experiment", {}).get("control_mode", "rca_with_regression_gates")
    if mode not in {"rca_only", "rca_with_regression_gates"}:
        raise ValueError(f"unsupported control_mode: {mode}")
    return Settings(p.parent.parent, raw)
