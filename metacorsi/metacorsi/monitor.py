from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class Observation:
    evidence_id: str
    kind: str
    content: str
    created_at: str
    metadata: dict[str, Any]


class Monitor:
    def __init__(self) -> None:
        self.observations: list[Observation] = []

    def record(self, kind: str, content: str, metadata: dict[str, Any] | None = None) -> Observation:
        created = datetime.now(timezone.utc).isoformat()
        digest = hashlib.sha256(f"{kind}|{content}|{created}".encode()).hexdigest()[:20]
        observation = Observation(f"obs-{digest}", kind, content, created, metadata or {})
        self.observations.append(observation)
        return observation

    def receipt(self) -> list[dict[str, Any]]:
        return [asdict(item) for item in self.observations]
