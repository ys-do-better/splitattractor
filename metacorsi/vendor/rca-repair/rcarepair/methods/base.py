"""Method families = the rows of the RCA taxonomy (deck page 1). The RCA loop picks one by what is in hand."""
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class MethodResult:
    localized: bool
    hops: list = field(default_factory=list)      # causal chain, violation first
    ranking: list = field(default_factory=list)   # (unit, score) for scoring methods
    missing: Optional[dict] = None                 # what evidence would let it go further
    summary: str = ""
    summary_zh: str = ""
    extra: Any = None


class Method:
    key = ""
    row = ""          # row name on page 1
    needs = ""        # evidence kind it needs (Evidence/in_hand vocabulary)
    needs_zh = ""     # the label on the page-4 flowchart edge
    does_zh = ""

    def applicable(self, have):
        return self.needs in have

    def run(self, target, ev):
        raise NotImplementedError
