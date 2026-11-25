"""Shared data of the two agents (unified algorithm draft, section 2)."""
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class Episode:
    """One step of experience, the episodic-memory record (deck page 7)."""
    o: str                      # observation
    a: str                      # action
    f: str                      # feedback
    t: str                      # where / when (file:line, step)
    meta: dict = field(default_factory=dict)
    expected: Any = None
    actual: Any = None
    delta: Any = None


@dataclass
class Signal:
    """A failure as first seen: sanitizer report, assertion, failed health check ..."""
    kind: str
    where: str
    report: dict
    reproducer: Optional[str] = None


class Evidence:
    """Everything in hand about this failure. Keys name the kind of evidence; the RCA picks a method from them."""

    def __init__(self):
        self.items = {}
        self.episodes = []

    def add(self, key, value):
        self.items[key] = value

    def has(self, key):
        return key in self.items

    def get(self, key, default=None):
        return self.items.get(key, default)

    def kinds(self):
        return set(self.items)


@dataclass
class Hop:
    """One link of a causal chain, from the violation back toward where the bad value came from."""
    kind: str                   # violation | bound | param | arg | local | write | origin
    func: str
    file: str
    line: int
    expr: str
    value: Any = None
    note: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class WhyStep:
    kind: str                   # bad_value | insufficient_guard | late_check | untrusted_source | ...
    text: str                   # one line, English
    text_zh: str                # one line, Chinese (slides)
    where: str
    evidence: dict = field(default_factory=dict)


@dataclass
class Diagnosis:
    """D = <L where, k column at fault (from the root step), k_s symptom column, W facts along the causal chain, iota,
    conf>, plus the walk up the build steps, the root step and the walk back (core/walk.py)"""
    L: dict
    k: str
    k_s: str
    W: list
    iota: dict
    conf: float
    method: str
    signature: dict
    chain: list = field(default_factory=list)
    from_case: bool = False
    root: dict = None           # the build step the walk ended at: {step, n, name, text_zh, kind}
    walk: list = field(default_factory=list)
    back: dict = None           # walk back: this case with the root step right; the class the same step made

    def to_json(self):
        d = asdict(self)
        return d
