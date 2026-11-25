"""The repair periodic table (Liu deck v14, page 3).

Columns are the kind of knowledge at fault, the same axis as the RCA taxonomy (page 1); rows are how deep a repair
reaches and the rights it needs. Each element carries the third axis (cost) and how much of the original it replaces
(shade). Solo elements cannot be combined with others.
"""
from dataclasses import dataclass

COLUMNS = ["Factual", "Conceptual", "Sequential", "Concurrent", "Probabilistic"]
COLUMN_LABEL = {
    "Factual": "Data, values, configuration",
    "Conceptual": "Models, rules, ownership",
    "Sequential": "Sequential",
    "Concurrent": "Concurrent, real-time, monitoring",
    "Probabilistic": "Probabilistic, learned (LLM)",
}
ROWS = ["black", "white", "spec"]  # ordered by depth; access level A also uses these names
ROW_LABEL = {"black": "Black box (user rights)", "white": "White box (developer rights)",
             "spec": "Specification (owner rights)"}


@dataclass(frozen=True)
class Element:
    sym: str
    name_zh: str
    name: str
    col: str
    row: str
    cost: int   # 1 low .. 4 highest, usual case; re-estimated per scenario
    shade: int  # 1 redo .. 5 replace everything
    solo: bool = False


_E = [
    # Factual
    ("Bl", "挡输入", "block input", "Factual", "black", 1, 2),
    ("Tn", "调参", "tune a parameter", "Factual", "black", 1, 2),
    ("Fb", "兜底", "fall back to a default", "Factual", "black", 1, 2),
    ("Rs", "重启", "restart", "Factual", "black", 2, 3),
    ("Sw", "整体替换", "swap the whole unit", "Factual", "black", 3, 4),
    ("Cl", "夹紧", "clamp a value", "Factual", "white", 2, 2),
    ("Rp", "修数据", "repair data", "Factual", "white", 2, 3),
    ("Sc", "换部件", "swap a component", "Factual", "white", 3, 4),
    # Conceptual
    ("Rb", "回滚", "roll back the release", "Conceptual", "black", 2, 3),
    ("Cd", "补条件", "add a condition", "Conceptual", "white", 2, 2),
    ("Lg", "改逻辑", "change logic", "Conceptual", "white", 2, 3),
    ("Sp", "改规约", "change the specification", "Conceptual", "spec", 3, 3),
    ("Rw", "推倒重来", "rewrite", "Conceptual", "spec", 4, 5),
    # Sequential
    ("By", "绕过", "bypass", "Sequential", "black", 1, 3),
    ("Ad", "补一步", "add a step", "Sequential", "white", 2, 2),
    ("Rm", "删一步", "remove a step", "Sequential", "white", 2, 3),
    ("Cp", "补偿", "compensate", "Sequential", "spec", 3, 2),
    # Concurrent
    ("Dg", "降级", "degrade", "Concurrent", "black", 1, 3),
    ("Lk", "加锁", "add a lock", "Concurrent", "white", 2, 2),
    ("Sy", "定顺序", "fix the order", "Concurrent", "white", 2, 2),
    # Probabilistic
    ("Rt", "重试", "retry", "Probabilistic", "black", 1, 1),
]
SOLO = {"By", "Rw"}
ELEMENTS = {e[0]: Element(*e, solo=e[0] in SOLO) for e in _E}
COMPOUNDS = {"TOCTOU": ("Lk", "Cd"), "bad release": ("Rb", "Rp"), "query of death": ("Rs", "Bl")}


def cell(col, row):
    return [e for e in ELEMENTS.values() if e.col == col and e.row == row]


def rows_within(access):
    """rows a repair may use with this access level: black < white < spec"""
    return ROWS[: ROWS.index(access) + 1]


def deeper(row):
    i = ROWS.index(row)
    return ROWS[i + 1] if i + 1 < len(ROWS) else None
