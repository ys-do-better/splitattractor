from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


_TEST_COMMAND = re.compile(
    r"(?:^|(?:&&|\|\||;|\|)\s*)(?:pytest|py\.test|go\s+test|cargo\s+test|"
    r"npm\s+(?:test|run\s+test)|pnpm\s+(?:test|run\s+test)|yarn\s+test|"
    r"bun\s+test|deno\s+test|gradle\w*\s+test|mvn\s+test|ctest)\b",
    re.I | re.M,
)
_FAILURE = re.compile(
    r"(?:\bFAILED\b|AssertionError|\bERRORS?\b|failures?\s*=|"
    r"---\s+FAIL:|\bFAIL\s+\S+|tests?\s+failed)",
    re.I,
)
_STRONG_FAILURE = re.compile(
    r"(?:Traceback \(most recent call last\)|AssertionError|^FAILED\s+|"
    r"^---\s+FAIL:|\bpanic:|segmentation fault|tests?\s+failed|"
    r"compilation failed|build failed)",
    re.I | re.M,
)
_FAILED_TEST_PATTERNS = (
    re.compile(r"^FAILED\s+([^\s]+)", re.M),
    re.compile(r"^---\s+FAIL:\s+([^\s(]+)", re.M),
    re.compile(r"^\s*FAIL\s+([^\s]+)", re.M),
    re.compile(r"^([^\s:]+::test[^\s]+).*?FAILED", re.M | re.I),
)
_LOCATION = re.compile(
    r"(?P<file>(?:[A-Za-z0-9_.@+-]+/)*[A-Za-z0-9_.@+-]+\."
    r"(?:py|pyi|go|rs|ts|tsx|js|jsx|java|kt|rb|php|c|cc|cpp|h|hpp))"
    r"(?::|\()(?P<line>\d+)(?::\d+|\))?"
)
_DIFF_FILE = re.compile(r"^\+\+\+\s+b/(.+)$", re.M)
_DIFF_HUNK = re.compile(r"^@@\s+[^+]*\+(\d+)(?:,(\d+))?\s+@@", re.M)


def command_text(commands: list[object]) -> str:
    return "\n".join(str(getattr(item, "keystrokes", "")) for item in commands).strip()


def is_test_command(command: str) -> bool:
    return bool(_TEST_COMMAND.search(command))


def observed_failure(output: str, timed_out: bool = False) -> bool:
    return timed_out or bool(_FAILURE.search(output))


def strong_failure(output: str) -> bool:
    return bool(_STRONG_FAILURE.search(output))


def failed_tests(output: str) -> tuple[str, ...]:
    found: list[str] = []
    for pattern in _FAILED_TEST_PATTERNS:
        found.extend(match.group(1).rstrip(":") for match in pattern.finditer(output))
    cleaned = [item for item in found
               if item.upper() not in {"FAIL", "FAILED", "ERROR"}
               and not item.startswith(("root@", "#"))]
    return tuple(dict.fromkeys(cleaned))


def changed_files(names: str, status: str = "") -> tuple[str, ...]:
    values = [line.strip() for line in names.splitlines() if line.strip()]
    for line in status.splitlines():
        if len(line) > 3:
            values.append(line[3:].strip().split(" -> ")[-1])
    return tuple(dict.fromkeys(values))


def code_locations(trace: str, diff: str = "") -> tuple[dict[str, object], ...]:
    located: list[dict[str, object]] = []
    for match in _LOCATION.finditer(trace):
        located.append({"file": match.group("file"), "line": int(match.group("line")),
                        "source": "trace"})

    current_file: str | None = None
    for line in diff.splitlines():
        file_match = _DIFF_FILE.match(line)
        if file_match:
            current_file = file_match.group(1)
            continue
        hunk_match = _DIFF_HUNK.match(line)
        if current_file and hunk_match:
            located.append({"file": current_file, "line": int(hunk_match.group(1)),
                            "source": "git-diff"})

    unique: dict[tuple[str, int], dict[str, object]] = {}
    for item in located:
        unique.setdefault((str(item["file"]), int(item["line"])), item)
    return tuple(list(unique.values())[:12])


def source_digest(status: str, diff: str) -> str:
    return hashlib.sha256(f"{status}\n{diff}".encode("utf-8", "replace")).hexdigest()


def test_family(command: str) -> str:
    match = _TEST_COMMAND.search(command)
    return re.sub(r"\s+", " ", match.group(0).lower()).strip() if match else ""


def test_signature(command: str) -> str:
    return re.sub(r"\s+", " ", command.strip()).lower()


@dataclass(frozen=True)
class RepositoryEvidence:
    status: str = ""
    diff_names: str = ""
    diff: str = ""

    @property
    def files(self) -> tuple[str, ...]:
        return changed_files(self.diff_names, self.status)

    @property
    def digest(self) -> str:
        return source_digest(self.status, self.diff)
