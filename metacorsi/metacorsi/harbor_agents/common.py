from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def harbor_executable() -> str:
    """Resolve Harbor on Windows or a native Linux installation."""
    configured = os.environ.get("METACORSI_HARBOR_EXE")
    if configured:
        return configured

    discovered = shutil.which("harbor")
    if discovered:
        return discovered

    adjacent = Path(sys.executable).with_name(
        "harbor.exe" if os.name == "nt" else "harbor"
    )
    if adjacent.exists():
        return str(adjacent)

    raise FileNotFoundError(
        "Harbor was not found. Install it in the active environment or set "
        "METACORSI_HARBOR_EXE."
    )


def dsh_executable() -> str:
    """Resolve the headless DSH CLI without assuming a Unix path."""
    configured = os.environ.get("METACORSI_DSH_EXE")
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.exists():
            return str(candidate.resolve())
        raise FileNotFoundError(f"METACORSI_DSH_EXE does not exist: {candidate}")

    discovered = shutil.which("dsh")
    if discovered:
        return discovered

    if os.name == "nt":
        candidates = [
            Path("D:/Deepseek/resources/runtime/cli/bin/dsh.cmd"),
            Path(os.environ.get("LOCALAPPDATA", ""))
            / "Programs/DeepSeek Harness/resources/runtime/cli/bin/dsh.cmd",
            Path(os.environ.get("PROGRAMFILES", ""))
            / "DeepSeek Harness/resources/runtime/cli/bin/dsh.cmd",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate.resolve())

    unix_default = Path("/usr/local/bin/dsh")
    if unix_default.is_file():
        return str(unix_default)

    raise FileNotFoundError(
        "DSH was not found. Install DeepSeek Harness, add dsh to PATH, or set "
        "METACORSI_DSH_EXE."
    )
