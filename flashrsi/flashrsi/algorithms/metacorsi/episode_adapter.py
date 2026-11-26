"""Translate a trusted FlashRSI DeepSWE event into native episode inputs."""

import hashlib
import json
from pathlib import Path


def verified_episode(trajectory, snapshot_id):
    task = trajectory.task
    if task is None or trajectory.status != "finished" or not trajectory.messages:
        return None
    verifier = trajectory.messages[-1]
    if (
        verifier.get("type") != "HarborVerifier"
        or verifier.get("verifier_environment_mode") != "separate"
        or verifier.get("trial_id") is None
        or verifier.get("error")
        or verifier.get("exception_type")
        or verifier.get("task_id") != task.task_id
        or not task.metadata.get("task_checksum")
        or verifier.get("task_checksum") != task.metadata["task_checksum"]
        or verifier.get("snapshot_id") != snapshot_id
    ):
        return None
    rewards = verifier.get("rewards") or {}
    reward = rewards.get("reward")
    if type(reward) not in (int, float) or not 0 <= reward <= 1:
        return None
    counts = {
        key: rewards.get(key)
        for key in ("f2p_passed", "f2p_total", "p2p_passed", "p2p_total")
    }
    # Native distillation requires counts. A missing group is not evidence of zero.
    if any(type(value) is not int for value in counts.values()) or any(
        not 0 <= counts[f"{kind}_passed"] <= counts[f"{kind}_total"]
        for kind in ("f2p", "p2p")
    ):
        return None
    archive = (verifier.get("artifacts") or {}).get("directory")
    if not archive:
        return None
    root = Path(archive).resolve()
    result = root / "result.json"
    if not result.is_file() or result.is_symlink():
        return None
    native = json.loads(result.read_bytes())
    if (
        native.get("task_name") != task.task_id
        or native.get("task_checksum") != task.metadata["task_checksum"]
        or native.get("exception_info")
        or native.get("id") != verifier["trial_id"]
        or (native.get("verifier_result") or {}).get("rewards") != rewards
    ):
        return None
    runtime = next(
        (item for item in trajectory.messages if item.get("type") == "MetaCoRSIMemory"),
        {},
    )
    patch = root / "artifacts/logs/artifacts/model.patch"
    data = patch.read_bytes() if patch.is_file() and not patch.is_symlink() else b""
    return {
        "task": task.task_id.removeprefix("datacurve/"),
        "reward": reward,
        **counts,
        "patch": data.decode("utf-8", errors="replace"),
        "patch_sha256": hashlib.sha256(data).hexdigest() if data else None,
        "source": str(root),
        "observed_at": result.stat().st_mtime,
        "steps": trajectory.usage.get("turns", 0),
        "groundings": list(runtime.get("groundings", [])),
        "final_analysis": trajectory.output,
    }
