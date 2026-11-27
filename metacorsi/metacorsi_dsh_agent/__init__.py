from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

try:
    from harbor.agents.base import BaseAgent
    from harbor.environments.base import BaseEnvironment
    from harbor.models.agent.context import AgentContext
except ModuleNotFoundError:  # pragma: no cover - allow offline unit testing without harbor
    class BaseAgent:  # type: ignore[no-redef]
        pass

    class BaseEnvironment:  # type: ignore[no-redef]
        pass

    class AgentContext:  # type: ignore[no-redef]
        pass

from metacorsi.brain import MetaCoRSIBrain
from metacorsi.contracts import FailureEvidence


_TEST_PROBE = re.compile(
    r"(?:\bpytest\b|\bvitest\b|\bjest\b|\bcargo\s+test\b|\bgo\s+test\b|"
    r"\b(?:pnpm|npm|yarn)\s+(?:run\s+)?test\b|\bdeno\s+test\b|\btox\b|"
    r"\bctest\b|\brspec\b|\bmvn\s+test\b|\bgradle\w*\s+.*test\b|"
    r"\bmake\s+(?:\w*test\w*)\b|METACORSI_(?:SMOKE|FOCUSED|BOUNDARY|PRESERVE|REGRESSION))",
    re.I,
)

_ANSWER_SEARCH_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bgit\s+(?:fsck|reflog)\b", re.I), "unreachable/reflog object search"),
    (re.compile(r"\bgit\s+rev-list\b[^\n]*(?:--all|--objects)", re.I), "all-object history search"),
    (re.compile(r"\bfind\s+/(?:\s|$)", re.I), "container-wide duplicate repository/test search"),
    (re.compile(r"/(?:gold|solution|solutions|oracle|verifier)(?:/|\b)", re.I),
     "gold/oracle/verifier path access"),
    (re.compile(r"\b(?:docker|podman)\s+(?:cp|exec|run)\b", re.I), "cross-container access"),
)


def _answer_search_violation(command: str) -> str | None:
    """Reject answer-seeking while preserving normal inspection inside /app."""
    for pattern, reason in _ANSWER_SEARCH_PATTERNS:
        if pattern.search(command):
            return reason
    return None


def _focused_probe(command: str) -> str | None:
    """Extract a replayable test segment; never turn tool failures into gates."""
    segments = [segment.strip() for segment in command.split("&&")]
    tests = [segment for segment in segments if _TEST_PROBE.search(segment)]
    if not tests:
        return None
    prefix = next((segment for segment in segments if re.match(r"^cd\s+", segment)), "")
    selected = " && ".join(tests)
    return f"{prefix} && {selected}" if prefix and prefix not in tests else selected


def _confirmed_focused_failure(commands: list[str], return_codes: list[int]) -> bool:
    """Only an executed test failure may unlock cross-task diagnostic memory."""
    return any(
        code != 0 and _TEST_PROBE.search(command)
        for command, code in zip(commands, return_codes)
    )


_EDIT_COMMAND = re.compile(
    r"(?:\bgit\s+apply\b|\bsed\s+-i\b|\bpython3?\s+-c\b.*(?:write_text|open\s*\(|replace)|"
    r"\bcat\s+<<|\bprintf\b.*>\s*\S+|\bpatch\b)",
    re.I | re.S,
)


def _classify_step_feedback(
    commands: list[str],
    return_codes: list[int],
    *,
    gate_Active: bool = False,
    edited_since_latch: bool = False,
    rca_gate_blocked: bool = False,
) -> dict[str, bool]:
    """Distinguish ordinary shell command errors from test counterevidence in V24."""
    has_test = any(bool(_TEST_PROBE.search(cmd)) for cmd in commands)
    test_failed = _confirmed_focused_failure(commands, return_codes)
    test_passed = any(
        code == 0 and bool(_TEST_PROBE.search(cmd))
        for cmd, code in zip(commands, return_codes)
    )
    edited = any(bool(_EDIT_COMMAND.search(cmd)) for cmd in commands)
    any_failed = any(code != 0 for code in return_codes)
    ordinary_failure = any_failed and not test_failed
    # Explicit counterevidence occurs when the RCA completion gate rejects the
    # candidate, or when an edit made under a latched hypothesis is tested and
    # the focused test still fails.
    explicit_counterevidence = rca_gate_blocked or (
        gate_Active and (edited_since_latch or edited) and test_failed
    )
    return {
        "has_test": has_test,
        "test_failed": test_failed,
        "test_passed": test_passed,
        "edited": edited,
        "ordinary_failure": ordinary_failure,
        "explicit_counterevidence": explicit_counterevidence,
    }


def _lock_grounding_receipt(
    receipt: list[dict], locked_abstract_id: str | None,
) -> tuple[list[dict], str | None, list[str]]:
    """Select one abstraction per episode and suppress later relation switching."""
    if locked_abstract_id is None:
        if not receipt:
            return [], None, []
        selected = receipt[0]
        return [selected], selected["abstract_id"], [
            item["abstract_id"] for item in receipt[1:]
        ]
    matching = [item for item in receipt if item["abstract_id"] == locked_abstract_id]
    suppressed = [
        item["abstract_id"] for item in receipt
        if item["abstract_id"] != locked_abstract_id
    ]
    return matching, locked_abstract_id, suppressed


def _json_object(text: str) -> dict:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates: list[dict] = []
    if fenced:
        try:
            candidates.append(json.loads(fenced.group(1)))
        except json.JSONDecodeError:
            pass
    decoder = json.JSONDecoder()
    # DSH may print a reasoning prelude containing illustrative braces. Scan
    # every object start and select the last valid command-plan object.
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("commands"), list):
            candidates.append(value)
    if not candidates:
        raise ValueError(f"DSH emitted no command-plan JSON (chars={len(text)})")
    return candidates[-1]


def _runtime_home(logs_dir: Path) -> Path:
    """Use a short stable path; Windows long paths are disabled on the test host."""
    configured = os.environ.get("METACORSI_DSH_RUNTIME_ROOT")
    if configured:
        root = Path(configured)
        trial_key = hashlib.sha256(str(logs_dir.resolve()).encode()).hexdigest()[:16]
        return root / trial_key
    return logs_dir / ".dsh-home"


def _successful_test_seen(history: list[dict]) -> bool:
    for item in history:
        for command, code in zip(item.get("commands", ()), item.get("return_codes", ())):
            if code == 0 and _TEST_PROBE.search(command):
                return True
    return False


def _protocol_fallback_plan(prompt: str) -> str:
    """Preserve a materialized candidate when provider reasoning has no final channel."""
    closing = "METACORSI RCA PHASE: CLOSE" in prompt
    return json.dumps({
        "analysis": "Provider reasoning exhausted the output cap; preserve and validate the materialized candidate.",
        "commands": [
            "cd /app && git status --short",
            "cd /app && git diff --check",
        ],
        "done": closing,
    })


def _phase_directive(step: int, max_steps: int, history: list[dict] | None = None,
                     candidate_seed: dict | None = None) -> str:
    """Convert RCA evidence into an execution policy, not another memory hint."""
    history = history or []
    seeded = bool(candidate_seed and candidate_seed.get("status") == "applied")
    tested = _successful_test_seen(history)
    if step <= 4:
        phase = "DIAGNOSE"
        rule = ("Localize the residual counterexample and one causal decision point. "
                "Do not redesign the complete feature.")
    elif not seeded and not tested and step <= 12:
        phase = "MATERIALIZE"
        rule = "Create a minimal vertical code slice now and wire its public export/registration/dispatch path."
    elif step <= 18:
        phase = "VERIFY"
        rule = "Run a focused end-to-end probe. Freeze working behavior and close only observed boundary gaps."
    else:
        phase = "CLOSE"
        rule = "Stop broad exploration. Finish integration, run the smallest decisive test, and preserve a git checkpoint."
    remaining = max_steps - step + 1
    stagnation = int((candidate_seed or {}).get("stagnation_count", 0))
    reflection = (candidate_seed or {}).get("reflection") or {}
    hypotheses = reflection.get("alternative_hypotheses") or ()
    escape = ""
    if stagnation and hypotheses:
        hypothesis = hypotheses[min(stagnation - 1, len(hypotheses) - 1)]
        escape = (f" STAGNATION={stagnation}: do not repeat the previous patch shape; actively test the "
                  f"alternative hypothesis: {hypothesis}.")
    command_limit = int(os.environ.get("METACORSI_DSH_COMMAND_MAX_CHARS", "240"))
    materialize = (
        f" During MATERIALIZE, use a direct edit command up to {command_limit} characters; "
        "do not spend a turn planning shell quoting."
        if phase == "MATERIALIZE" else ""
    )
    return (
        f"METACORSI RCA PHASE: {phase}; step={step}/{max_steps}; remaining={remaining}. "
        f"{rule} A successful shell command is not proof of capability; ensure an executable patch exists."
        f"{materialize}{escape}"
    )


def _candidate_for_session(registry: dict, session_id: str) -> dict | None:
    session = session_id.lower()
    trial_key = session.split("__", 1)[0]
    return next((value for task, value in (registry.get("tasks") or {}).items()
                 if task.lower().startswith(trial_key) or trial_key.startswith(task.lower())), None)


def _active_candidate(entry: dict | None) -> dict | None:
    if not entry:
        return None
    if entry.get("patch"):
        return entry
    candidates = entry.get("candidates") or ()
    active = entry.get("active_patch_digest")
    return next((item for item in candidates if item.get("patch_digest") == active),
                candidates[0] if candidates else None)


def _candidate_staging_commands(encoded: str, chunk_size: int = 8000) -> tuple[str, tuple[str, ...], str]:
    """Build Windows-safe commands for staging a base64 patch in Docker."""
    staging = "/tmp/metacorsi-candidate.b64"
    patch_path = "/tmp/metacorsi-candidate.patch"
    setup = f"cd /app && rm -f {staging} {patch_path} && : > {staging}"
    chunks = tuple(
        f"cd /app && printf '%s' '{encoded[offset:offset + chunk_size]}' >> {staging}"
        for offset in range(0, len(encoded), chunk_size)
    )
    apply = (
        f"cd /app && base64 -d {staging} > {patch_path} && "
        f"git apply --3way {patch_path} && git status --short"
    )
    return setup, chunks, apply


def _provider_only_env(source: dict[str, str]) -> dict[str, str]:
    """Prevent DSH account or unrelated provider credentials from becoming fallbacks."""
    child = source.copy()
    for key in (
        "DSH_TOKEN", "DSH_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY",
        "GOOGLE_API_KEY", "GEMINI_API_KEY", "AZURE_OPENAI_API_KEY",
        "OPENAI_API_KEY", "OPENAI_API_BASE", "OPENAI_API_BASE_URL",
    ):
        child.pop(key, None)
    required = ("METACORSI_DSH_API_KEY", "METACORSI_DSH_BASE_URL")
    missing = [key for key in required if not child.get(key)]
    if missing:
        raise RuntimeError("provider-only DSH environment missing: " + ", ".join(missing))
    child["DSH_TELEMETRY_MODE"] = "DISABLED"
    child["METACORSI_DSH_PROVIDER_ONLY"] = "1"
    return child


class AgentHarness(BaseAgent):
    """DSH plans; Harbor executes inside the DeepSWE Docker environment."""

    @staticmethod
    def name() -> str:
        return "metacorsi-dsh"

    def version(self) -> str:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    async def _audit_candidate(self, environment: BaseEnvironment) -> str:
        result = await environment.exec(
            "cd /app && echo METACORSI_CANDIDATE_AUDIT && git status --short && "
            "git diff --stat && git log -1 --oneline",
            timeout_sec=60,
        )
        return (result.stdout or "") + (result.stderr or "")

    async def _materialize_candidate(self, environment: BaseEnvironment, reason: str) -> dict:
        """Preserve an explored candidate so post-verifier RCA can inspect it.

        DeepSWE's model.patch collection follows git history.  Two near-passing
        recovery trajectories previously exhausted their step budget with a
        dirty worktree and yielded a zero-byte patch, erasing the implementation
        evidence that the self-evolution loop needed.
        """
        command = (
            "cd /app && "
            "if [ -n \"$(git status --porcelain)\" ]; then "
            "git config user.email metacorsi@localhost && "
            "git config user.name MetaCoRSI && git add -A && "
            f"git commit -m 'MetaCoRSI candidate checkpoint ({reason})'; "
            "fi; git status --short; git log -1 --oneline"
        )
        result = await environment.exec(command, timeout_sec=120)
        return {"reason": reason, "return_code": result.return_code,
                "output": ((result.stdout or "") + (result.stderr or ""))[-4000:]}

    async def _resume_candidate(self, environment: BaseEnvironment) -> dict | None:
        if os.environ.get("METACORSI_DISABLE_CANDIDATE_REPLAY") == "1":
            return {"status": "disabled-for-frozen-evaluation"}
        registry_path = os.environ.get("METACORSI_CANDIDATE_REGISTRY")
        if not registry_path or not Path(registry_path).exists():
            return None
        try:
            registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"status": "registry-unreadable"}
        # Harbor truncates long trial names (e.g. ``...directiv``), so use a
        # symmetric task-prefix match instead of requiring the full benchmark id.
        portfolio = _candidate_for_session(registry, self.session_id or "")
        match = _active_candidate(portfolio)
        if not match or not match.get("patch"):
            return None
        encoded = base64.b64encode(str(match["patch"]).encode()).decode()
        # Passing a large candidate as one docker-compose command crosses the
        # Windows CreateProcess command-line limit before Docker sees it.  The
        # payload is base64, so it is safe to stage in bounded shell chunks and
        # decode inside the task container.  This remains same-task candidate
        # continuation; no patch is shared between benchmark tasks.
        staging = "/tmp/metacorsi-candidate.b64"
        patch_path = "/tmp/metacorsi-candidate.patch"
        setup_command, chunk_commands, command = _candidate_staging_commands(encoded)
        result = await environment.exec(
            setup_command,
            timeout_sec=60,
        )
        if result.return_code != 0:
            return {"status": "staging-failed", "patch_digest": match.get("patch_digest")}
        for chunk_command in chunk_commands:
            result = await environment.exec(
                chunk_command, timeout_sec=60
            )
            if result.return_code != 0:
                await environment.exec(f"rm -f {staging} {patch_path}", timeout_sec=60)
                return {"status": "staging-failed", "patch_digest": match.get("patch_digest")}
        result = await environment.exec(command, timeout_sec=120)
        if result.return_code != 0:
            await environment.exec("cd /app && git reset --hard HEAD && git clean -fd", timeout_sec=60)
            # Some benchmark patches have stale index hashes but apply cleanly
            # by context. Fall back only after restoring the pristine task.
            result = await environment.exec(
                f"cd /app && git apply {patch_path} && git status --short",
                timeout_sec=120,
            )
        if result.return_code != 0:
            await environment.exec("cd /app && git reset --hard HEAD && git clean -fd", timeout_sec=60)
        return {"status": "applied" if result.return_code == 0 else "rejected",
                "patch_digest": match.get("patch_digest"),
                "prior_f2p": f"{match.get('f2p_passed')}/{match.get('f2p_total')}",
                "prior_p2p": f"{match.get('p2p_passed')}/{match.get('p2p_total')}",
                "root_kind": match.get("root_kind"),
                "stagnation_count": int((portfolio or {}).get("stagnation_count", 0)),
                "reflection": (portfolio or {}).get("reflection") or match.get("reflection"),
                "portfolio_size": len((portfolio or {}).get("candidates") or [match]),
                "output": ((result.stdout or "") + (result.stderr or ""))[-3000:]}

    def _ask(self, prompt: str) -> str:
        exe = os.environ["METACORSI_DSH_EXE"]
        patch = os.environ["METACORSI_DSH_PATCH"]
        if "METACORSI RCA PHASE: MATERIALIZE" in prompt:
            patch = os.environ.get("METACORSI_DSH_MATERIALIZE_PATCH", patch)
        child_env = _provider_only_env(dict(os.environ))
        # DSH's profile runtime owns live patch watchers and session stores.
        # Sharing one DSH_HOME across Harbor's parallel trials can serialize or
        # deadlock startup, so every trial receives a private runtime home.
        dsh_home = _runtime_home(self.logs_dir)
        dsh_home.mkdir(parents=True, exist_ok=True)
        child_env["DSH_HOME"] = str(dsh_home.resolve())
        # Read the task from stdin: DeepSWE requirements can exceed Windows'
        # command-line size and must never be silently truncated.
        call_timeout = int(os.environ.get("METACORSI_DSH_CALL_TIMEOUT", "360"))
        request_prompt = prompt
        attempts: list[dict] = []
        final = ""
        proc = None
        saw_max_tokens = False
        solver_session_id = getattr(self, "_dsh_session_id", None)
        formatter_mode = False
        solver_draft = ""
        for attempt in range(3):
            args = [exe, "--profile", "headless", "--patch", patch, "--json"]
            session_id = None if formatter_mode else solver_session_id
            if session_id:
                args += ["--session-id", session_id]
            args.append("-")
            proc = subprocess.run(
                args, input=request_prompt, cwd=str(self.logs_dir),
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=call_timeout, env=child_env,
            )
            events = []
            for raw in proc.stdout.splitlines():
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                events.append(event)
                if (not formatter_mode and event.get("type") == "session"
                        and event.get("sessionId")):
                    self._dsh_session_id = event["sessionId"]
                if event.get("type") == "final":
                    final = str(event.get("text") or "")
            attempts.append({"attempt": attempt + 1, "mode": "compiler" if formatter_mode else "solver",
                             "returncode": proc.returncode,
                             "events": events, "stderr": proc.stderr})
            output = (proc.stdout or "") + (proc.stderr or "")
            saw_max_tokens = saw_max_tokens or "max-tokens" in output
            valid_final = False
            if final:
                try:
                    valid_final = isinstance(_json_object(final), dict)
                except (json.JSONDecodeError, ValueError):
                    pass
            if valid_final:
                # DSH exits non-zero on a max-token finish even when the visible
                # final channel already contains a complete protocol object.
                break
            if attempt < 2 and "max-tokens" in output:
                # A provider-side 4096 cap is expected.  Preserve the session
                # and ask DSH to compress the unfinished plan into the strict
                # executor protocol instead of discarding the whole trajectory.
                command_limit = int(os.environ.get("METACORSI_DSH_COMMAND_MAX_CHARS", "240"))
                thinking = next((str(item.get("text") or "") for item in reversed(events)
                                 if item.get("type") == "thinking"), "")
                if thinking and not formatter_mode:
                    solver_draft = thinking[-6000:]
                if (os.environ.get("METACORSI_DSH_STATELESS_COMPILER") == "1"
                        and solver_draft):
                    formatter_mode = True
                    request_prompt = (
                        "You are a stateless command-plan compiler, not a problem solver. "
                        "Convert the solver draft below into ONE immediately executable JSON object. "
                        "Do not re-analyze or explain. Schema: analysis <= 120 chars; commands has 1-3 "
                        f"strings each <= {command_limit} chars; done=false. No markdown or heredoc. "
                        "Compact python -c, printf, sed, or git apply is allowed.\n\nSOLVER DRAFT:\n" +
                        solver_draft
                    )
                else:
                    request_prompt = (
                        "Your previous response reached the output limit. Do not repeat analysis. "
                        "Return ONLY one complete JSON object: analysis <= 160 characters; commands "
                        f"has at most 3 strings and each string <= {command_limit} characters; done is boolean. "
                        "No heredoc or markdown. Execute the smallest direct edit now; compact python -c, "
                        "printf, sed, or git apply is allowed."
                    )
                final = ""
                continue
            if attempt < 2:
                # A syntactically invalid final (commonly an unescaped grep
                # backslash) is a transport error, not a task failure. Keep the
                # solver session and ask it to re-encode the same action without
                # reopening the reasoning trajectory.
                invalid_tail = final[-1800:]
                request_prompt = (
                    "Your previous final was not valid JSON. Do not solve or analyze again. "
                    "Return the SAME command plan as one strict JSON object with properly escaped "
                    "backslashes and quotes; keys are analysis, commands, done. No markdown. "
                    "Invalid final to repair:\n" + invalid_tail
                )
                final = ""
                continue
            break
        with (self.logs_dir / "dsh_raw.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"session_id": getattr(self, "_dsh_session_id", None),
                                 "attempts": attempts}, ensure_ascii=False) + "\n")
        try:
            valid_final = bool(final) and isinstance(_json_object(final), dict)
        except (json.JSONDecodeError, ValueError):
            valid_final = False
        if proc is None or not valid_final:
            if saw_max_tokens:
                return _protocol_fallback_plan(prompt)
            raise RuntimeError(
                f"DSH failed rc={getattr(proc, 'returncode', None)}; runtime={dsh_home}; "
                f"stdout_tail={(getattr(proc, 'stdout', '') or '')[-600:]!r}; "
                f"stderr_tail={(getattr(proc, 'stderr', '') or '')[-600:]!r}"
            )
        return final

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        control_mode = os.environ.get(
            "METACORSI_CONTROL_MODE", "rca_with_regression_gates"
        )
        knowledge_path = os.environ.get("METACORSI_KNOWLEDGE_STORE")
        brain = MetaCoRSIBrain(
            control_mode=control_mode,
            knowledge_path=Path(knowledge_path) if knowledge_path else None,
        )
        generated = brain.generate_agent("dsh", self.session_id or "deepswe", instruction)
        k7_context = brain.skill_router.prompt(
            instruction, include_validation=control_mode != "rca_only"
        )
        cognitive_context = generated.context()
        candidate_seed = await self._resume_candidate(environment)
        initial = await environment.exec("pwd && find . -maxdepth 2 -type f | head -80", timeout_sec=60)
        observation = (initial.stdout or "") + (initial.stderr or "")
        v24_split_attractor = brain.ltm_policy in {"split_attractor_v24", "naive_latch_v24"}
        v24_predictive = brain.ltm_policy == "predictive_coding_v24"
        failure_locked = brain.ltm_policy == "failure_locked_v21"
        evidence_gated = brain.ltm_policy in {
            "evidence_gated_v20",
            "failure_locked_v21",
            "sleep_consolidation_v24",
            "engram_ensemble_v24",
        }
        if v24_split_attractor:
            # Step 0 starts with PFG-hDeltaK gate closed (g_0 = 0): task-local
            # Slow Backbone anchors remain available, while cross-task schemas
            # are held in background tracking until live evidence matches.
            evolved_context, sa_step0 = brain.step_working_memory(
                instruction,
                task_id=self.session_id or "deepswe",
                repository_evidence="",
                has_live_evidence=False,
            )
            grounding_history: list[dict] = [sa_step0.json()]
        elif v24_predictive:
            bundle0, pc_step0 = brain.predictive_coding.step(
                task_id=self.session_id or "deepswe",
                task_text=instruction,
                repository_evidence="",
            )
            brain.last_groundings = ()
            from metacorsi.knowledge.indexed_ltm import render_grounded_application

            evolved_context = render_grounded_application(
                bundle0, (), "predictive_coding_v24"
            )
            grounding_history = [pc_step0.json()]
        else:
            evolved_context = brain.evolved_knowledge_prompt(
                instruction,
                task_id=self.session_id or "deepswe",
                # V20 must not ground cross-task knowledge from a task statement or
                # a shallow file listing.  It starts with the local episode only and
                # re-runs grounding after the agent has collected repository evidence.
                repository_evidence="" if brain.ltm_policy in {"evidence_gated_v20", "failure_locked_v21"} else observation,
            )
            grounding_history = [{
                "step": 0,
                "stage": "initial",
                "evidence_digest": hashlib.sha256(b"").hexdigest(),
                "evidence_excerpt": "",
                "grounding": brain.grounding_receipt(),
            }]
        last_abstract_ids = {
            item["abstract_id"] for item in grounding_history[-1].get("grounding", ())
        }
        repository_evidence_window = ""
        sa_recent_observations: list[str] = []
        sa_stagnation_steps = 0
        sa_edited_since_latch = False
        sa_pending_counterevidence = False
        sa_ordinary_failure = False
        locked_abstract_id: str | None = None
        active_grounding_receipt: list[dict] = list(grounding_history[-1].get("grounding", ()))
        if candidate_seed:
            observation += "\n\nMETACORSI QUARANTINED CANDIDATE SEED:\n" + json.dumps(
                candidate_seed, ensure_ascii=False
            )
        history: list[dict] = []
        rca_focused_commands: list[str] = []
        base_steps = int(os.environ.get("METACORSI_DSH_MAX_STEPS", "24"))
        extension_steps = int(os.environ.get("METACORSI_DSH_MAX_EXTENSION", "6"))
        max_steps = base_steps
        hard_max = base_steps + max(0, extension_steps)
        materialization: dict | None = None
        completed = False
        for step in range(1, hard_max + 1):
            if step > max_steps:
                if extension_steps > 0 and _successful_test_seen(history):
                    max_steps = hard_max
                    if history:
                        history[-1]["budget_extension"] = {
                            "from": base_steps, "to": hard_max,
                            "reason": "materialized candidate has a successful focused test",
                        }
                else:
                    break
            if step in {6, 12, 18, 22, max_steps}:
                observation += "\n\n" + await self._audit_candidate(environment)
            knowledge_update = ""
            if v24_split_attractor and step > 1:
                live_window = "\n".join(sa_recent_observations[-2:])[-16000:]
                sa_directive, sa_step = brain.step_working_memory(
                    instruction,
                    task_id=self.session_id or "deepswe",
                    repository_evidence=live_window,
                    has_live_evidence=bool(live_window.strip()),
                    explicit_counterevidence=sa_pending_counterevidence,
                    stagnation_steps=sa_stagnation_steps,
                    ordinary_command_failure=sa_ordinary_failure,
                )
                sa_pending_counterevidence = False
                grounding_history.append(sa_step.json())
                if sa_step.transition == "disinhibitory_write":
                    sa_edited_since_latch = False
                    sa_stagnation_steps = 0
                elif sa_step.transition == "reinhibitory_erase":
                    sa_edited_since_latch = False
                    sa_stagnation_steps = 0
                if sa_step.active_grounding is not None:
                    active_grounding_receipt = [sa_step.active_grounding.json()]
                elif sa_step.transition == "reinhibitory_erase":
                    active_grounding_receipt = []
                if sa_step.transition in {
                    "disinhibitory_write",
                    "attractor_switch",
                    "reinhibitory_erase",
                }:
                    knowledge_update = sa_directive
            elif v24_predictive and step > 1:
                live_window = "\n".join(sa_recent_observations[-2:])[-16000:]
                _, pc_step = brain.predictive_coding.step(
                    task_id=self.session_id or "deepswe",
                    task_text=instruction,
                    repository_evidence=live_window,
                    explicit_counterevidence=sa_pending_counterevidence,
                    stagnation_steps=sa_stagnation_steps,
                )
                sa_pending_counterevidence = False
                grounding_history.append(pc_step.json())
                if pc_step.active_grounding is not None:
                    brain.last_groundings = (pc_step.active_grounding,)
                    active_grounding_receipt = [pc_step.active_grounding.json()]
                elif pc_step.transition == "falsification_reset":
                    brain.last_groundings = ()
                    active_grounding_receipt = []
                if pc_step.transition in {"surprise_driven_activation", "prediction_error_shift"} and pc_step.active_grounding:
                    rec = pc_step.active_grounding
                    knowledge_update = (
                        f"\n\n[PREDICTIVE-CODING SURPRISE UPDATE (surprise={pc_step.error_signal.surprise_magnitude:.3f}; "
                        f"type={pc_step.error_signal.error_type})]:\n"
                        + "\n".join(f"- {a}" for a in rec.instantiated_action)
                    )
                elif pc_step.transition == "falsification_reset":
                    knowledge_update = (
                        "\n\n[PREDICTIVE-CODING FALSIFICATION RESET]:\n"
                        f"Previous hypothesis was falsified by persistent prediction error ({pc_step.reason}). "
                        "Abandon that repair direction and re-derive causal error from current trace."
                    )
            elif evidence_gated and step > 1:
                repository_evidence = (repository_evidence_window + "\n" + observation)[-32000:]
                refreshed_context = brain.evolved_knowledge_prompt(
                    instruction,
                    task_id=self.session_id or "deepswe",
                    repository_evidence=repository_evidence,
                )
                receipt = brain.grounding_receipt()
                if failure_locked:
                    receipt, locked_abstract_id, suppressed = _lock_grounding_receipt(
                        receipt, locked_abstract_id
                    )
                    if suppressed:
                        grounding_history.append({
                            "step": step,
                            "stage": "relation-switch-suppressed",
                            "locked_abstract_id": locked_abstract_id,
                            "suppressed_abstract_ids": suppressed,
                            "evidence_digest": hashlib.sha256(
                                repository_evidence.encode("utf-8", errors="replace")
                            ).hexdigest(),
                            "evidence_excerpt": repository_evidence[-4000:],
                            "grounding": [],
                        })
                abstract_ids = (
                    {locked_abstract_id}
                    if failure_locked and locked_abstract_id is not None and not receipt
                    else {item["abstract_id"] for item in receipt}
                )
                if not failure_locked or (receipt and abstract_ids != last_abstract_ids):
                    evolved_context = refreshed_context
                if abstract_ids != last_abstract_ids:
                    grounding_history.append({
                        "step": step,
                        "stage": "repository-evidence-refresh",
                        "evidence_digest": hashlib.sha256(
                            repository_evidence.encode("utf-8", errors="replace")
                        ).hexdigest(),
                        "evidence_excerpt": repository_evidence[-4000:],
                        "grounding": receipt,
                    })
                    last_abstract_ids = abstract_ids
                    if receipt:
                        active_grounding_receipt = list(receipt)
                    if receipt:
                        knowledge_update = (
                            "\n\nMETACORSI EVIDENCE-GATED KNOWLEDGE UPDATE:\n"
                            + refreshed_context
                        )
                    elif not failure_locked:
                        knowledge_update = (
                            "\n\nMETACORSI EVIDENCE-GATED KNOWLEDGE RETRACTION:\n"
                            "The previous cross-task diagnostic prior is no longer supported "
                            "by the current repository-evidence window. Do not use it as a "
                            "repair premise; continue from task-local evidence and RCA."
                        )
            phase_directive = _phase_directive(step, max_steps, history, candidate_seed)
            command_limit = int(os.environ.get("METACORSI_DSH_COMMAND_MAX_CHARS", "240"))
            if getattr(self, "_dsh_session_id", None):
                prompt = f"""Continue the same DeepSWE task. The following is the newest output from commands
executed in the real Docker repository. Preserve conclusions already established in this session.
Return ONLY one JSON object with analysis, commands, and done. Use at most 3 commands,
each no longer than {command_limit} characters.

{phase_directive}

CONTAINER OBSERVATION (step {step}):
{observation[-16000:]}

RUNTIME ACTION SUMMARY:
{json.dumps(history[-5:], ensure_ascii=False)}
{knowledge_update}"""
            else:
                prompt = f"""You are the Target Agent inside a Meta-CoRSI/Harbor DeepSWE trial.
The real repository is in an isolated Docker environment. You cannot access it directly.
Return ONLY one JSON object: {{"analysis":"brief", "commands":["shell command"], "done":false}}.
Commands are executed in the task container. Use at most 3 safe shell commands per turn,
each no longer than {command_limit} characters. Inspect before editing; compact Python -c,
printf, sed, or git apply commands are allowed; run focused tests.
Set done=true only after implementing and testing the requested fix. Never use markdown.
Never search for gold/reference patches, hidden verifier tests, unreachable git objects,
or duplicate repository copies outside /app. Solve only from the public task, /app source,
/app tests, dependency documentation/source, and outputs of commands you run yourself.

TASK:
{instruction}

{k7_context}

{evolved_context}

METACORSI COGNITIVE CONTEXT:
{json.dumps(cognitive_context, ensure_ascii=False)}

RECENT CONTAINER OBSERVATION (step {step}):
{observation[-12000:]}

RECENT ACTION SUMMARY:
{json.dumps(history[-5:], ensure_ascii=False)}

{phase_directive}"""
            action = None
            parse_error = ""
            for attempt in range(4):
                retry_prompt = prompt if attempt == 0 else (
                    prompt + "\n\nYour previous response was empty or not valid command-plan JSON (" +
                    parse_error + "). Do not use tools or prose. Return the JSON object now."
                )
                answer = await asyncio.to_thread(self._ask, retry_prompt)
                try:
                    action = _json_object(answer)
                    break
                except ValueError as exc:
                    parse_error = str(exc)
            if action is None:
                raise RuntimeError("DSH command-plan protocol failed after 4 attempts: " + parse_error)
            commands = []
            for item in action.get("commands", []):
                if isinstance(item, str):
                    commands.append(item)
                elif isinstance(item, dict) and isinstance(item.get("command"), str):
                    commands.append(item["command"])
                if len(commands) == 3:
                    break
            outputs = []
            return_codes = []
            for command in commands:
                violation = _answer_search_violation(command)
                if violation:
                    return_codes.append(126)
                    outputs.append(
                        f"$ {command}\nrc=126\nMETACORSI ANSWER-ACCESS POLICY BLOCKED: {violation}. "
                        "Work only in /app from public source/tests and self-generated executions."
                    )
                    continue
                result = await environment.exec(command, timeout_sec=900)
                return_codes.append(result.return_code)
                outputs.append(f"$ {command}\nrc={result.return_code}\n{result.stdout or ''}\n{result.stderr or ''}")
            observation = "\n".join(outputs) if outputs else "No command executed."
            rca_record = None
            if any(code != 0 for code in return_codes):
                failed = tuple(c for c, code in zip(commands, return_codes) if code != 0)
                ev = FailureEvidence(task_id=self.session_id or "deepswe", assertion=observation[-6000:],
                    trace=observation[-12000:], failed_tests=failed, task_text=instruction,
                    metadata={"impacted_tests": list(failed)})
                if control_mode == "rca_only":
                    diagnosis = brain.analyze(ev)
                    rca_record = {
                        "control_mode": "rca_only",
                        "root_step": diagnosis.root_step,
                        "root_kind": diagnosis.root_kind,
                        "confidence": diagnosis.confidence,
                        "explanation": diagnosis.explanation,
                        "walk": diagnosis.raw.get("walk"),
                        "walk_back": diagnosis.raw.get("back"),
                    }
                    observation += "\n\nMETACORSI RCA DIRECTIVE:\n" + json.dumps(
                        rca_record, ensure_ascii=False
                    )
                else:
                    diagnosis, plan = brain.analyze_and_plan(ev)
                    rca_record = {"control_mode": "rca_with_regression_gates",
                                  "root_step": diagnosis.root_step,
                                  "confidence": diagnosis.confidence,
                                  "tier": plan.tier, "focused": list(plan.focused),
                                  "boundary": list(plan.boundary), "full": plan.full,
                                  "reason": plan.reason}
                    rca_focused_commands.extend(
                        probe for command in failed if (probe := _focused_probe(command)) is not None
                    )
                    observation += "\n\nMETACORSI RCA/REGRESSION DIRECTIVE:\n" + json.dumps(
                        rca_record, ensure_ascii=False
                    )
            if v24_split_attractor or v24_predictive:
                gate_active = (
                    brain.split_attractor.gate == 1
                    if v24_split_attractor
                    else (brain.predictive_coding.active_abstract_id is not None)
                )
                feedback = _classify_step_feedback(
                    commands,
                    return_codes,
                    gate_Active=gate_active,
                    edited_since_latch=sa_edited_since_latch,
                    rca_gate_blocked=False,
                )
                if feedback["edited"] and gate_active:
                    sa_edited_since_latch = True
                if feedback["test_passed"]:
                    sa_stagnation_steps = 0
                    sa_pending_counterevidence = False
                elif feedback["explicit_counterevidence"]:
                    sa_pending_counterevidence = True
                    sa_stagnation_steps += 1
                elif feedback["test_failed"] and gate_active:
                    sa_stagnation_steps += 1
                sa_ordinary_failure = feedback["ordinary_failure"]
                tagged_obs = observation
                if feedback["test_failed"]:
                    tagged_obs = "METACORSI_CONFIRMED_FAILURE_EVIDENCE\n" + observation
                sa_recent_observations.append(tagged_obs[-10000:])
            elif failure_locked:
                confirmed_failure = _confirmed_focused_failure(commands, return_codes)
                if confirmed_failure:
                    repository_evidence_window = (
                        repository_evidence_window
                        + "\nMETACORSI_CONFIRMED_FAILURE_EVIDENCE\n"
                        + observation
                    )[-32000:]
            elif evidence_gated:
                repository_evidence_window = (
                    repository_evidence_window + "\n" + observation
                )[-32000:]
            history.append({"step": step, "analysis": action.get("analysis", ""),
                            "commands": commands, "return_codes": [
                                int(re.search(r"rc=(-?\d+)", x).group(1)) for x in outputs],
                            "rca": rca_record})
            context.metadata = {"framework": "dsh", "model": self.model_name,
                                "control_mode": control_mode,
                                "steps": step, "history": history}
            if action.get("done"):
                if control_mode == "rca_only":
                    history[-1]["completion"] = "accepted_without_regression_gate"
                    materialization = await self._materialize_candidate(environment, "agent-done")
                    completed = True
                    break
                # RCA is the authority for focused regression. A Target Agent
                # cannot declare completion while a previously failing causal
                # probe has not been re-run successfully after repair.
                gate_commands = list(dict.fromkeys(rca_focused_commands))[-3:]
                gate_failures = []
                for command in gate_commands:
                    result = await environment.exec(command, timeout_sec=900)
                    if result.return_code != 0:
                        gate_failures.append(
                            f"$ {command}\nrc={result.return_code}\n{result.stdout or ''}\n{result.stderr or ''}"
                        )
                if gate_failures:
                    observation = "RCA FOCUSED GATE BLOCKED COMPLETION:\n" + "\n".join(gate_failures)
                    history[-1]["completion_blocked_by_rca_gate"] = True
                    if v24_split_attractor:
                        sa_pending_counterevidence = True
                        sa_recent_observations.append(
                            "METACORSI_CONFIRMED_FAILURE_EVIDENCE\n" + observation[-10000:]
                        )
                    continue
                history[-1]["rca_focused_gate"] = "passed" if gate_commands else "not-required"
                materialization = await self._materialize_candidate(environment, "agent-done")
                completed = True
                break
        if not completed:
            materialization = await self._materialize_candidate(environment, "step-budget-exhausted")
            if history:
                history[-1]["completion"] = "step_budget_exhausted_candidate_checkpointed"
        (self.logs_dir / "dsh_bridge.json").write_text(
            json.dumps({"model": self.model_name, "control_mode": control_mode,
                        "knowledge_store": knowledge_path,
                        "grounding": (active_grounding_receipt
                                      if (failure_locked or v24_split_attractor)
                                      else brain.grounding_receipt()),
                        "grounding_history": grounding_history,
                        "split_attractor_metrics": (
                            brain.split_attractor.metrics() if v24_split_attractor else None
                        ),
                        "candidate_seed": candidate_seed,
                        "evolved_knowledge": evolved_context,
                        "max_steps": max_steps, "candidate_materialization": materialization,
                        "cognitive_context": cognitive_context,
                        "brain": brain.snapshot(), "history": history}, ensure_ascii=False, indent=2),
            encoding="utf-8")


__all__ = ["AgentHarness"]
