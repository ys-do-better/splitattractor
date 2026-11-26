"""Run DeepSeek Harness (DSH) on the host while executing shell commands in Harbor."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from harbor.agents.base import BaseAgent

from flashrsi._records import _config_record
from flashrsi.agents.deepseek_harness import (
    _load_snapshot_config,
    _model_api,
    _parse_dsh_events,
    _provider_only_env,
    _render_snapshot_resources,
    _resolve_dsh_cli,
    _summarize_usage,
    _write_cordis_patch,
)
from flashrsi.core import Trajectory
from flashrsi.integrations._hosted_miniagent import _TRIAL

_TEST_PROBE = re.compile(
    r"(?:\bpytest\b|\bvitest\b|\bjest\b|\bcargo\s+test\b|\bgo\s+test\b|"
    r"\b(?:pnpm|npm|yarn)\s+(?:run\s+)?test\b|\bdeno\s+test\b|\btox\b|"
    r"\bctest\b|\brspec\b|\bmvn\s+test\b|\bgradle\w*\s+.*test\b|"
    r"\bmake\s+(?:\w*test\w*)\b|\bunittest\b|METACORSI_(?:SMOKE|FOCUSED|BOUNDARY|PRESERVE|REGRESSION))",
    re.I,
)

_ANSWER_SEARCH_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bgit\s+(?:fsck|reflog)\b", re.I), "unreachable/reflog object search"),
    (re.compile(r"\bgit\s+rev-list\b[^\n]*(?:--all|--objects)", re.I), "all-object history search"),
    (re.compile(r"\bfind\s+/(?:\s|$)", re.I), "container-wide duplicate repository/test search"),
    (
        re.compile(r"/(?:gold|solution|solutions|oracle|verifier)(?:/|\b)", re.I),
        "gold/oracle/verifier path access",
    ),
    (re.compile(r"\b(?:docker|podman)\s+(?:cp|exec|run)\b", re.I), "cross-container access"),
)

_PLANNER_PERSONA = (
    "You are a command planner for an external Docker executor. You have no local "
    "tools. Never inspect or modify the host workspace. For every turn, return one "
    "valid JSON object with analysis, commands, and done fields; return no markdown. "
    "Keep analysis under 160 characters and return at most three commands. "
    "During materialization, execute a direct edit immediately; compact python -c, "
    "printf, sed, or git apply is allowed. Avoid heredocs. Do not spend output "
    "explaining shell quoting."
)


def _answer_search_violation(command: str) -> str | None:
    for pattern, reason in _ANSWER_SEARCH_PATTERNS:
        if pattern.search(command):
            return reason
    return None


def _json_object(text: str) -> dict:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates: list[dict] = []
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
            if isinstance(parsed, dict) and isinstance(parsed.get("commands"), list):
                candidates.append(parsed)
        except json.JSONDecodeError:
            pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("commands"), list):
            candidates.append(value)
    if not candidates:
        raise ValueError(f"DSH emitted no command-plan JSON (chars={len(text)})")
    return candidates[-1]


def _successful_test_seen(history: list[dict]) -> bool:
    for item in history:
        for command, code in zip(item.get("commands", ()), item.get("return_codes", ())):
            if code == 0 and _TEST_PROBE.search(command):
                return True
    return False


def _protocol_fallback_plan(prompt: str) -> str:
    closing = "METACORSI RCA PHASE: CLOSE" in prompt
    return json.dumps(
        {
            "analysis": "Provider reasoning exhausted the output cap; preserve and validate the materialized candidate.",
            "commands": [
                "cd /app && git status --short",
                "cd /app && git diff --check",
            ],
            "done": closing,
        }
    )


def _phase_directive(
    step: int,
    max_steps: int,
    history: list[dict] | None = None,
    command_limit: int = 1600,
) -> str:
    history = history or []
    tested = _successful_test_seen(history)
    if step <= 4:
        phase = "DIAGNOSE"
        rule = (
            "Localize the residual counterexample and one causal decision point. "
            "Do not redesign the complete feature."
        )
    elif not tested and step <= 12:
        phase = "MATERIALIZE"
        rule = "Create a minimal vertical code slice now and wire its public export/registration/dispatch path."
    elif step <= 18:
        phase = "VERIFY"
        rule = "Run a focused end-to-end probe. Freeze working behavior and close only observed boundary gaps."
    else:
        phase = "CLOSE"
        rule = "Stop broad exploration. Finish integration, run the smallest decisive test, and preserve a git checkpoint."
    remaining = max_steps - step + 1
    materialize = (
        f" During MATERIALIZE, use a direct edit command up to {command_limit} characters; "
        "do not spend a turn planning shell quoting."
        if phase == "MATERIALIZE"
        else ""
    )
    return (
        f"METACORSI RCA PHASE: {phase}; step={step}/{max_steps}; remaining={remaining}. "
        f"{rule} A successful shell command is not proof of capability; ensure an executable patch exists."
        f"{materialize}"
    )


def _invoke_dsh_step(
    *,
    dsh_exe: str,
    profile: str,
    patch_default: Path,
    patch_materialize: Path,
    dsh_home: Path,
    logs_dir: Path,
    child_env: dict[str, str],
    prompt: str,
    session_id: str | None,
    call_timeout: int,
    command_limit: int,
    stateless_compiler: bool,
    step: int,
) -> tuple[str, str | None, list[dict], list[dict]]:
    patch = (
        patch_materialize
        if "METACORSI RCA PHASE: MATERIALIZE" in prompt
        else patch_default
    )
    dsh_home.mkdir(parents=True, exist_ok=True)
    env = dict(child_env)
    env["DSH_HOME"] = str(dsh_home.resolve())

    request_prompt = prompt
    attempts: list[dict] = []
    step_model_calls: list[dict] = []
    final = ""
    proc = None
    saw_max_tokens = False
    solver_session_id = session_id
    formatter_mode = False
    solver_draft = ""

    for attempt in range(3):
        args = [dsh_exe, "--profile", profile, "--patch", str(patch), "--json"]
        active_session = None if formatter_mode else solver_session_id
        if active_session:
            args += ["--session-id", active_session]
        args.append("-")
        proc = subprocess.run(
            args,
            input=request_prompt,
            cwd=str(logs_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=call_timeout,
            env=env,
        )
        parsed_final, new_session, events, calls = _parse_dsh_events(
            proc.stdout or "",
            call_prefix=f"step{step}-att{attempt + 1}",
        )
        for call in calls:
            call["call_id"] = f"step{step}:att{attempt + 1}:{call['call_id']}"
        step_model_calls.extend(calls)
        if not formatter_mode and new_session:
            solver_session_id = new_session
        if parsed_final:
            final = parsed_final
        attempts.append(
            {
                "attempt": attempt + 1,
                "mode": "compiler" if formatter_mode else "solver",
                "returncode": proc.returncode,
                "events": events,
                "stderr": proc.stderr,
            }
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        saw_max_tokens = saw_max_tokens or "max-tokens" in output
        valid_final = False
        if final:
            try:
                valid_final = isinstance(_json_object(final), dict)
            except (json.JSONDecodeError, ValueError):
                pass
        if valid_final:
            break
        if attempt < 2 and "max-tokens" in output:
            thinking = next(
                (
                    str(item.get("text") or "")
                    for item in reversed(events)
                    if item.get("type") == "thinking"
                ),
                "",
            )
            if thinking and not formatter_mode:
                solver_draft = thinking[-6000:]
            if stateless_compiler and solver_draft:
                formatter_mode = True
                request_prompt = (
                    "You are a stateless command-plan compiler, not a problem solver. "
                    "Convert the solver draft below into ONE immediately executable JSON object. "
                    "Do not re-analyze or explain. Schema: analysis <= 120 chars; commands has 1-3 "
                    f"strings each <= {command_limit} chars; done=false. No markdown or heredoc. "
                    "Compact python -c, printf, sed, or git apply is allowed.\n\nSOLVER DRAFT:\n"
                    + solver_draft
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

    with (logs_dir / "dsh_raw.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {"step": step, "session_id": solver_session_id, "attempts": attempts},
                ensure_ascii=False,
            )
            + "\n"
        )
    try:
        valid_final = bool(final) and isinstance(_json_object(final), dict)
    except (json.JSONDecodeError, ValueError):
        valid_final = False
    if proc is None or not valid_final:
        if saw_max_tokens:
            return (
                _protocol_fallback_plan(prompt),
                solver_session_id,
                attempts,
                step_model_calls,
            )
        raise RuntimeError(
            f"DSH failed rc={getattr(proc, 'returncode', None)}; "
            f"stdout_tail={(getattr(proc, 'stdout', '') or '')[-600:]!r}; "
            f"stderr_tail={(getattr(proc, 'stderr', '') or '')[-600:]!r}"
        )
    return final, solver_session_id, attempts, step_model_calls


class HostedDSHBridge(BaseAgent):
    """DSH plans on the host; Harbor executes shell commands inside the DeepSWE container."""

    @staticmethod
    def name() -> str:
        return "flashrsi-hosted-dsh"

    def version(self) -> str:
        return "1"

    async def setup(self, environment) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    async def _audit_candidate(self, environment) -> str:
        result = await environment.exec(
            "cd /app && echo METACORSI_CANDIDATE_AUDIT && git status --short && "
            "git diff --stat && git log -1 --oneline",
            timeout_sec=60,
        )
        return (result.stdout or "") + (result.stderr or "")

    async def _materialize_candidate(self, environment, reason: str) -> dict:
        command = (
            "cd /app && "
            'if [ -n "$(git status --porcelain)" ]; then '
            "git config user.email metacorsi@localhost && "
            "git config user.name MetaCoRSI && git add -A && "
            f"git commit -m 'MetaCoRSI candidate checkpoint ({reason})'; "
            "fi; git status --short; git log -1 --oneline"
        )
        result = await environment.exec(command, timeout_sec=120)
        return {
            "reason": reason,
            "return_code": result.return_code,
            "output": ((result.stdout or "") + (result.stderr or ""))[-4000:],
        }

    async def run(self, instruction: str, environment, context) -> None:
        trial = _TRIAL.get()
        target = trial.agent
        config = target.config
        llm = config["llm"]
        model = llm["model_name"]
        api_key = (
            llm.get("api_key")
            or os.environ.get("METACORSI_DSH_API_KEY")
            or os.environ.get("FLASHRSI_DSH_API_KEY")
        )
        base_url = llm.get("base_url") or "https://api.deepseek.com/v1"
        dsh_exe = _resolve_dsh_cli(config, required=True)
        api = _model_api(llm)
        reasoning = llm.get("reasoning_effort") or llm.get("effort")
        child_env = _provider_only_env(dict(os.environ), api_key, base_url)

        with (
            TemporaryDirectory(prefix="flashrsi-hosted-dsh-") as temporary,
            target.snapshot._export(full=True) as directory,
        ):
            runtime_root = Path(temporary).resolve()
            trial_key = hashlib.sha256(
                str(self.logs_dir.resolve()).encode()
            ).hexdigest()[:16]
            dsh_home = runtime_root / f"dsh-{trial_key}"
            snap_cfg = _load_snapshot_config(directory)
            system_prompt, extra_guidance, skill_dirs = _render_snapshot_resources(
                directory
            )
            persona_prefix = "\n\n".join(
                filter(None, (_PLANNER_PERSONA, system_prompt))
            )
            max_tokens = int(llm.get("max_tokens") or snap_cfg.get("max_tokens", 4096))
            materialize_tokens = int(
                snap_cfg.get("materialize_max_tokens", max(max_tokens, 8192))
            )
            patch_default = _write_cordis_patch(
                runtime_root / "dsh.default.patch.yml",
                model=model,
                max_tokens=max_tokens,
                api=api,
                reasoning_effort=reasoning,
                persona_prefix=persona_prefix,
                persona_suffix=extra_guidance,
                skill_dirs=skill_dirs,
                disable_host_tools=True,
            )
            patch_materialize = _write_cordis_patch(
                runtime_root / "dsh.materialize.patch.yml",
                model=model,
                max_tokens=materialize_tokens,
                api=api,
                reasoning_effort=reasoning,
                persona_prefix=persona_prefix,
                persona_suffix=extra_guidance,
                skill_dirs=skill_dirs,
                disable_host_tools=True,
            )

            initial = await environment.exec(
                "pwd && find . -maxdepth 2 -type f | head -80", timeout_sec=60
            )
            observation = (initial.stdout or "") + (initial.stderr or "")
            memory_rt = (
                await target.memory_runtime(trial.task, directory, observation)
                if hasattr(target, "memory_runtime")
                else None
            )
            memory_context = memory_rt.prompt_context() if memory_rt is not None else ""

            base_steps = int(config.get("max_turns", snap_cfg.get("max_turns", 24)))
            extension_steps = int(
                config.get("max_extension", snap_cfg.get("max_extension", 6))
            )
            call_timeout = int(
                config.get("call_timeout", snap_cfg.get("call_timeout", 360))
            )
            command_limit = int(
                config.get("command_max_chars", snap_cfg.get("command_max_chars", 1600))
            )
            stateless_compiler = bool(
                config.get(
                    "stateless_compiler", snap_cfg.get("stateless_compiler", False)
                )
            )
            use_phases = bool(snap_cfg.get("phase_directives", True))
            auto_commit = bool(snap_cfg.get("auto_commit", True))
            profile = str(snap_cfg.get("profile", "headless"))

            max_steps = base_steps
            hard_max = base_steps + max(0, extension_steps)
            history: list[dict] = []
            messages: list[dict] = [
                {
                    "type": "RunConfig",
                    "config": _config_record(config),
                    "snapshot_id": target.snapshot.id,
                }
            ]
            all_model_calls: list[dict] = []
            total_tool_calls = 0
            session_id: str | None = None
            materialization: dict | None = None
            completed = False
            final_analysis = ""

            for step in range(1, hard_max + 1):
                if step > max_steps:
                    if extension_steps > 0 and _successful_test_seen(history):
                        max_steps = hard_max
                        if history:
                            history[-1]["budget_extension"] = {
                                "from": base_steps,
                                "to": hard_max,
                                "reason": "materialized candidate has a successful focused test",
                            }
                    else:
                        break
                if step in {6, 12, 18, 22, max_steps}:
                    observation += "\n\n" + await self._audit_candidate(environment)

                phase_directive = (
                    _phase_directive(step, max_steps, history, command_limit)
                    if use_phases
                    else f"STEP {step}/{max_steps}."
                )
                if session_id:
                    step_prompt = (
                        "Continue the same DeepSWE task. The following is the newest output from commands\n"
                        "executed in the real Docker repository. Preserve conclusions already established in this session.\n"
                        "Return ONLY one JSON object with analysis, commands, and done. Use at most 3 commands,\n"
                        f"each no longer than {command_limit} characters.\n\n"
                        f"{phase_directive}\n\n"
                        f"CONTAINER OBSERVATION (step {step}):\n"
                        f"{observation[-16000:]}\n\n"
                        "RUNTIME ACTION SUMMARY:\n"
                        f"{json.dumps(history[-5:], ensure_ascii=False)}"
                    )
                else:
                    guidance_sections = "\n\n".join(
                        filter(None, (extra_guidance, memory_context))
                    )
                    step_prompt = (
                        "You are the Target Agent inside a FlashRSI/Harbor DeepSWE trial.\n"
                        "The real repository is in an isolated Docker environment. You cannot access it directly.\n"
                        'Return ONLY one JSON object: {"analysis":"brief", "commands":["shell command"], "done":false}.\n'
                        "Commands are executed in the task container. Use at most 3 safe shell commands per turn,\n"
                        f"each no longer than {command_limit} characters. Inspect before editing; compact Python -c,\n"
                        "printf, sed, or git apply commands are allowed; run focused tests.\n"
                        "Set done=true only after implementing, testing, and committing the requested fix. Never use markdown.\n"
                        "Never search for gold/reference patches, hidden verifier tests, unreachable git objects,\n"
                        "or duplicate repository copies outside /app. Solve only from the public task, /app source,\n"
                        "/app tests, dependency documentation/source, and outputs of commands you run yourself.\n\n"
                        f"TASK:\n{trial.prompt}\n\n"
                        + (f"{guidance_sections}\n\n" if guidance_sections else "")
                        + f"RECENT CONTAINER OBSERVATION (step {step}):\n{observation[-12000:]}\n\n"
                        + f"RECENT ACTION SUMMARY:\n{json.dumps(history[-5:], ensure_ascii=False)}\n\n"
                        + f"{phase_directive}"
                    )

                action = None
                parse_error = ""
                answer = ""
                for retry in range(4):
                    retry_prompt = (
                        step_prompt
                        if retry == 0
                        else (
                            step_prompt
                            + "\n\nYour previous response was empty or not valid command-plan JSON ("
                            + parse_error
                            + "). Do not use tools or prose. Return the JSON object now."
                        )
                    )
                    answer, session_id, _, step_calls = await asyncio.to_thread(
                        _invoke_dsh_step,
                        dsh_exe=dsh_exe,
                        profile=profile,
                        patch_default=patch_default,
                        patch_materialize=patch_materialize,
                        dsh_home=dsh_home,
                        logs_dir=self.logs_dir,
                        child_env=child_env,
                        prompt=retry_prompt,
                        session_id=session_id,
                        call_timeout=call_timeout,
                        command_limit=command_limit,
                        stateless_compiler=stateless_compiler,
                        step=step,
                    )
                    all_model_calls.extend(step_calls)
                    try:
                        action = _json_object(answer)
                        break
                    except ValueError as exc:
                        parse_error = str(exc)
                if action is None:
                    raise RuntimeError(
                        "DSH command-plan protocol failed after 4 attempts: "
                        + parse_error
                    )

                commands: list[str] = []
                for item in action.get("commands", []):
                    if isinstance(item, str):
                        commands.append(item)
                    elif isinstance(item, dict) and isinstance(item.get("command"), str):
                        commands.append(item["command"])
                    if len(commands) == 3:
                        break

                outputs: list[str] = []
                return_codes: list[int] = []
                for command in commands:
                    total_tool_calls += 1
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
                    outputs.append(
                        f"$ {command}\nrc={result.return_code}\n{result.stdout or ''}\n{result.stderr or ''}"
                    )
                    if (
                        memory_rt is not None
                        and result.return_code == 0
                        and _TEST_PROBE.search(command)
                    ):
                        memory_rt.focused_success = True

                observation = "\n".join(outputs) if outputs else "No command executed."
                rca_record = None
                if memory_rt is not None and any(code != 0 for code in return_codes):
                    failed_cmd = next(
                        c for c, code in zip(commands, return_codes) if code != 0
                    )
                    failed_code = next(code for code in return_codes if code != 0)
                    diagnosis = await memory_rt.analyze_failure(
                        step, failed_cmd, failed_code, observation
                    )
                    rca_record = {
                        "control_mode": "rca_only",
                        "root_step": diagnosis.get("root_step"),
                        "root_kind": diagnosis.get("root_kind"),
                        "confidence": diagnosis.get("confidence"),
                        "explanation": diagnosis.get("explanation"),
                    }
                    observation += "\n\nMETACORSI RCA DIRECTIVE:\n" + json.dumps(
                        rca_record, ensure_ascii=False
                    )

                final_analysis = str(action.get("analysis") or "")
                step_record = {
                    "step": step,
                    "analysis": final_analysis,
                    "commands": commands,
                    "return_codes": return_codes,
                    "done": bool(action.get("done")),
                    "rca": rca_record,
                }
                history.append(step_record)
                messages.append(
                    {
                        "role": "assistant",
                        "step": step,
                        "content": answer,
                        "analysis": final_analysis,
                        "commands": commands,
                        "done": bool(action.get("done")),
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "name": "shell",
                        "step": step,
                        "commands": commands,
                        "return_codes": return_codes,
                        "content": observation[-16000:],
                        "rca": rca_record,
                    }
                )
                if action.get("done"):
                    history[-1]["completion"] = "accepted_without_regression_gate"
                    if auto_commit:
                        materialization = await self._materialize_candidate(
                            environment, "agent-done"
                        )
                    completed = True
                    break

            if not completed:
                if auto_commit:
                    materialization = await self._materialize_candidate(
                        environment, "step-budget-exhausted"
                    )
                if history:
                    history[-1]["completion"] = (
                        "step_budget_exhausted_candidate_checkpointed"
                    )

            bridge_payload = {
                "model": model,
                "session_id": session_id,
                "max_steps": max_steps,
                "candidate_materialization": materialization,
                "history": history,
            }
            if memory_rt is not None:
                bridge_payload["grounding"] = memory_rt.grounding.get("groundings", [])
                bridge_payload["evolved_knowledge"] = memory_rt.grounding.get("prompt", "")
            (self.logs_dir / "dsh_bridge.json").write_text(
                json.dumps(bridge_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            context.metadata = {
                "framework": target.name,
                "model": model,
                "steps": len(history),
                "history": history,
            }
            messages.append(
                {
                    "type": "DSHExecution",
                    "session_id": session_id,
                    "steps": len(history),
                    "max_steps": max_steps,
                    "completed": completed,
                    "candidate_materialization": materialization,
                }
            )
            if memory_rt is not None:
                messages.append(memory_rt.receipt())

            usage = _summarize_usage(
                all_model_calls,
                turns=len(history),
                tool_calls=total_tool_calls,
                session_id=session_id,
            )
            trial.trajectory = Trajectory(
                status="finished",
                output=final_analysis or "Completed DSH execution",
                messages=tuple(messages),
                usage=usage,
            )
