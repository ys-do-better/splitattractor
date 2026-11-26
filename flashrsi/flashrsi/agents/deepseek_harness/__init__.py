"""DeepSeek Harness (DSH) adapter with versioned prompts, skills, config, and MetaCoRSI memory."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from flashrsi._records import _credential_key, _redact, _secrets
from flashrsi.agents.metacorsi import _NATIVE, MetaCoRSITarget, copy_active_memory
from flashrsi.core import Agent, Trajectory

_FORBIDDEN_RUNTIME_KEYS = ("node_path", "bun_path", "gateway_path")
_DEFAULT_CONFIG = {
    "profile": "headless",
    "max_turns": 24,
    "max_extension": 6,
    "call_timeout": 360,
    "command_max_chars": 1600,
    "max_tokens": 4096,
    "materialize_max_tokens": 8192,
    "stateless_compiler": False,
    "phase_directives": True,
    "auto_commit": True,
    "memory": {
        "enabled": False,
        "control_mode": "rca_only",
        "ltm_policy": "modular_reverse_v23",
        "memory_strategy": "cls_dual_store",
    },
}


def _resolve_dsh_cli(config, *, required: bool = False) -> str | None:
    explicit = config.get("cli_path")
    if explicit is not None:
        if not isinstance(explicit, (str, Path)) or not str(explicit).strip():
            raise ValueError("Set agent.cli_path to the executable DSH CLI")
        candidate = Path(explicit).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
        raise ValueError("Set agent.cli_path to the executable DSH CLI")

    for env_key in ("FLASHRSI_DSH_EXE", "METACORSI_DSH_EXE"):
        configured = os.environ.get(env_key)
        if configured:
            candidate = Path(configured).expanduser()
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate.resolve())
            if required:
                raise ValueError(f"{env_key} does not point to an executable DSH CLI")

    discovered = shutil.which("dsh")
    if discovered:
        return discovered

    candidates = [
        Path("/Applications/DeepSeek Harness.app/Contents/Resources/runtime/cli/bin/dsh"),
        Path("/usr/local/bin/dsh"),
    ]
    if os.name == "nt":
        candidates.extend(
            [
                Path("D:/Deepseek/resources/runtime/cli/bin/dsh.cmd"),
                Path(os.environ.get("LOCALAPPDATA", ""))
                / "Programs/DeepSeek Harness/resources/runtime/cli/bin/dsh.cmd",
                Path(os.environ.get("PROGRAMFILES", ""))
                / "DeepSeek Harness/resources/runtime/cli/bin/dsh.cmd",
            ]
        )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())

    if required:
        raise ValueError("Set agent.cli_path or METACORSI_DSH_EXE to the executable DSH CLI")
    return None


def _resolve_project(config, *, required: bool = False) -> Path | None:
    explicit = config.get("project_dir") or os.environ.get("METACORSI_PROJECT_DIR")
    candidates = []
    if explicit:
        base = Path(explicit).expanduser().resolve()
        candidates.extend([base, base / "metacorsi"])
    else:
        here = Path(__file__).resolve()
        for parent in here.parents:
            candidates.extend([parent / "metacorsi", parent / "metacorsi/metacorsi"])
    for candidate in candidates:
        if (candidate / "metacorsi/knowledge/evolution.py").is_file():
            return candidate.resolve()
    if explicit or required:
        raise ValueError("agent.project_dir does not contain MetaCoRSI evolution")
    return None


def _validate(config) -> None:
    for field in _FORBIDDEN_RUNTIME_KEYS:
        if config.get(field) is not None:
            raise ValueError(f"DeepSeek Harness does not accept {field}")
    llm = config.get("llm")
    if not isinstance(llm, dict) and not hasattr(llm, "get"):
        raise ValueError("DeepSeek Harness requires an llm mapping")
    model = llm.get("model_name")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("DeepSeek Harness requires llm.model_name")
    if not (
        llm.get("api_key")
        or os.environ.get("METACORSI_DSH_API_KEY")
        or os.environ.get("FLASHRSI_DSH_API_KEY")
    ):
        raise ValueError("DeepSeek Harness requires llm.api_key or METACORSI_DSH_API_KEY")
    if config.get("cli_path") is not None:
        _resolve_dsh_cli(config, required=True)
    for field, default, minimum in (
        ("max_turns", 24, 1),
        ("max_extension", 6, 0),
        ("timeout", 5400, 1),
    ):
        value = config.get(field, default)
        if type(value) is not int or value < minimum:
            raise ValueError(f"agent.{field} must be an integer >= {minimum}")
    scope = config.get("evaluation_scope", "heldout")
    if scope not in {"heldout", "historical"}:
        raise ValueError("agent.evaluation_scope must be heldout or historical")


def _load_snapshot_config(files_dir: Path) -> dict:
    cfg = json.loads(json.dumps(_DEFAULT_CONFIG))
    config_file = files_dir / "config.yaml"
    if config_file.is_file():
        loaded = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError("config.yaml must contain a YAML mapping")
        memory_cfg = loaded.get("memory")
        for key, value in loaded.items():
            if key == "memory" and isinstance(value, dict):
                cfg["memory"].update(value)
            else:
                cfg[key] = value
        if memory_cfg is not None and not isinstance(memory_cfg, dict):
            raise ValueError("config.yaml memory must be a mapping")
    return cfg


def _render_snapshot_resources(files_dir: Path) -> tuple[str, str, list[str]]:
    system_path = files_dir / "prompts/system.md"
    system_prompt = (
        system_path.read_text(encoding="utf-8").strip()
        if system_path.is_file()
        else ""
    )
    sections = []
    agents_md = files_dir / "AGENTS.md"
    if agents_md.is_file():
        text = agents_md.read_text(encoding="utf-8").strip()
        if text:
            sections.append(text)
    prompts_dir = files_dir / "prompts"
    if prompts_dir.is_dir():
        for path in sorted(prompts_dir.rglob("*.md")):
            if path == system_path or not path.is_file():
                continue
            text = path.read_text(encoding="utf-8").strip()
            if text:
                sections.append(text)
    skills_dir = files_dir / "skills"
    skill_dirs = [str(skills_dir.resolve())] if skills_dir.is_dir() else []
    if skills_dir.is_dir():
        skill_blocks = []
        for path in sorted(skills_dir.rglob("*.md")):
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8").strip()
            if text:
                rel = path.relative_to(files_dir).as_posix()
                skill_blocks.append(f"### Skill ({rel})\n{text}")
        if skill_blocks:
            sections.append("## Active Skills\n\n" + "\n\n".join(skill_blocks))
    return system_prompt, "\n\n".join(sections), skill_dirs


def _model_api(llm: dict) -> str:
    raw = llm.get("model_api") or "chat_completions"
    mapping = {
        "chat_completions": "openai-completions",
        "openai-completions": "openai-completions",
        "openai-responses": "openai-responses",
        "responses": "openai-responses",
        "anthropic-messages": "anthropic-messages",
        "anthropic": "anthropic-messages",
    }
    if raw not in mapping:
        raise ValueError(f"Unsupported DeepSeek Harness llm.model_api: {raw}")
    return mapping[raw]


def _write_cordis_patch(
    path: Path,
    *,
    model: str,
    max_tokens: int,
    api: str = "openai-completions",
    reasoning_effort: str | None = None,
    persona_prefix: str = "",
    persona_suffix: str = "",
    skill_dirs: list[str] | None = None,
    disable_host_tools: bool = True,
) -> Path:
    model_name = model.removeprefix("openai/")
    default_model_cfg: dict = {
        "provider": "metacorsi-deepseek",
        "model": model_name,
    }
    if reasoning_effort:
        default_model_cfg["reasoningEffort"] = reasoning_effort

    system_cfg: dict = {}
    if persona_prefix:
        system_cfg["personaPrefix"] = persona_prefix
    if persona_suffix:
        system_cfg["personaSuffix"] = persona_suffix

    lines = [
        "- id: llm-pi-ai",
        "  config:",
        "    providers:",
        "      metacorsi-deepseek:",
        "        apiKeyEnv: METACORSI_DSH_API_KEY",
        f"        api: {json.dumps(api)}",
        "        baseURL: !!js process.env.METACORSI_DSH_BASE_URL",
        f"        maxTokens: {int(max_tokens)}",
        "        compat:",
        "          supportsDeveloperRole: false",
        "          maxTokensField: max_tokens",
        "        models:",
        f"          - id: {json.dumps(model_name)}",
        "            input: [text]",
        f"            maxTokens: {int(max_tokens)}",
        "            compat:",
        "              thinkingFormat: deepseek",
        "- id: agent-default-model",
        "  config:",
        f"    provider: {json.dumps(default_model_cfg['provider'])}",
        f"    model: {json.dumps(default_model_cfg['model'])}",
    ]
    if reasoning_effort:
        lines.append(f"    reasoningEffort: {json.dumps(reasoning_effort)}")
    if system_cfg:
        lines.extend(["- id: system-prompt", "  config:"])
        for key, val in system_cfg.items():
            lines.append(f"    {key}: {json.dumps(val, ensure_ascii=False)}")
    if skill_dirs is not None:
        lines.extend(
            [
                "- id: skill-filesystem",
                "  config:",
                "    includeDefaultRoots: false",
                "    watch: false",
                f"    customSkillDirs: {json.dumps(skill_dirs, ensure_ascii=False)}",
            ]
        )
    if disable_host_tools:
        for tool_id in (
            "tool-pwsh",
            "tool-bash",
            "tool-jobs",
            "tool-fs",
            "tool-fs-search",
            "tool-skill",
            "tool-web",
            "tool-plugin-manager",
        ):
            lines.extend([f"- id: {tool_id}", "  disabled: true"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _parse_dsh_events(
    stdout: str, *, call_prefix: str = "dsh"
) -> tuple[str, str | None, list[dict], list[dict]]:
    events: list[dict] = []
    model_calls: list[dict] = []
    session_id: str | None = None
    final_text = ""
    last_text = ""
    for raw in (stdout or "").splitlines():
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        events.append(event)
        kind = event.get("type")
        if kind == "session" and event.get("sessionId"):
            session_id = str(event["sessionId"])
        elif kind == "text" and event.get("text"):
            last_text = str(event["text"])
        elif kind == "final":
            if event.get("sessionId"):
                session_id = str(event["sessionId"])
            if event.get("text") is not None:
                final_text = str(event.get("text") or "")
        elif kind == "status" and event.get("phase") == "step_end":
            usage = event.get("usage")
            if isinstance(usage, dict):
                inp = int(usage.get("inputTokens", usage.get("input_tokens", 0)) or 0)
                out = int(usage.get("outputTokens", usage.get("output_tokens", 0)) or 0)
                cache_read = int(
                    usage.get(
                        "cacheReadTokens",
                        usage.get(
                            "cacheReadInputTokens",
                            usage.get("cache_read_input_tokens", 0),
                        ),
                    )
                    or 0
                )
                cache_write = int(
                    usage.get(
                        "cacheWriteTokens",
                        usage.get(
                            "cacheCreationInputTokens",
                            usage.get("cache_creation_input_tokens", 0),
                        ),
                    )
                    or 0
                )
                total = int(
                    usage.get(
                        "totalTokens",
                        usage.get(
                            "total_tokens", inp + out + cache_read + cache_write
                        ),
                    )
                    or (inp + out + cache_read + cache_write)
                )
                turn_no = event.get("turn", len(model_calls) + 1)
                step_no = event.get("step", len(model_calls) + 1)
                call_id = f"{session_id or call_prefix}:t{turn_no}:s{step_no}:{len(model_calls) + 1}"
                model_calls.append(
                    {
                        "call_id": call_id,
                        "usage": {
                            "input_tokens": inp,
                            "output_tokens": out,
                            "cache_read_input_tokens": cache_read,
                            "cache_creation_input_tokens": cache_write,
                            "total_tokens": total,
                        },
                    }
                )
    return final_text or last_text, session_id, events, model_calls


def _summarize_usage(
    model_calls: list[dict],
    *,
    turns: int,
    tool_calls: int = 0,
    session_id: str | None = None,
) -> dict:
    inp = sum(c["usage"].get("input_tokens", 0) for c in model_calls)
    out = sum(c["usage"].get("output_tokens", 0) for c in model_calls)
    cache_read = sum(c["usage"].get("cache_read_input_tokens", 0) for c in model_calls)
    cache_write = sum(
        c["usage"].get("cache_creation_input_tokens", 0) for c in model_calls
    )
    total = sum(
        c["usage"].get(
            "total_tokens",
            c["usage"].get("input_tokens", 0)
            + c["usage"].get("output_tokens", 0)
            + c["usage"].get("cache_read_input_tokens", 0)
            + c["usage"].get("cache_creation_input_tokens", 0),
        )
        for c in model_calls
    )
    summary = {
        "model_calls": model_calls,
        "turns": turns,
        "tool_calls": tool_calls,
        "input_tokens": inp,
        "output_tokens": out,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_write,
        "total_tokens": total,
    }
    if session_id:
        summary["session_id"] = session_id
    return summary


def _provider_only_env(source: dict[str, str], api_key: str, base_url: str) -> dict[str, str]:
    child = source.copy()
    for key in (
        "DSH_TOKEN",
        "DSH_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_API_BASE",
        "OPENAI_API_BASE_URL",
    ):
        child.pop(key, None)
    if not api_key or not base_url:
        raise ValueError("DeepSeek Harness requires both api_key and base_url")
    child["METACORSI_DSH_API_KEY"] = api_key
    child["METACORSI_DSH_BASE_URL"] = base_url
    child["DSH_TELEMETRY_MODE"] = "DISABLED"
    child["METACORSI_DSH_PROVIDER_ONLY"] = "1"
    return child


class DSHMemoryRuntime:
    """Host-side MetaCoRSI CLS memory and Deep RCA runtime for DeepSeek Harness."""

    def __init__(self, target, task, memory: Path, grounding: dict, evidence: str):
        self.target = target
        self.base = {
            "memory": str(memory),
            "question": task.question,
            "task_id": task.task_id,
        }
        self.grounding = grounding
        self.evidence = evidence
        self.rca: list[dict] = []
        self.focused_success = False
        self.injection_bytes = len(self.prompt_context().encode("utf-8"))

    def prompt_context(self) -> str:
        return "\n\n".join(
            filter(
                None,
                (
                    self.grounding.get("k7_context"),
                    (self.grounding.get("prompt") or "").strip(),
                    "\n".join(self.grounding.get("processor_policy", ())),
                ),
            )
        )

    async def analyze_failure(self, step: int, command: str, code: int, observation: str) -> dict:
        diagnosis = await self.target.native(
            "analyze",
            {
                **self.base,
                "command": command,
                "observation": observation,
            },
        )
        self.rca.append({"step": step, "exit_code": code, "diagnosis": diagnosis})
        return diagnosis

    def receipt(self) -> dict:
        return {
            "type": "MetaCoRSIMemory",
            "strategy": "cls_dual_store",
            "policy": "modular_reverse_v23",
            "version": self.grounding["version"],
            "groundings": self.grounding["groundings"],
            "initial_evidence": self.evidence,
            "initial_retrieval": self.grounding["prompt"],
            "working_memory": self.grounding.get("working_memory"),
            "repository_probe_count": 1,
            "injection_bytes_per_call": self.injection_bytes,
            "rca": self.rca,
            "focused_test_seen": self.focused_success,
        }


class DeepSeekHarness(Agent):
    """DeepSeek Harness (dsh) adapter supporting general FlashRSI algorithms and MetaCoRSI."""

    name = "deepseek_harness"
    description = (
        "DeepSeek Harness (dsh) with versioned prompts, skills, config, and MetaCoRSI memory."
    )
    _container_dependencies = ("PyYAML",)
    instruction_file = "AGENTS.md"
    native = MetaCoRSITarget.native

    def __init__(self, config=None):
        config = dict(config or {})
        for field in _FORBIDDEN_RUNTIME_KEYS:
            if config.get(field) is not None:
                raise ValueError(f"DeepSeek Harness does not accept {field}")
        for field, default, minimum in (
            ("max_turns", 24, 1),
            ("max_extension", 6, 0),
            ("timeout", 5400, 1),
        ):
            value = config.get(field, default)
            if type(value) is not int or value < minimum:
                raise ValueError(f"agent.{field} must be an integer >= {minimum}")
        scope = config.get("evaluation_scope", "heldout")
        if scope not in {"heldout", "historical"}:
            raise ValueError("agent.evaluation_scope must be heldout or historical")

        seed = config.get("knowledge_dir")
        need_project = bool(
            seed or config.get("project_dir") or config.get("enable_memory")
        )
        self.project = _resolve_project(config, required=need_project)
        python = config.get("python") or os.environ.get("METACORSI_PYTHON")
        self.python = (
            Path(python).expanduser().absolute() if python else Path(sys.executable)
        )
        if not self.python.is_file():
            raise ValueError(
                "agent.python must point to a valid Python executable"
            )
        self.dsh = _resolve_dsh_cli(config, required=bool(config.get("cli_path")))
        harbor = {**config.get("harbor", {}), "hosted_dsh": True}
        cfg = {**config, "harbor": harbor}

        if seed and "source_dir" not in config:
            seed_path = Path(seed).expanduser().resolve()
            package_dir = Path(__file__).resolve().parent
            with TemporaryDirectory(prefix="flashrsi-dsh-seed-") as temporary:
                path = Path(temporary)
                shutil.copytree(package_dir, path, dirs_exist_ok=True)
                for py_file in path.glob("*.py"):
                    py_file.unlink()
                shutil.rmtree(path / "__pycache__", ignore_errors=True)
                shutil.rmtree(path / "memory", ignore_errors=True)
                copy_active_memory(seed_path, path / "memory")
                if self.project is not None:
                    env = {
                        key: value
                        for key, value in os.environ.items()
                        if not _credential_key(key) and not key.startswith("METACORSI_")
                    }
                    env.update(
                        PYTHONPATH=os.pathsep.join(
                            str(p)
                            for p in (self.project, self.project / "vendor/rca-repair")
                        ),
                        PYTHONDONTWRITEBYTECODE="1",
                        PYTHONUTF8="1",
                    )
                    result = subprocess.run(
                        [str(self.python), str(_NATIVE), "normalize_seed"],
                        input=json.dumps({"memory": str(path / "memory")}),
                        text=True,
                        capture_output=True,
                        env=env,
                        timeout=30,
                        check=False,
                    )
                    if result.returncode:
                        detail = result.stderr.strip().splitlines()
                        raise ValueError(
                            "Cannot initialize DeepSeek Harness memory"
                            + (f": {detail[-1]}" if detail else "")
                        )
                cfg_path = path / "config.yaml"
                loaded_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
                loaded_cfg.setdefault("memory", {})["enabled"] = True
                cfg_path.write_text(
                    yaml.safe_dump(loaded_cfg, sort_keys=False), encoding="utf-8"
                )
                super().__init__({**cfg, "source_dir": str(path)})
            del self.config["source_dir"]
        else:
            super().__init__(cfg)

    def _validate_compatible(self, candidate) -> None:
        if candidate.agent_type not in {"deepseek_harness", "dsh"}:
            raise ValueError("Snapshot belongs to a different Agent type")

    def memory_enabled(self, directory: Path | None = None) -> bool:
        if self.config.get("enable_memory") or self.config.get("knowledge_dir"):
            return True
        if directory is not None:
            try:
                snap_cfg = _load_snapshot_config(directory)
                return bool((snap_cfg.get("memory") or {}).get("enabled", False))
            except Exception:
                return False
        return False

    async def check_native(self, *, snapshot=None, forbidden_tasks=()):
        if self.project is None:
            self.project = _resolve_project(self.config, required=True)
        with (snapshot or self.snapshot)._export(full=True) as directory:
            inspected = await self.native(
                "inspect", {"memory": str(directory / "memory")}
            )
        if self.config.get("evaluation_scope", "heldout") == "heldout":
            overlap = set(inspected["tasks"]).intersection(
                task.task_id.removeprefix("datacurve/") for task in forbidden_tasks
            )
            if overlap:
                raise ValueError(
                    f"Held-out tasks appear in memory: {', '.join(sorted(overlap))}"
                )
        return inspected

    async def memory_runtime(self, task, directory: Path, evidence: str):
        if not self.memory_enabled(directory):
            return None
        if self.project is None:
            self.project = _resolve_project(self.config, required=True)
        memory = directory / "memory"
        if not (memory / "current.json").is_file():
            return None
        grounding = await self.native(
            "ground",
            {
                "memory": str(memory),
                "task_id": task.task_id,
                "question": task.question,
                "evidence": evidence,
            },
        )
        return DSHMemoryRuntime(self, task, memory, grounding, evidence)

    async def _run(self, prompt: str, working_dir: Path | None = None) -> Trajectory:
        config = self.config
        llm = config.get("llm") or {}
        model = llm.get("model_name")
        api_key = (
            llm.get("api_key")
            or os.environ.get("METACORSI_DSH_API_KEY")
            or os.environ.get("FLASHRSI_DSH_API_KEY")
        )
        base_url = llm.get("base_url") or "https://api.deepseek.com/v1"
        secrets = sorted(
            set(_secrets(os.environ) + _secrets(config)), key=len, reverse=True
        )
        try:
            if not isinstance(model, str) or not model.strip():
                raise ValueError("DeepSeek Harness requires llm.model_name")
            if not api_key:
                raise ValueError("DeepSeek Harness requires llm.api_key")
            dsh_exe = _resolve_dsh_cli(config, required=True)
            snap_cfg = _load_snapshot_config(self.files_dir)
            system_prompt, extra_guidance, skill_dirs = _render_snapshot_resources(
                self.files_dir
            )
            if config.get("_worker_system_prompt"):
                extra_guidance = "\n\n".join(
                    filter(None, (extra_guidance, config["_worker_system_prompt"]))
                )
            api = _model_api(llm)
        except Exception as exc:
            return Trajectory(
                status="error",
                error_message=_redact(str(exc), secrets),
            )

        max_tokens = int(llm.get("max_tokens") or snap_cfg.get("max_tokens", 4096))
        timeout = config.get("timeout", snap_cfg.get("call_timeout", 360))
        reasoning = llm.get("reasoning_effort") or llm.get("effort")
        profile = snap_cfg.get("profile", "headless")

        with TemporaryDirectory(prefix="flashrsi-dsh-run-") as temporary:
            runtime_root = Path(temporary).resolve()
            dsh_home = runtime_root / "dsh-home"
            dsh_home.mkdir(parents=True, exist_ok=True)
            task_dir = working_dir or (runtime_root / "work")
            task_dir.mkdir(parents=True, exist_ok=True)
            patch_path = _write_cordis_patch(
                runtime_root / "runtime.patch.yml",
                model=model,
                max_tokens=max_tokens,
                api=api,
                reasoning_effort=reasoning,
                persona_prefix=system_prompt,
                persona_suffix=extra_guidance,
                skill_dirs=skill_dirs,
                disable_host_tools=working_dir is None,
            )
            child_env = _provider_only_env(dict(os.environ), api_key, base_url)
            child_env["DSH_HOME"] = str(dsh_home)
            args = [dsh_exe, "--profile", str(profile), "--patch", str(patch_path), "--json"]
            if working_dir is not None:
                args.extend(
                    [
                        "--permission-mode",
                        "danger-full-access",
                        "--approval-policy",
                        "never",
                    ]
                )
            args.append("-")

            process = await asyncio.create_subprocess_exec(
                *args,
                cwd=str(task_dir),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=child_env,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(prompt.encode("utf-8")),
                    timeout=timeout,
                )
            except (asyncio.CancelledError, TimeoutError):
                if process.returncode is None:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.communicate(), 5)
                    except TimeoutError:
                        process.kill()
                        await process.communicate()
                if isinstance(sys.exc_info()[1], asyncio.CancelledError):
                    raise
                return Trajectory(
                    status="error",
                    error_message=f"DeepSeek Harness timed out after {timeout:g}s",
                )

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            final_text, session_id, events, model_calls = _parse_dsh_events(stdout)
            messages = [
                {"type": "RunInput", "prompt": prompt, "snapshot_id": self.snapshot.id},
                *events,
            ]
            usage = _summarize_usage(
                model_calls,
                turns=max(1, len(model_calls)),
                session_id=session_id,
            )
            if process.returncode and not final_text:
                detail = (stderr.strip() or stdout.strip())[-2000:]
                return Trajectory(
                    status="error",
                    messages=_redact(tuple(messages), secrets),
                    usage=_redact(usage, secrets),
                    error_message=_redact(
                        f"DeepSeek Harness exited with {process.returncode}: {detail}",
                        secrets,
                    ),
                )
            return Trajectory(
                status="finished",
                output=_redact(final_text, secrets),
                messages=_redact(tuple(messages), secrets),
                usage=_redact(usage, secrets),
            )


class DSH(DeepSeekHarness):
    """Short alias `dsh` for DeepSeekHarness."""

    name = "dsh"
    description = (
        "DeepSeek Harness (dsh) with versioned prompts, skills, config, and MetaCoRSI memory."
    )


__all__ = [
    "DSH",
    "DSHMemoryRuntime",
    "DeepSeekHarness",
    "_load_snapshot_config",
    "_model_api",
    "_parse_dsh_events",
    "_provider_only_env",
    "_render_snapshot_resources",
    "_resolve_dsh_cli",
    "_resolve_project",
    "_summarize_usage",
    "_validate",
    "_write_cordis_patch",
]
