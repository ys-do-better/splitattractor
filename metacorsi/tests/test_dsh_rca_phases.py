import json

from metacorsi_dsh_agent import (_active_candidate, _candidate_for_session,
                                 _candidate_staging_commands,
                                 _json_object, _phase_directive, _provider_only_env,
                                 _protocol_fallback_plan, _successful_test_seen)


def test_phase_directive_converges_from_diagnosis_to_closure():
    assert "DIAGNOSE" in _phase_directive(1, 24)
    assert "MATERIALIZE" in _phase_directive(8, 24)
    assert "VERIFY" in _phase_directive(16, 24)
    close = _phase_directive(23, 24)
    assert "CLOSE" in close
    assert "Stop broad exploration" in close
    assert "remaining=2" in close


def test_candidate_lookup_handles_harbor_truncated_trial_name():
    registry = {"tasks": {
        "bandit-structured-nosec-directives": {"patch": "bandit"},
        "clack-async-autocomplete-options": {"patch": "clack"},
    }}
    selected = _candidate_for_session(
        registry, "bandit-structured-nosec-directiv__5yjtTJd__agent"
    )
    assert selected == {"patch": "bandit"}


def test_candidate_portfolio_selects_active_patch():
    entry = {"active_patch_digest": "b", "candidates": [
        {"patch_digest": "a", "patch": "old"},
        {"patch_digest": "b", "patch": "best"},
    ]}
    assert _active_candidate(entry)["patch"] == "best"


def test_large_candidate_is_staged_in_windows_safe_chunks():
    payload = "A" * 100_000
    setup, chunks, apply = _candidate_staging_commands(payload)
    assert len(chunks) > 1
    assert max(map(len, (setup, *chunks, apply))) < 9000
    reconstructed = "".join(command.split("'")[3] for command in chunks)
    assert reconstructed == payload


def test_json_object_accepts_dsh_parameter_closing_suffix():
    text = '{"analysis":"inspect","commands":["git status"],"done":false}</｜DSML｜ parameter>'
    assert _json_object(text)["commands"] == ["git status"]


def test_protocol_fallback_only_finishes_in_close_phase():
    verify = json.loads(_protocol_fallback_plan("METACORSI RCA PHASE: VERIFY"))
    close = json.loads(_protocol_fallback_plan("METACORSI RCA PHASE: CLOSE"))
    assert verify["done"] is False
    assert close["done"] is True
    assert all("git" in command for command in close["commands"])


def test_successful_focused_test_is_progress_evidence():
    history = [{"commands": ["cd /app && pytest tests/test_api.py"], "return_codes": [0]}]
    assert _successful_test_seen(history)
    assert "VERIFY" in _phase_directive(10, 24, history)


def test_provider_only_env_removes_quota_fallback_credentials():
    env = _provider_only_env({
        "METACORSI_DSH_API_KEY": "owner-key",
        "METACORSI_DSH_BASE_URL": "https://provider.example/v1",
        "DSH_TOKEN": "dsh-quota-token",
        "OPENAI_API_KEY": "fallback-key",
        "ANTHROPIC_API_KEY": "fallback-key",
    })
    assert env["METACORSI_DSH_API_KEY"] == "owner-key"
    assert env["METACORSI_DSH_PROVIDER_ONLY"] == "1"
    assert env["DSH_TELEMETRY_MODE"] == "DISABLED"
    assert "DSH_TOKEN" not in env
    assert "OPENAI_API_KEY" not in env
    assert "ANTHROPIC_API_KEY" not in env
