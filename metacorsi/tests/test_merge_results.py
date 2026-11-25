import json

from metacorsi.harbor_agents.merge_results import merge
from metacorsi.finalize import _task_outcome


def _trial(job, name, reward, model="DeepSeek-V4.1-Flash-1017"):
    trial = job / f"{name}__trial"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(json.dumps({
        "task_name": f"datacurve/{name}",
        "config": {"agent": {"model_name": f"openai/{model}"}},
        "verifier_result": {"rewards": {} if reward is None else {
            "reward": reward, "f2p_total": 99, "p2p_total": 999,
        }},
        "exception_info": "infra" if reward is None else None,
    }), encoding="utf-8")
    return trial


def test_merge_excludes_infra_and_accepts_retry(tmp_path):
    tasks = tmp_path / "tasks.txt"
    tasks.write_text("one\ntwo\n", encoding="utf-8")
    main = tmp_path / "main"
    retry = tmp_path / "retry"
    _trial(main, "one", 1)
    _trial(main, "two", None)
    _trial(retry, "two", 0)

    result = merge(tasks, [main, retry], "dsh", "DeepSeek-V4.1-Flash-1017")

    assert result["complete"] is True
    assert result["valid_results"] == 2
    assert result["passes"] == 1
    assert len(result["excluded_infrastructure_attempts"]) == 1


def test_failure_classification_keeps_contract_and_regression_separate():
    base = {"reward": 0.0, "f2p_total": 10, "p2p_total": 100}
    assert _task_outcome({**base, "f2p_passed": 0, "p2p_passed": 100}) == \
        "no_verified_feature_gain"
    assert _task_outcome({**base, "f2p_passed": 9, "p2p_passed": 100}) == \
        "partial_contract_coverage"
    assert _task_outcome({**base, "f2p_passed": 10, "p2p_passed": 99}) == \
        "regression_after_feature_gain"
