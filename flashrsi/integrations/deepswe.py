"""Use Harbor's native collect/verifier phases with model-only network access."""

import os
from urllib.parse import urlsplit

from flashrsi.integrations.harbor import _run
from flashrsi.integrations.harbor import bind as bind_harbor


def _model_hosts(config):
    llm = config["llm"]
    if not llm.get("base_url"):
        raise ValueError(
            "DeepSWE requires agent.llm.base_url for its model-only allowlist"
        )
    hosts = []
    for field in ("base_url", "side_base_url"):
        value = llm.get(field)
        if value:
            url = urlsplit(value)
            if (
                url.scheme not in ("http", "https")
                or not url.hostname
                or url.username
                or url.password
            ):
                raise ValueError(f"agent.llm.{field} must be an HTTP(S) model endpoint")
            hosts.append(url.hostname)
    return list(dict.fromkeys(hosts))


def bind(agent, benchmark):
    from flashrsi.agents.metacorsi import MetaCoRSITarget

    metacorsi = isinstance(agent, MetaCoRSITarget)
    if metacorsi:
        if not (agent.config["llm"].get("api_key") or os.environ.get("METACORSI_DSH_API_KEY")):
            raise ValueError("Set agent.llm.api_key or METACORSI_DSH_API_KEY for MetaCoRSI")
        if agent.config["llm"].get("side_base_url"):
            raise ValueError("MetaCoRSI accepts a single DSH provider endpoint")
    else:
        bind_harbor(agent)
    _model_hosts(agent.config)

    async def run(target, task):
        if benchmark._tasks_by_id.get(task.task_id) != task:
            raise ValueError("DeepSWE requires a loaded benchmark Task")
        # config may change between rollouts; derive access from this run's
        # model endpoints, never from a benchmark-wide or Worker credential.
        hosts = _model_hosts(target.config)
        if metacorsi:
            return await target.run_task(task)
        return await _run(target, task, agent_allowed_hosts=hosts)

    agent._env_runner = run
