# MetaCoRSI on DeepSWE

Use `agent.name: metacorsi` for the local DSH harness and
`algorithm.name: metacorsi` for verifier-guided knowledge evolution. FlashRSI
owns task splitting, Snapshot selection, test evaluation and experiment records.
The sibling MetaCoRSI project supplies DSH execution, RCA, grounding and knowledge
distillation. Its native Harbor runs in a separate process.

Install FlashRSI with the `harbor` extra (Harbor 0.23.0) on Python 3.12
or newer. Set `agent.python` to a separate environment with MetaCoRSI's
Harbor 0.24.0 and `agent.project_dir` to its checkout. Set `agent.cli_path` or
`METACORSI_DSH_EXE` if DSH is outside PATH and its standard macOS install.
A running Docker Linux daemon and Compose are required. DSH calls the specified
model provider from the host. The agent and verifier containers retain the native
DeepSWE network restrictions.

The [example configuration](../examples/quickstarts/deepswe_metacorsi.yaml)
references the sibling workspace's MetaCoRSI source, Python environment and
local dev5 tasks. Adapt those paths and `agent.llm` to your installation.
Export `METACORSI_DSH_API_KEY` or configure `agent.llm.api_key` locally.
The native evolution routine uses verifier evidence and requires
`algorithm.worker.type: none`.

```bash
flashrsi run --config examples/quickstarts/deepswe_metacorsi.yaml --check-only
flashrsi run --config examples/quickstarts/deepswe_metacorsi.yaml
flashrsi evaluate --config examples/quickstarts/deepswe_metacorsi.yaml \
  --snapshot /path/to/run/snapshots/SNAPSHOT_ID.json
```

The initial Snapshot contains `agent.knowledge_dir` (K29 by default) with its
active LTM generation and the two DSH profiles. Put a hash-valid indexed
knowledge package in that directory. Keep the same seed and model ID when
loading a saved Snapshot. Raw job outputs and quarantined candidate patches
stay outside the Snapshot. Candidate patch replay is disabled. The source
knowledge directory is only read.

Each round runs train tasks with the selected Snapshot and distills their
independently verified episodes. The resulting candidate preserves the
seed's stable CLS backbone. It is selected when its reward on every dev task
matches or exceeds the incumbent's and the dev trials finish successfully.
The runner evaluates the selected Snapshot on test after all rounds.
Native logs, verifier results and patch artifacts are in the run's
`artifacts/harbor/` directory. Use `flashrsi evaluate` for a frozen run.

The bundled panel is a smoke configuration: the tasks overlap historical
K29 experience. For a generalization experiment, use test tasks independent
of the seed's history. FlashRSI reports one native verifier reward per task
and does not reproduce the historical K29 Pass@5 schedule.
