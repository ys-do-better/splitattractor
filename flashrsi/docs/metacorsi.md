# MetaCoRSI on DeepSWE

Use `agent.name: metacorsi` for the local DeepSeek Harness (DSH) and
`algorithm.name: metacorsi` for verifier-guided knowledge evolution. FlashRSI
owns task splitting, Snapshot selection, test evaluation and experiment records.
The sibling MetaCoRSI project supplies DSH execution, RCA, grounding and knowledge
distillation. Its native Harbor runs in a separate process.

Install FlashRSI with the `harbor` extra (Harbor 0.23.0) on Python 3.12
or newer. Set `agent.python` or `METACORSI_PYTHON` to a separate Python
environment with MetaCoRSI's Harbor 0.24.0. Set `agent.cli_path` or
`METACORSI_DSH_EXE` if DSH is outside PATH and its standard macOS install.
A running Docker Linux daemon and Compose are required. DSH calls the specified
model provider from the host. The agent and verifier containers retain the native
DeepSWE network restrictions.

The [example configuration](../examples/quickstarts/deepswe_metacorsi.yaml)
uses the sibling MetaCoRSI source in this monorepo and the pinned DeepSWE
revision. Set `--agent.project_dir` for a separate checkout and
`--benchmark.data_path` for local native tasks.
Export `METACORSI_DSH_API_KEY` or configure `agent.llm.api_key` locally.
The native evolution routine uses verifier evidence and requires
`algorithm.worker.type: none`.

```bash
cd flashrsi
export METACORSI_PYTHON=/path/to/metacorsi/.venv/bin/python
flashrsi run --config examples/quickstarts/deepswe_metacorsi.yaml \
  --benchmark.data_path /path/to/deep-swe/tasks --check-only
flashrsi run --config examples/quickstarts/deepswe_metacorsi.yaml \
  --benchmark.data_path /path/to/deep-swe/tasks
flashrsi evaluate --config examples/quickstarts/deepswe_metacorsi.yaml \
  --benchmark.data_path /path/to/deep-swe/tasks \
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

The configuration selects one task per role for a development smoke. The
local dev5 panel overlaps historical K29 experience. Use test tasks independent
of the seed's history. FlashRSI reports one native verifier reward per task
and does not reproduce the historical K29 Pass@5 schedule.

## Common MiniAgent comparison runtime

Use [the MiniAgent configuration](../examples/quickstarts/deepswe_metacorsi_miniagent.yaml)
for common-harness experiments and [the pilot launcher](../scripts/run_metacorsi_miniagent.py)
to prepare B0 through B3. The committed configuration points to the native source
in the sibling `metacorsi/` directory of this monorepo. Hosted MiniAgent uses the same Harbor
DeepSWE task and independent verifier for both targets. The MiniAgent process
and credential stay on the host, shells run in Docker, and the task's native
Git submission requirement applies to both targets.

B0 uses MiniAgent alone. B1 adds MetaCoRSI with the five native K7 priors and
no historical task episodes. B2 freezes an explicit historical seed. B3 starts
from K0 and updates only from its training split. The historical dev5 fixture
is for development. Use disjoint, audited task data for a generalization claim.

```bash
.venv/bin/python scripts/run_metacorsi_miniagent.py \
  --config examples/quickstarts/deepswe_metacorsi_miniagent.yaml \
  --arm B1 --data /path/to/deep-swe/tasks/abs-stepped-slices \
  --check-only
```

Real runs require `FLASHRSI_AGENT_API_KEY` or the local
`METACORSI_DSH_API_KEY` environment variable. Review
[benchmark evidence](../reports/02_benchmark_smoke_test.md),
[the runtime integration](../reports/04_runtime_integration.md),
and [evolution validation](../reports/05_evolution_validation.md) for the
scope of the completed checks.

For paired B3 tests, pass `--use-config-test-split` for the frozen arms and
keep the same data path and split seed. An audited warm seed can be used for
B2 and B3 with `--heldout-warm-seed`. Historical K29 cannot bypass that guard
for test tasks already found in its memory.
