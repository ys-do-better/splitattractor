# DeepSWE deep RCA integration

Upstream: `maoyixie/rca-repair@9d4df4a1dcde02ad5f7bdba8f0e697cfc9df4410`

## Runtime evidence path

1. The Target Agent observes a strong failure or a failing test command.
2. It captures the exact command output plus `/app` Git status, changed files,
   zero-context diff hunks, failed test names, and trace locations.
3. `DeepSWEAgentRCA` maps this evidence into the upstream five-step
   construction walk and records an evidence-quality score.
4. The as-is replay is the actual observed failure. The root-right replay stays
   pending until the same test command passes after the source digest changes.
5. Harbor's independent verifier closes the ledger after the trial. A full
   F2P/P2P pass produces `verifier_counterfactual_validation`; a failure rejects
   the runtime hypotheses or records that no runtime RCA was triggered.

No regression or candidate-promotion gate is enabled in `rca_only` mode.

## Two-sentinel result, 2026-11-05

Job: `wsl_deep_rca_fast2_v3_20261105`

| Task | F2P | P2P | Reward | RCA observation |
| --- | ---: | ---: | ---: | --- |
| `abs-stepped-slices` | 6/6 | 6/6 | 1 | Eight runtime diagnoses; concrete trace/diff localization; independent verifier closed walk-back as true. |
| `cattrs-partial-structuring-recovery` | 0/69 | 7/7 | 0 | No strong runtime failure was observed, so RCA did not activate; the agent exhausted 80 turns while continuing implementation review. |

Result: **1/2 (50%)**, with no infrastructure exceptions.

## Interpretation

ABS demonstrates the intended loop: real failing test -> evidence-backed RCA ->
localized repair -> independent hidden verifier confirmation. Cattrs demonstrates
the remaining trigger problem: an RCA system cannot diagnose evidence the target
agent never produces. The RCA-only processor policy now requires a minimal task
contract reproducer after the first source mutation and the exact same command
after repair, so future trials create the evidence needed for causal walk-back.
