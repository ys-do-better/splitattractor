# RCA-guided self-evolution v10

## Why the first knowledge loop was ineffective

The v8 loop remembered task words and aggregate F2P/P2P counts, but it did not change the
state transition policy. RCA ran only after a shell command returned non-zero. A trajectory
could therefore spend all 24 steps exploring, leave no transferable patch, and still be
distilled as an ordinary feature failure. Repeated trials also started from a clean task
image, so a 99.7% partial candidate was discarded and rebuilt from scratch.

## Revised brain loop

1. **Perceive evidence**: task identity, patch materialization, step budget, full verifier
   counterexamples, F2P and P2P are observed together.
2. **Post-verifier RCA**: classify the earliest contradicted stage as
   `patch_materialization_failure`, `contract_integration_failure`,
   `contract_boundary_gap`, `preservation_regression`, or a mixed failure.
3. **Separate knowledge from candidate state**: reusable P/M/R/F/S atoms remain versioned
   LTM; failed-but-partial patches enter a same-task quarantine and cannot become skills.
4. **Execute with phase deadlines**: `DIAGNOSE -> MATERIALIZE -> VERIFY -> CLOSE` prevents
   late broad exploration. Milestone audits expose an absent patch or public API path.
5. **Materialize before evaluation**: agent completion or budget exhaustion creates a git
   checkpoint so Harbor evaluates and records the actual candidate.
6. **Promote only verified knowledge**: reward=1 removes the task from quarantine and may
   create positive M/R/F atoms. Failed evidence creates only RCA recovery policies and
   guards tagged `negative-evidence`.

## Evidence from the near-pass recovery probe

- Clack improved from `77/82` to `80/82` with v8, then reached **82/82 F2P and 643/643
  P2P** under the RCA-phased brain. Its old candidate was rejected by `git apply`, so this
  success came from the distilled counterexamples and execution policy rather than replaying
  the previous answer.
- Bandit remained **69/69 F2P and 281/282 P2P** even after its partial candidate was resumed.
  This isolates the remaining weakness: the atom contained the test name but originally
  omitted the verifier assertion `expected [] != actual ['B602']`. v10 now retains the full
  counterexample so the next RCA can infer the statement-wide selector-union invariant.
- CSS Tree and Dasel were reclassified as execution/materialization failures because their
  24-step trajectories produced zero-byte patches. The new runner checkpoints dirty work,
  so this evidence-loss mode is closed for future trials.

The active production snapshot remains v8. `v10-rca-closure.json` is an experimental
candidate until it is evaluated on held-out tasks; same-task recovery is adaptation evidence,
not an unbiased generalization score.
