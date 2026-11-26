# Repository Diagnosis and Repair Skill

- Start by checking repository status and locating the exact module, symbol, and test target referenced by the task.
- Reproduce or probe the target behavior with a focused test command before making broad edits.
- Prefer compact, deterministic edits (`python3 -c`, `git apply`, or direct file edits) and verify syntax immediately.
- After passing the focused test, check edge cases and run relevant existing tests to prevent regressions.
- Commit your final changes to git so the task verifier can collect the patch from `HEAD`.

<!-- SLOW_UPDATE_START -->
Preserve existing public interfaces and passing behavior while fixing the root cause at its earliest causal decision point.
<!-- SLOW_UPDATE_END -->
