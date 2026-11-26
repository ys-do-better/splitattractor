# DeepSeek Harness Agent Instructions

Follow a disciplined diagnose-materialize-verify-close workflow:

1. **Diagnose**: Localize the concrete failure or requirement and identify the smallest causal decision point before editing.
2. **Materialize**: Implement a minimal vertical repair or feature slice and wire its public entry points.
3. **Verify**: Run focused tests covering the new or repaired behavior as well as boundary and regression cases.
4. **Close**: Ensure the worktree is clean or committed as required by the task before finishing.
