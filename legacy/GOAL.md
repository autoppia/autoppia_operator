# Goal

Make `autoppia_operator` work again on the `daryxx` branch against the refactored `autoppia_iwa` `daryxx` branch.

Success means all of the following are true:

1. The miner/operator contract still works on `daryxx`.
   `main.py` exposes `app`, `/health` is healthy, `/act` returns valid actions, and `python check.py` passes.
2. The operator code and local evaluator are compatible with the current `autoppia_iwa` `daryxx` layout.
   Broken imports, moved modules, renamed interfaces, and stale paths must be fixed rather than worked around with fake stubs.
3. Local evaluation works again.
   `eval.py` must run from this repo against the current sibling `autoppia_iwa` checkout without relying on the old pre-refactor structure.
4. There is measured evidence that the migrated operator can solve at least one real task.
   Store that evidence in `data/eval_migration_smoke.json`.

## Manager Policy

The Manager may update `.arbos/SPEC.md`, `.arbos/tests/`, `.arbos/PROMPT.md`, and `.arbos/SUPERVISOR_PROMPT.md` if they are stale or miscalibrated.

The Manager must not dilute the goal into "clean up docs", "improve style", or training/RL work that is unrelated to restoring operator + eval compatibility on `daryxx`.
