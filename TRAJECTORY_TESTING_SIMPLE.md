# Mini Documentation: Trajectory Testing (Strict Replay)

Short reference for how we validate trajectories in `autoppia_operator`.

## Scope

We validate trajectories in `src/operator/agents/fsm/trajectory.py` by checking:

- action mapping is valid
- actions execute without runtime errors
- final benchmark score is non-zero

## Main tool

- Script: `scripts/test_trajectory_task_score.py`
- Mode: strict replay

## Assumptions

- You are in `AUTOPPIA/Autoppia_repos/autoppia_operator`
- Venv is already activated
- Demo web/backend/event stack is running

## Commands

Single use case:

```bash
python scripts/test_trajectory_task_score.py \
--web-project-id autobooks \
--use-case SEARCH_BOOK \
--expect-non-zero \
--iwa-log-level ERROR
```

One task by id:

```bash
python scripts/test_trajectory_task_score.py \
--web-project-id autobooks \
--task-id <TASK_ID> \
--expect-non-zero \
--iwa-log-level ERROR
```

All use cases in a project (batch):

```bash
python scripts/test_trajectory_task_score.py \
--web-project-id autobooks \
--expect-non-zero \
--iwa-log-level ERROR
```

Explicit batch flag (equivalent):

```bash
python scripts/test_trajectory_task_score.py \
--web-project-id autobooks \
--all-use-cases \
--expect-non-zero \
--iwa-log-level ERROR
```

## Current behavior notes

- If `--use-case` and `--task-id` are omitted, batch mode runs automatically.
- URL/seed comes from trajectory (`NavigateAction` / trajectory `url`).
- Changing seed can break criteria matching and produce `score=0.0`.
- `--task-cache` is optional; cache is auto-detected/generated if needed.

## How to read results

- `exec_ok=True` for all steps + `Final score=0.0`:
  - actions execute, but event criteria are not satisfied.
- selector timeout / action error:
  - selector or mapping issue.
- `Final score > 0`:
  - trajectory passes for that task/use case.

## Typical workflow

1. Test one use case.
2. Fix trajectory or mapping.
3. Re-run same use case.
4. Run project batch to confirm full health.
