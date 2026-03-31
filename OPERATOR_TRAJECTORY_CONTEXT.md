# Operator Trajectory Context

Last update: 2026-03-31

## Scope

This note is a quick handoff context for continuing trajectory work in `autoppia_operator`, especially for `autocinema` and `autobooks`.

## Where Trajectories Live

- Main file: `src/operator/agents/fsm/trajectory.py`
- Canonical variable: `TRAJECTORIES`
- Current structure:
  - `project_id: "p01_autocinema"`
  - `project_id: "p02_autobooks"`
- Both projects are unified in the same `TRAJECTORIES` list.

## How Selection Works

`get_trajectory_bootstrap_actions(...)` chooses one trajectory by score:

- project match (normalized alias, e.g. `autobooks` <-> `p02_autobooks`)
- exact `use_case` match
- prompt token overlap

Then `_apply_prompt_overrides(...)` adapts placeholders from the real task prompt:

- `<username>`, `<password>`, email, comments
- `__SEARCH_QUERY__`, `__RATING__`, `__AUTHOR__`, `__GENRE__`, etc.

## Test Script Used

- Script: `scripts/test_trajectory_task_score.py`
- Purpose: run one benchmark task with trajectory actions only and print step-by-step score evolution.

Key flags:

- `--web-project-id`
- `--use-case` or `--task-id`
- `--task-cache` (important: choose cache for correct project)
- `--seed`
- `--expect-non-zero`
- `--iwa-log-level INFO|ERROR`
- `--keep-navigate` (disabled by default)

## Important Pitfalls

1. Wrong `web-project-id` typo
- `autoboooks` fails.
- Correct: `autobooks`.

2. Wrong cache for project
- Default cache (`data/task_cache/tasks_cache.json`) currently contains only `autocinema`.
- For Autobooks tests, use cache like `/tmp/autobooks_tasks_cache_seed1.json`.

3. Seed behavior
- Script forces task URL seed via `_force_seed_in_url(...)`.
- By default it removes `NavigateAction` to mimic FSM bootstrap.
- If `--keep-navigate` is used, navigate URLs inside trajectory must match seed or evaluator can fail with `Seed mismatch`.

4. Dynamic constraints
- Many benchmark prompts are constraint-based (`equals`, `not_equals`, ranges).
- A trajectory may execute fine but still score `0` if chosen entity/text does not satisfy task constraints.

## Recent Fix Applied

In `trajectory.py`, search-query extraction was improved for prompts like:

- `query not_equals 'X'`

Now the adapter avoids using the forbidden value as the search text, which fixed `SEARCH_BOOK` scoring in tested seed.

## Autobooks Status (seed=1, task cache seed1)

Validated snapshot during this session:

- Passing (`1.0`):
  - `SEARCH_BOOK`
  - `ADD_TO_READING_LIST`

- Failing / needs adjustment:
  - `EDIT_BOOK`
  - `PURCHASE_BOOK`
  - `REMOVE_FROM_READING_LIST`
  - `VIEW_CART_BOOK`
  - `ADD_TO_CART_BOOK`
  - `REMOVE_FROM_CART_BOOK`

- Special case:
  - `FILTER_BOOK`, `BOOK_DETAIL` are currently navigate-centric.
  - Without `--keep-navigate`, they may show `No trajectory actions found`.
  - With `--keep-navigate`, they can fail on seed mismatch if trajectory URL seed differs.

## Recommended Command Templates

Autobooks:

```bash
.venv/bin/python scripts/test_trajectory_task_score.py \
  --task-cache /tmp/autobooks_tasks_cache_seed1.json \
  --web-project-id autobooks \
  --use-case SEARCH_BOOK \
  --seed 1 \
  --expect-non-zero \
  --iwa-log-level ERROR
```

Autocinema:

```bash
.venv/bin/python scripts/test_trajectory_task_score.py \
  --web-project-id autocinema \
  --use-case SEARCH_FILM \
  --seed 1 \
  --expect-non-zero \
  --iwa-log-level ERROR
```

Debug with full internal signals:

```bash
.venv/bin/python scripts/test_trajectory_task_score.py \
  --task-cache /tmp/autobooks_tasks_cache_seed1.json \
  --web-project-id autobooks \
  --use-case PURCHASE_BOOK \
  --seed 1 \
  --iwa-log-level INFO
```

## Next Work Plan

1. For each failing use case, run one task with `--iwa-log-level INFO`.
2. Check first failing step:
   - selector mismatch vs. DOM
   - action order mismatch
   - task-constraint mismatch
3. Adjust trajectory to be constraint-robust (prefer placeholders and stable selectors).
4. Re-run with `--expect-non-zero`.
5. Repeat for remaining use cases.

