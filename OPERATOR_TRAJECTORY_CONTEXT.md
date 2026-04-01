# Operator Trajectory Context

Last update: 2026-04-01

## Scope

This is the handoff context to continue trajectory work in `autoppia_operator`.
Current focus has been strict replay workflow consolidation plus trajectories for:

- `autobooks`
- `autozone`
- `autodining`
- `autocrm` (started today)

## Current Replay Model (Important)

Trajectory testing is now strict replay-oriented:

- trajectory source: `src/operator/agents/fsm/trajectory.py`
- selection API: `get_trajectory_replay_bundle(...)`
- prompt adaptation: `_apply_prompt_overrides(...)`
- runtime mapper/executor: `src/operator/runtime/trajectory_executor.py`
- test runner: `scripts/test_trajectory_task_score.py`

### Key behavior now

- URL/seed comes from trajectory (`NavigateAction` / trajectory `url`).
- No CLI seed override in trajectory score script.
- If you change seed in trajectory URL, many use cases can fail by criteria mismatch.
- If `--use-case` and `--task-id` are omitted, the script runs all use cases in batch for the selected project.

## Script Updates Implemented Today

File: `scripts/test_trajectory_task_score.py`

- Added `--all-use-cases`.
- Added automatic batch mode when no `--use-case` and no `--task-id`.
- Batch summary includes total/passed/failed and failed use case list.
- Kept single-use-case mode unchanged.

## Runtime Updates Implemented Today

File: `src/operator/runtime/trajectory_executor.py`

- `SelectAction` support was added end-to-end:
  - accepted in supported action types
  - mapped to IWA payload
  - executed with Playwright `select_option`

## Trajectory Adapter Updates Implemented Today

File: `src/operator/agents/fsm/trajectory.py`

- Cleaned and reinforced strict replay overrides for `autodining`.
- Added robust prompt extraction for negative constraints (`not`, `not_equals`, etc.).
- Fixed selector/XPath generation issues (including invalid XPath in phone input case).
- Added controlled no-op key steps in `SEARCH_RESTAURANT` to cover debounce-based event logging.

## IWA Alignment Changes Done

Files in `autoppia_iwa`:

- `autoppia_iwa/src/demo_webs/projects/autodining_4/data.py`
- `autoppia_iwa/src/demo_webs/projects/autodining_4/events.py`

Purpose:

- align constants and event parsing with real `web_4_autodining` frontend behavior and labels.

## Validation Snapshot

### Autobooks / Autozone

- Work was migrated to strict replay style and validated in this session series.

### Autodining

- Final strict replay sweep: `20/20` use cases green.
- Cache used/regenerated: `data/task_cache/autodining_tasks_cache.json`.

### AutoCRM (new start)

- Recording intake started.
- First received recording: `SEARCH_MATTER` (`Estate`) on `http://localhost:8004/?seed=1`.
- Next requested recording: `FILTER_MATTER_STATUS` with prompt:
  - `Filter matters to only show those with status 'Active'.`

## Commands (Short, from `autoppia_operator` with venv active)

Single use case:

```bash
python scripts/test_trajectory_task_score.py --web-project-id autodining --use-case VIEW_RESTAURANT --expect-non-zero --iwa-log-level ERROR
```

Whole project (batch):

```bash
python scripts/test_trajectory_task_score.py --web-project-id autodining --expect-non-zero --iwa-log-level ERROR
```

Explicit batch flag:

```bash
python scripts/test_trajectory_task_score.py --web-project-id autodining --all-use-cases --expect-non-zero --iwa-log-level ERROR
```

## Next Session Plan

1. Continue `autocrm` recordings use case by use case (1x1).
2. Convert each recording into strict replay trajectory entry.
3. Run single-use-case score checks.
4. Run full `autocrm` batch and close gaps to green.
