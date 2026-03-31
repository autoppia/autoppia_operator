# Mini Documentation: How We Test `TRAJECTORIES`

This is the short process we use to validate trajectory quality in `autoppia_operator`.

## Scope

We test whether trajectories in `src/operator/agents/fsm/trajectory.py`:

- map correctly to executable actions
- execute without action errors
- produce non-zero benchmark score

## Infrastructure prerequisites

Before running tests, make sure the environment is deployed:

- Autoppia IWA demo web stack must be up (at least autocinema frontend/backend).
- The task URL must be reachable (example: `http://localhost:8000/?seed=1`).
- Backend event service must be reachable by evaluator (used to validate score/events).

Agent deployment requirement depends on test type:

- Trajectory-only test (`scripts/test_trajectory_task_score.py`):
  - agent server is **not required** (actions come directly from trajectories).
- Full benchmark through `eval.py` (`/act` loop):
  - agent server **is required**, either:
    - local auto-start (`START_AGENT_SERVER=1`), or
    - external endpoint (`AGENT_BASE_URL=http://...`).

## Components

- Trajectory source:
  - `src/operator/agents/fsm/trajectory.py`
  - function: `get_trajectory_bootstrap_actions(...)`
- Mapper/adapter:
  - `src/operator/runtime/trajectory_executor.py`
  - converts trajectory dicts into IWA executable actions
- Debug test script:
  - `scripts/test_trajectory_task_score.py`
  - runs one benchmark task end-to-end using trajectory actions only

## What the debug script checks

For one selected task, it:

1. loads task from cache (`data/task_cache/tasks_cache.json`)
2. gets trajectory actions for the selected use case
3. maps actions through `TrajectoryExecutor`
4. executes actions step-by-step in evaluator
5. prints:
   - execution status per step (`exec_ok`)
   - exact action error text (if present)
   - final score (`0` or non-zero)

## Recommended command (from repo root)

```bash
LLM_PROVIDER=openai OPENAI_API_KEY=dummy \
PYTHONPATH=../autoppia_iwa:. \
.venv/bin/python scripts/test_trajectory_task_score.py \
--web-project-id autocinema \
--use-case LOGIN \
--seed 1 \
--web-agent-id 1 \
--expect-non-zero
```

## Test one specific task id

```bash
LLM_PROVIDER=openai OPENAI_API_KEY=dummy \
PYTHONPATH=../autoppia_iwa:. \
.venv/bin/python scripts/test_trajectory_task_score.py \
--task-id <TASK_ID> \
--web-agent-id 1 \
--expect-non-zero
```

## How we interpret outcomes

- All steps `exec_ok=True` but final score is `0.0`:
  - action schema/format is likely fine
  - task constraints/event validation are not satisfied yet
- Selector timeout or action error:
  - trajectory selector or mapping issue
- Final score `> 0`:
  - trajectory works for that task

## Practical workflow we followed

1. Run single-use-case debug (`LOGIN`, `REGISTRATION`, etc.)
2. Fix trajectory or runtime mapping
3. Re-run same use case until stable
4. Run broader autocinema sweep and compare average score/non-zero use cases

## Get task IDs quickly

Use this to list task ids by project/use case from cache:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path("data/task_cache/tasks_cache.json")
data = json.loads(p.read_text())
tasks = data["tasks"] if isinstance(data, dict) else data
for t in tasks:
    uc = t.get("use_case", {})
    uc_name = uc.get("name") if isinstance(uc, dict) else uc
    if str(t.get("web_project_id")) == "autocinema" and str(uc_name).upper() == "REGISTRATION":
        print(t.get("id"), "->", t.get("prompt"))
PY
```

## Important pitfall: stale task cache

`scripts/test_trajectory_task_score.py` reads tasks from local cache (`data/task_cache/tasks_cache.json`), not directly from live source code.

If you changed IWA criteria/events (for example, registration criteria), cache may still contain old constraints and you can get misleading `score=0`.

Always verify selected task prompt + event criteria in script output before concluding trajectory is wrong.


