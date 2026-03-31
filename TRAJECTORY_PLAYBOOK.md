# Trajectory Playbook (Creation, Usage, Testing)

This guide explains how trajectories work in `autoppia_operator` and how to create, run, and debug them end-to-end.

It is written so another agent can execute the process without prior context.

## 1. What a trajectory is

A trajectory is a fixed list of UI actions for a specific use case (for example `LOGIN`, `REGISTRATION`, `ADD_FILM`) stored in:

- `src/operator/agents/fsm/trajectory.py`

Main structure:

- `TRAJECTORIES`: list of projects (for autocinema, project id is `p01_autocinema`)
- each project has `trajectories`
- each trajectory has:
  - `use_case`
  - `prompt` (example text)
  - `actions` (list of action dictionaries)
  - `has_success`

## 2. Runtime flow (important)

The execution path is:

1. `get_trajectory_bootstrap_actions(...)` reads the trajectory actions.
2. `TrajectoryExecutor` maps trajectory dictionaries into executable IWA actions.
3. Evaluator runs actions step by step on the web.
4. Benchmark score is computed from tests/event criteria.

Key files:

- Trajectories: `src/operator/agents/fsm/trajectory.py`
- Mapper: `src/operator/runtime/trajectory_executor.py`
- Single-task debugger: `scripts/test_trajectory_task_score.py`

## 3. Prerequisites

To test only action format/mapping logic:

- You can inspect and unit-test mapping without full benchmark runtime.

To test real benchmark score (`0` vs non-zero):

- Autoppia IWA demo web stack must be running (frontend/backend/event services).
- Task URL must be reachable (example: `http://localhost:8000/?seed=1`).

For full benchmark (`eval.py`, `/act` loop):

- IWA services must be running.
- Operator agent server must be running too.

## 4. How to create a trajectory from a browser recording

Given a recording JSON:

1. Keep only relevant steps (`navigate`, `click`, `change`, optional key presses).
2. Ignore `setViewport` and other non-action setup steps.
3. Convert to trajectory action schema:
   - `click` -> `ClickAction`
   - `change` -> `TypeAction`
   - `navigate` -> `NavigateAction`
4. Prefer stable selectors:
   - first choice: `attributeValueSelector` by `id`
   - fallback: `xpathSelector`
5. Replace hardcoded values with placeholders when required by benchmark:
   - for login: `user<web_agent_id>`, `Passw0rd!`
   - for registration username/email: `newuser<web_agent_id>`, `newuser<web_agent_id>@gmail.com`
6. Add the new trajectory under `p01_autocinema` with correct `use_case`.

Minimal action examples:

```json
{
  "type": "ClickAction",
  "selector": {
    "type": "attributeValueSelector",
    "attribute": "id",
    "value": "login-sign-in-button",
    "case_sensitive": false
  }
}
```

```json
{
  "type": "TypeAction",
  "selector": {
    "type": "attributeValueSelector",
    "attribute": "id",
    "value": "login-username-input",
    "case_sensitive": false
  },
  "text": "user<web_agent_id>"
}
```

## 5. Quick local validation after editing trajectories

Check that the use case exists and actions are loaded:

```bash
.venv/bin/python - <<'PY'
from src.operator.agents.fsm.trajectory import get_trajectory_bootstrap_actions
actions = get_trajectory_bootstrap_actions(
    web_project_id="autocinema",
    use_case="LOGIN",
    prompt="",
    max_actions=20,
)
print("count:", len(actions))
print(actions[:2])
PY
```

## 6. Score testing for one use case

Run trajectory-only evaluator for one use case:

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

For registration, use an id that matches `newuser<web_agent_id>` in your run:

```bash
LLM_PROVIDER=openai OPENAI_API_KEY=dummy \
PYTHONPATH=../autoppia_iwa:. \
.venv/bin/python scripts/test_trajectory_task_score.py \
--web-project-id autocinema \
--use-case REGISTRATION \
--seed 1 \
--web-agent-id 177 \
--expect-non-zero
```

## 7. Test by task id

Get task ids from cache:

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
    if str(t.get("web_project_id")) == "autocinema":
        print(uc_name, t.get("id"))
PY
```

Run by id:

```bash
LLM_PROVIDER=openai OPENAI_API_KEY=dummy \
PYTHONPATH=../autoppia_iwa:. \
.venv/bin/python scripts/test_trajectory_task_score.py \
--task-id <TASK_ID> \
--web-agent-id 1 \
--expect-non-zero
```

## 8. How to read results

The script prints:

- `exec_ok` per step
- action errors (if any)
- `Final score`

Interpretation:

- `exec_ok=True` in all steps + `Final score=0.0`
  - mapping/actions probably work
  - failure is usually criteria/event mismatch
- selector timeout/action error
  - trajectory selector or mapper issue
- `Final score>0`
  - trajectory passes that benchmark task

## 9. Common pitfall: stale task cache

`scripts/test_trajectory_task_score.py` reads `data/task_cache/tasks_cache.json`.

If IWA criteria changed but cache is old, task prompt/tests may still use old constraints, causing misleading `score=0`.

Always verify printed task details:

- selected task prompt
- `Criteria` printed by evaluator logs

If needed, regenerate cache:

```bash
.venv/bin/python scripts/eval/generate_tasks.py --project-id autocinema --prompts-per-use-case 1
```

## 10. Criteria vs event payload rule

A trajectory can execute perfectly and still score 0 if benchmark criteria ask for fields not present in the real backend event payload.

For any failing use case:

1. inspect backend event payload fields
2. inspect event validation criteria in IWA project files
3. align criteria with real payload fields (without changing benchmark intent)

## 11. Definition of done (trajectory is "correct")

A trajectory is considered correct when:

1. Actions are loaded by `get_trajectory_bootstrap_actions(...)`.
2. Mapper converts actions without schema errors.
3. Steps execute without critical action errors.
4. Final benchmark score is non-zero for the target task/use case.
5. Result is reproducible with same seed.

## 12. Suggested working loop

1. Create/adjust one trajectory.
2. Run single-use-case score test.
3. Fix selectors/placeholders/mapping.
4. Re-test same use case until non-zero.
5. Move to next use case.
6. Run an autocinema sweep at the end.

