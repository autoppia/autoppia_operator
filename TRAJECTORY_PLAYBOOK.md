# Trajectory Playbook (Creation, Usage, Testing)

This guide explains how trajectories work in `autoppia_operator` and how to create, run, and debug them end-to-end.

It is written so another agent can execute the process without prior context.

## 0. Quick Commands (Short)

Assumption:

- you are already in `AUTOPPIA/Autoppia_repos/autoppia_operator`
- venv is already activated

Run one strict replay trajectory (Autobooks):

```bash
python scripts/test_trajectory_task_score.py --web-project-id autobooks --use-case SEARCH_BOOK --expect-non-zero --iwa-log-level ERROR
```

Run all use cases in one project (strict replay batch):

```bash
python scripts/test_trajectory_task_score.py --web-project-id autobooks --expect-non-zero --iwa-log-level ERROR
```

Restart operator agent:

```bash
pkill -f "uvicorn main:app" || true
python -m uvicorn main:app --host 0.0.0.0 --port "${AGENT_PORT:-9000}"
```

Health check:

```bash
curl -sS "http://127.0.0.1:${AGENT_PORT:-9000}/health"
```

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

1. `get_trajectory_replay_bundle(...)` selects trajectory + full action list.
2. `TrajectoryExecutor` maps trajectory dictionaries into executable IWA actions.
3. Evaluator runs actions step by step on the web.
4. Benchmark score is computed from tests/event criteria.

Key files:

- Trajectories: `src/operator/agents/fsm/trajectory.py`
- Mapper: `src/operator/runtime/trajectory_executor.py`
- Strict replay tester (single and batch): `scripts/test_trajectory_task_score.py`

## 3. Prerequisites

Assumption for all commands in this document:

- you are already in `AUTOPPIA/Autoppia_repos/autoppia_operator`
- your virtual environment is already activated

To test only action format/mapping logic:

- You can inspect and unit-test mapping without full benchmark runtime.

To test real benchmark score (`0` vs non-zero):

- Autoppia IWA demo web stack must be running (frontend/backend/event services).
- Task URL must be reachable (example: `http://localhost:8000/?seed=1`).

For full benchmark (`eval.py`, `/act` loop):

- IWA services must be running.
- Operator agent server must be running too.

### 3.1 Services To Start Before Testing Trajectories

Minimum runtime required for strict replay scoring:

- demo webs backend (`webs_server`) on `8090`
- demo webs frontend(s) on `8000+` (at least the project you test)
- operator agent on `9000`

If demo webs are not already up:

```bash
cd ../autoppia_webs_demo
./scripts/setup.sh
```

Start/restart operator agent:

```bash
cd ../autoppia_operator
pkill -f "uvicorn main:app" || true
python -m uvicorn main:app --host 0.0.0.0 --port "${AGENT_PORT:-9000}"
```

Quick health checks:

```bash
curl -s http://127.0.0.1:9000/health
curl -s http://127.0.0.1:8090/health
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/
```

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
6. Add the new trajectory under the correct project block (`p01_autocinema`, `p02_autobooks`, `p03_autozone`, `p04_autodining`, etc.) with correct `use_case`.

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

Check that the use case exists and a strict replay bundle is loaded:

```bash
python - <<'PY'
from src.operator.agents.fsm.trajectory import get_trajectory_replay_bundle
bundle = get_trajectory_replay_bundle(
    web_project_id="autocinema",
    use_case="LOGIN",
    prompt="",
    apply_prompt_overrides=True,
)
actions = bundle.get("actions", [])
print("url:", bundle.get("url"))
print("count:", len(actions))
print(actions[:2])
PY
```

## 6. Score testing (single use case or full project batch)

Run strict replay evaluator for one use case:

```bash
python scripts/test_trajectory_task_score.py \
--web-project-id autocinema \
--use-case LOGIN \
--web-agent-id 1 \
--expect-non-zero
```

For Autobooks:

```bash
python scripts/test_trajectory_task_score.py \
--web-project-id autobooks \
--use-case SEARCH_BOOK \
--expect-non-zero \
--iwa-log-level ERROR
```

Run all Autobooks use cases (no `--use-case`):

```bash
python scripts/test_trajectory_task_score.py \
--web-project-id autobooks \
--expect-non-zero \
--iwa-log-level ERROR
```

Equivalent explicit batch mode:

```bash
python scripts/test_trajectory_task_score.py \
--web-project-id autobooks \
--all-use-cases \
--expect-non-zero \
--iwa-log-level ERROR
```

### 6.1 Why this command is long (and what each argument does)

`scripts/test_trajectory_task_score.py` is a debugging/evaluation runner, not a short UX command.
It asks for several arguments to make runs explicit and reproducible.

Most important arguments:

- `--web-project-id`: project id to filter tasks (`autocinema`, `autobooks`, etc.).
- `--use-case`: use case to test (`SEARCH_BOOK`, `ADD_BOOK`, `LOGIN`, etc.). If omitted (and no `--task-id`), the script runs all use cases for that project.
- `--all-use-cases`: explicit batch mode for all use cases in the project (optional, same behavior as omitting `--use-case` and `--task-id`).
- `--task-cache`: optional. If omitted, script auto-detects a cache that contains the requested project and auto-generates one if needed.

Seed behavior in strict replay:

- reset URL comes from trajectory (`NavigateAction` / trajectory `url`)
- the script does not override trajectory seed via CLI

Common supporting arguments:

- `--expect-non-zero`: exits with error code if final score stays `0.0` (useful for quick pass/fail).
- `--iwa-log-level ERROR|INFO|...`: controls evaluator verbosity.
- `--web-agent-id`: useful when prompts/criteria depend on `web_agent_id`.
- `--raw-placeholders`: disables prompt-based placeholder resolution and runs raw trajectory values as-is.

Environment variables used in examples:

- None required for normal strict replay usage of this script.
- If auto-cache generation is needed and no usable cache exists, ensure `autoppia_operator/.env` has a valid `OPENAI_API_KEY`.

Common mistakes:

- Typo in project id: `autoboooks` (wrong) vs `autobooks` (correct).
- Passing a cache that does not contain the selected project.

Autobooks example (single use case):

```bash
python scripts/test_trajectory_task_score.py \
--web-project-id autobooks \
--use-case SEARCH_BOOK \
--expect-non-zero \
--iwa-log-level ERROR
```

Optional shortcut alias for Autobooks:

```bash
alias test_ab='python scripts/test_trajectory_task_score.py --web-project-id autobooks --expect-non-zero --iwa-log-level ERROR'
```

Then run:

```bash
test_ab --use-case SEARCH_BOOK
```

### 6.2 Placeholder behavior in strict replay

This script runs strict replay by default.

Default behavior:

- keeps `NavigateAction` and action order as stored
- uses trajectory URL/seed for reset
- resolves placeholders from task prompt before execution

If you want raw literal values from the trajectory file:

```bash
--raw-placeholders
```

## 7. Test by task id

Get task ids from cache:

```bash
python - <<'PY'
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
python scripts/test_trajectory_task_score.py \
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
python scripts/eval/generate_tasks.py --project-id autocinema --prompts-per-use-case 1
```

## 10. Criteria vs event payload rule

A trajectory can execute perfectly and still score 0 if benchmark criteria ask for fields not present in the real backend event payload.

For any failing use case:

1. inspect backend event payload fields
2. inspect event validation criteria in IWA project files
3. align criteria with real payload fields (without changing benchmark intent)

## 11. Definition of done (trajectory is "correct")

A trajectory is considered correct when:

1. Actions are loaded by `get_trajectory_replay_bundle(...)`.
2. Mapper converts actions without schema errors.
3. Steps execute without critical action errors.
4. Final benchmark score is non-zero for the target task/use case.
5. Result is reproducible with the trajectory's recorded URL/seed.

## 12. Suggested working loop

1. Create/adjust one trajectory.
2. Run single-use-case score test.
3. Fix selectors/placeholders/mapping.
4. Re-test same use case until non-zero.
5. Move to next use case.
6. Run an autocinema sweep at the end.
