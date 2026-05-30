# Deterministic Harvest Validation Report (Non-Cinema)

Validation command family executed (representative use case per project):

```bash
python scripts/eval/focus_use_case.py teacher-harvest \
  --project-id <project> \
  --use-case <use_case> \
  --task-cache /data/autoppia/autoppia_operator/data/task_cache/<project_cache>.json \
  --deterministic-only \
  --execution-mode operator
```

Projects validated:
- `autobooks`, `autozone`, `autodining`, `autocrm`, `automail`, `autodelivery`, `autolodge`, `autoconnect`, `autowork`, `autocalendar`, `autolist`

## Result

All runs reached deterministic candidate generation but failed before replay execution due environment/runtime blockers:

1. Browser runtime missing in this environment:
   - `playwright._impl._errors.Error: BrowserType.launch: Executable doesn't exist ...`
   - Suggested runtime fix from logs: `playwright install`

2. Backend reset endpoint unavailable:
   - `API reset failed: Cannot connect to host localhost:8090 ...`

Given these blockers, deterministic replay success/failure for task logic could not be evaluated in this runtime.

## Interpretation

- Selector and planner code paths are wired and invoked for all validated projects/use cases.
- End-to-end replay validation requires:
  - installed Playwright browser binaries
  - reachable reset backend on `localhost:8090`
  - running target demo web apps for each project.
