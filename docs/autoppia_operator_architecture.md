# Autoppia Operator Architecture

`autoppia_operator` is the trajectory-discovery system.

The operator is not the distilled `/act` model. The operator is the strongest
offline agent we can build: a Claude/code process with access to browser tools,
IWA, task caches, evaluator feedback, backend events, screenshots, traces, and
repo context. Its job is to discover a successful trajectory for a task it has
not seen before.

## Roles

### Autoppia Operator

- Runs as a Claude/IWA process, not as a subnet `/act` endpoint.
- Can inspect the IWA task, web project source, DOM, selectors, screenshots, and
  backend events.
- Iterates against the evaluator until IWA reports success.
- Emits verified trajectory artifacts only when `success=true` and `score >= 1.0`.

Canonical entrypoint:

```bash
python -m training.autoppia_operator.claude_operator \
  --web-project-id autocinema \
  --use-case LOGIN \
  --seed 1 \
  --task-cache data/task_cache/autocinema_tasks.json \
  --output-dir data
```

Compatibility wrapper:

```bash
python scripts/training/autoppia_operator.py ...
```

### Distilled `/act` Agent

- Consumes verified operator trajectories.
- Trains a Qwen/LoRA adapter or other cheap runtime policy.
- Implements the subnet `/act` contract.
- Should not be used as the data-discovery mechanism.

## Artifact Contract

Each operator run writes under:

```text
data/operator_runs/<web_project>/<use_case>/seed_<NNNN>/
```

Expected files:

- `task_context.json`
- `attempts.jsonl`
- `trajectory.json`
- `final_report.json`
- `iwa_score.json`
- `backend_events.json`
- `screenshots/`
- `manifest.json`

The manifest uses `artifact_version=autoppia-operator.v1` and has
`distillation_ready=true` only when IWA verifies the run.

## Migration Rule

Existing `training/harvester*.py` files are legacy compatibility backends. New
operator-facing code should import from `training.autoppia_operator`. Distillation
code should live under `training.distilled_agent` or existing fine-tuning modules.
