# Harvester Guide — autoppia_operator

A step-by-step operational guide for generating, accumulating, and verifying gold trajectories
for any web project within the `autoppia_operator` framework.

> **Scope:** This guide is project-agnostic. Replace `<PROJECT_ID>` and `<USE_CASE>` with the
> actual values for your target web project (e.g. `autohealth`, `SEARCH_DOCTORS`).

---

## Table of Contents

1. [Architecture overview](#1-architecture-overview)
2. [Prerequisites](#2-prerequisites)
3. [Step 1 — Generate tasks](#3-step-1--generate-tasks)
4. [Step 2 — Collect gold trajectories](#4-step-2--collect-gold-trajectories)
5. [Step 3 — Export to IWA format](#5-step-3--export-to-iwa-format)
6. [Step 4 — Verify with IWA benchmark](#6-step-4--verify-with-iwa-benchmark)
7. [End-to-end shell script](#7-end-to-end-shell-script)
8. [Output file structure](#8-output-file-structure)
9. [Adding a new project](#9-adding-a-new-project)
10. [Troubleshooting](#10-troubleshooting)
11. [Project reference table](#11-project-reference-table)

---

## 1. Architecture overview

The pipeline has two independent stages:

```
[Task cache]
     │
     ▼
[Deterministic harvester]  ──────────────────────────────────┐
     │ IWA trajectory plan + enriched selectors              │
     │                                                       │
     ▼                                                       ▼
[collect_rows_for_seeds]  →  data/<project>/<use_case>/      │
     │                         gold/episodes.jsonl           │
     │  (accumulates across runs with merge_existing=True)   │
     ▼                                                       │
[export_gold_to_iwa.py]   →  autoppia_iwa/src/...            │
     │                         harvested_trajectories/       │
     ▼                                                       │
[verify_harvested_*.py]   ←  IWA AsyncStatefulEvaluator  ────┘
     │  (replays actions, checks test events)
     ▼
  PASS / FAIL per trajectory
```

**Gold definition:** a trajectory is gold when `success=True` **and** `score >= 1.0` as reported
by the operator's step engine. The IWA verify script is a second, stricter gate — it replays
actions against the live web app and checks event-based tests.

---

## 2. Prerequisites

### Repos and venvs

```
Autoppia_repos/
├── autoppia_operator/    ← main repo (this guide)
│   └── .venv/            ← Python venv
└── autoppia_iwa/         ← IWA benchmark repo
    └── .venv/            ← separate venv
```

Both repos must be cloned and their venvs activated for the respective commands.

### Environment variables

Create `autoppia_operator/.env` (copy from `.env.example` if present):

```bash
OPENAI_API_KEY=sk-...       # Required for Claude/GPT-guided harvesting
ANTHROPIC_API_KEY=sk-...    # Optional; only needed if using Claude provider
```

### Web app running

The web app for the target project must be running locally on its designated port before
collecting trajectories or running IWA verification.

| Project | Port |
|---------|------|
| autocinema | 8000 |
| autobooks | 8001 |
| autozone | 8002 |
| autodining | 8003 |
| autocrm | 8004 |
| automail | 8005 |
| autodelivery | 8006 |
| autolodge | 8007 |
| autoconnect | 8008 |
| autowork | 8009 |
| autocalendar | 8010 |
| autolist | 8011 |
| autodrive | 8012 |
| autohealth | 8013 |

Check availability: `curl -s -o /dev/null -w "%{http_code}" http://localhost:<PORT>/`

### Task cache

A task cache JSON file must exist at `data/task_cache/<PROJECT_ID>_tasks_cache.json`.
See [Step 1](#3-step-1--generate-tasks) to create one if missing.

---

## 3. Step 1 — Generate tasks

Tasks describe *what* the agent should accomplish: a prompt, a starting URL, and test criteria.
They are consumed by the harvester to drive trajectory collection.

### Option A — Generate via IWA pipeline (recommended for new projects)

Requires `OPENAI_API_KEY` and the `autoppia_iwa` venv.

```bash
cd autoppia_operator
.venv/bin/python scripts/eval/generate_tasks.py \
    --project-id <PROJECT_ID> \
    --prompts-per-use-case 1 \
    --out data/task_cache/<PROJECT_ID>_tasks_cache.json
```

Multiple projects in one shot:

```bash
.venv/bin/python scripts/eval/generate_tasks.py \
    --project-ids autohealth,autodrive \
    --prompts-per-use-case 1 \
    --out data/task_cache/multi_tasks_cache.json
```

### Option B — Use a pre-built cache

If a cache already exists (e.g. `data/task_cache/autohealth_tasks_cache.json`), skip generation.
The harvester will discover it automatically via `default_task_cache_for_project(<PROJECT_ID>)`.

### Verifying the cache

```bash
.venv/bin/python - <<'EOF'
import json
from pathlib import Path

cache = Path("data/task_cache/<PROJECT_ID>_tasks_cache.json")
d = json.loads(cache.read_text())
tasks = d.get("tasks", [])
print(f"Total tasks: {len(tasks)}")
for t in tasks:
    uc = t.get("use_case", {})
    name = uc.get("name", uc) if isinstance(uc, dict) else uc
    print(f"  {name} → {t.get('url')}")
EOF
```

Expected output: one task per use case, each with a `?seed=N` URL.

---

## 4. Step 2 — Collect gold trajectories

### Single use case

Use `focus_use_case.py collect` when working on one use case at a time or debugging:

```bash
cd autoppia_operator
.venv/bin/python scripts/eval/focus_use_case.py collect \
    --project-id <PROJECT_ID> \
    --use-case <USE_CASE> \
    --task-cache data/task_cache/<PROJECT_ID>_tasks_cache.json \
    --seeds 1..10 \
    --execution-mode operator
```

Example (autohealth, SEARCH_DOCTORS, seeds 1 to 5):

```bash
.venv/bin/python scripts/eval/focus_use_case.py collect \
    --project-id autohealth \
    --use-case SEARCH_DOCTORS \
    --task-cache data/task_cache/autohealth_tasks_cache.json \
    --seeds 1..5 \
    --execution-mode operator
```

### All use cases in a project

Use `harvest_suite.py` to iterate over every registered use case:

```bash
cd autoppia_operator
.venv/bin/python scripts/eval/harvest_suite.py \
    --project-id <PROJECT_ID> \
    --use-cases all \
    --seeds 1..10 \
    --execution-mode operator
```

Example (autodrive, all use cases):

```bash
.venv/bin/python scripts/eval/harvest_suite.py \
    --project-id autodrive \
    --use-cases all \
    --seeds 1..10 \
    --execution-mode operator
```

Useful flags:

| Flag | Default | Effect |
|------|---------|--------|
| `--target-gold-per-use-case N` | 0 (disabled) | Skip a use case once it has N gold episodes |
| `--max-seeds-per-use-case N` | 0 (no cap) | Run at most N seeds per use case |
| `--no-merge-existing` | off | Overwrite instead of accumulating |
| `--use-cases A,B,C` | `all` | Run only the listed use cases |

### Deterministic-only mode (zero-AI, fastest)

Uses pre-recorded IWA trajectories to generate candidates without any LLM calls.
This is the recommended starting point for any IWA-backed project (autohealth, autodrive, etc.)
because it is fast, free, and reproducible.

**Single use case:**

```bash
.venv/bin/python scripts/eval/focus_use_case.py teacher-harvest \
    --project-id <PROJECT_ID> \
    --use-case <USE_CASE> \
    --task-cache data/task_cache/<PROJECT_ID>_tasks_cache.json \
    --deterministic-only \
    --execution-mode operator
```

**All use cases at once (via harvest_suite):**

```bash
.venv/bin/python scripts/eval/harvest_suite.py \
    --project-id <PROJECT_ID> \
    --use-cases all \
    --seeds 1..10 \
    --deterministic-only \
    --execution-mode operator
```

> **Important:** `--deterministic-only` is required to generate a valid `candidate_path` in each
> gold episode. Without it, `export_gold_to_iwa.py` will skip those episodes
> ("all episodes failed to build rows").

### How accumulation works

Each run calls `write_harvest_artifacts(merge_existing=True)`, which:

1. Reads existing `data/<PROJECT_ID>/<USE_CASE>/gold/attempts.jsonl`
2. Appends new rows, deduplicating by `(seed, attempt_name)`
3. Re-filters for gold (success=True, score≥1.0)
4. Overwrites `gold/episodes.jsonl` with the merged gold set

Run repeatedly with different seed ranges to accumulate coverage.

### Checking collection results

```bash
.venv/bin/python - <<'EOF'
import json
from pathlib import Path

project = "<PROJECT_ID>"
data_root = Path(f"data/{project}")
for episodes in sorted(data_root.rglob("gold/episodes.jsonl")):
    use_case = episodes.parent.parent.name
    rows = [json.loads(l) for l in episodes.read_text().splitlines() if l.strip()]
    print(f"{use_case}: {len(rows)} gold episode(s)")
EOF
```

---

## 5. Step 3 — Export to IWA format

Converts the operator's flat gold episodes into the nested format expected by the IWA
benchmark's verify script.

```bash
cd autoppia_operator
.venv/bin/python scripts/eval/export_gold_to_iwa.py \
    --project-id <PROJECT_ID> \
    --data-root data \
    --task-cache data/task_cache/<PROJECT_ID>_tasks_cache.json \
    --iwa-root /home/rodriggod/AUTOPPIA/Autoppia_repos
```

Optional flags:

| Flag | Effect |
|------|--------|
| `--use-cases SEARCH_DOCTORS,CONTACT_DOCTOR` | Export only listed use cases |
| `--limit N` | Cap at N trajectories per use case |

**Output location:**

```
autoppia_iwa/src/demo_webs/projects/<FOLDER>/harvested_trajectories/<USE_CASE>/successful_trajectories.jsonl
```

Where `<FOLDER>` is the IWA folder for the project (e.g. `p14_autohealth`, `p13_autodrive`).
See [Project reference table](#11-project-reference-table) for the full mapping.

**Format transformation:**

| Operator format | IWA format |
|-----------------|------------|
| `selector_candidates` (list) | `selector` (first candidate) |
| flat row | nested `task`, `tests`, `actions` |

---

## 6. Step 4 — Verify with IWA benchmark

Replays trajectories against the live web app using the IWA `AsyncStatefulEvaluator`.
Each trajectory is replayed action-by-action; test events are checked against expected criteria.

```bash
cd autoppia_iwa
.venv/bin/python scripts/verify_harvested_<PROJECT_ID>_gold.py \
    --harvested-root src/demo_webs/projects/<FOLDER>/harvested_trajectories \
    --limit-per-use-case 1
```

Examples:

```bash
# Autohealth — verify 1 trajectory per use case
cd autoppia_iwa && .venv/bin/python scripts/verify_harvested_autohealth_gold.py \
    --harvested-root src/demo_webs/projects/p14_autohealth/harvested_trajectories \
    --limit-per-use-case 1

# Autocinema — verify all (limit 0 = unlimited)
cd autoppia_iwa && .venv/bin/python scripts/verify_harvested_autocinema_gold.py \
    --harvested-root src/demo_webs/projects/p01_autocinema/harvested_trajectories \
    --limit-per-use-case 0
```

Optional flags:

| Flag | Effect |
|------|--------|
| `--use-cases A,B` | Filter to specific use cases |
| `--limit-per-use-case 0` | Verify all trajectories (no cap) |

**Success criteria:**

A trajectory **passes** if:
- All test events are triggered in the correct order
- `score = 1.0` as reported by `AsyncStatefulEvaluator`

A trajectory **fails** if:
- The start URL is not reachable (`url_not_reachable`)
- An action fails (element not found, navigation error)
- A test event is not triggered or has wrong data

---

## 7. End-to-end shell script

For each project, a convenience script at the `Autoppia_repos/` root combines Steps 3 and 4:

```bash
# Syntax: ./check_<PROJECT_ID>_gold_iwa.sh [LIMIT_PER_USE_CASE] [USE_CASE_FILTER]

# Run 1 trajectory per use case (default)
./check_autohealth_gold_iwa.sh

# Run 3 trajectories per use case
./check_autohealth_gold_iwa.sh 3

# Run only SEARCH_DOCTORS, 2 trajectories
./check_autohealth_gold_iwa.sh 2 SEARCH_DOCTORS
```

To create a script for a new project, copy and adapt:

```bash
cp check_autohealth_gold_iwa.sh check_<PROJECT_ID>_gold_iwa.sh
# Edit: change --project-id, --task-cache, --harvested-root, and the verify script name
```

---

## 8. Output file structure

All harvested data lives under `data/<PROJECT_ID>/<USE_CASE>/`:

```
data/<PROJECT_ID>/<USE_CASE>/
├── candidates/
│   └── seed_<NNNN>_<attempt_name>.json   # TrajectoryCandidate with full action list
├── gold/
│   ├── attempts.jsonl    # Every attempt (gold + failed), deduped by (seed, attempt_name)
│   ├── episodes.jsonl    # Gold only (success=True, score≥1.0)
│   ├── summary.json      # Aggregated stats (gold count, seeds covered, etc.)
│   └── runs/             # Per-seed run reports
├── harvester/
│   ├── claude_runs/      # Claude brief and feedback per seed
│   └── deterministic_runs/  # Deterministic plan outputs per seed
├── task_cache/           # Per-seed overridden task cache files
└── traces/               # Episode traces for each gold candidate
```

IWA export output:

```
autoppia_iwa/src/demo_webs/projects/<FOLDER>/harvested_trajectories/
└── <USE_CASE>/
    └── successful_trajectories.jsonl   # Exported gold in IWA format
```

---

## 9. Adding a new project

Follow these steps to onboard any IWA-backed project that has trajectories in
`autoppia_iwa.src.demo_webs.trajectory_registry`.

### 9.1 Verify IWA trajectories exist

```bash
cd autoppia_operator
.venv/bin/python - <<'EOF'
import sys; sys.path.insert(0, "../autoppia_iwa")
import training._iwa_path
from autoppia_iwa.src.demo_webs.trajectory_registry import get_trajectory_map
from training.deterministic_harvester.builders.iwa_enriched_planner import iwa_enriched_action_builders

PROJECT = "<PROJECT_ID>"
m = get_trajectory_map(PROJECT)
print(f"{PROJECT} IWA use cases ({len(m)}):", sorted(m.keys()))
builders = iwa_enriched_action_builders(PROJECT)
print(f"Enriched builders ready: {len(builders)}")
EOF
```

If `len(builders) > 0`, the project can use the IWA-backed path.

### 9.2 Create the task cache

```bash
.venv/bin/python scripts/eval/generate_tasks.py \
    --project-id <PROJECT_ID> \
    --prompts-per-use-case 1 \
    --out data/task_cache/<PROJECT_ID>_tasks_cache.json
```

Or write a minimal cache by hand (see `data/task_cache/autohealth_tasks_cache.json` as template).
One task per use case is enough to run the deterministic harvester.

### 9.3 Create the builder module

Create `training/deterministic_harvester/builders/<PROJECT_ID>.py`:

```python
"""Deterministic plans for `<PROJECT_ID>` backed by IWA trajectories."""
from __future__ import annotations

import training._iwa_path  # noqa: F401
from training.deterministic_harvester.builders.common import DeterministicPlan
from training.deterministic_harvester.builders.iwa_enriched_planner import (
    build_iwa_enriched_deterministic_plan,
    iwa_enriched_action_builders,
)
from training.deterministic_harvester.normalizer import DeterministicTaskObjective
from training.deterministic_harvester.trajectory_selectors import list_iwa_use_cases

_PROJECT = "<PROJECT_ID>"
_SOURCE = "iwa_p<NN>_<PROJECT_ID>"   # e.g. "iwa_p14_autohealth"
_PROMPT = "IWA p<NN> <PROJECT_ID> trajectory for"

<PROJECT_ID_UPPER>_PLAN_BUILDERS = iwa_enriched_action_builders(_PROJECT)


def list_<PROJECT_ID>_iwa_use_cases() -> frozenset[str]:
    return list_iwa_use_cases(_PROJECT)


def build_<PROJECT_ID>_plan(objective: DeterministicTaskObjective) -> DeterministicPlan:
    return build_iwa_enriched_deterministic_plan(
        _PROJECT,
        objective,
        iwa_source_label=_SOURCE,
        prompt_preamble=_PROMPT,
    )


__all__ = [
    "<PROJECT_ID_UPPER>_PLAN_BUILDERS",
    "build_<PROJECT_ID>_plan",
    "list_<PROJECT_ID>_iwa_use_cases",
]
```

### 9.4 Register in the builders registry

In `training/deterministic_harvester/builders/registry.py`, add two lines:

```python
# Import
from training.deterministic_harvester.builders.<PROJECT_ID> import <PROJECT_ID_UPPER>_PLAN_BUILDERS, build_<PROJECT_ID>_plan

# In DETERMINISTIC_PLAN_BUILDERS dict
**{("<PROJECT_ID>", use_case): build_<PROJECT_ID>_plan for use_case in <PROJECT_ID_UPPER>_PLAN_BUILDERS},
```

### 9.5 Add the IWA folder mapping

In `scripts/eval/export_gold_to_iwa.py`, add to `_IWA_PROJECT_FOLDER`:

```python
"<PROJECT_ID>": "p<NN>_<PROJECT_ID>",
```

### 9.6 Create verify script in autoppia_iwa

```bash
cp autoppia_iwa/scripts/verify_harvested_autohealth_gold.py \
   autoppia_iwa/scripts/verify_harvested_<PROJECT_ID>_gold.py
```

Edit the new file:
- Change the `--harvested-root` default to `src/demo_webs/projects/p<NN>_<PROJECT_ID>/harvested_trajectories`
- Update the description string

### 9.7 Create end-to-end shell script

```bash
cp check_autohealth_gold_iwa.sh check_<PROJECT_ID>_gold_iwa.sh
chmod +x check_<PROJECT_ID>_gold_iwa.sh
```

Edit:
- `--project-id <PROJECT_ID>`
- `--task-cache data/task_cache/<PROJECT_ID>_tasks_cache.json`
- Script name in verify step: `verify_harvested_<PROJECT_ID>_gold.py`
- `--harvested-root src/demo_webs/projects/p<NN>_<PROJECT_ID>/harvested_trajectories`

### 9.8 Collect and verify

```bash
# Collect (deterministic, fast, no LLM)
.venv/bin/python scripts/eval/harvest_suite.py \
    --project-id <PROJECT_ID> \
    --use-cases all \
    --seeds 1 \
    --execution-mode operator

# Export + verify
./check_<PROJECT_ID>_gold_iwa.sh 1
```

---

## 10. Troubleshooting

### `ValueError: Unsupported deterministic use case: <USE_CASE>`

The `(project_id, use_case)` pair is not in `DETERMINISTIC_PLAN_BUILDERS`. Check:
1. The builder file exists and is registered in `registry.py`
2. The use case name matches IWA exactly (run `list_iwa_use_cases("<PROJECT_ID>")` to verify)

### `FileNotFoundError: No task cache found for project_id=<PROJECT_ID>`

Create the task cache file at `data/task_cache/<PROJECT_ID>_tasks_cache.json` (see Step 1).

### `url_not_reachable` in IWA verify

The web app is not running on the expected port. Start the app and retry.
Check the port in the [project reference table](#11-project-reference-table).

### `candidate has no actions, skipping`

The candidate JSON exists but has an empty `actions` list. This happens when the deterministic
planner returned no steps (usually because the use case is not in the builder). Add it to the
registry and re-collect.

### All episodes have `score=0.0` but `success=False`

The operator's step engine could not match any selector. Inspect the candidate actions at
`data/<PROJECT_ID>/<USE_CASE>/candidates/seed_<NNNN>_deterministic_01.json` and check
`selector_candidates` — if all are XPath-only, the enrichment step may have failed.

### `target_already_met` in harvest_suite output

The `--target-gold-per-use-case` threshold was already reached for that use case.
Use `--no-merge-existing` or increase the target to force more collection.

---

## 11. Project reference table

| Project ID | IWA folder | Port | Task cache file |
|------------|-----------|------|-----------------|
| autocinema | p01_autocinema | 8000 | `tasks_cache.json` |
| autobooks | p02_autobooks | 8001 | `autobooks_tasks_cache.json` |
| autozone | p03_autozone | 8002 | `autozone_tasks_cache.json` |
| autodining | p04_autodining | 8003 | `autodining_tasks_cache.json` |
| autocrm | p05_autocrm | 8004 | `autocrm_tasks_cache.json` |
| automail | p06_automail | 8005 | `automail_tasks_cache.json` |
| autodelivery | p07_autodelivery | 8006 | `autodelivery_tasks_cache.json` |
| autolodge | p08_autolodge | 8007 | — |
| autoconnect | p09_autoconnect | 8008 | — |
| autowork | p10_autowork | 8009 | — |
| autocalendar | p11_autocalendar | 8010 | — |
| autolist | p12_autolist | 8011 | — |
| autodrive | p13_autodrive | 8012 | `autodrive_tasks_cache.json` |
| autohealth | p14_autohealth | 8013 | `autohealth_tasks_cache.json` |

`—` means the task cache must be generated via `generate_tasks.py` before use.
