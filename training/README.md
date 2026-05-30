# Training Package

`training/` has two separate responsibilities:

1. `training.autoppia_operator`: the Claude/IWA trajectory-discovery operator.
   This is the main web agent for finding verified successful trajectories on
   never-seen tasks. It is not constrained by `/act`.
2. Distillation/fine-tuning modules: consume verified operator trajectories,
   export SFT/RL data, fine-tune a Qwen/LoRA adapter, serve it, and evaluate the
   resulting `/act` runtime.

Do not mix the operator with the distilled model. The operator creates gold
trajectories; the distilled model learns to imitate them.

Kept modules:
- `training/autoppia_operator/`: canonical Autoppia Operator namespace and artifact contract.
- `training/autoppia_operator/briefs.py`: Claude brief generation for trajectory discovery.
- `training/autoppia_operator/discovery.py`: Claude/evaluator retry loop for verified trajectories.
- `training/claude_code_harvester.py`: compatibility shim for old imports.
- `training/claude_guided_harvester.py`: teacher-guided action hints for focused harvests.
- `training/harvester.py`: legacy compatibility backend for baseline, code-aware, and guided collection.
- `training/focus_pipeline.py`: compatibility wrapper around the unified harvester and eval helpers.
- `training/layout.py`: canonical `data/<web_project>/<use_case>/...` layout.
- `training/dagger.py`: generic DAgger correction loop across use cases.
- `training/login_dagger.py` and `training/login_rule_corrections.py`: compatibility wrappers and login-specific heuristics.
- `training/format_for_sft.py`: SFT export aligned to the operator runtime format.
- `training/finetune_bu.py`: LoRA fine-tuning on RunPod / A100.
- `training/hf_openai_server.py`: OpenAI-compatible serving wrapper for a base model + adapter.
- `training/post_finetune_eval.py`: post-train evaluation against the real evaluator.

Current supported workflow:
1. run the Autoppia Operator until IWA verifies `success=true` and `score=1.0`
2. save canonical operator artifacts under `data/operator_runs/...`
3. export verified trajectories to SFT/RL data
4. fine-tune a Qwen/LoRA distilled agent
5. serve the adapter behind `/act`
6. evaluate on holdout seeds

Canonical operator CLI:

```bash
python -m training.autoppia_operator.claude_operator \
  --web-project-id autocinema \
  --use-case LOGIN \
  --seed 1 \
  --task-cache data/task_cache/autocinema_tasks.json \
  --output-dir data
```

Canonical data rule:
- only trace-backed, replayable trajectories count as canonical gold
- every gold row must have real `result_path` and `trace_file`
- every gold row must carry provenance fields so later dataset repairs and audits are possible

Focused CLI flow:
1. `scripts/eval/focus_use_case.py collect`
2. `scripts/eval/focus_use_case.py consolidate-gold`
3. `scripts/eval/focus_use_case.py export-sft`
4. `scripts/eval/focus_use_case.py validate-dataset`
5. `scripts/eval/focus_use_case.py train`
6. `scripts/eval/focus_use_case.py eval`

Deterministic-only teacher-harvest flow (zero-AI):
1. Provide a task cache file scoped to the project/use case (e.g. `data/task_cache/autocinema_50_tasks.json`, `data/task_cache/autobooks_tasks.json` for web2, `data/task_cache/autozone_tasks.json` for web3, `data/task_cache/autodining_tasks.json` for web4, `data/task_cache/autocrm_tasks.json` / `automail_tasks.json` / `autolodge_tasks.json` for web5–web7, `autodelivery_tasks.json` / `autowork_tasks.json` / `autoconnect_tasks.json` for web8–web10, or `autocalendar_tasks.json` / `autolist_tasks.json` for web11–web12). **IWA `demo_web_projects` web1–web12** (typical local ports from IWA `trajectories.py` `BASE` values): `autocinema` (n/a), `autobooks` 8001, `autozone` 8002, `autodining` 8003, `autocrm` 8004, `automail` 8005, `autodelivery` 8006, `autolodge` 8007, `autoconnect` 8008, `autowork` 8009, `autocalendar` 8010, `autolist` 8011. Ensure the app for that project is running at the host/port in the task `url`; otherwise replays can fail with connection errors.
2. Run:
   - `python scripts/eval/focus_use_case.py teacher-harvest --project-id <project_id> --use-case <use_case> --task-cache <cache.json> --deterministic-only`
   - Autobooks example: `python scripts/eval/focus_use_case.py teacher-harvest --project-id autobooks --use-case SEARCH_BOOK --task-cache data/task_cache/autobooks_tasks.json --deterministic-only --execution-mode operator`
   - Autozone example: `python scripts/eval/focus_use_case.py teacher-harvest --project-id autozone --use-case SEARCH_PRODUCT --task-cache data/task_cache/autozone_tasks.json --deterministic-only --execution-mode operator`
   - Autodining example: `python scripts/eval/focus_use_case.py teacher-harvest --project-id autodining --use-case SEARCH_RESTAURANT --task-cache data/task_cache/autodining_tasks.json --deterministic-only --execution-mode operator`
   - Autocrm (web5) example: `python scripts/eval/focus_use_case.py teacher-harvest --project-id autocrm --use-case SEARCH_MATTER --task-cache data/task_cache/autocrm_tasks.json --deterministic-only --execution-mode operator`
   - Automail (web6) example: `python scripts/eval/focus_use_case.py teacher-harvest --project-id automail --use-case SEARCH_EMAIL --task-cache data/task_cache/automail_tasks.json --deterministic-only --execution-mode operator`
   - Autolodge (web7) example: `python scripts/eval/focus_use_case.py teacher-harvest --project-id autolodge --use-case SEARCH_HOTEL --task-cache data/task_cache/autolodge_tasks.json --deterministic-only --execution-mode operator`
   - Autodelivery (web8) example: `python scripts/eval/focus_use_case.py teacher-harvest --project-id autodelivery --use-case SEARCH_DELIVERY_RESTAURANT --task-cache data/task_cache/autodelivery_tasks.json --deterministic-only --execution-mode operator`
   - Autowork (web9) example: `python scripts/eval/focus_use_case.py teacher-harvest --project-id autowork --use-case SEARCH_SKILL --task-cache data/task_cache/autowork_tasks.json --deterministic-only --execution-mode operator`
   - Autoconnect (web10) example: `python scripts/eval/focus_use_case.py teacher-harvest --project-id autoconnect --use-case SEARCH_USERS --task-cache data/task_cache/autoconnect_tasks.json --deterministic-only --execution-mode operator`
   - Autocalendar (web11) example: `python scripts/eval/focus_use_case.py teacher-harvest --project-id autocalendar --use-case SEARCH_SUBMIT --task-cache data/task_cache/autocalendar_tasks.json --deterministic-only --execution-mode operator`
   - Autolist (web12) example: `python scripts/eval/focus_use_case.py teacher-harvest --project-id autolist --use-case AUTOLIST_ADD_TASK_CLICKED --task-cache data/task_cache/autolist_tasks.json --deterministic-only --execution-mode operator`
3. The run resolves seeds from the provided cache for that project/use case, generates deterministic plans, replays them, and writes canonical harvest artifacts under `data/<project>/<use_case>/gold`. For `autobooks`, gold plans are sourced from IWA `p02_autobooks/trajectories.py` via `training/deterministic_harvester/builders/autobooks.py`. For `autozone`, use IWA `p03_autozone/trajectories.py` via `training/deterministic_harvester/builders/autozone.py`. For `autodining`, use IWA `p04_autodining/trajectories.py` via `training/deterministic_harvester/builders/autodining.py`. For `autocrm`, `automail`, and `autolodge`, use IWA `p05_autocrm`, `p06_automail`, and `p08_autolodge` `trajectories.py` via `autocrm.py`, `automail.py`, and `autolodge.py` respectively. For `autodelivery`, `autowork`, and `autoconnect`, use IWA `p07_autodelivery`, `p10_autowork`, and `p09_autoconnect` via `autodelivery.py`, `autowork.py`, and `autoconnect.py`. For `autocalendar` and `autolist`, use IWA `p11_autocalendar` and `p12_autolist` via `autocalendar.py` and `autolist.py`.

Multi-use-case bootstrap:
- `scripts/eval/harvest_suite.py --use-cases all --seeds 1..10 --strategy baseline`
- Use `--target-gold-per-use-case` and `--max-seeds-per-use-case` to cap exploratory collection while you measure which use cases already generalize well.

Implementation note:
- `training/harvester.py` is the source of truth for collection, artifact writing, consolidation, and SFT export.
- `training/focus_pipeline.py` remains only to preserve compatibility for older imports and tests while the repo finishes migrating.
- Deterministic planner logic is organized under `training/deterministic_harvester/builders/`, with shared helpers in `common.py`, project-specific logic in `autocinema.py` and IWA-backed `autobooks.py` / `autodining.py` / `autocrm.py` / `automail.py` / `autolodge.py` / `autodelivery.py` / `autowork.py` / `autoconnect.py` / `autocalendar.py` / `autolist.py` / `autozone.py` (convert IWA `trajectories.py` actions), plus `iwa_planned_actions.py` for IWA action shaping, and a thin registry/facade in `planners.py`.

Deterministic selector strategy (non-cinema rollout):
- Use semantic selectors first (`id`, `class`, `placeholder`, `text`) and keep XPath as ordered fallback only.
- `training/deterministic_harvester/trajectory_selectors.py` now derives semantic selector candidates from raw trajectory selectors, including XPath parsing for common attribute/text patterns.
- Curated per-use-case text fallbacks are applied where trajectories are too positional to infer stable semantic anchors.
- Coverage and ordering checks are enforced by `tests/test_semantic_selector_coverage.py`.
- Baseline project/use-case coverage matrix is tracked in `training/deterministic_harvester/selector_coverage_matrix.md`.

Current trusted dataset:
- `data/autocinema/login`

Current trusted model artifact:
- `models/bu-30b-login-500-lora`
