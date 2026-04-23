# Training Package

`training/` is now centered on one clean path: harvest a single use case against the
real `84.247.180.192` evaluator, distill it into SFT data, fine-tune a LoRA adapter,
serve it, and evaluate it again.

Kept modules:
- `training/claude_code_harvester.py`: Claude-focused harvesting entrypoint.
- `training/claude_guided_harvester.py`: teacher-guided action hints for focused harvests.
- `training/harvester.py`: unified harvest orchestration for baseline, code-aware, and guided collection.
- `training/focus_pipeline.py`: compatibility wrapper around the unified harvester and eval helpers.
- `training/layout.py`: canonical `data/<web_project>/<use_case>/...` layout.
- `training/dagger.py`: generic DAgger correction loop across use cases.
- `training/login_dagger.py` and `training/login_rule_corrections.py`: compatibility wrappers and login-specific heuristics.
- `training/format_for_sft.py`: SFT export aligned to the operator runtime format.
- `training/finetune_bu.py`: LoRA fine-tuning on RunPod / A100.
- `training/hf_openai_server.py`: OpenAI-compatible serving wrapper for a base model + adapter.
- `training/post_finetune_eval.py`: post-train evaluation against the real evaluator.

Current supported workflow:
1. collect gold trajectories with evaluator `score=1.0`
2. add DAgger or rule corrections if needed
3. export SFT
4. fine-tune
5. serve the adapter
6. evaluate on holdout seeds

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
1. Provide a task cache file scoped to the project/use case (e.g. `data/task_cache/autocinema_50_tasks.json`, `data/task_cache/autobooks_tasks.json` for web2, or `data/task_cache/autozone_tasks.json` for web3).
2. Run:
   - `python scripts/eval/focus_use_case.py teacher-harvest --project-id <project_id> --use-case <use_case> --task-cache <cache.json> --deterministic-only`
   - Autobooks example: `python scripts/eval/focus_use_case.py teacher-harvest --project-id autobooks --use-case SEARCH_BOOK --task-cache data/task_cache/autobooks_tasks.json --deterministic-only --execution-mode operator`
   - Autozone example: `python scripts/eval/focus_use_case.py teacher-harvest --project-id autozone --use-case SEARCH_PRODUCT --task-cache data/task_cache/autozone_tasks.json --deterministic-only --execution-mode operator`
3. The run resolves seeds from the provided cache for that project/use case, generates deterministic plans, replays them, and writes canonical harvest artifacts under `data/<project>/<use_case>/gold`. For `autobooks`, gold plans are sourced from IWA `p02_autobooks/trajectories.py` via `training/deterministic_harvester/builders/autobooks.py`. For `autozone`, use IWA `p03_autozone/trajectories.py` via `training/deterministic_harvester/builders/autozone.py`.

Multi-use-case bootstrap:
- `scripts/eval/harvest_suite.py --use-cases all --seeds 1..10 --strategy baseline`
- Use `--target-gold-per-use-case` and `--max-seeds-per-use-case` to cap exploratory collection while you measure which use cases already generalize well.

Implementation note:
- `training/harvester.py` is the source of truth for collection, artifact writing, consolidation, and SFT export.
- `training/focus_pipeline.py` remains only to preserve compatibility for older imports and tests while the repo finishes migrating.
- Deterministic planner logic is organized under `training/deterministic_harvester/builders/`, with shared helpers in `common.py`, project-specific logic in `autocinema.py` and IWA-backed `autobooks.py` / `autozone.py` (convert IWA `trajectories.py` actions), plus `iwa_planned_actions.py` for IWA action shaping, and a thin registry/facade in `planners.py`.

Current trusted dataset:
- `data/autocinema/login`

Current trusted model artifact:
- `models/bu-30b-login-500-lora`
