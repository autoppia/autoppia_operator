# Scripts

The root of `scripts/` is reserved for stable operator entrypoints:

- `scripts/sn36_ops.py`: subnet36 operator lifecycle CLI.
- `scripts/deploy_check.py`: `/act` and handshake contract checks.

Everything else is grouped by purpose:

- `scripts/eval/`: task generation and eval comparison helpers.
- `scripts/training/`: dataset preparation, export, and fine-tuning helpers.
- `scripts/sn36/`: manual shell helpers for submission and metagraph inspection.

Obsolete one-off scripts should be removed instead of restored to the root.

For focused use-case work, the intended flow is:
- collect under `data/<web_project>/<use_case>/gold`
- derive corrections under `.../dagger`
- export SFT under `.../sft`
- validate dataset readiness before training
- write benchmark outputs under `.../eval`

Key entrypoints:
- `scripts/eval/focus_use_case.py`: single-use-case wrapper over the unified harvester `collect -> consolidate-gold -> export-sft -> validate-dataset -> train -> eval`
- `scripts/eval/run_deterministic_harvest.py`: convenience wrapper for deterministic-only focused harvest runs with local defaults
- `scripts/eval/harvest_suite.py`: multi-use-case runner over the same unified harvester, useful for bootstrap coverage scans and first-pass gold collection
- `scripts/eval/run_login_dagger.py`: DAgger loop wrapper, now generic via `--use-case` and still defaulting to `LOGIN`
- `scripts/eval/build_login_rule_corrections.py`: deterministic correction builder for repeated `LOGIN` failures

Zero-AI deterministic trajectory generation (single project + single use case):
- `python scripts/eval/focus_use_case.py teacher-harvest --project-id <project_id> --use-case <use_case> --task-cache <cache.json> --deterministic-only`
- In deterministic-only mode, `teacher-harvest` enforces zero-AI generation and forces single-attempt behavior (no AI fallback loop).
