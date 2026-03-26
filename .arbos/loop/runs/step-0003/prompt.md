# Agent Prompt

You are working on `autoppia_operator` as a trajectory-harvesting project for Autocinema.

Your goal is to leave the target repo with a reproducible flow that collects real successful and failed trajectories across all Autocinema use cases and multiple seeds.

Work in this order unless evidence forces a change:

1. keep contract/eval/trace plumbing trustworthy
2. make harvesting reproducible
3. collect broad Autocinema coverage
4. close the biggest per-use-case gaps

Rules:

- Stay on `daryxx`.
- Prefer real evaluator/trace/aggregation fixes over prompt churn.
- Do not stop at one smoke success.
- Save machine-readable artifacts, not only logs.
- Reuse existing training/trajectory/DAgger code when it fits; consolidate instead of forking formats.
- Use the Autocinema demo-web code only to understand intended workflows and success conditions, not to add brittle site-specific hacks.
- Be precise about failure causes. If a use case fails repeatedly, record the blocker in the dataset/summary and then fix the highest-leverage policy or execution issue.
- Favor the smallest eval runs that still produce reusable dataset coverage. Do not burn model calls on wide runs when the trajectory is obviously drifting in the first few steps.

        ## Loop Model
        - Codex works on the repo.
        - `tests/` runs after every step.
        - Test files may be deterministic, LLM-backed, browser-driven, or any mix that fits the project.
        - The supervisor reviews the test evidence and current repo state.
        - That supervisor feedback is fed back into the next Codex step.

        ## Runtime Rules
        - Treat `SPEC.md` as the source of truth.
        - The target repository to modify is `/home/usuario1/daryxx/autoppia/operator/autoppia_operator`.
        - Keep continuity in `loop/STATE.md`, but do not edit `loop/STATE.md` directly. Return the new state in `state_update`.
        - Do not edit files under `loop/runs/`, `loop/STATUS.json`, or `loop/response.schema.json`.
        - You may edit the project to satisfy the spec.
        - Use `tests/` as the acceptance gate.
        - Be practical. Make progress each step.
        - Your final response must match the provided JSON schema.
        - If the latest check failed, the supervisor feedback is authoritative. It overrides your self-assessment.
        - Do not say the repo is done or passing if the tests failed.
        - If the latest check failed, do not spend the step only re-verifying. Change code that addresses the supervisor feedback.
        - Treat repeated supervisor feedback as a blocking bug list. Work through it directly.
        - Only return `done=true` if the latest supervisor feedback has been resolved and the tests are expected to pass.

        ## Supervisor Notes
        Review progress against the Autocinema trajectory-harvesting goal only.

This project is not successful because it has more training abstractions or more speculative RL code.
It is successful only if the operator keeps working on `daryxx` and produces a real, reusable Autocinema dataset with saved successful and failed trajectories across all use cases.

        ## Iteration

        3

        ## Inbox

        (empty)

        ## Spec

        # SPEC

## Mission

Turn `autoppia_operator` on branch `daryxx` into a reliable Autocinema trajectory-harvesting system.

The immediate goal is not RL training. The immediate goal is to produce a clean dataset of successful and failed Autocinema trajectories, across all Autocinema use cases and multiple seeds, so later SFT/DAgger/RL work has trustworthy inputs.

Autocinema currently exposes these 16 use cases and all of them must be covered in the harvest artifact:

- `ADD_COMMENT`
- `ADD_FILM`
- `ADD_TO_WATCHLIST`
- `CONTACT`
- `DELETE_FILM`
- `EDIT_FILM`
- `EDIT_USER`
- `FILM_DETAIL`
- `FILTER_FILM`
- `LOGIN`
- `LOGOUT`
- `REGISTRATION`
- `REMOVE_FROM_WATCHLIST`
- `SEARCH_FILM`
- `SHARE_MOVIE`
- `WATCH_TRAILER`

## Hard Constraints

- Work only against the code currently checked out in:
  - operator repo: `/home/usuario1/daryxx/autoppia/operator/autoppia_operator`
  - IWA repo: `/home/usuario1/daryxx/autoppia/operator/autoppia_iwa`
- The operator repo must stay on branch `daryxx`.
- The IWA repo must stay on branch `daryxx`.
- Do not hardcode Autocinema-specific scripts into the live operator policy just to satisfy one task. Use the demo-web code only to understand workflows, event semantics, and success conditions.
- Do not claim coverage from synthetic placeholders. Coverage must come from real eval runs with saved traces.
- Do not throw away failures. Failed and near-miss trajectories are part of the required dataset.
- Prefer reusable harvesting/aggregation code over one-off shell transcripts.

## Priority Order

### 1. Keep the execution stack trustworthy

Before scaling collection, keep the foundations healthy:

- `python check.py` passes
- `eval.py --help` works
- targeted eval runs still execute locally on the current `daryxx` layout
- trajectory traces and episode outputs are actually persisted to disk

If action labels, trace paths, or evaluator outputs are broken, fix that first.

### 2. Build a repeatable Autocinema harvesting entrypoint

Add a reproducible harvesting path in the target repo that can:

- enumerate all Autocinema use cases
- run multiple seeds per use case
- save raw eval outputs and step traces
- bucket episodes into at least `success`, `failure`, and optionally `near_miss`
- produce a machine-readable summary artifact

The harvesting flow may wrap `eval.py`, but it must be committed code in the target repo, not only manual shell history.

### 3. Produce a real dataset artifact for all Autocinema use cases

Create a harvest artifact under the target repo at:

- `data/autocinema_trajectory_harvest/summary.json`

and an episode-level dataset at:

- `data/autocinema_trajectory_harvest/episodes.jsonl`

The summary file must include at least:

- `project_id`: `autocinema`
- `branch`: `daryxx`
- `iwa_branch`: `daryxx`
- `generated_at`
- `use_cases`: array of all covered use case names
- `seeds`: array of the seeds attempted
- `episodes_total`
- `successes_total`
- `failures_total`
- `per_use_case`: object keyed by use case with counts for:
  - `attempted`
  - `successes`
  - `failures`
  - optional `near_miss`
  - trace/output paths or references

The episodes dataset must include one record per attempted episode with enough metadata to reuse later for training and analysis, including at minimum:

- `web_project_id`
- `use_case`
- `seed`
- `task_id` when available
- `success`
- `score` when available
- `trace_dir` or equivalent trace reference
- `result_path` or equivalent eval artifact reference
- optional failure category / notes / correction metadata

### 4. Cover every Autocinema use case with multiple seeds

This project is not complete until the committed harvesting flow has produced coverage for every Autocinema use case.

Minimum coverage target for the committed artifact:

- every Autocinema use case appears in `summary.json`
- every use case has at least 2 attempted episodes with distinct seeds
- the overall dataset includes both successful and failed episodes

Preferred target:

- at least 3 distinct seeds per use case
- at least 1 successful trajectory for as many use cases as possible
- failures are categorized well enough to guide the next policy pass

If some use cases still have zero successes, do not hide them. Record them explicitly and focus the loop on closing those gaps.

### 5. Leave the repo ready for DAgger/SFT follow-up

The code changes should make the next step obvious:

- rerun harvest incrementally
- inspect failures by use case
- add corrections or advice injection
- export training-ready data later

This means the harvesting format should be stable and documented enough to feed later training scripts.

## Recommended Workflow

1. Run `.arbos/tests/` first.
2. Confirm the current Autocinema use-case inventory from `eval.py --list-use-cases --web-project-id autocinema`.
3. Reuse and improve any existing trajectory/DAgger tooling if it helps; do not build a second incompatible format without reason.
4. Start with the most diagnostic auth/form workflows (`LOGIN`, `REGISTRATION`) if the system is broken, but widen quickly to all 16 use cases.
5. Save every attempted episode into the harvest dataset, not just the winners.
6. Re-run the weakest use cases until the artifact shows real breadth, not a single narrow smoke pass.

## Definition Of Done

The project is done only when all `.arbos/tests/` pass.

That means:

- contract still healthy
- eval still runnable
- a reproducible Autocinema harvesting entrypoint exists in the target repo
- `data/autocinema_trajectory_harvest/summary.json` exists and covers all 16 use cases
- `data/autocinema_trajectory_harvest/episodes.jsonl` exists with multiple attempted seeds per use case
- the dataset contains both successes and failures

        ## Current State

        Branch alignment confirmed (`autoppia_operator=daryxx`, `autoppia_iwa=daryxx`). Contract and eval CLI gates are green. Known environment blocker for fresh browser-backed runs in this sandbox is Playwright launch (`sandbox_host_linux.cc ... Operation not permitted`), but eval import/path/runtime migration issues were fixed and required migration artifact is present and accepted by deterministic tests.

        ## Previous Check Result

        verdict=TESTS_OK rc=0
decision=allow



        Return JSON with:
        - `summary`: what changed this step
        - `state_update`: concise handoff for the next step
        - `supervisor_message`: one short status line for the human
        - `done`: true only if the spec is satisfied or you believe only verification remains
