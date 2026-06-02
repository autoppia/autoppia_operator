# Agent Prompt

You are running a strong Autocinema harvesting campaign for `autoppia_operator`.

Your objective is concrete:

- get to at least 10 successful trajectories for each of the 16 Autocinema use cases
- each success must be from a distinct seed
- keep failures and near-misses too
- keep every episode replayable and trace-backed

You are allowed to improve the harvest loop itself while doing this.

## Working Rules

- Stay on target repo branch `arbos`, based on `main`.
- Prefer improvements that make harvesting more effective over broad blind reruns.
- Use the demo-web code in the repo to understand intended flows and success conditions.
- Use DAgger-style corrections or advice when the base policy keeps drifting.
- Favor focused use-case loops on the weakest workflows over expensive wide evals that obviously drift.
- Save machine-readable artifacts, not only shell logs.
- Keep the committed harvest as the source of truth.

## Priorities

1. keep `check.py`, local eval, and trace persistence healthy
2. improve harvesting / advice / correction loop quality
3. close the weakest use cases first
4. push every use case to 10 successful distinct-seed trajectories

## Strong Preferences

- Reuse and improve existing scripts, formats, and training helpers instead of inventing parallel formats.
- Record why failures happen.
- When advice improves a trajectory, capture that correction path in a reusable way.
- Treat old score-only result files as weak evidence unless they have matching replayable traces.

## Do Not

- do not stop because “tests pass”
- do not stop because a few use cases look good
- do not hide use cases with zero or low success
- do not hardcode brittle Autocinema-only scripts into the live policy just to hit one seed

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
        - Write `state_update` as short natural-language handoff notes, not JSON. Focus on current objective, the blocker that matters most, and the next concrete attempt.
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

        7

        ## Inbox

        (empty)

        ## Spec

        # SPEC

## Mission

Turn `autoppia_operator` into a strong Autocinema trajectory-harvesting and correction loop.

The goal is not just to "have a harvest script". The goal is to leave the repo with a reproducible system that can drive Autocinema use case by use case, use every legitimate debugging aid available, and accumulate a training-quality dataset with broad successful coverage.

This run is complete only when the committed dataset contains at least 10 successful trajectories for every Autocinema use case, each from distinct seeds, with replayable traces and machine-readable provenance.

## Target Repo And Branch Discipline

- target repo: `/home/usuario1/daryxx/autoppia/operator/autoppia_operator`
- work only on target repo branch `arbos`
- that branch must start from `main`
- do not silently switch back to `daryxx`

If branch handling is wrong, fix it first.

## Autocinema Use Cases

All 16 Autocinema use cases must be covered:

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

## What Is Allowed

Use everything legitimate that helps collect correct trajectories:

- improve the harvesting loop
- improve trace persistence and aggregation
- improve the operator policy
- inject advice / DAgger-style hints
- inspect Autocinema demo-web code under the operator repo to understand intended workflows, validation semantics, and success conditions
- use the existing trajectory, SFT, reward, and eval code if it helps

## What Is Not Allowed

- do not fake success counts
- do not count score-only result rows without replayable traces
- do not hardcode brittle task scripts directly into the live policy just to satisfy one seed
- do not hide failed use cases or collapse them out of the dataset
- do not stop after a few smoke wins
- do not declare success because tests pass while the dataset target is still far away

## Dataset Goal

The final harvest artifact must be under:

- `data/autocinema_trajectory_harvest/summary.json`
- `data/autocinema_trajectory_harvest/episodes.jsonl`
- `data/autocinema_trajectory_harvest/collection_manifest.json`
- `data/autocinema_trajectory_harvest/golden_seeds.json`

And it must satisfy:

- every use case present
- at least 10 successful trajectories per use case
- those 10 successes must use distinct seeds
- every kept episode is trace-backed and replayable
- the dataset still retains failures and near-misses

This implies a minimum of:

- `successes_total >= 160`

If a use case requires many failed attempts and advice iterations before reaching 10 successes, keep those failures in the dataset and keep iterating.

## Required Summary Fields

`summary.json` must include at least:

- `project_id = "autocinema"`
- `branch = "arbos"`
- `iwa_branch`
- `generated_at`
- `require_trace_files = true`
- `use_cases`
- `seeds`
- `episodes_total`
- `successes_total`
- `failures_total`
- `replayable_episodes_total`
- `per_use_case`

For each use case in `per_use_case`, record at minimum:

- `attempted`
- `successes`
- `failures`
- optional `near_miss`
- `distinct_seeds`
- `successful_seeds`
- `golden_seed_count`
- references to trace / output roots when possible

## DAgger / Advice Requirements

This run should not rely on blind harvesting alone. It must leave a reusable correction path.

At minimum, the repo should end with:

- a harvesting entrypoint that can run fresh evals and save traces
- an advice / correction mechanism or artifact format that captures when the base policy drifted and what the corrective guidance was
- machine-readable metadata showing which episodes were harvested under plain policy vs corrected / advised policy when that information exists

The exact implementation can vary, but the run must improve the loop itself as it learns what works.

## Demo-Web Code Use

Use the Autocinema demo-web code inside the operator repo to:

- understand intended page flows
- understand what constitutes task success
- identify stable local workflows
- diagnose why specific seeds or use cases fail

Do not use it to bake in one-off site hacks that only work for a single prompt.

## Recommended Workflow

1. Keep `python check.py` and the local eval path healthy.
2. Inspect the current harvest and identify weakest use cases.
3. Improve the harvest loop itself before burning large eval batches.
4. Use small focused runs on weak use cases and seeds.
5. When the base policy drifts, inject advice / corrected continuation rather than only collecting repeated failures.
6. Keep building the trace-backed dataset and summary as the single source of truth.
7. Repeat until every use case reaches 10 successful distinct-seed trajectories.

## Definition Of Done

The project is done only when all `.arbos/tests/` pass.

That means:

- branch discipline is correct
- the harvest flow is reproducible
- all 16 use cases are present
- every use case has at least 10 successful trajectories from distinct seeds
- the dataset contains failures too
- all committed episodes are replayable
- provenance and golden seeds are saved
- the harvesting loop itself is improved enough to support further DAgger/SFT/RL work

        ## Current State

        ## Working Memory

Branch remains `arbos`. New policy helpers: `_extract_seed_from_url`, `_extract_prompt_title_literal`, `_page_mentions_title`, `_preferred_seed_stable_navigation`. Regression tests are in `tests/test_policy_exemplars.py` and pass with `tests/test_autocinema_harvest.py`. Focus next on the post-login/add-to-watchlist continuation: the live probe improved step 0/1 routing (login/search path instead of featured-detail loop) but later drifted to unrelated nav like `/contact`. Inspect the fresh runtime behavior for `ADD_TO_WATCHLIST`, then extend deterministic local-workflow guards after auth/search so the agent stays on movie-finding and watchlist controls. Dataset gate is still far from target: `summary.json` remains below the required 160 successes, so `.arbos/tests/test_03_eval_migration_progress.py` would still fail.

## Recent History

- step 1: Last supervisor decision: DENY | outcome=TESTS_FAIL. | check_ok=False | codex_rc=0
- step 2: Last supervisor decision: DENY | outcome=TESTS_FAIL. | check_ok=False | codex_rc=0
- step 3: Last supervisor decision: DENY | outcome=TESTS_FAIL. | check_ok=False | codex_rc=0
- step 4: Last supervisor decision: DENY | outcome=TESTS_FAIL. | check_ok=False | codex_rc=0
- step 5: Last supervisor decision: DENY | outcome=TESTS_FAIL. | check_ok=False | codex_rc=0

        ## Previous Check Result

        verdict=TESTS_FAIL rc=1
decision=deny



        Return JSON with:
        - `summary`: what changed this step
        - `state_update`: concise handoff for the next step
        - `supervisor_message`: one short status line for the human
        - `done`: true only if the spec is satisfied or you believe only verification remains
