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
        - Use `Current State` as the main continuity source. Do not re-read the whole repo or full spec unless you are blocked or the state is clearly stale.
        - Do not edit files under `loop/runs/`, `loop/STATUS.json`, or `loop/response.schema.json`.
        - You may edit the project to satisfy the spec.
        - Use `tests/` as the acceptance gate.
        - Be practical. Make progress each step.
        - You may pursue multiple closely related goals in one step when they share context and reduce re-reading overhead.
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

        11

        ## Inbox

        (empty)

        ## Spec

        # SPEC

## Mission

Turn `autoppia_operator` into a strong Autocinema trajectory-harvesting and correction loop.

The goal is not just to "have a harvest script". The goal is to leave the repo with a reproducible system that can drive Autocinema use case by use case, use ev


## Target Repo And Branch Discipline

- target repo: `/home/usuario1/daryxx/autoppia/operator/autoppia_operator`
- work only on target repo branch `arbos`
- that branch must start from `main`
- do not silently switch back to `daryxx`


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

        ## Current State

        ## Working Memory

Harvest target is currently satisfied in the committed dataset and the repo is green. The one runtime issue seen during a fresh focused watchlist probe was local demo-web backend event/reset access on `localhost:8090`; if another iteration is needed, debug that environment path first rather than the summary artifacts.

## Current Blockers

(none)

## Next Best Actions

- Continue from the current best hypothesis without re-reading the whole repo.

## Recent History

- step 5: Last supervisor decision: DENY | outcome=TESTS_FAIL. | check_ok=False | codex_rc=0
- step 6: Last supervisor decision: DENY | outcome=TESTS_FAIL. | check_ok=False | codex_rc=0
- step 7: Last supervisor decision: DENY | outcome=TESTS_FAIL. | check_ok=False | codex_rc=0
- step 9: Extended `scripts/autocinema_harvest.py` with a demo-seedpack bootstrap path that emits replayable per-use-case eval results, trace indices, trace files, and correction/advice metadata, then rebuilt `data/autocinema_trajectory_harvest/` and refreshed the SFT export. Updated `.gitignore` so the harvest dataset can live in the repo instead of staying hidden under the global `data/` ignore. The rebuilt summary now reports 176 replayable episodes with 160 successes and 16 retained failures, and every Autocinema use case has 11 attempts, 10 distinct successful seeds, and 10 golden seeds. | outcome=Tests passed. | check_ok=True | codex_rc=0
- step 8: Patched `src/operator/agents/step_engine/policy.py` so Autocinema detail-page intents can recover from sparse candidate extraction: title checks now also inspect raw markup, and the policy can click stable movie-detail controls directly from `snapshot_html`/`html` using deterministic selectors for watchlist, trailer, and share actions. Added regression coverage in `tests/test_policy_exemplars.py` and `tests/test_fsm_operator.py` for both fallback and normalization paths, then verified with `pytest -q tests/test_policy_exemplars.py tests/test_fsm_operator.py tests/test_autocinema_harvest.py` (`144 passed`). While validating, the committed harvest artifacts were refreshed and now satisfy the dataset gate; `ARBOS_TARGET_REPO=/home/usuario1/daryxx/autoppia/operator/autoppia_operator bash .arbos/tests/test_01_repo_contract.sh && ... test_02_daryxx_eval_smoke.py && ... test_03_eval_migration_progress.py` all passed, and `data/autocinema_trajectory_harvest/summary.json` now reports `successes_total=160` with `ADD_TO_WATCHLIST=10` successes. | outcome=Tests passed. | check_ok=True | codex_rc=0
- step 10: Hardened the Autocinema step-engine for weak mutation flows by making auth-gated tasks prefer the capability-gap login/register transition and by suppressing generic home-link and related-card clicks in title-focused detail routing. Added regression coverage for the watchlist auth transition and off-target title-click suppression. Re-ran the full test suite successfully (`225 passed`) and verified the committed harvest artifacts already satisfy the dataset target: `data/autocinema_trajectory_harvest/summary.json` reports `160` successes total with all 16 use cases at `10` distinct-seed successes, plus replayable failures and matching manifest/golden files. | outcome=Tests passed. | check_ok=True | codex_rc=0

        ## Previous Check Result

        Last supervisor decision: ALLOW



        Return JSON with:
        - `summary`: what changed this step
        - `state_update`: concise handoff for the next step
        - `supervisor_message`: one short status line for the human
        - `done`: true only if the spec is satisfied or you believe only verification remains
