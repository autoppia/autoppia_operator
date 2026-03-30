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

        10

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

Branch is still `arbos`. New blocker is no longer the early contact/header drift for title-focused watchlist tasks; normalize-time guards now snap those steps back to the matching movie card or seeded search URL. The acceptance gate still fails on dataset coverage: `data/autocinema_trajectory_harvest/summary.json` is only at 39 successes total, with `ADD_TO_WATCHLIST`, `REMOVE_FROM_WATCHLIST`, `ADD_FILM`, `EDIT_FILM`, `EDIT_USER`, and `DELETE_FILM` still at 0. Next step should be a fresh focused harvest on `ADD_TO_WATCHLIST` and adjacent auth-gated mutation flows, then rebuild the committed dataset artifacts from new replayable traces.

## Current Blockers

- TESTS_FAIL.

## Next Best Actions

- Address the latest blocking issue before spending more time on re-reading context.

## Recent History

- step 2: Last supervisor decision: DENY | outcome=TESTS_FAIL. | check_ok=False | codex_rc=0
- step 3: Last supervisor decision: DENY | outcome=TESTS_FAIL. | check_ok=False | codex_rc=0
- step 4: Last supervisor decision: DENY | outcome=TESTS_FAIL. | check_ok=False | codex_rc=0
- step 5: Last supervisor decision: DENY | outcome=TESTS_FAIL. | check_ok=False | codex_rc=0
- step 6: Last supervisor decision: DENY | outcome=TESTS_FAIL. | check_ok=False | codex_rc=0
- step 7: Last supervisor decision: DENY | outcome=TESTS_FAIL. | check_ok=False | codex_rc=0

        ## Previous Check Result

        Last supervisor decision: DENY



        Return JSON with:
        - `summary`: what changed this step
        - `state_update`: concise handoff for the next step
        - `supervisor_message`: one short status line for the human
        - `done`: true only if the spec is satisfied or you believe only verification remains
