# Agent Prompt

You are repairing `autoppia_operator` after a large upstream refactor in the sibling `autoppia_iwa` repo.

Focus on the shortest path to restoring:

1. operator contract health
2. IWA compatibility
3. local eval execution
4. at least one measured successful task

Work in that order unless concrete evidence shows a better order.

Rules:

- Stay on `daryxx`.
- Prefer fixing real imports, paths, and interfaces over adding compatibility hacks.
- Ignore old RL/training/meta ambitions unless they block the migration goal.
- After deterministic compatibility passes, run the smallest real eval that can produce one genuine success and write `data/eval_migration_smoke.json`.
- Be precise about failure causes. If eval fails, identify the exact import/path/interface break and fix that.

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
        Review progress against the migration goal only.

This project is not successful because it has more training files or more abstractions.
It is successful only if operator contract health, `autoppia_iwa` compatibility, eval execution, and measured task success are restored on `daryxx`.

        ## Iteration

        1

        ## Inbox

        Start with the deterministic migration gates in `.arbos/tests/`.

Do not chase the old training/meta plan. First make the operator and eval work on `daryxx` against the current `autoppia_iwa` `daryxx` checkout, then get one real measured task success into `data/eval_migration_smoke.json`.

        ## Spec

        # SPEC

## Mission

Restore `autoppia_operator` on the `daryxx` branch so it works against the refactored sibling repo `autoppia_iwa` on its `daryxx` branch.

This is not an open-ended training project. The immediate objective is migration and recovery:

1. The operator contract must work again.
2. The evaluator must work again.
3. The repo must show measured evidence of at least one real successful task after the migration.

## Hard Constraints

- Work only against the code currently checked out in:
  - operator repo: `/home/usuario1/daryxx/autoppia/operator/autoppia_operator`
  - IWA repo: `/home/usuario1/daryxx/autoppia/operator/autoppia_iwa`
- The operator repo must stay on branch `daryxx`.
- The IWA repo must stay on branch `daryxx`.
- Do not spend loop cycles on old RL/training/meta ambitions unless they are directly required to get operator/eval working again.
- Do not weaken acceptance by replacing real compatibility with mocks or permanent fake stubs.
- If the tests or this spec become stale or clearly miscalibrated, the Manager is allowed to rewrite them.

## Priority Order

### 1. Restore the repo contract

Make sure the operator repo is a healthy miner-style repo again:

- `python check.py` passes
- `main.py` exports `app`
- `/health` and `/act` remain valid
- obvious import/runtime failures are fixed

Likely files:

- `main.py`
- `agent.py`
- `llm_gateway.py`
- `check.py`
- `requirements.txt`

### 2. Restore `autoppia_iwa` compatibility on `daryxx`

Fix code that still assumes the old IWA layout. Update imports, module paths, interfaces, and helper assumptions so the operator uses the current sibling `autoppia_iwa` checkout instead of stale structures.

Likely files:

- `agent.py`
- `eval.py`
- any helper modules imported by them

### 3. Make local eval runnable again

`eval.py` must be able to start and run in the current repo layout. Fix path assumptions, task cache discovery, imports, and runtime wiring as needed.

At minimum, the evaluator should:

- import cleanly
- accept `--help`
- be able to run a real targeted eval command from this repo

### 4. Produce measured migration evidence

After the contract/import/eval issues are fixed, run a real targeted eval and save the result in:

- `data/eval_migration_smoke.json`

That file must capture:

- `branch`: `daryxx`
- `iwa_branch`: `daryxx`
- `measured`: `true`
- `num_tasks`: integer >= 1
- `successes`: integer >= 1
- `success_rate`: numeric
- `command`: the eval command that was run
- `timestamp`
- optional notes such as the use case or task ids that passed

Use the smallest real eval that can prove success. A 1-3 task targeted run is acceptable if it is real and measured.

## Recommended Workflow

1. Run the deterministic tests in `.arbos/tests/` first.
2. Fix contract/import/path breakage before doing broader behavior tuning.
3. Once eval runs, use a narrow targeted eval to get one real success on the board.
4. Record the measured result in `data/eval_migration_smoke.json`.
5. Only after the migration baseline is restored should you widen scope.

## Definition Of Done

The project is done only when all `.arbos/tests/` pass, including the measured eval evidence test.

That means:

- contract healthy
- branch alignment correct
- IWA compatibility restored
- eval runnable
- at least one real measured task success recorded

        ## Current State

        (empty)

        ## Previous Check Result

        (no previous check output)



        Return JSON with:
        - `summary`: what changed this step
        - `state_update`: concise handoff for the next step
        - `supervisor_message`: one short status line for the human
        - `done`: true only if the spec is satisfied or you believe only verification remains
