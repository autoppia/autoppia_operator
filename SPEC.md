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
