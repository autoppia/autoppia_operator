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
