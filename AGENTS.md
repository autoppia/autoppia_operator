# AGENTS.md - sn36 Auto-Mining Playbook (daryxx)

This repository is the operator side of Autoppia Subnet 36. This guide explains how Codex should behave when iterating code and preparing submissions for subnet36.

## Repo assumptions

- The validator requires `POST /act`, `GET /health`, and a valid `/act` payload.
- Source of truth for runtime contracts is `main.py`, `agent.py`, `check.py`, `scripts/deploy_check.py`, and `README.md`.
- Any change should keep `requirements.txt` aligned with sandbox expectations.
- Branch focus: `daryxx`.

## Concept map: what this repo contributes to sn36

- `agent.py` receives validator tasks and returns IWA action payloads.
- `autoppia_iwa` is used for planning, action validation, and evaluation scoring.
- The validator judges the remote repo behavior by cloning the branch from `GITHUB_URL`, so miner updates must be committed and pushed.
- The local score does not automatically equal subnet score, but local eval is the required first gate.
- In this repo, Codex controls workflow orchestration and submission; it does not host on-chain payment/eval-count APIs.

## Review of subnet36 operational flow (runtime-level view)

At a high level, the lifecycle is:

1. Miner edits code in their forked repo and commits changes.
2. Miner pushes the commit (and optionally passes a specific `--github` pointer).
3. Miner calls `autoppia-miner-cli submit` to update committed identity metadata (`agent_name`, `github_url`, optional `agent_image`) for their hotkey on subnet36.
4. Subnet validator picks up the miner record from the subnet metagraph and schedules evaluator runs from `autoppia_web_agents_subnet` task queues.
5. Validator clones/instantiates that exact repo reference and runs the eval pipeline against requested tasks/use-cases or web projects.
6. Evaluator emits per-task artifacts (success, error, score, traces) to IWAP; IWAP stores rounds and run logs.
7. Subnet contract uses evaluation outputs to compute and post incentives; chain incentive is then reflected in metagraph stats.
8. Miner observes:
   - on-chain peer comparison via `bittensor_tools` (`metagraph`, `my-miner`, `uid`, `uid-stats`);
   - off-chain score trend via IWAP (`iwap.*` tools).

Practical implication for operators:
- `scripts/sn36_ops.py` and MCP tools automate **local validation + metadata submission**, not direct payment history extraction.
- If you need true validator-level eval counts, consult subnet runtime or dashboard APIs in the runtime repo/API; this repo currently models only subset telemetry through IWAP and on-chain metagraph data.

### Review of `autoppia_web_agents_subnet` runtime flow (practical)

- `autoppia-miner-cli submit` updates on-chain miner metadata only (`agent_name`, `github_url`, optional `agent_image`).
- The validator reads this metadata from metagraph state, then queues and executes eval tasks from subnet runtime project/use-case/task configuration.
- Evaluations are run against the exact repo reference (URL/ref) submitted to ensure deterministic reproducibility.
- Evaluation workers submit run/task outputs to IWAP (`validator-rounds`, `agent-runs`, `task-logs` style endpoints).
- Incentive accounting happens in validator/subnet runtime logic after run completion, then appears on-chain in metagraph incentive fields.

## Strict operator SOP (preflight → eval → cycle → submit → IWAP verification)

1. `python scripts/sn36_ops.py preflight`
2. `python scripts/sn36_ops.py eval --project-id <project-id> --success-threshold 0.70 --avg-score-threshold 0.60`
3. `python scripts/sn36_ops.py cycle --github-url <repo/tree/or/commit> --agent-name <agent-name> --submit`  
   (runs preflight + eval and only submits on gate pass)
4. `python tools/bittensor_tools.py my-miner`
5. `python tools/iwap_tools.py last-round`
6. `python tools/iwap_tools.py season-tasks --season-id 36`
7. Decision:
   - pass trend: continue iterating with smaller targeted task scope
   - degrade trend: revert/fix and re-run from step 2

Hard constraints:
- No submission without passing eval gate.
- No secret values in logs or file outputs.
- Use explicit user confirmation before any branch reset/force/push in ambiguous states.

Ready-to-run Codex orchestration prompt:

```text
Run the subnet36 strict SOP. Use local wallet variables SN36_HOTKEY and SN36_COLDKEY only.
Execute preflight, then local eval using current user scope.
If eval passes, execute cycle with submit using provided github reference.
After submit, run my-miner and IWAP verification (last-round + season-tasks).
Report decision state: SUBMIT_OK, WAIT_FOR_ROUND, or REVISE_BEFORE_RESUBMIT.
```

## Concept map: subnet36 workflow

- Reference repos:
  - Autoppia subnet runtime: https://github.com/autoppia/autoppia_web_agents_subnet
  - IWAP dashboard backend: https://github.com/autoppia/autoppia_bittensor_dashboard_backend
  - Product/ops entrypoint: https://autoppia.com

- Miner metadata is committed on-chain via `autoppia-miner-cli submit`.
- `AGENT_NAME`, `GITHUB_URL` and optional `AGENT_IMAGE` are the required committed fields.
- Handshake behavior is defined by `AGENT_NAME` and `GITHUB_URL` only. If missing or invalid, validator score can drop to zero.
- Recommended `GITHUB_URL` form: explicit `tree/<ref>` or `commit/<sha>`.
- `autoppia-miner-cli` defaults target netuid 36 on `finney`.

## Concept map: IWAP and why it matters

IWAP is the dashboard/round telemetry used by subnet tooling.

- The subnet validator pushes batch evaluations and task logs to IWAP at round boundaries.
- You use IWAP to confirm real scoring trends (task success and score progression) after validator evaluations.
- We use the existing repo contract in `autoppia_web_agents_subnet` where current paths include:
  - `POST /api/v1/validator-rounds/auth-check`
  - `POST /api/v1/validator-rounds/start`
  - `POST /api/v1/validator-rounds/{validator_round_id}/tasks`
  - `POST /api/v1/validator-rounds/{validator_round_id}/agent-runs/start`
  - `POST /api/v1/validator-runs/{run}/evaluations/batch`
- `POST /api/v1/task-logs`
- `POST /api/v1/validator-rounds/{validator_round_id}/finish`

Current local tooling status:
- IWAP calls are modeled as mock-first (`tools/iwap_tools.py`) until the backend endpoints are stabilized.
- Bittensor tooling (`tools/bittensor_tools.py`) is live and reads on-chain metagraph data for subnet36.
- Miner-CLI wrappers in this repo do not call any payment API and do not provide per-validator eval counts directly.

## Concept map: autoppia.com context

`autoppia.com` is the business-facing domain the agent may operate in many flows. The key constraint is strict protocol compliance in `/act` and stable task execution, not UI-level coupling.

## Commands and tools available for Codex

- `python check.py`
  - Hard repo gate: `main` app and endpoint contracts.
- `python scripts/deploy_check.py`
  - Local contract + subnet handshake env checks.
- `python scripts/sn36_ops.py preflight`
  - Run both checks with one command.
- `python scripts/sn36_ops.py eval`
  - Run local eval and apply pass/fail gates.
- `python scripts/sn36_ops.py iwap --base-url <IWAP API base URL>`
  - Pull recent IWAP run/task/log metrics.
- `python scripts/sn36_ops.py submit`
  - Call `autoppia-miner-cli submit` with validated inputs.
- `python scripts/sn36_ops.py cycle`
  - Run eval gate -> optional submit -> optional IWAP probe.
- `bash scripts/sn36_submit.sh`
  - Shell wrapper for quick miner-cli submission with environment defaults.
- `bash scripts/sn36_metaanchor.sh`
  - Metagraph read helper, prints local hotkey UID and top incentive peers.
- `python tools/bittensor_tools.py metagraph`
  - Fetch current metagraph snapshot for subnet36 and inspect UID/hotkey/incentive.
- `python tools/bittensor_tools.py my-miner`
  - Resolve local miner hotkey (`SN36_HOTKEY`) to UID and inspect incentive/stake.
- `python tools/bittensor_tools.py uid --hotkey <hotkey> --netuid 36`
  - Explicitly resolve a hotkey to UID and stats.
- `python tools/bittensor_tools.py uid-stats --uid <uid>`
  - Get incentive/stake distribution context for a specific UID.
- `python tools/iwap_tools.py last-round`
  - Get latest mock IWAP result (no API required now).
- `python tools/iwap_tools.py season-results --season-id 36`
  - Aggregate mock IWAP scores and success by UID for a season.
- `python tools/iwap_tools.py top-uids`
  - Get mock leaderboard over available rounds.
- `python tools/iwap_tools.py rounds --limit 5`
  - View mock recent IWAP rounds.
- `python tools/iwap_tools.py season-tasks --season-id 36`
  - Return per-task season rows (uid/task_id/score), with live fallback when base URL is provided.

## Codex prompt to start or resume auto-mining

Use this exact prompt format in an agent run when onboarding a miner repo:

```text
You are a Subnet 36 operator agent for Autoppia.
Environment:
- branch: daryxx
- subnet: 36
- network: finney
- local keys are configured as SN36_COLDKEY and SN36_HOTKEY in the env and must not be printed.

Goal: iterate on repo code, run local eval, and submit only when safe.

Procedure:
1) run preflight and local checks: python scripts/sn36_ops.py preflight
2) run evaluator: python scripts/sn36_ops.py eval with requested scope (project/use-case/task/json/repeat/concurrency)
3) if pass criteria hold, run:
   python scripts/sn36_ops.py submit --github-url <repo/tree/or/commit> --agent-name <name> --agent-image <url optional>
4) confirm on-chain identity + metagraph mapping:
   python tools/bittensor_tools.py my-miner
5) optionally run IWAP check:
   python tools/iwap_tools.py last-round
   python tools/iwap_tools.py season-tasks --season-id 36
6) if trend improves, continue with another cycle; if not, revert, fix, re-run eval, and resubmit.

Constraints:
- never print or persist secrets.
- do not call miner submission when local eval fails.
- never mutate branch or call submit without explicit user confirmation when in uncertain state.
```

## How evaluator scope can be configured locally

- Use all project tasks (default): `python scripts/sn36_ops.py eval`
- Restrict by web project: `--project-id <project_id>`
- Restrict by use case: `--use-case <LOGIN|SEARCH|...>` or `--all-use-cases`
- Run one task from file: `--task-id <task_id>`
- Run custom task set via JSON: `--task-cache /abs/path/tasks.json`
- Control repetitions: `--repeat N`
- Control concurrency: `--task-concurrency N`
- Use explicit gates: `--success-threshold`, `--avg-score-threshold`, `--max-cost`
- Optional round/season labels for miner-cli submit: `--season`, `--target_round`

Example:

```bash
python scripts/sn36_ops.py eval \
  --project-id autocinema \
  --use-case LOGIN \
  --task-cache ./tasks.json \
  --repeat 2 \
  --task-concurrency 2 \
  --success-threshold 0.7 \
  --avg-score-threshold 0.6
```

## MCP tooling (Codex agent bridge)

- On forked users: copy `.env.template` (or `.env.example`) to `.env`, fill values, then run MCP in that directory.
- The MCP server will auto-load `.env` when present.

- MCP server script: `python tools/mcp_server.py`
- Machine-readable tool inventory: `python tools/mcp_server.py --list-tools`

The MCP server dispatches to existing tools in this repo and is suitable for codex-driven automation:

- Bittensor: `bittensor.metagraph`, `bittensor.uid`, `bittensor.uid-stats`, `bittensor.my-miner`
- IWAP: `iwap.last-round`, `iwap.season-results`, `iwap.top-uids`, `iwap.rounds`, `iwap.season-tasks`
- sn36: `sn36.preflight`, `sn36.eval`, `sn36.submit`, `sn36.cycle`, `sn36.iwap`
- runpod: `runpod.list_pods`, `runpod.get_pod`, `runpod.create_pod`, `runpod.stop_pod`, `runpod.resume_pod`, `runpod.terminate_pod`, `runpod.balance`, `runpod.list_gpu_types`, `runpod.graphql`
- smtp: `smtp.send`, `smtp.check`

RunPod safety rules:
- Set `RUNPOD_API_KEY` in environment only.
- Mutating tools (`runpod.create_pod`, `runpod.stop_pod`, `runpod.resume_pod`, `runpod.terminate_pod`, and `runpod.graphql` mutations) require `allow_side_effects=true` in tool arguments.

SMTP notification flow:
- Configure SMTP keys in `.env`/`.env.example`:
  - `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `SMTP_FROM`
  - `SMTP_DEFAULT_TO`, `SMTP_USE_SSL`, `SMTP_USE_TLS`, `SMTP_TIMEOUT`
- `smtp.send` sends email. Use `smtp.check` before critical notifications.
- Use `--dry-run` to validate payload and recipients without sending.

Use this configuration shape with a local agent client:

```json
{
  "mcpServers": {
    "miner_mcp": {
      "command": "python",
      "args": ["/home/usuario1/autoppia/operator/autoppia_operator/tools/mcp_server.py"],
      "env": {
        "SN36_HOTKEY": "(local hotkey from wallet)",
        "SN36_COLDKEY": "(local coldkey from wallet)",
        "SN36_NETWORK": "finney",
        "SN36_NETUID": "36"
      }
    }
  }
}
```

## How to use `autoppia-miner-cli` (dev branch flow)

- Source of behavior is the subnet runtime workflow in `autoppia_web_agents_subnet` (dev branch expectations for submit fields shown here are stable in this repo tooling).
- Commit metadata fields required by validator scoring are `github_url` and `agent_name` (with optional `agent_image`).
- Always run with local wallet env set (do not hardcode secrets):
  - `SN36_HOTKEY`
  - `SN36_COLDKEY`

Submission command used by operator scripts:

```bash
autoppia-miner-cli submit \
  --github "<GITHUB_URL>" \
  --agent.name "<AGENT_NAME>" \
  --wallet.name "${SN36_COLDKEY}" \
  --wallet.hotkey "${SN36_HOTKEY}" \
  --subtensor.network "${SN36_NETWORK}" \
  --netuid "${SN36_NETUID}" \
  [--agent.image "<AGENT_IMAGE>"] \
  [--subtensor.chain_endpoint "${SN36_CHAIN_ENDPOINT}"] \
  [--season N] [--target_round N]
```

The wrapper entry points map to:

- `python scripts/sn36_ops.py submit --github-url ... --agent-name ...` (single-shot submit)
- `python scripts/sn36_ops.py cycle --github-url ... --agent-name ... --submit` (preflight + eval gate + submit)

Quick shell helper:

- `bash scripts/sn36_submit.sh` (reads `SN36_*` env vars and runs `autoppia-miner-cli submit`)

### Direct answer: miner-cli payment / per-validator counts on dev branch

- This repository’s `sn36_ops.py` + MCP wrappers do **not** call any miner-cli endpoint or SDK function for
  - payment history,
  - per-validator assigned eval count,
  - per-validator reward breakdown.
- For those views you must inspect the subnet runtime/API that executes validator-side evaluator jobs.
- In operator practice, use:
  - chain-level incentive signals via `bittensor_tools` (`uid-stats`), and
  - IWAP run/task metrics via `iwap.season-tasks` / `iwap.last-round`.

## Local key policy (required by user)

- Codex must assume coldkey/hotkey are already configured locally.
- Do not write secrets into repo files.
- Prefer using environment overrides over prompts for:
  - `SN36_COLDKEY` (default: `default`)
  - `SN36_HOTKEY` (default: `default`)
  - `SN36_NETUID` (default: `36`)
  - `SN36_NETWORK` (default: `finney`)
- Use explicit values in command output; never guess key names from code changes.

## Codex automation loop

1. `python scripts/sn36_ops.py preflight`
2. `python scripts/sn36_ops.py eval --project-id autocinema --num-tasks 10 --success-threshold 0.70 --avg-score-threshold 0.60`
   - `--project-id` limits by web project.
   - `--use-case LOGIN` or `--all-use-cases` for use-case selection.
   - `--task-id <id>` runs one explicit task.
   - `--repeat N` executes each task N times.
   - `--task-cache /abs/path/tasks.json` evaluates against a custom task JSON.
   - `--task-concurrency N` controls concurrent episodes.
3. If pass: `python scripts/sn36_ops.py cycle --github-url <repo/tree/or/commit> --submit`
4. Poll IWAP metrics with `python scripts/sn36_ops.py iwap` or `python tools/iwap_tools.py last-round`.
5. Resolve current miner UID/incentive with `python tools/bittensor_tools.py my-miner`.
6. If IWAP and local trends improve, continue; if regress, revert and stop iteration.

## Decision thresholds (defaults)

Defaults are intentionally conservative.

- Success rate threshold: `0.70`
- Average score threshold: `0.60`
- Over-cost and error spikes require human review; do not keep iterating on repeated hard failures.

## Safe automation rules

- No credential values in logs.
- If local eval fails, do not call submit.
- Use dry run mode for any action that mutates branch or submits to chain.
- Require explicit user confirmation before changing wallet-sensitive behavior.

## Environment variables used by tooling
- `IWAP_BASE_URL`, `IWAP_API_TOKEN` (IWAP reads)
- `SN36_COLDKEY`, `SN36_HOTKEY`, `SN36_NETUID`, `SN36_NETWORK`
- `SN36_CHAIN_ENDPOINT`, `SN36_GITHUB_URL`, `SN36_AGENT_NAME`, `SN36_AGENT_IMAGE`
- `SN36_TARGET_ROUND`, `SN36_SEASON`, `SN36_MINER_CLI`, `SN36_TOP_NODES`
- `IWAP_MOCK_DATA` (override mock IWAP payload path used by `tools/iwap_tools.py`)
- `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `SMTP_FROM`, `SMTP_DEFAULT_TO`, `SMTP_USE_SSL`, `SMTP_USE_TLS`, `SMTP_TIMEOUT`
- `RUNPOD_API_KEY` (RunPod tooling)

## Fork users: ready-to-copy Codex MCP JSON

```json
{
  "mcpServers": {
    "miner_mcp": {
      "command": "python",
      "args": ["/home/usuario1/autoppia/operator/autoppia_operator/tools/mcp_server.py"],
      "env": {
        "SN36_NETWORK": "finney",
        "SN36_NETUID": "36",
        "SN36_COLDKEY": "<your-coldkey-name>",
        "SN36_HOTKEY": "<your-hotkey-name>",
        "SN36_CHAIN_ENDPOINT": "wss://entrypoint-finney.opentensor.ai:443",
        "IWAP_BASE_URL": "https://api-leaderboard.autoppia.com",
        "IWAP_API_TOKEN": "<optional>",
        "RUNPOD_API_KEY": "<optional>"
      }
    }
  }
}
```

Security guardrail: keep `.env` secrets out of VCS and share only placeholder MCP JSON.

### Quick Codex bootstrap (operator copy/paste)

For fork users, this is the shortest path:

```text
1) cp .env.template .env
2) fill SN36_COLDKEY and SN36_HOTKEY (and IWAP/MCP keys if used)
3) configure MCP key `miner_mcp` in client
4) run:
   python tools/mcp_server.py --list-tools
   python scripts/sn36_ops.py preflight
   python scripts/sn36_ops.py eval --project-id autocinema
5) ask Codex for strict SOP execution:
   preflight -> eval -> cycle/submit -> my-miner -> IWAP verification
```

### Environment ambiguity guardrails (important)

- `SN36_COLDKEY`/`SN36_HOTKEY` must match the local wallet loaded on the miner node.
- `IWAP_MOCK_DATA` is an explicit local override and should be left empty for real IWAP reads.
- `tools/iwap_tools.py` is mock-first:
  - mock data is used by default,
  - live IWAP data is used only when the call is made with explicit live preference.
- If you set both local mock override and live URL, local mock wins unless the wrapper opts into live fetch first.
- Never hardcode secrets; share only placeholder values in MCP JSON and docs.

## Useful constants to inspect while coding

- `scripts/deploy_check.py` for required route/payload checks.
- `agent.py` and `main.py` for endpoint contracts.
- `eval.py` for local eval score fields and output keys.
- `training/iwap_client.py` and `training/pipeline.py` for IWAP ingestion field names (`runId`, `taskId`, payload URL, metadata score fields).
