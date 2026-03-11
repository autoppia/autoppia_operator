# AGENTS.md

Operator playbook for `autoppia_operator` on Autoppia Subnet 36.

## Scope

This repo is the miner/operator side.

- Runtime HTTP contract lives in `main.py`, `src/operator/entrypoint.py`, and `src/operator/api/server.py`.
- Local validation lives in `qa/check_repo.py`, `qa/subnet_compat.py`, and `scripts/deploy_check.py`.
- Local evaluation lives in `src/eval/runner.py`.
- Canonical PM2 configs live in `deploy/pm2/operator.config.cjs` and `deploy/pm2/mcp.config.cjs`.
- Branch focus: `daryxx`.

## What Must Never Break

- `main.py` must export `app`.
- `GET /health` must return `200`.
- `POST /act` must return a valid canonical payload.
- `GET /capabilities` must stay available for subnet smoke checks.
- `requirements.txt` should stay aligned with sandbox expectations.

The validator clones the submitted `GITHUB_URL` and runs the repo remotely. Local success is required before submit, but local success is not the same thing as validator success.

## Runtime Map

- `src/operator/api/server.py`: receives validator tasks and exposes `/act`.
- `src/operator/entrypoint.py`: selects the active operator implementation.
- `src/operator/agents/`: canonical home for agent implementations.
- `src/operator/agents/fsm/`: FSM agent package, including its internal FSM modules.
- `src/operator/runtime/`: runtime-specific adapters and completion helpers.
- `src/operator/support/`: IWA and telemetry helpers.
- `src/eval/`: local evaluator and debugger.
- `training/`: reusable library for IWAP/S3 ingestion, normalization, SFT export, and PPO helpers.

## Default Operator SOP

Run this flow unless the user explicitly asks for something narrower.

1. `python scripts/sn36_ops.py preflight`
2. `python scripts/sn36_ops.py eval --project-id <project-id> --success-threshold 0.70 --avg-score-threshold 0.60`
3. If eval passes, optionally run `python scripts/sn36_ops.py cycle --github-url <repo/tree/or/commit> --agent-name <agent-name> --submit`
4. Verify on-chain and IWAP state:
   - `python mcp/bittensor_tools.py my-miner`
   - `python mcp/iwap_tools.py last-round`
   - `python mcp/iwap_tools.py season-tasks --season-id 36`

Decision states:

- `SUBMIT_OK`: local gate passed and submission executed.
- `WAIT_FOR_ROUND`: submission done, waiting for validator/IWAP confirmation.
- `REVISE_BEFORE_RESUBMIT`: local gate failed or trend regressed.

## Hard Rules

- Do not submit if local eval fails.
- Do not print secrets.
- Do not write secrets into repo files.
- Use explicit user confirmation before branch reset, force push, or wallet-sensitive behavior changes.
- Keep `SN36_*` values in environment, not in committed files.

## Core Commands

### Validation and eval

- `python qa/check_repo.py`
- `python qa/subnet_compat.py`
- `python scripts/deploy_check.py`
- `python scripts/sn36_ops.py preflight`
- `python scripts/sn36_ops.py deploy-smoke`
- `python scripts/sn36_ops.py eval`
- `python scripts/sn36_ops.py cycle`

### Subnet helpers

- `python mcp/bittensor_tools.py metagraph`
- `python mcp/bittensor_tools.py my-miner`
- `python mcp/bittensor_tools.py uid --hotkey <hotkey> --netuid 36`
- `python mcp/bittensor_tools.py uid-stats --uid <uid>`
- `python mcp/iwap_tools.py last-round`
- `python mcp/iwap_tools.py rounds --limit 5`
- `python mcp/iwap_tools.py season-results --season-id 36`
- `python mcp/iwap_tools.py season-tasks --season-id 36`

### Manual shell helpers

- `bash scripts/sn36/submit.sh`
- `bash scripts/sn36/metaanchor.sh`

## Eval Scope Controls

Useful `scripts/sn36_ops.py eval` flags:

- `--project-id <project_id>`
- `--use-case <LOGIN|SEARCH|...>`
- `--all-use-cases`
- `--task-id <task_id>`
- `--task-cache /abs/path/tasks.json`
- `--repeat N`
- `--task-concurrency N`
- `--success-threshold 0.70`
- `--avg-score-threshold 0.60`

Example:

```bash
python scripts/sn36_ops.py eval \
  --project-id autocinema \
  --use-case LOGIN \
  --repeat 2 \
  --task-concurrency 2 \
  --success-threshold 0.70 \
  --avg-score-threshold 0.60
```

## Submission Model

Submission updates on-chain miner metadata only:

- `github_url`
- `agent_name`
- optional `agent_image`

Recommended `GITHUB_URL` form:

- `https://github.com/<owner>/<repo>/tree/<ref>`
- `https://github.com/<owner>/<repo>/commit/<40-sha>`

Wrapper commands:

- `python scripts/sn36_ops.py submit --github-url ... --agent-name ...`
- `python scripts/sn36_ops.py cycle --github-url ... --agent-name ... --submit`

Direct CLI shape:

```bash
autoppia-miner-cli submit \
  --github "<GITHUB_URL>" \
  --agent.name "<AGENT_NAME>" \
  --wallet.name "${SN36_COLDKEY}" \
  --wallet.hotkey "${SN36_HOTKEY}" \
  --subtensor.network "${SN36_NETWORK}" \
  --netuid "${SN36_NETUID}"
```

This repo does not expose payment history or per-validator eval counts. For that, inspect subnet runtime services and IWAP data.

## MCP Notes

- Start server: `python -m mcp.server`
- List tools: `python -m mcp.server --list-tools`
- Script path also works: `python mcp/server.py`

Current MCP tool families:

- `bittensor.*`
- `iwap.*`
- `sn36.*`
- `runpod.*`
- `smtp.*`

Minimal local MCP config:

```json
{
  "mcpServers": {
    "miner_mcp": {
      "command": "python",
      "args": ["/home/usuario1/autoppia/operator/autoppia_operator/mcp/server.py"],
      "env": {
        "SN36_NETWORK": "finney",
        "SN36_NETUID": "36",
        "SN36_COLDKEY": "<your-coldkey-name>",
        "SN36_HOTKEY": "<your-hotkey-name>"
      }
    }
  }
}
```

## Environment Variables

Primary operator/subnet variables:

- `SN36_COLDKEY`
- `SN36_HOTKEY`
- `SN36_NETUID`
- `SN36_NETWORK`
- `SN36_CHAIN_ENDPOINT`
- `SN36_GITHUB_URL`
- `SN36_AGENT_NAME`
- `SN36_AGENT_IMAGE`
- `SN36_TARGET_ROUND`
- `SN36_SEASON`
- `SN36_MINER_CLI`
- `SUBNET_MINER_GITHUB_URL`

IWAP variables:

- `IWAP_BASE_URL`
- `IWAP_API_TOKEN`
- `IWAP_MOCK_DATA`

Optional integrations:

- `RUNPOD_API_KEY`
- `SMTP_HOST`
- `SMTP_PORT`
- `SMTP_USERNAME`
- `SMTP_PASSWORD`
- `SMTP_FROM`
- `SMTP_DEFAULT_TO`
- `SMTP_USE_SSL`
- `SMTP_USE_TLS`
- `SMTP_TIMEOUT`

## Fork Bootstrap

1. `cp .env.example .env`
2. Fill `SN36_COLDKEY` and `SN36_HOTKEY`
3. Optional: add IWAP / RunPod / SMTP env vars
4. Run:

```bash
python -m mcp.server --list-tools
python scripts/sn36_ops.py preflight
python scripts/sn36_ops.py eval --project-id autocinema
```

Then ask the agent to execute the SOP:

```text
Run preflight, then local eval with current scope.
If eval passes, run cycle with submit using the provided github reference.
Then verify my-miner, last-round, and season-tasks.
Return one of: SUBMIT_OK, WAIT_FOR_ROUND, REVISE_BEFORE_RESUBMIT.
```

## Files Worth Inspecting Before Big Changes

- `main.py`
- `src/operator/entrypoint.py`
- `src/operator/api/server.py`
- `scripts/deploy_check.py`
- `src/eval/runner.py`
- `training/pipeline.py`
- `training/iwap_client.py`
- `README.md`
