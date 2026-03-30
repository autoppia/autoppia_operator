# sn36 Tools

This folder contains Codex operator helper tools to reduce the manual steps in subnet36 iteration.

## `bittensor_tools.py`

Run Bittensor SDK actions for subnet36 metagraph visibility.

- `metagraph`
  - `python tools/bittensor_tools.py metagraph`
  - Get latest metagraph snapshot (uid/hotkey/incentive/stake).
  - Uses `SN36_NETUID`, `SN36_NETWORK`, and optional `SN36_CHAIN_ENDPOINT`.

- `uid`
  - `python tools/bittensor_tools.py uid --uid 123`
  - `python tools/bittensor_tools.py uid --hotkey ${SN36_HOTKEY}`
  - Resolve hotkey/uid and incentive metadata.

- `uid-stats`
  - `python tools/bittensor_tools.py uid-stats --uid 123`
  - Pull distribution metrics for the given UID.

- `my-miner`
  - `python tools/bittensor_tools.py my-miner --hotkey ${SN36_HOTKEY}`
  - Resolve the local hotkey to UID(s) using `SN36_HOTKEY`.

## `iwap_tools.py`

Mock-first IWAP helper for getting trend data while the endpoint implementation is being finalized.

- `last-round`
  - `python tools/iwap_tools.py last-round`
  - Show the latest round payload from mock data.

- `season-results`
  - `python tools/iwap_tools.py season-results --season-id 36`
  - Aggregate per-UID season performance (scores and success).

- `top-uids`
  - `python tools/iwap_tools.py top-uids --limit 20`
  - Leaderboard of avg scores across rounds (mock).

- `rounds`
  - `python tools/iwap_tools.py rounds --limit 5`
  - List the most recent rounds (mock, descending by default).

- `season-tasks`
  - `python tools/iwap_tools.py season-tasks --season-id 36`
  - Pull per-task rows for a season (UID, task id, score).
  - `python tools/iwap_tools.py season-tasks --season-id 36 --uid 11 --min-score 0.6 --limit 20`
  - In this repo, `--base-url` plus `--prefer-live` attempts live IWAP endpoint; if live fetch fails, it falls back to mock data.

## Mock data

Mock data is embedded in `iwap_tools.py` and can be overridden with:

- `IWAP_MOCK_DATA=/abs/path/custom_iwap.json`

Use this to test the automation loop until the production IWAP endpoints are ready.

## MCP wrapper (`mcp_server.py`)

Use this repo-local MCP server to expose the tools above to Codex-compatible clients.

- Start stdio server:
  - `python tools/mcp_server.py`
- Print MCP tool specs (JSON payload):
  - `python tools/mcp_server.py --list-tools`
- Optional repo root override:
  - `python tools/mcp_server.py --root /abs/path/to/autoppia_operator`

Example MCP client config:

```json
{
  "mcpServers": {
    "miner_mcp": {
      "command": "/usr/bin/python3",
      "args": ["/home/usuario1/autoppia/operator/autoppia_operator/tools/mcp_server.py"],
      "env": {
        "SN36_NETUID": "36",
        "SN36_NETWORK": "finney",
        "SN36_HOTKEY": "your_hotkey"
      }
    }
  }
}
```

## Codex bootstrap check

When onboarding a new fork:

```bash
python tools/mcp_server.py --list-tools
python tools/mcp_server.py --root /home/usuario1/autoppia/operator/autoppia_operator
```

Use the exact server key `miner_mcp` in your MCP settings. This repo avoids hardcoded secrets and reads credentials from env.

The MCP wrapper exposes these method handlers:

- `initialize`
- `tools/list`
- `tools/call`

Tool-call names are namespaced, e.g. `bittensor.metagraph`, `iwap.season-tasks`, and `sn36.cycle`.

## `runpod_tools.py`

RunPod pod and account tooling for operator automation:

- `list-pods`
  - `python tools/runpod_tools.py list-pods`
- `get-pod --pod-id <pod_id>`
  - `python tools/runpod_tools.py get-pod --pod-id <pod_id>`
- `create-pod --name ... --gpu-type-id ...`
  - `python tools/runpod_tools.py create-pod --name miner-dev --gpu-type-id <GPU_TYPE_ID> --allow-side-effects`
- `stop-pod --pod-id <pod_id> --allow-side-effects`
- `resume-pod --pod-id <pod_id> --gpu-count 1 --allow-side-effects`
- `terminate-pod --pod-id <pod_id> --allow-side-effects`
- `get-balance`
  - `python tools/runpod_tools.py get-balance`
- `list-gpu-types`
  - `python tools/runpod_tools.py list-gpu-types`
- `graphql --query '<graphql>'`
  - direct GraphQL (mutations require `--allow-side-effects`)

Authentication: set `RUNPOD_API_KEY` in environment. No token values are embedded in repo.

## `smtp_tools.py` (email notifications)

Basic email tooling for operator/agent notification loops.

- `check`
  - `python tools/smtp_tools.py check`
  - Validate SMTP connectivity from configured env.
- `send`
  - `python tools/smtp_tools.py send --to "you@example.com" --subject "Update" --body "Done"`
  - Use `--cc`, `--bcc`, `--html`, or `--dry-run` for testing.

SMTP settings are env-based:

- `SMTP_HOST`
- `SMTP_PORT`
- `SMTP_USERNAME`
- `SMTP_PASSWORD`
- `SMTP_FROM`
- `SMTP_DEFAULT_TO`
- `SMTP_USE_SSL`
- `SMTP_USE_TLS`
- `SMTP_TIMEOUT`

MCP names:
- `smtp.send`
- `smtp.check`
