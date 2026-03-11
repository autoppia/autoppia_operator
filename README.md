# autoppia_operator (Miner Example)

This repo is a minimal FastAPI web-agent service intended to run as a **miner** in the Autoppia web-agents subnet.

## What the validator runs

The validator starts your container with:

```bash
uvicorn main:app --host 0.0.0.0 --port $SANDBOX_AGENT_PORT
```

So the only hard requirements are:
- `main.py` exports `app`
- `GET /health` returns 200
- `POST /act` returns canonical IWA payload:
  `{ "tool_calls": [...], "content": "...", "done": false, "state_out": {} }`

### Sandbox dependencies / requirements.txt

The subnet validator runs your repo inside a sandbox image with a fixed Python environment.

This repo's `requirements.txt` is intended to be **identical** to the sandbox requirements shipped in the subnet (`autoppia_web_agents_subnet/opensource/sandbox/requirements.txt`).

If you change dependencies here, also update the subnet sandbox image (otherwise your miner may work locally but fail in production).


## Gateway / OpenAI routing

In subnet production, miners **must not** call OpenAI directly. The validator runs a local HTTP proxy (the "sandbox gateway") and injects:

- `OPENAI_BASE_URL=http://sandbox-gateway:9000/openai/v1`

Your agent should send requests to `${OPENAI_BASE_URL}/chat/completions` (or the equivalent endpoint) so the gateway can enforce policy, collect metrics, and use the validator-controlled provider keys.

### Task id propagation (required)

Every LLM request must include the header:

- `IWA-Task-ID: <task_id>`

Where `<task_id>` is the `task_id` value received in `POST /act`.

This is how the gateway correlates all model calls to a single evaluation episode.

### Where it is implemented here

- `llm_gateway.py`: minimal OpenAI-compatible client that:
  - reads `OPENAI_BASE_URL`
  - injects `IWA-Task-ID`
  - only uses `OPENAI_API_KEY` when you are **not** using the sandbox gateway
- `agent.py`: calls `openai_chat_completions(task_id=..., ...)` so the header is always present.


## Agent flow

`POST /act` (`agent.py`) receives:
- `task_id`: used for `IWA-Task-ID` header
- `prompt`: natural-language task
- `url`: current page URL
- `snapshot_html`: current page HTML
- `step_index`: current step number
- `history`: last actions (best-effort)

The API layer is intentionally thin: it forwards to `AutoppiaOperator` (class implementing `IWebAgent.act()` from IWA).

Current canonical `/act` response keys:
- `protocol_version` (currently `1.0`)
- `tool_calls` (namespaced tool calls, e.g. `browser.click`, `user.request_input`)
- `content` (optional final user-facing text)
- `done` (boolean completion hint)
- `state_out` (state roundtrip object)

The operator then:
1. Extracts interactive candidates from HTML (buttons/links/inputs, etc.).
2. Ranks candidates against the task.
3. Builds a compact deterministic **Page IR** (forms, headings, links, cards, CTAs) plus deltas from previous step.
4. Optionally runs a completion-check LLM call (small model, step/repeat-gated by default) to decide if task is already complete.
5. Maintains RAM subgoal memory per `task_id` (inferred milestones, done/blocked tracking) and injects active-subgoal hint into planning.
6. Calls the planner LLM to choose the next browser action sequence (up to `FSM_MAX_ACTIONS_PER_STEP`, default `3`) using a browser-use-like observation with indexed interactive elements and recent action/results context.
7. Returns tool calls for browser/user interaction and sets `done/content` when the task is complete.

Preferred browser tool naming follows browser-use conventions:
- `browser.search`
- `browser.navigate`
- `browser.go_back`
- `browser.click`
- `browser.dblclick`
- `browser.rightclick`
- `browser.middleclick`
- `browser.tripleclick`
- `browser.input`
- `browser.select_dropdown`
- `browser.dropdown_options`
- `browser.hover`
- `browser.scroll`
- `browser.wait`
- `browser.screenshot`
- `browser.send_keys`
- `browser.hold_key`
- `browser.extract`
- `browser.done`

`browser.extract` persists its returned value into execution traces via IWA `ActionExecutionResult.action_output`, so downstream evaluation and training logs can inspect those tool results.

Runtime constraints for the planner:
- Up to `FSM_MAX_ACTIONS_PER_STEP` browser actions per step (default `3`, hard cap `5`).
- Multi-action steps should stay inside one local workflow.
- Interactive element references should prefer `arguments.index` from the indexed shortlist.
- Tabs and file tools are not supported in this runtime.
- Unavailable tools include `switch_tab`, `close_tab`, `upload_file`, `read_file`, `write_file`, and `replace_file`.

### Completion checker defaults

- `AGENT_ENABLE_COMPLETION_CHECK=1` (enabled by default)
- `AGENT_COMPLETION_MODEL=gpt-4o-mini` (smaller than typical planner model)
- `AGENT_COMPLETION_MIN_CONFIDENCE=0.82`

Planner reliability/cost behavior is now opinionated by default (not env-tuned):
- candidate extraction/ranking budgeted internally
- deterministic repair path (no extra planner call)
- repeated URL+decision patterns hard-stop with `done=true`

Credential placeholders like `<username>` / `<password>` are handled by IWA (the evaluator replaces placeholders in actions before execution).

## Built-in inspection tools

The planner can request tools before choosing an action:
- `search_text`
- `visible_text`
- `extract_tables`
- `extract_entities`
- `css_select`
- `xpath_select`
- `extract_forms`
- `list_links`
- `list_candidates`
- `list_cards`
- `find_card` (query-focused card retrieval)

## Local eval

This repo includes a local evaluator:

```bash
python eval.py --model gpt-5.2 --num-tasks 5 --distinct-use-cases
```

List available web projects/use cases from the configured cache:

```bash
python eval.py --list-web-projects
python eval.py --list-use-cases --web-project-id autocinema
```

Run one web project with all use cases:

```bash
python eval.py --web-project-id autocinema --all-use-cases --tasks-per-use-case 1 --max-steps 15
```

Save full `/act` traces (request/response + metadata per step):

```bash
python eval.py --model gpt-5.2 --web-project-id autocinema --all-use-cases --save-act-traces
# optional: add --include-reasoning to request reasoning in each /act trace
```

Run the local operator debugger for one task/use case:

```bash
python scripts/debug_operator_run.py --web-project-id autocinema --use-case LOGIN --model gpt-5.2
```

Inspect an existing trace bundle without rerunning eval:

```bash
python scripts/debug_operator_run.py --reuse-trace data/debug_runs/_smoke_trace
```

Capture screenshots into the debug trace bundle:

```bash
python scripts/debug_operator_run.py --web-project-id autocinema --use-case LOGIN --capture-screenshot
```

Task generation helper (writes the cache consumed by `eval.py`):

```bash
python scripts/generate_tasks.py --project-id autocinema --prompts-per-use-case 1
```

Outputs are written to `data/` (gitignored).


## Model comparison

To compare multiple models/providers on the same task set, use:

```bash
python scripts/compare_eval.py --runs openai:gpt-5.2 openai:gpt-4o-mini --num-tasks 5 --distinct-use-cases
```

Anthropic example (requires `ANTHROPIC_API_KEY` in your env):

```bash
python scripts/compare_eval.py --runs anthropic:claude-sonnet-4 --num-tasks 5 --distinct-use-cases
```

Outputs:
- `data/compare/<provider>__<model>.json`
- `data/compare/compare_summary.json`

## IWAP/S3 Training Scripts

`autoppia_operator/training` now includes:
- typed trajectory classes (`TaskInfo`, `StepRecord`, `TrajectoryRecord`, ...)
- IWAP API ingestion
- direct S3 ingestion (`s3://bucket/prefix`) for task-log payloads
- SFT exports (`sft_train.jsonl`, `sft_val.jsonl`)
- PPO bootstrap export (`ppo_bootstrap.jsonl`)
- optional online PPO rollout collector with IWA `StatefulEvaluator` as reward loop

Recommended orchestration-friendly flow (modular, no monolithic auto-pipeline):

1) List S3 task-log objects

```bash
python scripts/s3_list_task_logs.py \
  --s3-bucket my-iwap-logs \
  --s3-prefix task-logs/ \
  --out data/training/s3_manifest.jsonl
```

2) Normalize S3 payloads into cleaned trajectories

```bash
python scripts/s3_normalize_trajectories.py \
  --s3-bucket my-iwap-logs \
  --s3-prefix task-logs/ \
  --out-dir data/training/cleaned
```

3) Export SFT files from cleaned trajectories

```bash
python scripts/export_sft_from_cleaned.py \
  --cleaned-jsonl data/training/cleaned/cleaned_trajectories.jsonl \
  --out-dir data/training/sft
```

4) Export PPO bootstrap transitions from cleaned trajectories

```bash
python scripts/export_ppo_bootstrap_from_cleaned.py \
  --cleaned-jsonl data/training/cleaned/cleaned_trajectories.jsonl \
  --out-jsonl data/training/ppo/ppo_bootstrap.jsonl
```

Alternative single command still exists if needed:

```bash
python scripts/build_iwap_training_dataset.py \
  --source iwap-api \
  --base-url http://127.0.0.1:8000 \
  --max-runs 200 \
  --min-eval-score 0.5
```

Optional auth:
- pass `--token <bearer-token>`, or
- set `IWAP_API_TOKEN` in your environment.

or:

```bash
python scripts/build_iwap_training_dataset.py \
  --source s3 \
  --s3-bucket my-iwap-logs \
  --s3-prefix task-logs/ \
  --s3-region us-east-1
```

Note:
- S3 mode requires `boto3` available on the training machine.

Core outputs:
- `cleaned_trajectories.jsonl`: normalized + class-mapped trajectories.
- `sft_all.jsonl`: chat-format SFT samples.
- `sft_train.jsonl` / `sft_val.jsonl`: fine-tuning splits.
- `ppo_bootstrap.jsonl`: transition rows usable for PPO/offline-RL bootstrap.
- `manifest` JSON files with stats + file paths.

### Collect PPO-style online rollouts (LLM exploration + StatefulEvaluator reward)

```bash
python scripts/run_iwa_ppo_loop.py \
  --tasks-json data/tasks/tasks.json \
  --num-episodes 50 \
  --max-steps 25 \
  --epsilon 0.05
```

Outputs (timestamped under `data/training/ppo/`):
- `ppo_episodes.jsonl`
- `ppo_transitions.jsonl`
- `ppo_summary.json`

## Repo self-check

```bash
python check.py
```

This validates entrypoints, endpoint shapes, and scans for obvious secrets.

⚠️ deploy_check may fail locally if handshake env vars are missing:
- set `MINER_AGENT_NAME` (or `AGENT_NAME`)
- set `MINER_GITHUB_URL` (or `AGENT_GITHUB_URL` / `GITHUB_URL`)
- optional: `AGENT_IMAGE`, `AGENT_VERSION`

If those are absent, subnet metadata can be considered missing during handshake.

## Fork-ready production onboarding (recommended)

This repo is ready for fork-and-iterate workflows with these defaults for subnet36:

1. Copy env template:

```bash
cp .env.template .env
```

2. Fill local secrets only:

- `SN36_COLDKEY`
- `SN36_HOTKEY`
- optional SMTP/RunPod variables if used
- `IWAP_BASE_URL=https://api-leaderboard.autoppia.com`
- `IWAP_API_TOKEN` (only if the IWAP endpoint requires auth)

3. Run local gate and score loop:

```bash
python scripts/sn36_ops.py preflight
python scripts/sn36_ops.py eval --project-id autocinema
python scripts/sn36_ops.py cycle --github-url <repo/tree/or/commit> --agent-name "<AGENT_NAME>" --submit
```

4. Verify:

```bash
python tools/bittensor_tools.py my-miner
python tools/iwap_tools.py last-round
python tools/iwap_tools.py season-tasks --season-id 36
```

### Production IWAP settings

- Backend recommended in forked setups: `IWAP_BASE_URL=https://api-leaderboard.autoppia.com`
- In this repo tooling, IWAP metrics are mock-first by default.
- For live fetch, call scripts with `--prefer-live` (or wire that behavior in your wrapper) and:
  - `IWAP_BASE_URL=https://api-leaderboard.autoppia.com`
  - `IWAP_API_TOKEN` only if the endpoint requires auth
- If live fetch fails, tools fall back to mock payload for continuity.
- For smoke validation, you can keep `IWAP_API_TOKEN` empty and use mock mode.

### MCP naming for clients

Use either:

- `miner_mcp` (recommended)
- `mcp`

as the `mcpServers` key in client configs. Do not use `daryxx_mcp`.

Example:

```json
{
  "mcpServers": {
    "miner_mcp": {
      "command": "python",
      "args": ["/home/usuario1/autoppia/operator/autoppia_operator/tools/mcp_server.py"]
    }
  }
}
```

### Known production caveats (fork users)

- `SN36_*` must match local wallet state; wrong hotkey/coldkey names are the most common reason for blank `my-miner` output.
- `IWAP_MOCK_DATA` (if set) overrides `IWAP_BASE_URL` and forces mock reads.
- `tools/iwap_tools.py` uses:
  - mock path by default,
  - live path only when `--base-url` is present and `--prefer-live` is set.
- Keep `IWAP_API_TOKEN` out of docs and commit history; set it only in local environment.

## Quick Codex + MCP setup (3-step)

Recommended for users who want Codex to run the auto-mining loop:

1) Install/prepare environment

```bash
cp .env.template .env
# Fill only local values:
# SN36_COLDKEY, SN36_HOTKEY, (optional) IWAP_API_TOKEN, SMTP/RUNPOD vars
```

2) Configure your MCP client with `miner_mcp` (or `mcp`)

```json
{
  "mcpServers": {
    "miner_mcp": {
      "command": "python",
      "args": ["/home/usuario1/autoppia/operator/autoppia_operator/tools/mcp_server.py"],
      "env": {
        "SN36_COLDKEY": "<your-coldkey>",
        "SN36_HOTKEY": "<your-hotkey>",
        "SN36_NETWORK": "finney",
        "SN36_NETUID": "36",
        "IWAP_BASE_URL": "https://api-leaderboard.autoppia.com"
      }
    }
  }
}
```

3) Verify and start loop

```bash
python tools/mcp_server.py --list-tools
python scripts/sn36_ops.py preflight
python scripts/sn36_ops.py eval --project-id autocinema --success-threshold 0.70 --avg-score-threshold 0.60
```

Then ask Codex to follow the SOP in [AGENTS.md](/home/usuario1/autoppia/operator/autoppia_operator/AGENTS.md) for `preflight → eval → cycle → submit → IWAP verification`.
