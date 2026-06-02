# AGENTS.md

Operator playbook for `autoppia_operator` on Autoppia Subnet 36.

## Scope

This repo is the miner/operator side.

- Runtime HTTP contract lives in `main.py`, `src/operator/entrypoint.py`, and `src/operator/api/server.py`.
- Local validation lives in `qa/check_repo.py`, `qa/subnet_compat.py`, and `scripts/deploy_check.py`.
- Local evaluation lives in `src/eval/runner.py`.
- Canonical PM2 configs live in `deploy/pm2/operator.config.cjs` and `deploy/pm2/mcp.config.cjs`.
- Branch focus: `daryxx`.


## Research Context

This repo is not only a subnet operator; it is also the current research sandbox for training and serving a stronger web agent for the IWA benchmark.

Current project goal:
- Train and serve a generalist agent that performs strongly across IWA demo-web use cases.
- Prioritise methods that generalise across demo webs and tasks, not one-off task hacks.
- Use the real benchmark environment whenever possible.

Current benchmark focus:
- `autocinema`
- Current hard use cases under active iteration: `CONTACT`, `LOGIN`, then broader multi-use-case campaigns.

## Demo Webs And Evaluator

Use the real remote demo-web endpoint unless the user explicitly says otherwise:
- `DEMO_WEBS_ENDPOINT=http://84.247.180.192`
- Typical Autocinema frontend URL: `http://84.247.180.192:8000`

Do not assume local docker deployment is wanted.
- The current default is to use the remote `84.247.180.192` demo-web server.
- Do not spin up local demo-web containers unless the user explicitly asks for that.

Canonical evaluator path in this repo:
- `src/operator/eval/session.py`
- It wraps `autoppia_iwa.src.evaluation.stateful_evaluator.TaskExecutionSession`
- This stateful evaluator is the thing to trust for real task scoring and browser execution.

If you need to inspect benchmark internals, check these sibling repos:
- `../autoppia_iwa`
- `../autoppia_webs_demo`

Absolute paths in this workspace:
- `/home/usuario1/daryxx/autoppia/operator/autoppia_iwa`
- `/home/usuario1/daryxx/autoppia/operator/autoppia_webs_demo`

Use those repos to inspect:
- task definitions
- use cases
- demo-web implementations
- stateful evaluator behavior
- event and score logic

## Clean Architecture Direction

The intended long-term split is:
- `training method`: generic as possible, reusable across demo webs and use cases
- `inference method`: clean policy/runtime path for serving the trained model
- `eval method`: easy to run against the real demo-web benchmark

Important principle:
- Inference should stay clean.
- Avoid use-case-specific heuristics, overrides, or ad-hoc rules in inference whenever possible.
- Training should also move toward generic methods, not brittle task-specific recipes.

Practical interpretation:
- We may temporarily use pragmatic structure during research to diagnose failure modes.
- But the target design is a general policy that works on the benchmark because it was trained well, not because inference is hardcoded for each task.

## Inference Rules

The active inference operator must stay clean.

Hard rule:
- The active inference path must be just the model choosing browser tools from the current observation.
- No task-specific planners, workflow scripts, route overrides, or selector heuristics are allowed in the active inference operator.
- If a behavior depends on hand-written task logic, it does not belong in the active inference path.

Current intended meaning of `clean inference`:
- model sees task + browser state + candidates + history
- model chooses tool calls
- runtime executes them
- nothing else decides the task flow

Research consequence:
- Do not improve benchmark scores by silently adding use-case heuristics to the active operator.
- If heuristics are needed for data collection, keep them outside the active inference path and label them clearly as teacher/harvest infrastructure.

Current runtime split:
- Clean active operator: `src/operator/agents/operator.py`
- Archived heuristic teacher runtime: `src/operator/agents/heuristic_runtime/structured_heuristic_operator.py`
- Runtime selection entrypoint: `src/operator/entrypoint.py`

Runtime env intent:
- `WEB_AGENT_RUNTIME=structured` -> clean model-only inference path
- `WEB_AGENT_RUNTIME=heuristic_structured` -> heuristic teacher runtime for harvest/debug only

The heuristic runtime is not the serving target.
It exists to generate trajectories and diagnose the benchmark.

## Current Clean Paths

Current clean inference path:
- `src/operator/agents/operator.py`
- enabled via `WEB_AGENT_RUNTIME=structured`
- wired from `src/operator/entrypoint.py`

Current heuristic teacher path:
- `src/operator/agents/heuristic_runtime/structured_heuristic_operator.py`
- enabled via `WEB_AGENT_RUNTIME=heuristic_structured`
- use only for harvest, teacher rollouts, and debugging

Current eval wrapper:
- `scripts/eval/eval_structured_operator.py`
- This should stay as the simple entrypoint for running the clean inference operator against the benchmark.

Current single-use-case training wrapper:
- `scripts/training/run_single_usecase_rl_pipeline.py`
- intended pipeline:
  - heuristic teacher harvest on random seeds
  - runtime-aligned SFT export
  - SFT fine-tune
  - PPO refinement on clean inference

Current campaign / legacy wrappers:
- `scripts/training/run_autocinema_training.py`
- `scripts/eval/run_autocinema_campaign.py`

Current RL / online training area:
- `training/rl/`
- notable files:
  - `training/rl/contact_env.py`
  - `training/rl/contact_ppo_trainer.py`
  - `training/rl/generic_env.py`
  - `training/rl/generic_ppo_trainer.py`
  - `training/rl/reward.py`
  - `training/rl/judge.py`

## Preferred Training Method

The target method for a new demo web / use case is:

1. Run teacher harvest on real benchmark seeds.
- Use the archived heuristic runtime only as a teacher.
- Collect successful trajectories with full traces.

2. Export runtime-aligned SFT data.
- Convert replayable successful traces into `messages` JSONL.
- Keep the format aligned with the clean inference runtime.

3. Fine-tune the base model with SFT.
- Prefer LoRA / QLoRA on the existing BU base model.
- Use RunPod A100 for serious training runs.

4. Run PPO on clean inference only.
- The actor uses the clean operator path only.
- PPO may use:
  - evaluator score
  - dense reward shaping
  - LLM judge reward
  - teacher BC loss from harvested traces
- PPO must not rely on heuristic inference-time routing.

5. Evaluate holdout seeds using clean inference.
- Success only counts if the clean inference runtime improves.

Summary:
- heuristic runtime is allowed for teacher data collection
- active inference must remain model-only
- learning should move capability from teacher traces into the model, not into runtime logic

## RL Status

Current research question:
- Can RL over real demo-web episodes learn useful behavior on IWA tasks using evaluator score as reward, optionally densified with an LLM judge?

Current RL ingredients explored here:
- evaluator score / success as reward signal
- PPO-style online updates
- optional dense reward shaping in `training/rl/reward.py`
- optional LLM judge in `training/rl/judge.py`
- current default judge model there is `gpt-4o` via `CONTACT_RL_JUDGE_MODEL`

Current conclusion status:
- RL infrastructure is working technically.
- It is useful for experimentation.
- The open question is still whether score + judge are sufficient to train a generalist benchmark agent without inference-time heuristics.

## Infrastructure Notes

Available GPU path:
- RunPod is available and this repo already contains RunPod tooling.
- There is A100 support in the training stack.
- Relevant files include:
  - `training/finetune_bu.py`
  - `training/runpod_job.py`
  - `training/runpod_config.py`
  - `training/runpod_bootstrap_status.json`

When GPU training is needed, prefer the existing RunPod/A100 flow rather than inventing a new one.

For the current single-use-case pipeline:
- local CPU machine is acceptable for harvest and SFT export smoke tests
- serious SFT and PPO runs should go to RunPod A100
- do not pretend local no-GPU runs are meaningful PPO training

## Working Rules For Future Agents

If working on this repo, assume the following unless the user says otherwise:
- Use `84.247.180.192` as the demo-web server.
- Use the real stateful evaluator for serious validation.
- Keep active inference clean and model-only.
- Keep training methods as generic as possible.
- Prefer changes that improve the benchmark method, not only one task.
- If heuristics are introduced, keep them outside active inference and label them explicitly as teacher/harvest logic.
- Do not claim a use case is solved until it passes in real eval against the benchmark environment.

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
