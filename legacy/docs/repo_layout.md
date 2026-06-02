# Repo Layout

## Core runtime

- `src/operator/`: operator runtime split by responsibility (`entrypoint`, `api/`, `agents/`, `runtime/`, `support/`).
- `infra/`: shared platform services (`llm_gateway`, `pricing`).
- `src/eval/`: evaluator code.
- Root entrypoint: `main.py`; core implementation lives under `src/` and `qa/`.
- `deploy/pm2/`: canonical PM2 configs (`operator.config.cjs`, `mcp.config.cjs`).

## Debugger

- `src/eval/debugger/app.py`: debugger API server.
- `src/eval/debugger/static/`: debugger frontend assets.
- CLI entrypoint: `operator-debugger` (after `pip install -e .`) or `python -m src.eval.debugger`.

## MCP

- `mcp/`: MCP server and helper tools (`server.py`, `bittensor_tools.py`, `iwap_tools.py`, `runpod_tools.py`, `smtp_tools.py`).
- PM2 start command: `pm2 start deploy/pm2/mcp.config.cjs`.

## Scripts

- `scripts/sn36_ops.py`: main operator lifecycle CLI (`preflight`, `eval`, `cycle`, `submit`, `deploy-smoke`).
- `scripts/deploy_check.py`: local HTTP contract and handshake validation.
- `scripts/eval/`: eval-scoped helpers (`generate_tasks.py`, `compare_eval.py`, `analyze_model_runs.py`).
- `scripts/training/`: dataset ingestion/export and fine-tuning helpers.
- `scripts/sn36/`: shell wrappers for manual subnet operations.

## Training

- `training/`: reusable Python package for trajectory normalization, IWAP/S3 ingestion, dataset export, and PPO collection helpers.
