#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

GITHUB_URL="${SN36_GITHUB_URL:?set SN36_GITHUB_URL first}"
AGENT_NAME="${SN36_AGENT_NAME:?set SN36_AGENT_NAME first}"
AGENT_IMAGE="${SN36_AGENT_IMAGE:-}"
WALLET_NAME="${SN36_COLDKEY:-default}"
WALLET_HOTKEY="${SN36_HOTKEY:-default}"
NETWORK="${SN36_NETWORK:-finney}"
NETUID="${SN36_NETUID:-36}"
CHAIN_ENDPOINT="${SN36_CHAIN_ENDPOINT:-}"
SEASON="${SN36_SEASON:-}"
TARGET_ROUND="${SN36_TARGET_ROUND:-}"
CLI="${SN36_MINER_CLI:-autoppia-miner-cli}"

if ! command -v "$CLI" >/dev/null 2>&1; then
  echo "[sn36_submit] missing CLI: $CLI"
  echo "Install from autoppia_web_agents_subnet, then retry."
  exit 1
fi

CMD=(
  "$CLI" submit
  --github "$GITHUB_URL"
  --agent.name "$AGENT_NAME"
  --wallet.name "$WALLET_NAME"
  --wallet.hotkey "$WALLET_HOTKEY"
  --subtensor.network "$NETWORK"
  --netuid "$NETUID"
)

if [[ -n "${AGENT_IMAGE}" ]]; then
  CMD+=(--agent.image "$AGENT_IMAGE")
fi
if [[ -n "${CHAIN_ENDPOINT}" ]]; then
  CMD+=(--subtensor.chain_endpoint "$CHAIN_ENDPOINT")
fi
if [[ -n "${SEASON}" ]]; then
  CMD+=(--season "$SEASON")
fi
if [[ -n "${TARGET_ROUND}" ]]; then
  CMD+=(--target_round "$TARGET_ROUND")
fi

printf '[sn36_submit] %s\n' "${CMD[*]}"
"${CMD[@]}"
