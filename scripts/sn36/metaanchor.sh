#!/usr/bin/env bash
set -euo pipefail

NETUID="${SN36_NETUID:-36}"
NETWORK="${SN36_NETWORK:-finney}"
COLDKEY="${SN36_COLDKEY:-default}"
HOTKEY="${SN36_HOTKEY:-default}"
CHAIN_ENDPOINT="${SN36_CHAIN_ENDPOINT:-}"

if command -v btcli >/dev/null 2>&1; then
  echo "[sn36_metagraph] btcli detected"
  echo "Useful btcli commands:"
  echo "  btcli metagraph --netuid ${NETUID} --network ${NETWORK}"
  echo "  btcli wallet show --name ${COLDKEY} --hotkey ${HOTKEY}"
  echo
fi

if ! command -v python >/dev/null 2>&1; then
  echo "[sn36_metagraph] python not found; cannot run Bittensor SDK introspection."
  exit 1
fi

python - <<'PY'
import importlib.util
import os

spec = importlib.util.find_spec("bittensor")
if spec is None:
    raise SystemExit(
        "[sn36_metagraph] bittensor package not found. Install with: pip install bittensor"
    )

import bittensor as bt

netuid = int(os.getenv("SN36_NETUID", "36"))
network = os.getenv("SN36_NETWORK", "finney")
coldkey = os.getenv("SN36_COLDKEY", "default")
hotkey = os.getenv("SN36_HOTKEY", "default")
chain_endpoint = os.getenv("SN36_CHAIN_ENDPOINT", "")
top_n = int(os.getenv("SN36_TOP_NODES", "10"))

print(f"[sn36_metagraph] bittensor {getattr(bt, '__version__', 'unknown')}")
print(f"[sn36_metagraph] loading metagraph netuid={netuid}, network={network}")

params = {"network": network}
if chain_endpoint:
    params["chain_endpoint"] = chain_endpoint

subtensor = bt.subtensor(**params)
metagraph = subtensor.metagraph(netuid=netuid, lite=True)

hotkeys = metagraph.hotkeys if hasattr(metagraph, "hotkeys") else []
uids = list(range(len(hotkeys)))
incentives = (
    metagraph.incentive
    if hasattr(metagraph, "incentive")
    else (metagraph.incentives if hasattr(metagraph, "incentives") else [])
)
stake = (
    metagraph.stake
    if hasattr(metagraph, "stake")
    else (metagraph.stakes if hasattr(metagraph, "stakes") else [])
)

print(f"[sn36_metagraph] hotkeys={len(hotkeys)} uids={len(uids)}")

matches = [uid for uid in uids if str(hotkeys[uid]) == str(hotkey)]
if matches:
    uid = matches[0]
    metric = float(incentives[uid]) if incentives else 0.0
    stake_v = float(stake[uid]) if stake else 0.0
    print(
        f"[sn36_metagraph] local_hotkey match uid={uid} coldkey={coldkey} hotkey={hotkey} "
        f"incentive={metric:.6f} stake={stake_v:.6f}"
    )
else:
    print(f"[sn36_metagraph] local_hotkey not found for HOTKEY={hotkey}")

if incentives:
    ranking = sorted(
        [
            {"uid": uid, "hotkey": str(hotkeys[uid]), "incentive": float(incentives[uid])}
            for uid in uids
        ],
        key=lambda item: item["incentive"],
        reverse=True,
    )
    top = ranking[:max(1, top_n)]
    print(f"[sn36_metagraph] top-{len(top)} incentives:")
    for item in top:
        print(
            f"  uid={item['uid']} hotkey={item['hotkey']} incentive={item['incentive']:.6f}"
        )
else:
    print("[sn36_metagraph] incentive vector unavailable in this Bittensor/metagraph version")
PY
