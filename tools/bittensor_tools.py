#!/usr/bin/env python3
"""Bittensor helper tools for subnet 36 operator workflows."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import dataclass
from typing import Any


def _json_dump(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, dict):
        return list(value.values())
    if hasattr(value, "tolist"):
        try:
            return list(value.tolist())
        except Exception:
            pass
    try:
        return list(value)
    except Exception:
        return []


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


@dataclass
class MetaRow:
    uid: int
    hotkey: str
    incentive: float
    stake: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "hotkey": self.hotkey,
            "incentive": self.incentive,
            "stake": self.stake,
            "rank": self.rank,
        }


def _load_metagraph(
    netuid: int, network: str, chain_endpoint: str | None
) -> tuple[dict[str, Any], list[str], list[float], list[float]]:
    try:
        import bittensor as bt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "bittensor is not installed in this environment. Install with: pip install bittensor"
        ) from exc

    params = {"network": network}
    if chain_endpoint:
        params["chain_endpoint"] = chain_endpoint

    subtensor = bt.subtensor(**params)
    metagraph = subtensor.metagraph(netuid=netuid, lite=True)

    hotkeys = _to_list(getattr(metagraph, "hotkeys", []))
    incentives = _to_list(getattr(metagraph, "incentive", []))
    stakes = _to_list(getattr(metagraph, "stake", []))

    metas: dict[str, Any] = {"netuid": int(netuid), "network": network}
    if chain_endpoint:
        metas["chain_endpoint"] = chain_endpoint
    return metas, [str(h) for h in hotkeys], [_float(v) for v in incentives], [_float(v) for v in stakes]


def _build_snapshot(netuid: int, network: str, chain_endpoint: str | None) -> list[MetaRow]:
    _, hotkeys, incentives, stakes = _load_metagraph(netuid=netuid, network=network, chain_endpoint=chain_endpoint)

    # Rank by incentive descending
    order = sorted(range(len(hotkeys)), key=lambda uid: incentives[uid] if uid < len(incentives) else 0.0, reverse=True)
    rank_for_uid: dict[int, int] = {uid: idx + 1 for idx, uid in enumerate(order)}

    rows: list[MetaRow] = []
    for uid in range(len(hotkeys)):
        rows.append(
            MetaRow(
                uid=uid,
                hotkey=hotkeys[uid],
                incentive=incentives[uid] if uid < len(incentives) else 0.0,
                stake=stakes[uid] if uid < len(stakes) else 0.0,
                rank=rank_for_uid.get(uid, len(hotkeys)),
            )
        )
    return rows


def cmd_metagraph(args: argparse.Namespace) -> int:
    rows = _build_snapshot(args.netuid, args.network, args.chain_endpoint)

    payload = {
        "ok": True,
        "netuid": args.netuid,
        "network": args.network,
        "chain_endpoint": args.chain_endpoint or "",
        "count": len(rows),
        "rows": [row.to_dict() for row in rows[: max(1, int(args.limit))]],
    }
    if args.include_all:
        payload["rows"] = [row.to_dict() for row in rows]

    print(_json_dump(payload))
    return 0


def cmd_uid_lookup(args: argparse.Namespace) -> int:
    rows = _build_snapshot(args.netuid, args.network, args.chain_endpoint)

    matches: list[MetaRow] = []
    if args.uid is not None:
        matches = [row for row in rows if row.uid == int(args.uid)]
    else:
        target = str(args.hotkey or "").strip()
        if not target:
            raise RuntimeError("Either --uid or --hotkey is required.")
        matches = [row for row in rows if row.hotkey == target]

    if not matches and args.hotkey:
        matches = [row for row in rows if row.hotkey.endswith(f":{args.hotkey}") or args.hotkey in row.hotkey]

    payload = {
        "ok": bool(matches),
        "netuid": args.netuid,
        "network": args.network,
        "query": {"uid": args.uid, "hotkey": args.hotkey},
        "count": len(matches),
        "matches": [row.to_dict() for row in matches],
    }
    if not matches:
        payload["message"] = "No matching miner found in metagraph snapshot."
    print(_json_dump(payload))
    return 0 if matches else 1


def cmd_uid_stats(args: argparse.Namespace) -> int:
    rows = _build_snapshot(args.netuid, args.network, args.chain_endpoint)
    if not rows:
        raise RuntimeError("No metagraph rows returned.")
    if args.uid < 0 or args.uid >= len(rows):
        raise RuntimeError(f"UID out of range: {args.uid}")

    row = rows[args.uid]
    incentives = [r.incentive for r in rows]
    payload = {
        "ok": True,
        "netuid": args.netuid,
        "network": args.network,
        "uid": row.uid,
        "hotkey": row.hotkey,
        "incentive": row.incentive,
        "stake": row.stake,
        "rank": row.rank,
        "distribution": {
            "min_incentive": min(incentives) if incentives else 0.0,
            "max_incentive": max(incentives) if incentives else 0.0,
            "avg_incentive": statistics.mean(incentives) if incentives else 0.0,
            "median_incentive": statistics.median(incentives) if incentives else 0.0,
        },
    }
    print(_json_dump(payload))
    return 0


def cmd_user_miner(args: argparse.Namespace) -> int:
    hotkey = (args.hotkey or os.getenv("SN36_HOTKEY", "")).strip()
    if not hotkey:
        raise RuntimeError(
            "SN36_HOTKEY is not set. Run with --hotkey or set SN36_HOTKEY env var."
        )

    rows = _build_snapshot(args.netuid, args.network, args.chain_endpoint)
    matches = [row for row in rows if row.hotkey == hotkey]
    payload = {
        "ok": bool(matches),
        "netuid": args.netuid,
        "network": args.network,
        "hotkey": hotkey,
        "miner_count": len(matches),
        "rows": [row.to_dict() for row in matches],
        "message": (
            "Found one or more local UID candidates for this hotkey."
            if matches
            else "No metagraph match for SN36_HOTKEY."
        ),
    }
    print(_json_dump(payload))
    return 0 if matches else 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autoppia sn36 Bittensor helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_metagraph = subparsers.add_parser("metagraph", help="fetch metagraph snapshot for subnet")
    p_metagraph.add_argument("--netuid", type=int, default=int(os.getenv("SN36_NETUID", "36")))
    p_metagraph.add_argument("--network", default=os.getenv("SN36_NETWORK", "finney"))
    p_metagraph.add_argument("--chain-endpoint", default=os.getenv("SN36_CHAIN_ENDPOINT", ""))
    p_metagraph.add_argument("--limit", type=int, default=20, help="rows to print when include_all is false")
    p_metagraph.add_argument("--include-all", action="store_true", help="print full metagraph rows")
    p_metagraph.set_defaults(func=cmd_metagraph)

    p_uid = subparsers.add_parser("uid", help="lookup incentive by hotkey or uid")
    p_uid.add_argument("--netuid", type=int, default=int(os.getenv("SN36_NETUID", "36")))
    p_uid.add_argument("--network", default=os.getenv("SN36_NETWORK", "finney"))
    p_uid.add_argument("--chain-endpoint", default=os.getenv("SN36_CHAIN_ENDPOINT", ""))
    p_uid_group = p_uid.add_mutually_exclusive_group(required=True)
    p_uid_group.add_argument("--uid", type=int)
    p_uid_group.add_argument("--hotkey")
    p_uid.set_defaults(func=cmd_uid_lookup)

    p_stats = subparsers.add_parser("uid-stats", help="get incentive/stake stats for a UID")
    p_stats.add_argument("--uid", type=int, required=True)
    p_stats.add_argument("--netuid", type=int, default=int(os.getenv("SN36_NETUID", "36")))
    p_stats.add_argument("--network", default=os.getenv("SN36_NETWORK", "finney"))
    p_stats.add_argument("--chain-endpoint", default=os.getenv("SN36_CHAIN_ENDPOINT", ""))
    p_stats.set_defaults(func=cmd_uid_stats)

    p_user = subparsers.add_parser("my-miner", help="resolve SN36_HOTKEY to local UID(s)")
    p_user.add_argument("--hotkey", default=None)
    p_user.add_argument("--netuid", type=int, default=int(os.getenv("SN36_NETUID", "36")))
    p_user.add_argument("--network", default=os.getenv("SN36_NETWORK", "finney"))
    p_user.add_argument("--chain-endpoint", default=os.getenv("SN36_CHAIN_ENDPOINT", ""))
    p_user.set_defaults(func=cmd_user_miner)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except RuntimeError as exc:
        print(f"[bittensor-tools] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
