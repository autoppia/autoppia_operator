#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OPERATOR_ROOT = SCRIPT_DIR.parents[1]
if str(OPERATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(OPERATOR_ROOT))

from training.exporters import export_ppo_bootstrap, load_cleaned_trajectories


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export PPO bootstrap transitions from cleaned trajectory records")
    ap.add_argument("--cleaned-jsonl", required=True)
    ap.add_argument("--out-jsonl", required=True, help="Output path for ppo_bootstrap.jsonl")
    ap.add_argument("--terminal-bonus", type=float, default=1.0)
    return ap.parse_args()


def main() -> None:
    args = _args()
    records = load_cleaned_trajectories(Path(args.cleaned_jsonl))
    result = export_ppo_bootstrap(
        records=records,
        out_path=Path(args.out_jsonl).resolve(),
        terminal_bonus=float(args.terminal_bonus),
    )

    manifest_path = Path(args.out_jsonl).resolve().with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    payload = dict(result.get("stats") or {})
    payload["manifest"] = str(manifest_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
