from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.rl.splits import build_seed_splits


def main() -> int:
    parser = argparse.ArgumentParser(description="Build reproducible CONTACT RL seed splits.")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=1000)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--out", default="data/rl/contact_seed_splits.json")
    args = parser.parse_args()
    splits = build_seed_splits(
        start=int(args.start),
        end=int(args.end),
        train_ratio=float(args.train_ratio),
        dev_ratio=float(args.dev_ratio),
        random_seed=int(args.random_seed),
    )
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = splits.to_dict()
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), **{k: len(v) for k, v in payload.items()}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
