from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.rl.export_rollouts import export_rollouts_to_weighted_sft


def main() -> int:
    parser = argparse.ArgumentParser(description="Export CONTACT RL rollouts to weighted SFT-style JSONL.")
    parser.add_argument("--rollout-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--min-return", type=float, default=0.0)
    args = parser.parse_args()
    summary = export_rollouts_to_weighted_sft(
        rollout_dir=Path(args.rollout_dir),
        out_path=Path(args.out),
        gamma=float(args.gamma),
        min_return=float(args.min_return),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
