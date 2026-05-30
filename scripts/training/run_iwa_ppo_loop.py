#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
OPERATOR_ROOT = SCRIPT_DIR.parents[1]
if str(OPERATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(OPERATOR_ROOT))

from training import (
    IWAStatefulPPOCollector,
    OperatorLLMPolicy,
    PPOLoopConfig,
    export_ppo_collection,
    load_tasks,
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=("Collect PPO-style rollouts using AutoppiaOperator exploration policy and IWA StatefulEvaluator rewards."))
    ap.add_argument(
        "--tasks-json",
        required=True,
        help="Path to JSON file with tasks or {'tasks':[...]} payload",
    )
    ap.add_argument("--out-dir", default="data/training/ppo", help="Output base directory")
    ap.add_argument(
        "--model",
        default=None,
        help="Optional model override passed to operator policy",
    )
    ap.add_argument("--num-episodes", type=int, default=20, help="Number of episodes to collect")
    ap.add_argument("--max-steps", type=int, default=25, help="Max steps per episode")
    ap.add_argument("--epsilon", type=float, default=0.05, help="Exploration epsilon")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--task-limit", type=int, default=None, help="Optional cap on loaded tasks")
    ap.add_argument(
        "--web-agent-id-prefix",
        default="autoppia-ppo",
        help="Evaluator web_agent_id prefix",
    )
    return ap.parse_args()


def main() -> None:
    load_dotenv(OPERATOR_ROOT / ".env", override=False)
    args = _parse_args()

    tasks = load_tasks(Path(args.tasks_json), limit=args.task_limit)
    if not tasks:
        raise SystemExit("No tasks could be loaded")

    policy = OperatorLLMPolicy(model_override=args.model)
    config = PPOLoopConfig(
        num_episodes=int(args.num_episodes),
        max_steps_per_episode=int(args.max_steps),
        epsilon_exploration=float(args.epsilon),
        random_seed=int(args.seed),
        web_agent_id_prefix=str(args.web_agent_id_prefix),
    )

    collector = IWAStatefulPPOCollector(policy=policy, config=config)
    episodes = collector.collect(tasks)

    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).resolve() / stamp
    files = export_ppo_collection(out_dir, episodes)

    summary = json.loads(Path(files["summary"]).read_text(encoding="utf-8"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[done] PPO collection written to {out_dir}")


if __name__ == "__main__":
    main()
