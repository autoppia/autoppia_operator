from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.rl.contact_env import DEFAULT_CONTACT_TASK_CACHE_DIR, rollout_contact_seed
from training.rl.rollout_store import RolloutStore


def parse_seed_spec(raw: str) -> list[int]:
    text = str(raw or "").strip()
    if not text:
        return []
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ".." in part:
            start_s, end_s = part.split("..", 1)
            start_i = int(start_s)
            end_i = int(end_s)
            step = 1 if end_i >= start_i else -1
            seeds.extend(list(range(start_i, end_i + step, step)))
        else:
            seeds.append(int(part))
    seen: set[int] = set()
    ordered: list[int] = []
    for seed in seeds:
        if seed not in seen:
            seen.add(seed)
            ordered.append(seed)
    return ordered


async def _run_episode(seed: int, *, model: str, max_steps: int, task_cache_dir: Path) -> dict:
    return await rollout_contact_seed(
        seed=seed,
        model_override=model,
        max_steps=max_steps,
        task_cache_dir=task_cache_dir,
    )


async def _run_all(seeds: list[int], *, model: str, max_steps: int, concurrency: int, task_cache_dir: Path) -> list[dict]:
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _guarded(seed: int) -> dict:
        async with sem:
            try:
                return await _run_episode(seed, model=model, max_steps=max_steps, task_cache_dir=task_cache_dir)
            except Exception as exc:
                return {"seed": int(seed), "success": False, "score": 0.0, "error": str(exc), "steps": []}

    return await asyncio.gather(*[_guarded(seed) for seed in seeds])


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CONTACT RL rollouts with the real StepEngine policy and stateful evaluator.")
    parser.add_argument("--seeds", default="1..10", help="Seed list like 1..10,15,20")
    parser.add_argument("--sample-count", type=int, default=0, help="If >0, sample this many seeds from --seed-range")
    parser.add_argument("--seed-range", default="1..1000", help="Range used with --sample-count")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--model", default="", help="Policy model override passed to StepEngine")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--task-cache-dir", default=str(DEFAULT_CONTACT_TASK_CACHE_DIR))
    parser.add_argument("--out-dir", default=str(Path("data") / "rl" / "contact_rollouts"))
    args = parser.parse_args()

    if int(args.sample_count or 0) > 0:
        universe = parse_seed_spec(str(args.seed_range))
        rng = random.Random(int(args.random_seed))
        seeds = sorted(rng.sample(universe, min(len(universe), int(args.sample_count))))
    else:
        seeds = parse_seed_spec(str(args.seeds))
    if not seeds:
        raise SystemExit("no seeds selected")

    out_dir = Path(args.out_dir).expanduser().resolve()
    store = RolloutStore(out_dir)
    task_cache_dir = Path(args.task_cache_dir).expanduser().resolve()
    episodes = asyncio.run(
        _run_all(
            seeds,
            model=str(args.model or ""),
            max_steps=int(args.max_steps),
            concurrency=int(args.concurrency),
            task_cache_dir=task_cache_dir,
        )
    )
    success_count = sum(1 for item in episodes if bool(item.get("success")))
    avg_score = (sum(float(item.get("score") or 0.0) for item in episodes) / len(episodes)) if episodes else 0.0
    for episode in episodes:
        store.write_episode(seed=int(episode.get("seed") or 0), episode=episode)
    summary = {
        "model": str(args.model or ""),
        "seeds": seeds,
        "episodes": len(episodes),
        "successes": success_count,
        "success_rate": (success_count / len(episodes)) if episodes else 0.0,
        "avg_score": avg_score,
        "out_dir": str(out_dir),
    }
    store.write_summary(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
