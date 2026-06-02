"""Batch trajectory collection script.

Runs the operator on IWA tasks and captures trajectories using
TrajectoryCollector. Produces episodes.jsonl for downstream formatting.

Usage:
    python -m training.collect_trajectories --num-tasks 10
    COLLECT_TRAJECTORIES=1 python -m training.collect_trajectories
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from .collector import TrajectoryCollector

logger = logging.getLogger(__name__)


def run_collection(
    num_tasks: int = 5,
    output_dir: str = "data/trajectories",
    websites: Optional[List[str]] = None,
) -> str:
    """Run trajectory collection on IWA tasks.

    Args:
        num_tasks: Number of tasks to run.
        output_dir: Directory for episodes.jsonl output.
        websites: List of IWA website URLs to use.

    Returns:
        Path to the episodes.jsonl file.
    """
    if websites is None:
        websites = [
            "http://autocinema.iwa.cloud",
            "http://autobooks.iwa.cloud",
            "http://autodining.iwa.cloud",
        ]

    collector = TrajectoryCollector(output_dir=output_dir)
    output_path = os.path.join(output_dir, "episodes.jsonl")

    logger.info("Starting trajectory collection: %d tasks across %d websites",
                num_tasks, len(websites))

    # The actual integration with eval.py happens when COLLECT_TRAJECTORIES=1
    # is set and the FSMEngine hooks are active. This function provides
    # the batch orchestration entry point.
    #
    # For standalone collection, import and run the eval pipeline:
    try:
        _run_eval_loop(collector, num_tasks, websites)
    except ImportError as e:
        logger.warning("Eval pipeline not available for standalone collection: %s", e)
        logger.info("Set COLLECT_TRAJECTORIES=1 and run eval.py directly instead.")
    except Exception as e:
        logger.error("Collection error: %s", e, exc_info=True)

    logger.info("Collection complete. Output: %s", output_path)
    return output_path


def _run_eval_loop(
    collector: TrajectoryCollector,
    num_tasks: int,
    websites: List[str],
) -> None:
    """Run the eval pipeline with trajectory collection enabled.

    This imports the eval module and runs tasks while the collector
    hooks capture trajectory data.
    """
    os.environ["COLLECT_TRAJECTORIES"] = "1"

    # Try to import the eval pipeline
    # This will work when running from the repo root
    try:
        from eval import run_eval  # type: ignore
    except ImportError:
        logger.info("eval.run_eval not available — use eval.py directly with "
                     "COLLECT_TRAJECTORIES=1 env var")
        raise

    for i in range(num_tasks):
        website = websites[i % len(websites)]
        logger.info("Task %d/%d on %s", i + 1, num_tasks, website)
        try:
            run_eval(website_url=website, collector=collector)
        except Exception as e:
            logger.error("Task %d failed: %s", i + 1, e)
            if collector.is_collecting:
                collector.end_episode(success=False, final_score=0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect trajectories from IWA tasks")
    parser.add_argument("--num-tasks", type=int, default=5)
    parser.add_argument("--output-dir", default="data/trajectories")
    parser.add_argument("--websites", nargs="+", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_collection(
        num_tasks=args.num_tasks,
        output_dir=args.output_dir,
        websites=args.websites,
    )


if __name__ == "__main__":
    main()
