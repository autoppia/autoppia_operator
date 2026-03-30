"""A/B evaluation: compare heuristic vs learned components on operator runs.

Runs the operator with different ranker/verifier configurations and compares
performance metrics (task success rate, score, steps, speed).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ABConfig:
    """Configuration for A/B evaluation.

    Attributes:
        ranker_model_path: Path to trained ranker model (None = heuristic only).
        verifier_model_path: Path to trained verifier model (None = heuristic only).
        n_tasks: Number of tasks to evaluate.
        output_dir: Directory for evaluation results.
    """

    ranker_model_path: Optional[str] = None
    verifier_model_path: Optional[str] = None
    n_tasks: int = 50
    output_dir: str = "eval/results"


@dataclass
class RunResult:
    """Result of a single task evaluation run."""

    task_id: str = ""
    config_name: str = ""
    success: bool = False
    final_score: float = 0.0
    total_steps: int = 0
    duration_ms: float = 0.0
    error: str = ""


@dataclass
class ABResult:
    """Aggregate A/B evaluation result."""

    config_a: str = "heuristic"
    config_b: str = "learned"
    n_tasks: int = 0
    a_results: List[RunResult] = field(default_factory=list)
    b_results: List[RunResult] = field(default_factory=list)

    @property
    def a_success_rate(self) -> float:
        if not self.a_results:
            return 0.0
        return sum(1 for r in self.a_results if r.success) / len(self.a_results)

    @property
    def b_success_rate(self) -> float:
        if not self.b_results:
            return 0.0
        return sum(1 for r in self.b_results if r.success) / len(self.b_results)

    @property
    def a_avg_score(self) -> float:
        if not self.a_results:
            return 0.0
        return sum(r.final_score for r in self.a_results) / len(self.a_results)

    @property
    def b_avg_score(self) -> float:
        if not self.b_results:
            return 0.0
        return sum(r.final_score for r in self.b_results) / len(self.b_results)

    @property
    def a_avg_steps(self) -> float:
        if not self.a_results:
            return 0.0
        return sum(r.total_steps for r in self.a_results) / len(self.a_results)

    @property
    def b_avg_steps(self) -> float:
        if not self.b_results:
            return 0.0
        return sum(r.total_steps for r in self.b_results) / len(self.b_results)

    def summary(self) -> Dict[str, Any]:
        """Generate a comparison summary."""
        return {
            "n_tasks": self.n_tasks,
            "config_a": {
                "name": self.config_a,
                "success_rate": round(self.a_success_rate, 3),
                "avg_score": round(self.a_avg_score, 3),
                "avg_steps": round(self.a_avg_steps, 1),
            },
            "config_b": {
                "name": self.config_b,
                "success_rate": round(self.b_success_rate, 3),
                "avg_score": round(self.b_avg_score, 3),
                "avg_steps": round(self.b_avg_steps, 1),
            },
            "improvement": {
                "success_rate_delta": round(self.b_success_rate - self.a_success_rate, 3),
                "avg_score_delta": round(self.b_avg_score - self.a_avg_score, 3),
                "avg_steps_delta": round(self.b_avg_steps - self.a_avg_steps, 1),
            },
        }


class ABEvaluator:
    """Runs A/B evaluation comparing heuristic vs learned components.

    Usage:
        evaluator = ABEvaluator(operator_factory=make_operator)
        result = evaluator.evaluate(tasks, config)
    """

    def __init__(
        self,
        operator_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        """Initialize evaluator.

        Args:
            operator_factory: Callable that creates an FSMOperator with given config.
                Signature: (use_learned: bool, ranker_path: str, verifier_path: str) -> operator
        """
        self._factory = operator_factory

    def evaluate(
        self,
        tasks: List[Dict[str, Any]],
        config: Optional[ABConfig] = None,
    ) -> ABResult:
        """Run A/B evaluation on a set of tasks.

        Each task is run twice: once with heuristic, once with learned components.

        Args:
            tasks: List of task dicts with task_id, prompt, url, etc.
            config: Evaluation configuration.

        Returns:
            ABResult with comparison metrics.
        """
        config = config or ABConfig()
        result = ABResult(
            config_a="heuristic",
            config_b="learned",
            n_tasks=len(tasks),
        )

        for task in tasks[:config.n_tasks]:
            # Run with heuristic
            a_run = self._run_task(task, use_learned=False, config=config)
            result.a_results.append(a_run)

            # Run with learned
            b_run = self._run_task(task, use_learned=True, config=config)
            result.b_results.append(b_run)

        # Save results
        self._save_results(result, config)
        return result

    def _run_task(
        self,
        task: Dict[str, Any],
        use_learned: bool,
        config: ABConfig,
    ) -> RunResult:
        """Run a single task with given configuration."""
        config_name = "learned" if use_learned else "heuristic"
        task_id = str(task.get("task_id", ""))

        if self._factory is None:
            return RunResult(
                task_id=task_id,
                config_name=config_name,
                error="No operator factory configured",
            )

        try:
            operator = self._factory(
                use_learned=use_learned,
                ranker_path=config.ranker_model_path or "",
                verifier_path=config.verifier_model_path or "",
            )

            start = time.monotonic()
            result = operator.run(payload=task)
            duration = (time.monotonic() - start) * 1000

            return RunResult(
                task_id=task_id,
                config_name=config_name,
                success=bool(result.get("success", False)),
                final_score=float(result.get("score", 0.0)),
                total_steps=int(result.get("steps", 0)),
                duration_ms=duration,
            )

        except Exception as e:
            return RunResult(
                task_id=task_id,
                config_name=config_name,
                error=str(e),
            )

    def _save_results(self, result: ABResult, config: ABConfig) -> None:
        """Save evaluation results to disk."""
        os.makedirs(config.output_dir, exist_ok=True)
        path = os.path.join(config.output_dir, "ab_results.json")
        with open(path, "w") as f:
            json.dump(result.summary(), f, indent=2)


# Alias expected by CHECK.sh
ABTest = ABEvaluator
