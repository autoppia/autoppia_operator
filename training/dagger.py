"""DAgger (Dataset Aggregation) oracle for querying an expert on uncertain states.

The DAgger oracle intercepts action selection when the agent is uncertain,
in a loop, or about to take an irreversible action. It queries a stronger
model for the preferred action and records the correction for training.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .dagger_config import DAggerConfig
from .schema import FullStepRecord


class DAggerOracle:
    """Oracle that queries an expert model on uncertain/failed states.

    Integrates with the FSM engine to provide expert corrections.
    Records (state, expert_action) pairs for training set aggregation.
    """

    def __init__(
        self,
        config: Optional[DAggerConfig] = None,
        expert_call: Optional[Callable[..., Dict[str, Any]]] = None,
        output_dir: str = "data/trajectories",
    ) -> None:
        self.config = config or DAggerConfig()
        self._expert_call = expert_call
        self.output_dir = output_dir
        self._query_count: int = 0
        self._episode_query_count: int = 0
        self._corrections: List[Dict[str, Any]] = []

    def reset_episode(self) -> None:
        """Reset per-episode counters."""
        self._episode_query_count = 0

    def should_query(
        self,
        uncertainty: float = 0.0,
        loop_count: int = 0,
        steps_without_progress: int = 0,
        action_type: str = "",
    ) -> Tuple[bool, str]:
        """Determine whether to query the expert for this state.

        Returns (should_query, reason).
        """
        if not self.config.enabled:
            return False, "dagger_disabled"

        if self._query_count >= self.config.max_total_queries:
            return False, "global_budget_exhausted"

        if self._episode_query_count >= self.config.max_queries_per_episode:
            return False, "episode_budget_exhausted"

        if uncertainty >= self.config.uncertainty_threshold:
            return True, f"high_uncertainty({uncertainty:.2f})"

        if loop_count >= self.config.loop_threshold:
            return True, f"loop_detected({loop_count})"

        if steps_without_progress >= self.config.no_progress_threshold:
            return True, f"no_progress({steps_without_progress})"

        if self.config.query_on_irreversible and self.config.irreversible_action_types:
            if action_type in self.config.irreversible_action_types:
                return True, f"irreversible_action({action_type})"

        return False, "not_needed"

    def query_expert(
        self,
        task: str,
        state: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        reason: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Query the expert model for the preferred action.

        Args:
            task: Task description.
            state: Current agent state.
            candidates: Available candidates.
            reason: Why the query is triggered.

        Returns:
            Expert's preferred action dict, or None if query fails.
        """
        if self._expert_call is None:
            return None

        self._query_count += 1
        self._episode_query_count += 1

        # Build expert prompt
        prompt = self._build_expert_prompt(task, state, candidates)

        try:
            response = self._expert_call(
                prompt=prompt,
                model=self.config.expert_model,
            )
            expert_action = self._parse_expert_response(response, candidates)

            # Record the correction
            correction = {
                "task": task,
                "state_summary": {
                    "url": state.get("url", ""),
                    "step_index": state.get("step_index", 0),
                },
                "candidate_count": len(candidates),
                "expert_action": expert_action,
                "reason": reason,
                "timestamp": time.time(),
            }
            self._corrections.append(correction)
            return expert_action

        except Exception:
            return None

    def _build_expert_prompt(
        self,
        task: str,
        state: Dict[str, Any],
        candidates: List[Dict[str, Any]],
    ) -> str:
        """Build a prompt for the expert oracle."""
        cand_lines = []
        for i, c in enumerate(candidates[:15]):
            text = str(c.get("text", ""))[:80]
            ctype = c.get("element_type", "")
            role = c.get("role", "")
            cand_lines.append(f"  [{i}] {ctype}/{role}: {text}")

        return (
            f"Task: {task}\n"
            f"URL: {state.get('url', '')}\n"
            f"Step: {state.get('step_index', 0)}\n"
            f"Candidates:\n" + "\n".join(cand_lines) + "\n\n"
            f"Which candidate should be selected? Reply with just the index number."
        )

    def _parse_expert_response(
        self,
        response: Dict[str, Any],
        candidates: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Parse expert response to extract the chosen candidate."""
        text = str(response.get("text", response.get("content", "")))
        # Extract first integer from response
        for token in text.split():
            token = token.strip("[]().,")
            if token.isdigit():
                idx = int(token)
                if 0 <= idx < len(candidates):
                    return candidates[idx]
        return None

    def save_corrections(self) -> int:
        """Save all collected corrections to disk. Returns count saved."""
        if not self._corrections:
            return 0
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, "dagger_corrections.jsonl")
        with open(path, "a") as f:
            for corr in self._corrections:
                f.write(json.dumps(corr, ensure_ascii=False, default=str) + "\n")
        count = len(self._corrections)
        self._corrections.clear()
        return count

    @property
    def total_queries(self) -> int:
        """Total number of expert queries made."""
        return self._query_count
