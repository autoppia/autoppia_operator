"""Trajectory collector that hooks into the FSM engine to produce EpisodeRecords."""
from __future__ import annotations

import atexit
import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .schema import (
    CandidateRecord,
    EpisodeRecord,
    FullStepRecord,
    ValidationEvent,
)


class TrajectoryCollector:
    """Collects trajectory data from FSM engine runs and produces EpisodeRecords.

    Usage:
        collector = TrajectoryCollector(output_dir="data/trajectories")
        collector.begin_episode(task_id="t1", task_text="Search for shoes", ...)
        collector.record_step(step_index=0, ...)
        episode = collector.end_episode(success=True, final_score=0.85)
    """

    def __init__(self, output_dir: str = "data/trajectories") -> None:
        self.output_dir = output_dir
        self._current_episode: Optional[EpisodeRecord] = None
        self._episode_start_time: float = 0.0
        self._step_start_time: float = 0.0
        self._cumulative_reward: float = 0.0
        self._visited_urls: set[str] = set()
        self._previous_action_type: str = ""
        self._loop_count: int = 0
        self._last_url: str = ""
        atexit.register(self._flush_on_exit)

    def _flush_on_exit(self) -> None:
        """Flush any in-progress episode on process exit."""
        if self._current_episode is not None:
            self.end_episode(success=False, final_score=0.0)

    def begin_episode(
        self,
        task_id: str,
        task_text: str,
        task_type: str = "",
        website_url: str = "",
        ranker_type: str = "heuristic",
        verifier_type: str = "heuristic",
        agent_version: str = "",
    ) -> None:
        """Start collecting a new episode."""
        self._current_episode = EpisodeRecord(
            task_id=task_id,
            task_text=task_text,
            task_type=task_type,
            website_url=website_url,
            episode_id=str(uuid.uuid4()),
            ranker_type=ranker_type,
            verifier_type=verifier_type,
            agent_version=agent_version,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._episode_start_time = time.monotonic()
        self._cumulative_reward = 0.0
        self._visited_urls = set()
        self._previous_action_type = ""
        self._loop_count = 0
        self._last_url = ""

    def begin_step(self) -> None:
        """Mark the start of a step for timing."""
        self._step_start_time = time.monotonic()

    def record_step(
        self,
        step_index: int,
        url: str = "",
        page_title: str = "",
        page_graph_summary: Optional[Dict[str, Any]] = None,
        dom_node_count: int = 0,
        candidates: Optional[List[Dict[str, Any]]] = None,
        chosen_action: Optional[Dict[str, Any]] = None,
        chosen_action_type: str = "",
        chosen_candidate_id: str = "",
        expert_action: Optional[Dict[str, Any]] = None,
        expert_candidate_id: Optional[str] = None,
        validation_events: Optional[List[Dict[str, Any]]] = None,
        score_delta: float = 0.0,
        verifier_status: str = "",
        verifier_confidence: float = 0.0,
        made_progress: bool = False,
        error_type: str = "",
        error_details: str = "",
        token_usage: int = 0,
    ) -> None:
        """Record a single step in the current episode."""
        if self._current_episode is None:
            return

        # Track URL changes
        url_changed = url != self._last_url and self._last_url != ""
        if url:
            self._visited_urls.add(url)
            self._last_url = url

        # Track loops (same action type repeated)
        if chosen_action_type == self._previous_action_type:
            self._loop_count += 1
        else:
            self._loop_count = 0

        # Accumulate reward
        self._cumulative_reward += score_delta

        # Calculate step duration
        step_duration_ms = 0.0
        if self._step_start_time > 0:
            step_duration_ms = (time.monotonic() - self._step_start_time) * 1000

        # Build candidate records
        candidate_records: List[CandidateRecord] = []
        for c in (candidates or []):
            if not isinstance(c, dict):
                continue
            candidate_records.append(CandidateRecord(
                candidate_id=str(c.get("candidate_id", "")),
                element_type=str(c.get("element_type", "")),
                role=str(c.get("role", "")),
                text=str(c.get("text", ""))[:200],
                features=c.get("features", {}),
                score=float(c.get("score", 0.0)),
                was_chosen=str(c.get("candidate_id", "")) == chosen_candidate_id,
            ))

        # Build validation event records
        ve_records: List[ValidationEvent] = []
        for ve in (validation_events or []):
            if not isinstance(ve, dict):
                continue
            ve_records.append(ValidationEvent(
                source=str(ve.get("source", "")),
                event_type=str(ve.get("event_type", "")),
                success=bool(ve.get("success", False)),
                details=ve.get("details", {}),
                timestamp=str(ve.get("timestamp", "")),
            ))

        step = FullStepRecord(
            task_id=self._current_episode.task_id,
            task_text=self._current_episode.task_text,
            task_type=self._current_episode.task_type,
            step_index=step_index,
            timestamp=datetime.now(timezone.utc).isoformat(),
            url=url,
            page_title=page_title,
            page_graph_summary=page_graph_summary or {},
            dom_node_count=dom_node_count,
            candidates=candidate_records,
            candidate_count=len(candidate_records),
            chosen_action_type=chosen_action_type,
            chosen_action=chosen_action or {},
            chosen_candidate_id=chosen_candidate_id,
            expert_action=expert_action,
            expert_candidate_id=expert_candidate_id,
            validation_events=ve_records,
            score_delta=score_delta,
            cumulative_reward=self._cumulative_reward,
            verifier_status=verifier_status,
            verifier_confidence=verifier_confidence,
            made_progress=made_progress,
            error_type=error_type,
            error_details=error_details,
            step_duration_ms=step_duration_ms,
            token_usage=token_usage,
            url_changed=url_changed,
            dom_changed=bool(page_graph_summary),
            previous_action_type=self._previous_action_type,
            loop_count=self._loop_count,
            unique_urls_visited=len(self._visited_urls),
        )

        self._current_episode.steps.append(step)
        self._previous_action_type = chosen_action_type

    def end_episode(
        self,
        success: bool = False,
        final_score: float = 0.0,
    ) -> Optional[EpisodeRecord]:
        """Finalize and return the current episode. Optionally writes to disk."""
        if self._current_episode is None:
            return None

        episode = self._current_episode
        episode.success = success
        episode.final_score = final_score
        episode.total_steps = len(episode.steps)
        episode.total_duration_ms = (time.monotonic() - self._episode_start_time) * 1000
        episode.total_tokens = sum(s.token_usage for s in episode.steps)

        # Write to disk
        self._write_episode(episode)

        self._current_episode = None
        return episode

    def _write_episode(self, episode: EpisodeRecord) -> None:
        """Write episode to JSONL file."""
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, "episodes.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(episode.to_dict(), ensure_ascii=False) + "\n")

    @property
    def is_collecting(self) -> bool:
        """Whether an episode is currently being collected."""
        return self._current_episode is not None
