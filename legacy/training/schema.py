"""Lightweight training schema used by legacy exporters."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CandidateRecord:
    candidate_id: str = ""
    features: dict[str, float] = field(default_factory=dict)
    text: str = ""
    type: str = ""


@dataclass
class ValidationEvent:
    source: str = ""
    success: bool = False
    message: str = ""


@dataclass
class FullStepRecord:
    task_id: str = ""
    task_text: str = ""
    step_index: int = 0
    url: str = ""
    url_changed: bool = False
    dom_changed: bool = False
    chosen_action_type: str = ""
    chosen_action: dict[str, Any] = field(default_factory=dict)
    chosen_candidate_id: str = ""
    candidate_count: int = 0
    loop_count: int = 0
    unique_urls_visited: int = 0
    previous_action_type: str = ""
    score_delta: float = 0.0
    dom_node_count: int = 0
    page_title: str = ""
    verifier_status: str = ""
    made_progress: bool = False
    expert_action: dict[str, Any] | None = None
    expert_candidate_id: str | None = None
    error_type: str | None = None
    candidates: list[CandidateRecord] = field(default_factory=list)
    validation_events: list[ValidationEvent] = field(default_factory=list)


@dataclass
class EpisodeRecord:
    task_id: str = ""
    task_text: str = ""
    success: bool = False
    steps: list[FullStepRecord] = field(default_factory=list)
