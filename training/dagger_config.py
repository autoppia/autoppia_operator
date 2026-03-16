"""Configuration for DAgger (Dataset Aggregation) integration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal


@dataclass
class DAggerConfig:
    """Configuration for the DAgger oracle and selective querying.

    Attributes:
        expert_model: Which model to use as the expert oracle.
        uncertainty_threshold: Query expert when ranker uncertainty exceeds this.
        loop_threshold: Query expert after this many consecutive same-action loops.
        no_progress_threshold: Query expert after N steps without progress.
        query_on_irreversible: Whether to query before irreversible actions.
        max_queries_per_episode: Budget limit per episode.
        max_total_queries: Global budget limit.
        enabled: Whether DAgger is active.
    """

    expert_model: str = "claude-sonnet-4-20250514"
    uncertainty_threshold: float = 0.7
    loop_threshold: int = 3
    no_progress_threshold: int = 4
    query_on_irreversible: bool = True
    max_queries_per_episode: int = 10
    max_total_queries: int = 500
    enabled: bool = False

    # Action types to flag as potentially irreversible (triggers expert review).
    # These are action type identifiers from the FSM action schema, not keywords.
    irreversible_action_types: List[str] = field(default_factory=list)
