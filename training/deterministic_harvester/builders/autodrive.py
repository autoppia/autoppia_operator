"""Deterministic plans for `autodrive` backed by IWA trajectories."""

from __future__ import annotations

import training._iwa_path  # noqa: F401
from training.deterministic_harvester.builders.common import DeterministicPlan
from training.deterministic_harvester.builders.iwa_enriched_planner import (
    build_iwa_enriched_deterministic_plan,
    iwa_enriched_action_builders,
)
from training.deterministic_harvester.normalizer import DeterministicTaskObjective
from training.deterministic_harvester.trajectory_selectors import list_iwa_use_cases

_PROJECT = "autodrive"
_SOURCE = "iwa_p13_autodrive"
_PROMPT = "IWA p13 autodrive trajectory for"

AUTODRIVE_PLAN_BUILDERS = iwa_enriched_action_builders(_PROJECT)


def list_autodrive_iwa_use_cases() -> frozenset[str]:
    return list_iwa_use_cases(_PROJECT)


def build_autodrive_plan(objective: DeterministicTaskObjective) -> DeterministicPlan:
    return build_iwa_enriched_deterministic_plan(
        _PROJECT,
        objective,
        iwa_source_label=_SOURCE,
        prompt_preamble=_PROMPT,
    )


__all__ = [
    "AUTODRIVE_PLAN_BUILDERS",
    "build_autodrive_plan",
    "list_autodrive_iwa_use_cases",
]
