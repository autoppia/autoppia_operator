"""Deterministic plans for `autohealth` with enriched selector candidates."""

from __future__ import annotations

import training._iwa_path  # noqa: F401
from training.deterministic_harvester.builders.common import DeterministicPlan
from training.deterministic_harvester.builders.iwa_enriched_planner import (
    build_iwa_enriched_deterministic_plan,
    iwa_enriched_action_builders,
)
from training.deterministic_harvester.normalizer import DeterministicTaskObjective
from training.deterministic_harvester.trajectory_selectors import list_iwa_use_cases

_PROJECT = "autohealth"
_SOURCE = "iwa_p14_autohealth"
_PROMPT = "IWA p14 autohealth trajectory for"

AUTOHEALTH_PLAN_BUILDERS = iwa_enriched_action_builders(_PROJECT)


def list_autohealth_iwa_use_cases() -> frozenset[str]:
    return list_iwa_use_cases(_PROJECT)


def build_autohealth_plan(objective: DeterministicTaskObjective) -> DeterministicPlan:
    return build_iwa_enriched_deterministic_plan(
        _PROJECT,
        objective,
        iwa_source_label=_SOURCE,
        prompt_preamble=_PROMPT,
    )


__all__ = [
    "AUTOHEALTH_PLAN_BUILDERS",
    "build_autohealth_plan",
    "list_autohealth_iwa_use_cases",
]
