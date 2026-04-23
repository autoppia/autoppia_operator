"""Deterministic plans for `autocalendar` — same layout as :mod:`autocinema` (per-use-case action builders + enriched selectors)."""

from __future__ import annotations

import training._iwa_path  # noqa: F401
from training.deterministic_harvester.builders.common import DeterministicPlan
from training.deterministic_harvester.builders.iwa_enriched_planner import (
    build_iwa_enriched_deterministic_plan,
    iwa_enriched_action_builders,
)
from training.deterministic_harvester.normalizer import DeterministicTaskObjective
from training.deterministic_harvester.trajectory_selectors import list_iwa_use_cases

_PROJECT = "autocalendar"
_SOURCE = "iwa_p11_autocalendar"
_PROMPT = "IWA p11 autocalendar trajectory for"

AUTOCALENDAR_PLAN_BUILDERS = iwa_enriched_action_builders(_PROJECT)


def list_autocalendar_iwa_use_cases() -> frozenset[str]:
    return list_iwa_use_cases(_PROJECT)


def build_autocalendar_plan(objective: DeterministicTaskObjective) -> DeterministicPlan:
    return build_iwa_enriched_deterministic_plan(
        _PROJECT,
        objective,
        iwa_source_label=_SOURCE,
        prompt_preamble=_PROMPT,
    )


__all__ = [
    "AUTOCALENDAR_PLAN_BUILDERS",
    "build_autocalendar_plan",
    "list_autocalendar_iwa_use_cases",
]
