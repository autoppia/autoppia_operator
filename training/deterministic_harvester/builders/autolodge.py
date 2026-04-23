"""Deterministic plans for `autolodge` — same layout as :mod:`autocinema` (per-use-case action builders + enriched selectors)."""

from __future__ import annotations

import training._iwa_path  # noqa: F401
from training.deterministic_harvester.builders.common import DeterministicPlan
from training.deterministic_harvester.builders.iwa_enriched_planner import (
    build_iwa_enriched_deterministic_plan,
    iwa_enriched_action_builders,
)
from training.deterministic_harvester.normalizer import DeterministicTaskObjective
from training.deterministic_harvester.trajectory_selectors import list_iwa_use_cases

_PROJECT = "autolodge"
_SOURCE = "iwa_p08_autolodge"
_PROMPT = "IWA p08 autolodge trajectory for"

AUTOLODGE_PLAN_BUILDERS = iwa_enriched_action_builders(_PROJECT)


def list_autolodge_iwa_use_cases() -> frozenset[str]:
    return list_iwa_use_cases(_PROJECT)


def build_autolodge_plan(objective: DeterministicTaskObjective) -> DeterministicPlan:
    return build_iwa_enriched_deterministic_plan(
        _PROJECT,
        objective,
        iwa_source_label=_SOURCE,
        prompt_preamble=_PROMPT,
    )


__all__ = [
    "AUTOLODGE_PLAN_BUILDERS",
    "build_autolodge_plan",
    "list_autolodge_iwa_use_cases",
]
