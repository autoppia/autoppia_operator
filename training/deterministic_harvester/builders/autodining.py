"""Deterministic plans for `autodining` — same layout as :mod:`autocinema` (per-use-case action builders + enriched selectors)."""

from __future__ import annotations

import training._iwa_path  # noqa: F401
from training.deterministic_harvester.builders.common import DeterministicPlan
from training.deterministic_harvester.builders.iwa_enriched_planner import (
    build_iwa_enriched_deterministic_plan,
    iwa_enriched_action_builders,
)
from training.deterministic_harvester.normalizer import DeterministicTaskObjective
from training.deterministic_harvester.trajectory_selectors import list_iwa_use_cases

_PROJECT = "autodining"
_SOURCE = "iwa_p04_autodining"
_PROMPT = "IWA p04 autodining trajectory for"

AUTODINING_PLAN_BUILDERS = iwa_enriched_action_builders(_PROJECT)


def list_autodining_iwa_use_cases() -> frozenset[str]:
    return list_iwa_use_cases(_PROJECT)


def build_autodining_plan(objective: DeterministicTaskObjective) -> DeterministicPlan:
    return build_iwa_enriched_deterministic_plan(
        _PROJECT,
        objective,
        iwa_source_label=_SOURCE,
        prompt_preamble=_PROMPT,
    )


__all__ = [
    "AUTODINING_PLAN_BUILDERS",
    "build_autodining_plan",
    "list_autodining_iwa_use_cases",
]
