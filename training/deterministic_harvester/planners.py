from __future__ import annotations

from training.deterministic_harvester.builders.common import DeterministicPlan
from training.deterministic_harvester.builders.registry import build_registered_plan
from training.deterministic_harvester.normalizer import DeterministicTaskObjective


def build_deterministic_plan(objective: DeterministicTaskObjective) -> DeterministicPlan:
    return build_registered_plan(objective)


__all__ = ["DeterministicPlan", "build_deterministic_plan"]
