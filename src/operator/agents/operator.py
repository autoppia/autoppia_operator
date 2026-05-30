from __future__ import annotations

from src.operator.agents.step_engine import ApifiedWebAgent as StepApifiedWebAgent
from src.operator.agents.heuristic_runtime.structured_heuristic_operator import (
    StructuredInferenceOperator as HeuristicStructuredInferenceOperator,
)


class CleanModelInferenceOperator(StepApifiedWebAgent):
    """Clean inference path: the model decides tool calls directly."""

    @staticmethod
    def _runtime_impl() -> str:
        return "clean_model_inference"


# Keep the historical export name for deterministic structured tests/tools.
StructuredInferenceOperator = HeuristicStructuredInferenceOperator

__all__ = [
    "CleanModelInferenceOperator",
    "StructuredInferenceOperator",
]
