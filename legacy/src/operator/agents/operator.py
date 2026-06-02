from __future__ import annotations

from src.operator.agents.step_engine import ApifiedWebAgent as StepApifiedWebAgent


class CleanModelInferenceOperator(StepApifiedWebAgent):
    """Clean inference path: the model decides tool calls directly."""

    @staticmethod
    def _runtime_impl() -> str:
        return "clean_model_inference"


# Keep the historical export name so entrypoint/import sites stay stable.
StructuredInferenceOperator = CleanModelInferenceOperator

__all__ = [
    "CleanModelInferenceOperator",
    "StructuredInferenceOperator",
]
