from __future__ import annotations

import os

from src.operator.agents import ApifiedWebAgent, FSMApifiedWebAgent, StepApifiedWebAgent, StepEngine, _FSM_OPERATOR, _STEP_ENGINE
from src.operator.agents.operator import CleanModelInferenceOperator
from src.operator.api.step_protocol import _normalize_demo_url, _sanitize_action_payload, _task_from_payload


class AutoppiaOperator(ApifiedWebAgent):
    """Concrete subnet operator exposed by the HTTP server."""


def _build_operator() -> object:
    runtime = str(os.getenv("WEB_AGENT_RUNTIME", "") or os.getenv("AUTOPPIA_OPERATOR_RUNTIME", "")).strip().lower()
    if runtime in {"structured", "structured_inference", "operator", "clean", "clean_model"}:
        return CleanModelInferenceOperator(id=os.getenv("WEB_AGENT_ID", "1"), name="CleanModelInferenceOperator")
    if runtime in {"heuristic", "heuristic_structured", "structured_heuristic"}:
        from src.operator.agents.heuristic_runtime.structured_heuristic_operator import StructuredInferenceOperator as HeuristicStructuredInferenceOperator
        return HeuristicStructuredInferenceOperator(id=os.getenv("WEB_AGENT_ID", "1"), name="HeuristicStructuredInferenceOperator")
    if runtime in {"custom", "custom_operator", "skills", "skills_operator"}:
        from src.operator.custom_operator import build_custom_operator
        return build_custom_operator()
    return AutoppiaOperator(id=os.getenv("WEB_AGENT_ID", "1"), name="AutoppiaOperator")


OPERATOR = _build_operator()

FSMAgent = FSMApifiedWebAgent
StepAgent = StepApifiedWebAgent

__all__ = [
    "ApifiedWebAgent",
    "AutoppiaOperator",
    "FSMAgent",
    "StepAgent",
    "StepEngine",
    "OPERATOR",
    "_STEP_ENGINE",
    "_FSM_OPERATOR",
    "_normalize_demo_url",
    "_sanitize_action_payload",
    "_task_from_payload",
]
