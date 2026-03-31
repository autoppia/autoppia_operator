from __future__ import annotations

from src.operator.agents.step_engine import _STEP_ENGINE, ApifiedWebAgent as StepApifiedWebAgent, StepEngine

ApifiedWebAgent = StepApifiedWebAgent
FSMApifiedWebAgent = StepApifiedWebAgent
FSMOperator = StepEngine
_FSM_OPERATOR = _STEP_ENGINE

__all__ = [
    "_FSM_OPERATOR",
    "_STEP_ENGINE",
    "ApifiedWebAgent",
    "FSMApifiedWebAgent",
    "FSMOperator",
    "StepApifiedWebAgent",
    "StepEngine",
]
