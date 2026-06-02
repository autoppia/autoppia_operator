from __future__ import annotations

from src.operator.agents.step_engine import ApifiedWebAgent as StepApifiedWebAgent
from src.operator.agents.step_engine import StepEngine, _STEP_ENGINE

ApifiedWebAgent = StepApifiedWebAgent
FSMApifiedWebAgent = StepApifiedWebAgent
FSMOperator = StepEngine
_FSM_OPERATOR = _STEP_ENGINE

__all__ = [
    "ApifiedWebAgent",
    "StepApifiedWebAgent",
    "StepEngine",
    "FSMApifiedWebAgent",
    "FSMOperator",
    "_STEP_ENGINE",
    "_FSM_OPERATOR",
]
