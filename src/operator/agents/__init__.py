from __future__ import annotations

from src.operator.agents.fsm import ApifiedWebAgent as FSMApifiedWebAgent
from src.operator.agents.fsm import FSMOperator, _FSM_OPERATOR
ApifiedWebAgent = FSMApifiedWebAgent

__all__ = [
    "ApifiedWebAgent",
    "FSMApifiedWebAgent",
    "FSMOperator",
    "_FSM_OPERATOR",
]
