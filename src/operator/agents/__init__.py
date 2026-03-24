from __future__ import annotations

from src.operator.agents.fsm import _FSM_OPERATOR, ApifiedWebAgent as FSMApifiedWebAgent, FSMOperator

ApifiedWebAgent = FSMApifiedWebAgent

__all__ = [
    "_FSM_OPERATOR",
    "ApifiedWebAgent",
    "FSMApifiedWebAgent",
    "FSMOperator",
]
