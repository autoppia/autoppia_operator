from __future__ import annotations

import os

from src.operator.agents import ApifiedWebAgent, FSMApifiedWebAgent, StepApifiedWebAgent, StepEngine, _FSM_OPERATOR, _STEP_ENGINE
from src.operator.api.step_protocol import _normalize_demo_url, _sanitize_action_payload, _task_from_payload


class AutoppiaOperator(ApifiedWebAgent):
    """Concrete subnet operator exposed by the HTTP server."""


OPERATOR = AutoppiaOperator(id=os.getenv("WEB_AGENT_ID", "1"), name="AutoppiaOperator")

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
