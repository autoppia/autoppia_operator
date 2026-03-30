from __future__ import annotations

import os

from src.operator.agents import ApifiedWebAgent, FSMApifiedWebAgent, _FSM_OPERATOR
from src.operator.api.act_protocol import _normalize_demo_url, _sanitize_action_payload, _task_from_payload


class AutoppiaOperator(ApifiedWebAgent):
    """Concrete subnet operator exposed by the HTTP server."""


OPERATOR = AutoppiaOperator(id=os.getenv("WEB_AGENT_ID", "1"), name="AutoppiaOperator")

FSMAgent = FSMApifiedWebAgent

__all__ = [
    "ApifiedWebAgent",
    "AutoppiaOperator",
    "FSMAgent",
    "OPERATOR",
    "_FSM_OPERATOR",
    "_normalize_demo_url",
    "_sanitize_action_payload",
    "_task_from_payload",
]
