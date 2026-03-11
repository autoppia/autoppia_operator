from __future__ import annotations

import os

from src.operator.agents import FSMApifiedWebAgent
from src.operator.agents import _FSM_OPERATOR as _CANONICAL_FSM_OPERATOR
from src.operator.api.act_protocol import _normalize_demo_url, _sanitize_action_payload, _task_from_payload
from src.operator.runtime.fsm_adapter import run_fsm_operator
from src.operator.api.server import app  # noqa: E402

_FSM_OPERATOR = _CANONICAL_FSM_OPERATOR


class _CompatFSMApifiedWebAgent(FSMApifiedWebAgent):
    async def act_from_payload(self, payload: dict[str, object]) -> dict[str, object]:
        model_override = str(payload.get("model") or "").strip()
        return run_fsm_operator(_FSM_OPERATOR, payload, model_override=model_override)

ApifiedWebAgent = _CompatFSMApifiedWebAgent


class AutoppiaOperator(ApifiedWebAgent):
    """Compatibility entrypoint for callers that still import src.operator.agent."""


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
    "app",
]
