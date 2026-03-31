from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from src.operator.agents.step_engine.models import CanonicalBrowserState
from src.operator.runtime.fsm_adapter import normalize_fsm_output as normalize_step_output
from src.operator.support.utils import env_bool
from src.operator.support.telemetry import logger


def build_step_payload(payload: dict[str, Any]) -> CanonicalBrowserState:
    from src.operator.runtime.fsm_adapter import build_fsm_payload

    return CanonicalBrowserState.model_validate(build_fsm_payload(payload))


def run_step_engine(step_engine: Any, payload: dict[str, Any], *, model_override: str) -> dict[str, Any]:
    try:
        out = step_engine.run(payload=build_step_payload(payload), model_override=model_override)
    except Exception as exc:
        logger.exception(
            f"[AGENT_TRACE] step_engine_failed task_id={str(payload.get('task_id') or '')} "
            f"step_index={int(payload.get('step_index') or 0)} err={str(exc)}"
        )
        raise HTTPException(status_code=500, detail="step_engine_failed")
    return normalize_step_output(out, model_override=model_override, return_metrics=env_bool("AGENT_RETURN_METRICS", False))
