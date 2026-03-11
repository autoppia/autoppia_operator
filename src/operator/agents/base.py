from __future__ import annotations

from typing import Any

import json
import os
import time

from src.operator.api.act_protocol import (
    _act_http_response,
    _collect_supported_tool_definitions,
    _normalize_allowed_tool_names,
    _normalize_demo_url,
    _sanitize_action_payload,
    _serialize_use_case,
    use_vision,
)
from src.operator.support.iwa import BaseAction, IWA_ACT_PROTOCOL_VERSION, IWebAgent, Task
from src.operator.support.telemetry import (
    attach_operator_metrics,
    log_act_failure,
    log_act_finish,
    log_act_start,
    logger,
)


class BaseApifiedWebAgent(IWebAgent):
    def __init__(self, id: str = "1", name: str = "AutoppiaOperator") -> None:
        self.id = str(id)
        self.name = str(name)

    async def act(
        self,
        *,
        task: Task,
        snapshot_html: str,
        screenshot: str | bytes | None = None,
        url: str,
        step_index: int,
        history: list[dict[str, Any]] | None = None,
        state: dict[str, Any] | None = None,
    ) -> list[BaseAction]:
        task_id = str(getattr(task, "id", "") or "")
        create_action_fn = getattr(BaseAction, "create_action", None)
        payload: dict[str, Any] = {
            "task_id": task_id,
            "prompt": str(getattr(task, "prompt", "") or ""),
            "snapshot_html": snapshot_html,
            "screenshot": screenshot,
            "url": url,
            "web_project_id": str(getattr(task, "web_project_id", "") or ""),
            "use_case": _serialize_use_case(getattr(task, "use_case", None)),
            "step_index": int(step_index),
            "history": history or [],
        }
        if isinstance(state, dict):
            payload["state_in"] = dict(state)
        raw = await self.act_from_payload(payload)
        normalized = self._normalize_actions(raw, task_id=task_id, step_index=int(step_index))

        actions: list[BaseAction] = []
        for action in normalized:
            try:
                converted = create_action_fn(action) if callable(create_action_fn) else None
            except Exception as exc:
                logger.error(
                    f"[AGENT_TRACE] create_action failed task_id={task_id} step_index={int(step_index)} "
                    f"action_type={str(action.get('type') or '')} err={str(exc)} "
                    f"payload={json.dumps(action, ensure_ascii=True)[:500]}"
                )
                continue
            if converted is not None:
                actions.append(converted)
        return actions

    async def step(
        self,
        *,
        task: Task,
        snapshot_html: str,
        screenshot: str | bytes | None = None,
        url: str,
        step_index: int,
        history: list[dict[str, Any]] | None = None,
        state: dict[str, Any] | None = None,
    ) -> list[BaseAction]:
        return await self.act(
            task=task,
            snapshot_html=snapshot_html,
            screenshot=screenshot,
            url=url,
            step_index=step_index,
            history=history,
            state=state,
        )

    async def step_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self.act_from_payload(payload)

    async def solve_task(self, task: Task) -> Any:
        raise NotImplementedError("solve_task is not supported by this miner; use iterative /act calls")

    @staticmethod
    def supported_tool_definitions() -> list[dict[str, Any]]:
        return _collect_supported_tool_definitions()

    def capabilities_payload(self) -> dict[str, Any]:
        return {
            "name": str(self.name or "AutoppiaOperator"),
            "protocol_version": IWA_ACT_PROTOCOL_VERSION,
            "act_endpoint": "/act",
            "step_endpoint": "/step",
            "use_vision": use_vision(),
            "supported_response_formats": ["tool_calls"],
            "supports_request_user_input": True,
            "supports_state_roundtrip": True,
            "tool_definitions": self.supported_tool_definitions(),
        }

    def _normalize_actions(self, raw_resp: Any, *, task_id: str, step_index: int) -> list[dict[str, Any]]:
        actions = raw_resp.get("actions") if isinstance(raw_resp, dict) else []
        normalized: list[dict[str, Any]] = []
        for action in actions if isinstance(actions, list) else []:
            try:
                payload = action if isinstance(action, dict) else action.model_dump(exclude_none=True)
            except Exception as exc:
                logger.error(
                    f"[AGENT_TRACE] /act action normalization failed task_id={task_id} step_index={step_index} "
                    f"err={str(exc)} raw={str(action)[:500]}"
                )
                continue
            normalized.append(_sanitize_action_payload(payload))
        return normalized

    async def respond_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        ctx = log_act_start(payload, normalize_url=_normalize_demo_url)
        started_at = time.monotonic()
        task_id = str(ctx.get("task_id") or "")
        step_index = int(ctx.get("step_index") or 0)
        try:
            raw_resp = await self.step_from_payload(payload)
            normalized = self._normalize_actions(raw_resp, task_id=task_id, step_index=step_index)
            allowed_tool_names = _normalize_allowed_tool_names(payload.get("allowed_tools"))
            response_payload = _act_http_response(
                raw_resp if isinstance(raw_resp, dict) else {},
                normalized,
                allowed_tool_names=allowed_tool_names,
            )
            response_payload = attach_operator_metrics(response_payload, started_at=started_at)
            log_act_finish(ctx, started_at, response_payload)
            return response_payload
        except Exception as exc:
            log_act_failure(ctx, started_at, exc)
            raise
