from __future__ import annotations

from typing import Any

import logging
import time

from src.operator.support.utils import env_bool

logger = logging.getLogger("autoppia_operator")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

LOG_DECISIONS = env_bool("AGENT_LOG_DECISIONS", False)


def log_trace(message: str) -> None:
    if LOG_DECISIONS:
        logger.info(f"[AGENT_TRACE] {message}")


def payload_log_context(payload: dict[str, Any], *, normalize_url: callable) -> dict[str, Any]:
    prompt = str(payload.get("prompt") or payload.get("task_prompt") or "")
    html = str(payload.get("snapshot_html") or "")
    history = payload.get("history") if isinstance(payload.get("history"), list) else []
    allowed_tools = payload.get("allowed_tools") if isinstance(payload.get("allowed_tools"), list) else []
    state_in = payload.get("state_in") if isinstance(payload.get("state_in"), dict) else {}
    screenshot = payload.get("screenshot")
    return {
        "task_id": str(payload.get("task_id") or ""),
        "step_index": int(payload.get("step_index") or 0),
        "url": normalize_url(str(payload.get("url") or "")),
        "prompt_len": len(prompt),
        "html_len": len(html),
        "history_len": len(history),
        "allowed_tools_len": len(allowed_tools),
        "state_keys": len(state_in),
        "has_screenshot": bool(screenshot),
    }


def log_act_start(payload: dict[str, Any], *, normalize_url: callable) -> dict[str, Any]:
    ctx = payload_log_context(payload, normalize_url=normalize_url)
    logger.info(
        "[ACT] start task_id=%s step=%s url=%s prompt_len=%s html_len=%s history_len=%s allowed_tools=%s state_keys=%s screenshot=%s",
        ctx["task_id"],
        ctx["step_index"],
        ctx["url"],
        ctx["prompt_len"],
        ctx["html_len"],
        ctx["history_len"],
        ctx["allowed_tools_len"],
        ctx["state_keys"],
        int(bool(ctx["has_screenshot"])),
    )
    return ctx


def log_act_finish(ctx: dict[str, Any], started_at: float, response_payload: dict[str, Any]) -> None:
    tool_calls = response_payload.get("tool_calls") if isinstance(response_payload.get("tool_calls"), list) else []
    state_out = response_payload.get("state_out") if isinstance(response_payload.get("state_out"), dict) else {}
    content = str(response_payload.get("content") or "") if isinstance(response_payload, dict) else ""
    reasoning = str(response_payload.get("reasoning") or "") if isinstance(response_payload, dict) else ""
    operator_metrics = (
        (response_payload.get("metrics") or {}).get("operator")
        if isinstance(response_payload.get("metrics"), dict)
        else {}
    )
    duration_ms = int((operator_metrics or {}).get("duration_ms") or 0) or int((time.monotonic() - started_at) * 1000)
    logger.info(
        "[ACT] finish task_id=%s step=%s done=%s tool_calls=%s content_len=%s reasoning_len=%s state_out_keys=%s duration_ms=%s",
        ctx.get("task_id", ""),
        ctx.get("step_index", 0),
        int(bool(response_payload.get("done"))),
        len(tool_calls),
        len(content),
        len(reasoning),
        len(state_out),
        duration_ms,
    )


def log_act_failure(ctx: dict[str, Any], started_at: float, exc: Exception) -> None:
    duration_ms = int((time.monotonic() - started_at) * 1000)
    logger.exception(
        "[ACT] failure task_id=%s step=%s url=%s duration_ms=%s err_type=%s err=%s",
        ctx.get("task_id", ""),
        ctx.get("step_index", 0),
        ctx.get("url", ""),
        duration_ms,
        type(exc).__name__,
        str(exc),
    )


def attach_operator_metrics(response_payload: dict[str, Any], *, started_at: float) -> dict[str, Any]:
    if not isinstance(response_payload, dict):
        return response_payload
    if not isinstance(response_payload.get("metrics"), dict):
        return response_payload
    duration_ms = int((time.monotonic() - started_at) * 1000)
    metrics = response_payload.get("metrics") if isinstance(response_payload.get("metrics"), dict) else {}
    operator_metrics = metrics.get("operator") if isinstance(metrics.get("operator"), dict) else {}
    operator_metrics["duration_ms"] = int(duration_ms)
    metrics["operator"] = operator_metrics
    response_payload["metrics"] = metrics
    return response_payload
