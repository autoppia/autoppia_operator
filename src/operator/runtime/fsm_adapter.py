from __future__ import annotations

from typing import Any

import os

from fastapi import HTTPException

from src.operator.api.act_protocol import _normalize_demo_url, env_bool
from src.operator.support.iwa import IWA_ACT_PROTOCOL_VERSION
from src.operator.support.telemetry import logger
from infra.pricing import estimate_cost_usd


def build_fsm_payload(payload: dict[str, Any]) -> dict[str, Any]:
    include_reasoning = str(payload.get("include_reasoning") or payload.get("return_reasoning") or "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    return {
        "task_id": str(payload.get("task_id") or ""),
        "prompt": str(payload.get("prompt") or payload.get("task_prompt") or ""),
        "url": _normalize_demo_url(str(payload.get("url") or "")),
        "web_project_id": str(payload.get("web_project_id") or ""),
        "use_case": payload.get("use_case") if isinstance(payload.get("use_case"), dict) else {},
        "step_index": int(payload.get("step_index") or 0),
        "snapshot_html": str(payload.get("snapshot_html") or ""),
        "screenshot": payload.get("screenshot"),
        "history": payload.get("history") if isinstance(payload.get("history"), list) else [],
        "state_in": payload.get("state_in") if isinstance(payload.get("state_in"), dict) else {},
        "allowed_tools": payload.get("allowed_tools"),
        "include_reasoning": include_reasoning,
    }


def normalize_fsm_output(
    out: Any,
    *,
    model_override: str,
    return_metrics: bool,
) -> dict[str, Any]:
    if not isinstance(out, dict):
        raise HTTPException(status_code=500, detail="fsm_operator_invalid_response")

    normalized = dict(out)
    normalized["protocol_version"] = str(normalized.get("protocol_version") or IWA_ACT_PROTOCOL_VERSION)
    normalized["state_out"] = normalized.get("state_out") if isinstance(normalized.get("state_out"), dict) else {}
    normalized["actions"] = normalized.get("actions") if isinstance(normalized.get("actions"), list) else []

    usage = normalized.get("usage") if isinstance(normalized.get("usage"), dict) else None
    model_name = str(normalized.get("model") or model_override or os.getenv("OPENAI_MODEL", "gpt-5.2")).strip()
    if usage is not None:
        try:
            normalized_usage = {
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "total_tokens": int(usage.get("total_tokens") or 0),
            }
        except Exception:
            normalized_usage = None
        if normalized_usage is not None:
            normalized["usage"] = normalized_usage
            normalized["total_tokens"] = int(normalized_usage.get("total_tokens") or 0)
            try:
                estimated_cost, _ = estimate_cost_usd(model_name, normalized_usage)
                normalized["estimated_cost_usd"] = float(estimated_cost)
            except Exception:
                pass

    if model_name:
        normalized["model"] = model_name

    if return_metrics:
        llm_usages = [dict(normalized["usage"])] if isinstance(normalized.get("usage"), dict) else []
        helper_models = [str(m).strip() for m in list(normalized.get("helper_models") or []) if str(m).strip()]
        raw_call_breakdown = normalized.get("call_breakdown") if isinstance(normalized.get("call_breakdown"), dict) else {}
        call_breakdown = {
            "policy_llm_calls": int(raw_call_breakdown.get("policy_llm_calls") or 0),
            "obs_extract_llm_calls": int(raw_call_breakdown.get("obs_extract_llm_calls") or 0),
            "vision_llm_calls": int(raw_call_breakdown.get("vision_llm_calls") or 0),
        }
        explicit_llm_calls = sum(call_breakdown.values())
        raw_usage_breakdown = normalized.get("usage_breakdown") if isinstance(normalized.get("usage_breakdown"), dict) else {}
        usage_breakdown = {
            "policy": {
                "prompt_tokens": int(((raw_usage_breakdown.get("policy") or {}).get("prompt_tokens") or 0)),
                "completion_tokens": int(((raw_usage_breakdown.get("policy") or {}).get("completion_tokens") or 0)),
                "total_tokens": int(((raw_usage_breakdown.get("policy") or {}).get("total_tokens") or 0)),
            },
            "obs_extract": {
                "prompt_tokens": int(((raw_usage_breakdown.get("obs_extract") or {}).get("prompt_tokens") or 0)),
                "completion_tokens": int(((raw_usage_breakdown.get("obs_extract") or {}).get("completion_tokens") or 0)),
                "total_tokens": int(((raw_usage_breakdown.get("obs_extract") or {}).get("total_tokens") or 0)),
            },
            "vision": {
                "prompt_tokens": int(((raw_usage_breakdown.get("vision") or {}).get("prompt_tokens") or 0)),
                "completion_tokens": int(((raw_usage_breakdown.get("vision") or {}).get("completion_tokens") or 0)),
                "total_tokens": int(((raw_usage_breakdown.get("vision") or {}).get("total_tokens") or 0)),
            },
        }
        normalized["metrics"] = {
            "llm": {
                "llm_calls": int(explicit_llm_calls or len(llm_usages)),
                "llm_usages": llm_usages,
                "model": model_name,
                "helper_models": helper_models,
                "call_breakdown": call_breakdown,
                "usage_breakdown": usage_breakdown,
            }
        }
    return normalized


def run_fsm_operator(fsm_operator: Any, payload: dict[str, Any], *, model_override: str) -> dict[str, Any]:
    try:
        out = fsm_operator.run(payload=build_fsm_payload(payload), model_override=model_override)
    except Exception as exc:
        logger.exception(
            f"[AGENT_TRACE] strict_fsm_failed task_id={str(payload.get('task_id') or '')} "
            f"step_index={int(payload.get('step_index') or 0)} err={str(exc)}"
        )
        raise HTTPException(status_code=500, detail="fsm_operator_failed")
    return normalize_fsm_output(out, model_override=model_override, return_metrics=env_bool("AGENT_RETURN_METRICS", False))
