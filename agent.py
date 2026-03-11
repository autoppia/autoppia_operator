from __future__ import annotations

from typing import Any

import inspect
import json
import logging
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from fsm_operator import FSMOperator
from legacy_operator import ApifiedWebAgent as LegacyApifiedWebAgent
from llm_gateway import openai_chat_completions, openai_vision_chat_completions
from pricing import estimate_cost_usd

# Default this branch to OpenAI via the validator gateway.
os.environ.setdefault("LLM_PROVIDER", "openai")
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)

try:
    from autoppia_iwa.src.data_generation.tasks.classes import Task
    from autoppia_iwa.src.execution.actions.base import BaseAction
    from autoppia_iwa.src.web_agents.act_protocol import ACT_PROTOCOL_VERSION as IWA_ACT_PROTOCOL_VERSION
    from autoppia_iwa.src.web_agents.classes import IWebAgent
    import autoppia_iwa.src.execution.actions.actions  # noqa: F401

    _AUTOPPIA_IWA_IMPORT_OK = True
    _AUTOPPIA_IWA_IMPORT_ERROR = ""
except Exception:  # pragma: no cover
    IWebAgent = object  # type: ignore[assignment]
    Task = Any  # type: ignore[assignment]
    BaseAction = Any  # type: ignore[assignment]
    IWA_ACT_PROTOCOL_VERSION = "1.0"
    _AUTOPPIA_IWA_IMPORT_OK = False
    _AUTOPPIA_IWA_IMPORT_ERROR = "autoppia_iwa import failed in miner runtime"


app = FastAPI(title="Autoppia Web Agent API")
logger = logging.getLogger("autoppia_operator")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

_cors_kwargs = {
    "allow_origins": [
        "http://127.0.0.1",
        "http://localhost",
        "http://127.0.0.1:5060",
        "http://localhost:5060",
    ],
    "allow_origin_regex": r"chrome-extension://.*",
    "allow_credentials": False,
    "allow_methods": ["*"],
    "allow_headers": ["*"],
}
if "allow_private_network" in inspect.signature(CORSMiddleware.__init__).parameters:
    _cors_kwargs["allow_private_network"] = True
app.add_middleware(CORSMiddleware, **_cors_kwargs)



def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _use_vision() -> bool:
    return _env_bool("USE_VISION", False) or _env_bool("AGENT_USE_VISION", False)


def _is_tool_enabled(name: str) -> bool:
    canonical = _canonical_browser_tool_name(name)
    if not canonical:
        return False
    if canonical == "browser.screenshot" and not _use_vision():
        return False
    return True


_LOG_DECISIONS = _env_bool("AGENT_LOG_DECISIONS", False)
_FSM_OPERATOR = FSMOperator(
    llm_call=openai_chat_completions,
    vision_call=(openai_vision_chat_completions if _use_vision() else None),
)
_LEGACY_OPERATOR = LegacyApifiedWebAgent(id=os.getenv("WEB_AGENT_ID", "1"), name="AutoppiaOperatorMain")



def _log_trace(message: str) -> None:
    if _LOG_DECISIONS:
        logger.info(f"[AGENT_TRACE] {message}")


def _payload_log_context(payload: dict[str, Any]) -> dict[str, Any]:
    prompt = str(payload.get("prompt") or payload.get("task_prompt") or "")
    html = str(payload.get("snapshot_html") or "")
    history = payload.get("history") if isinstance(payload.get("history"), list) else []
    allowed_tools = payload.get("allowed_tools") if isinstance(payload.get("allowed_tools"), list) else []
    state_in = payload.get("state_in") if isinstance(payload.get("state_in"), dict) else {}
    screenshot = payload.get("screenshot")
    return {
        "task_id": str(payload.get("task_id") or ""),
        "step_index": int(payload.get("step_index") or 0),
        "url": _normalize_demo_url(str(payload.get("url") or "")),
        "prompt_len": len(prompt),
        "html_len": len(html),
        "history_len": len(history),
        "allowed_tools_len": len(allowed_tools),
        "state_keys": len(state_in),
        "has_screenshot": bool(screenshot),
    }


def _log_act_start(payload: dict[str, Any]) -> dict[str, Any]:
    ctx = _payload_log_context(payload)
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


def _log_act_finish(ctx: dict[str, Any], started_at: float, response_payload: dict[str, Any]) -> None:
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


def _log_act_failure(ctx: dict[str, Any], started_at: float, exc: Exception) -> None:
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


def _attach_operator_metrics(response_payload: dict[str, Any], *, started_at: float) -> dict[str, Any]:
    if not isinstance(response_payload, dict):
        return response_payload
    duration_ms = int((time.monotonic() - started_at) * 1000)
    metrics = response_payload.get("metrics") if isinstance(response_payload.get("metrics"), dict) else {}
    operator_metrics = metrics.get("operator") if isinstance(metrics.get("operator"), dict) else {}
    operator_metrics["duration_ms"] = int(duration_ms)
    metrics["operator"] = operator_metrics
    response_payload["metrics"] = metrics
    return response_payload


if not _AUTOPPIA_IWA_IMPORT_OK:
    logger.error(f"[AGENT_TRACE] autoppia_iwa import failed: {_AUTOPPIA_IWA_IMPORT_ERROR}")



def _normalize_demo_url(raw_url: str | None) -> str:
    normalized = str(raw_url or "").strip()
    if not normalized:
        return normalized
    if not _env_bool("AGENT_FORCE_LOCALHOST_URLS", False):
        return normalized
    try:
        if "://" not in normalized:
            if not normalized.startswith("/"):
                if "." in normalized or ":" in normalized:
                    parsed = urlsplit(f"http://{normalized}")
                    path = parsed.path or ""
                    return urlunsplit(("http", "localhost", path, parsed.query, parsed.fragment))
                normalized = f"/{normalized}"
            return f"http://localhost{normalized}"
        parsed = urlsplit(normalized)
        path = parsed.path or ""
        return urlunsplit(("http", "localhost", path, parsed.query, parsed.fragment))
    except Exception:
        return "http://localhost"



def _is_navigate_action_type(action_type: Any) -> bool:
    return str(action_type or "").strip().lower() in {"navigateaction", "navigate"}



def _is_done_action_type(action_type: Any) -> bool:
    return str(action_type or "").strip().lower() in {"doneaction", "finishaction", "done", "finish"}



def _is_request_user_input_action_type(action_type: Any) -> bool:
    return str(action_type or "").strip().lower().replace("-", "_") in {
        "requestuserinputaction",
        "request_user_input",
        "request_user_input_action",
    }



def _sanitize_action_payload(action_payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(action_payload or {})
    selector = _normalize_selector_payload(payload.get("selector"))
    if isinstance(selector, dict):
        payload["selector"] = selector
    elif "selector" in payload:
        payload.pop("selector", None)
    if _is_navigate_action_type(payload.get("type")):
        payload["url"] = _normalize_demo_url(payload.get("url"))
    return payload


def _normalize_selector_payload(raw_selector: Any) -> dict[str, Any] | None:
    if not isinstance(raw_selector, dict):
        return None
    selector = dict(raw_selector)
    raw_type = str(selector.get("type") or "").strip()
    sel_type = raw_type.lower()
    case_sensitive = bool(selector.get("case_sensitive", False))

    def first_text(*keys: str) -> str:
        for key in keys:
            value = _candidate_text(selector.get(key))
            if value:
                return value[:400]
        return ""

    def normalize_xpath(value: str) -> str:
        cleaned = str(value or "").strip()
        if cleaned.lower().startswith("xpath="):
            cleaned = cleaned[6:].strip()
        if cleaned.startswith("///"):
            cleaned = "//" + cleaned.lstrip("/")
        elif cleaned.startswith("/") and not cleaned.startswith("//"):
            cleaned = cleaned.lstrip("/")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    allowed_types = {"attributevalueselector", "tagcontainsselector", "xpathselector"}
    if sel_type in {"text", "textselector", "textcontains", "textcontainsselector", "linktext", "partiallinktext"}:
        value = first_text("value", "text", "label", "name", "query")
        if not value:
            return None
        return {"type": "tagContainsSelector", "value": value, "case_sensitive": case_sensitive}
    if sel_type in {"attribute", "attributevalueselector"}:
        attribute = first_text("attribute", "attr", "name")
        value = first_text("value", "text", "label")
        if not attribute or not value:
            return None
        return {
            "type": "attributeValueSelector",
            "attribute": attribute,
            "value": value,
            "case_sensitive": case_sensitive,
        }
    if sel_type == "xpathselector":
        value = normalize_xpath(first_text("value", "text", "xpath"))
        if not value:
            return None
        return {"type": "xpathSelector", "value": value, "case_sensitive": case_sensitive}
    if sel_type == "tagcontainsselector":
        value = first_text("value", "text", "label")
        if not value:
            return None
        return {"type": "tagContainsSelector", "value": value, "case_sensitive": case_sensitive}
    if sel_type in {"id", "class", "name", "href", "placeholder", "aria-label", "aria_label", "title", "role", "value", "type"}:
        value = first_text("value", "text", "label")
        if not value:
            return None
        attribute = "aria-label" if sel_type == "aria_label" else raw_type
        return {
            "type": "attributeValueSelector",
            "attribute": attribute,
            "value": value,
            "case_sensitive": case_sensitive,
        }
    if sel_type not in allowed_types:
        attribute = first_text("attribute", "attr", "name")
        value = first_text("value", "text", "label", "query")
        if attribute and value:
            return {
                "type": "attributeValueSelector",
                "attribute": attribute,
                "value": value,
                "case_sensitive": case_sensitive,
            }
        if value:
            return {"type": "tagContainsSelector", "value": value, "case_sensitive": case_sensitive}
        return None
    return dict(selector)



def _candidate_text(value: Any) -> str | None:
    if isinstance(value, str):
        collapsed = " ".join(value.strip().split())
        return collapsed if collapsed else None
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        for item in value:
            parsed = _candidate_text(item)
            if parsed:
                return parsed
        return None
    if isinstance(value, dict):
        for key in ("content", "final_text", "final_answer", "summary", "answer", "result", "output", "text", "message"):
            parsed = _candidate_text(value.get(key))
            if parsed:
                return parsed
    return None



def _extract_result_text_from_action(action: dict[str, Any] | None) -> str | None:
    return _candidate_text(action) if isinstance(action, dict) else None


def _task_from_payload(payload: dict[str, Any]) -> Task:
    task_payload = {
        "id": str(payload.get("task_id") or ""),
        "url": _normalize_demo_url(str(payload.get("url") or "")),
        "prompt": str(payload.get("prompt") or payload.get("task_prompt") or ""),
        "web_project_id": payload.get("web_project_id"),
    }
    try:
        if isinstance(Task, type):
            return Task(**task_payload)
    except Exception:
        pass
    return SimpleNamespace(**task_payload)


def _canonical_browser_tool_name(raw_name: str) -> str:
    name = str(raw_name or "").strip().lower()
    if not name:
        return ""
    if name == "user.request_input":
        return name
    suffix = name[8:] if name.startswith("browser.") else name
    if suffix == "evaluate":
        return ""
    alias = {
        "search": "search",
        "navigate": "navigate",
        "go_back": "go_back",
        "click": "click",
        "dblclick": "dblclick",
        "rightclick": "rightclick",
        "middleclick": "middleclick",
        "tripleclick": "tripleclick",
        "input": "input",
        "scroll": "scroll",
        "wait": "wait",
        "select_dropdown": "select_dropdown",
        "dropdown_options": "dropdown_options",
        "hover": "hover",
        "screenshot": "screenshot",
        "send_keys": "send_keys",
        "hold_key": "hold_key",
        "extract": "extract",
        "done": "done",
    }
    return f"browser.{alias.get(suffix, suffix)}"



def _action_to_tool_call(action: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(action, dict):
        return None
    action_type = str(action.get("type") or "").strip()
    if not action_type or _is_done_action_type(action_type):
        return None
    arguments = {key: value for key, value in action.items() if key != "type"}
    if _is_request_user_input_action_type(action_type):
        return {"name": "user.request_input", "arguments": arguments}
    mapping = {
        "navigateaction": "browser.navigate",
        "clickaction": "browser.click",
        "gobackaction": "browser.go_back",
        "searchaction": "browser.search",
        "typeaction": "browser.input",
        "fillaction": "browser.input",
        "scrollaction": "browser.scroll",
        "waitaction": "browser.wait",
        "selectaction": "browser.select_dropdown",
        "selectdropdownoptionaction": "browser.select_dropdown",
        "hoveraction": "browser.hover",
        "screenshotaction": "browser.screenshot",
        "getdropdownoptionsaction": "browser.dropdown_options",
        "doubleclickaction": "browser.dblclick",
        "rightclickaction": "browser.rightclick",
        "middleclickaction": "browser.middleclick",
        "tripleclickaction": "browser.tripleclick",
        "holdkeyaction": "browser.hold_key",
        "sendkeysaction": "browser.send_keys",
        "sendkeysiwaaction": "browser.send_keys",
        "extractaction": "browser.extract",
    }
    tool_name = mapping.get(action_type.lower())
    if not tool_name:
        suffix = action_type[:-6] if action_type.endswith("Action") else action_type
        tool_name = f"browser.{suffix.strip().lower()}"
    tool_name = _canonical_browser_tool_name(tool_name)
    if not _is_tool_enabled(tool_name):
        return None
    return _normalize_tool_call_payload({"name": tool_name, "arguments": arguments})


def _normalize_tool_call_payload(tool_call: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(tool_call, dict):
        return None
    name = _canonical_browser_tool_name(str(tool_call.get("name") or "").strip())
    if not name:
        return None
    if not _is_tool_enabled(name):
        return None
    arguments = dict(tool_call.get("arguments") or {}) if isinstance(tool_call.get("arguments"), dict) else {}
    selector = _normalize_selector_payload(arguments.get("selector"))
    if isinstance(selector, dict):
        arguments["selector"] = selector
    elif "selector" in arguments:
        arguments.pop("selector", None)
    if name == "browser.select_dropdown":
        value = _candidate_text(arguments.get("text")) or _candidate_text(arguments.get("value"))
        if value:
            arguments["text"] = value
        arguments.pop("value", None)
    return {"name": name, "arguments": arguments}


def _serialize_use_case(raw: Any) -> dict[str, str]:
    if isinstance(raw, dict):
        out = {
            "name": str(raw.get("name") or "").strip()[:80],
            "description": str(raw.get("description") or "").strip()[:400],
        }
        return {k: v for k, v in out.items() if v}
    out = {
        "name": str(getattr(raw, "name", "") or "").strip()[:80],
        "description": str(getattr(raw, "description", "") or "").strip()[:400],
    }
    return {k: v for k, v in out.items() if v}


@app.get("/health", summary="Health check")
async def health() -> dict[str, str]:
    return {"status": "healthy"}


class ApifiedWebAgent(IWebAgent):
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

    @staticmethod
    def supported_tool_definitions() -> list[dict[str, Any]]:
        return _collect_supported_tool_definitions()

    def capabilities_payload(self) -> dict[str, Any]:
        return {
            "name": str(self.name or "AutoppiaOperator"),
            "protocol_version": IWA_ACT_PROTOCOL_VERSION,
            "act_endpoint": "/act",
            "step_endpoint": "/step",
            "use_vision": _use_vision(),
            "supported_response_formats": ["tool_calls"],
            "supports_request_user_input": True,
            "supports_state_roundtrip": True,
            "tool_definitions": self.supported_tool_definitions(),
        }

    @staticmethod
    def _runtime_impl() -> str:
        raw = str(os.getenv("AGENT_RUNTIME_IMPL", "fsm") or "fsm").strip().lower()
        if raw in {"main", "legacy"}:
            return "main"
        return "fsm"

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
        ctx = _log_act_start(payload)
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
            response_payload = _attach_operator_metrics(response_payload, started_at=started_at)
            _log_act_finish(ctx, started_at, response_payload)
            return response_payload
        except Exception as exc:
            _log_act_failure(ctx, started_at, exc)
            raise

    async def act_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        runtime_impl = self._runtime_impl()
        if runtime_impl == "main":
            return await _LEGACY_OPERATOR.act_from_payload(payload)

        task_id = str(payload.get("task_id") or "")
        prompt = str(payload.get("prompt") or payload.get("task_prompt") or "")
        model_override = str(payload.get("model") or "").strip()
        url = _normalize_demo_url(str(payload.get("url") or ""))
        step_index = int(payload.get("step_index") or 0)
        history = payload.get("history") if isinstance(payload.get("history"), list) else []
        state_in = payload.get("state_in") if isinstance(payload.get("state_in"), dict) else {}
        include_reasoning = str(payload.get("include_reasoning") or payload.get("return_reasoning") or "").strip().lower() in {"1", "true", "yes"}
        return_metrics = _env_bool("AGENT_RETURN_METRICS", False)

        fsm_payload = {
            "task_id": task_id,
            "prompt": prompt,
            "url": url,
            "web_project_id": str(payload.get("web_project_id") or ""),
            "use_case": payload.get("use_case") if isinstance(payload.get("use_case"), dict) else {},
            "step_index": step_index,
            "snapshot_html": str(payload.get("snapshot_html") or ""),
            "screenshot": payload.get("screenshot"),
            "history": history,
            "state_in": state_in,
            "allowed_tools": payload.get("allowed_tools"),
            "include_reasoning": include_reasoning,
        }

        try:
            out = _FSM_OPERATOR.run(payload=fsm_payload, model_override=model_override)
        except Exception as exc:
            logger.exception(f"[AGENT_TRACE] strict_fsm_failed task_id={task_id} step_index={step_index} err={str(exc)}")
            raise HTTPException(status_code=500, detail="fsm_operator_failed")

        if not isinstance(out, dict):
            raise HTTPException(status_code=500, detail="fsm_operator_invalid_response")

        out["protocol_version"] = str(out.get("protocol_version") or IWA_ACT_PROTOCOL_VERSION)
        out["state_out"] = out.get("state_out") if isinstance(out.get("state_out"), dict) else {}
        out["actions"] = out.get("actions") if isinstance(out.get("actions"), list) else []

        usage = out.get("usage") if isinstance(out.get("usage"), dict) else None
        model_name = str(out.get("model") or model_override or os.getenv("OPENAI_MODEL", "gpt-5.2")).strip()
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
                out["usage"] = normalized_usage
                out["total_tokens"] = int(normalized_usage.get("total_tokens") or 0)
                try:
                    estimated_cost, _ = estimate_cost_usd(model_name, normalized_usage)
                    out["estimated_cost_usd"] = float(estimated_cost)
                except Exception:
                    pass

        if model_name:
            out["model"] = model_name

        if return_metrics:
            llm_usages = [dict(out["usage"])] if isinstance(out.get("usage"), dict) else []
            helper_models = [str(m).strip() for m in list(out.get("helper_models") or []) if str(m).strip()]
            raw_call_breakdown = out.get("call_breakdown") if isinstance(out.get("call_breakdown"), dict) else {}
            call_breakdown = {
                "policy_llm_calls": int(raw_call_breakdown.get("policy_llm_calls") or 0),
                "obs_extract_llm_calls": int(raw_call_breakdown.get("obs_extract_llm_calls") or 0),
                "vision_llm_calls": int(raw_call_breakdown.get("vision_llm_calls") or 0),
            }
            explicit_llm_calls = sum(call_breakdown.values())
            raw_usage_breakdown = out.get("usage_breakdown") if isinstance(out.get("usage_breakdown"), dict) else {}
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
            out["metrics"] = {
                "llm": {
                    "llm_calls": int(explicit_llm_calls or len(llm_usages)),
                    "llm_usages": llm_usages,
                    "model": model_name,
                    "helper_models": helper_models,
                    "call_breakdown": call_breakdown,
                    "usage_breakdown": usage_breakdown,
                }
            }
        return out


AutoppiaOperator = ApifiedWebAgent
OPERATOR = AutoppiaOperator(id=os.getenv("WEB_AGENT_ID", "1"), name="AutoppiaOperator")



def _collect_supported_tool_definitions() -> list[dict[str, Any]]:
    defs_fn = getattr(BaseAction, "all_function_definitions", None)
    if not callable(defs_fn):
        return []
    try:
        definitions = defs_fn()
    except Exception:
        return []
    if not isinstance(definitions, list):
        return []

    out: list[dict[str, Any]] = []
    for item in definitions:
        if not isinstance(item, dict):
            continue
        function_payload = item.get("function") if isinstance(item.get("function"), dict) else {}
        raw_name = str(function_payload.get("name") or "").strip()
        if not raw_name or raw_name == "done":
            continue
        namespaced = "user.request_input" if raw_name == "request_user_input" else f"browser.{raw_name}"
        if not _is_tool_enabled(namespaced):
            continue
        out.append(
            {
                "name": namespaced,
                "description": str(function_payload.get("description") or ""),
                "parameters": function_payload.get("parameters") if isinstance(function_payload.get("parameters"), dict) else {},
            }
        )
    return out



def _normalize_allowed_tool_names(allowed_tools: Any) -> set[str]:
    if not isinstance(allowed_tools, list):
        return set()
    out: set[str] = set()
    for item in allowed_tools:
        if not isinstance(item, dict):
            continue
        raw_name = str(item.get("name") or "").strip()
        if not raw_name and isinstance(item.get("function"), dict):
            function_name = str((item.get("function") or {}).get("name") or "").strip()
            if function_name:
                raw_name = "user.request_input" if function_name == "request_user_input" else _canonical_browser_tool_name(f"browser.{function_name}")
        if raw_name:
            canonical = _canonical_browser_tool_name(raw_name)
            if canonical and _is_tool_enabled(canonical):
                out.add(canonical)
    return out



def _act_http_response(
    raw_resp: dict[str, Any],
    actions: list[dict[str, Any]],
    *,
    allowed_tool_names: set[str] | None = None,
) -> dict[str, Any]:
    allowed = set(allowed_tool_names or [])
    content = _candidate_text(raw_resp.get("content"))
    done_from_actions = False
    tool_calls: list[dict[str, Any]] = []

    raw_tool_calls = raw_resp.get("tool_calls")
    if isinstance(raw_tool_calls, list):
        for raw_call in raw_tool_calls:
            if not isinstance(raw_call, dict):
                continue
            name = str(raw_call.get("name") or "").strip()
            if not name or not _is_tool_enabled(name) or (allowed and name not in allowed):
                continue
            normalized_call = _normalize_tool_call_payload(raw_call)
            if isinstance(normalized_call, dict):
                tool_calls.append(normalized_call)
    else:
        for action in actions:
            if not isinstance(action, dict):
                continue
            if _is_done_action_type(action.get("type")):
                done_from_actions = True
                if not content:
                    content = _extract_result_text_from_action(action)
                continue
            call = _action_to_tool_call(action)
            if not isinstance(call, dict):
                continue
            name = str(call.get("name") or "").strip()
            if allowed and name not in allowed:
                continue
            tool_calls.append(call)

    out: dict[str, Any] = {
        "protocol_version": str(raw_resp.get("protocol_version") or IWA_ACT_PROTOCOL_VERSION),
        "tool_calls": tool_calls,
        "content": (content if (bool(raw_resp.get("done")) or done_from_actions) and isinstance(content, str) and content.strip() else None),
        "reasoning": str(raw_resp.get("reasoning")).strip()[:200] if isinstance(raw_resp.get("reasoning"), str) and str(raw_resp.get("reasoning")).strip() else None,
        "state_out": raw_resp.get("state_out") if isinstance(raw_resp.get("state_out"), dict) else {},
        "done": bool(raw_resp.get("done")) or done_from_actions,
    }
    if isinstance(raw_resp.get("error"), str) and str(raw_resp.get("error")).strip():
        out["error"] = str(raw_resp.get("error")).strip()[:400]
    if isinstance(raw_resp.get("metrics"), dict):
        out["metrics"] = raw_resp.get("metrics")
        helper_models = [str(m).strip() for m in list(raw_resp.get("helper_models") or []) if str(m).strip()]
        llm_metrics = out["metrics"].get("llm") if isinstance(out["metrics"].get("llm"), dict) else None
        if helper_models and isinstance(llm_metrics, dict):
            llm_metrics["helper_models"] = helper_models
        if isinstance(raw_resp.get("call_breakdown"), dict) and isinstance(llm_metrics, dict):
            llm_metrics["call_breakdown"] = raw_resp.get("call_breakdown")
        if isinstance(raw_resp.get("usage_breakdown"), dict) and isinstance(llm_metrics, dict):
            llm_metrics["usage_breakdown"] = raw_resp.get("usage_breakdown")
    if isinstance(raw_resp.get("usage"), dict):
        out["usage"] = raw_resp.get("usage")
    if isinstance(raw_resp.get("total_tokens"), int):
        out["total_tokens"] = int(raw_resp.get("total_tokens") or 0)
    if isinstance(raw_resp.get("model"), str) and str(raw_resp.get("model")).strip():
        out["model"] = str(raw_resp.get("model")).strip()
    if isinstance(raw_resp.get("estimated_cost_usd"), (int, float)):
        out["estimated_cost_usd"] = float(raw_resp.get("estimated_cost_usd"))
    if isinstance(raw_resp.get("helper_models"), list):
        out["helper_models"] = [str(m).strip() for m in raw_resp.get("helper_models") if str(m).strip()]
    if isinstance(raw_resp.get("call_breakdown"), dict):
        out["call_breakdown"] = raw_resp.get("call_breakdown")
    if isinstance(raw_resp.get("usage_breakdown"), dict):
        out["usage_breakdown"] = raw_resp.get("usage_breakdown")
    if isinstance(raw_resp.get("action_rationales"), list):
        out["action_rationales"] = raw_resp.get("action_rationales")
    return out


@app.get("/capabilities", summary="Operator capabilities and protocol metadata")
async def capabilities() -> dict[str, Any]:
    return OPERATOR.capabilities_payload()


@app.post("/act", summary="Decide next agent actions")
async def act(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    try:
        return await OPERATOR.respond_from_payload(payload)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"act_failed:{type(exc).__name__}") from exc


@app.post("/step", summary="Alias for /act")
async def step(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    try:
        return await OPERATOR.respond_from_payload(payload)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"step_failed:{type(exc).__name__}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("agent:app", host="0.0.0.0", port=8000, reload=True)
