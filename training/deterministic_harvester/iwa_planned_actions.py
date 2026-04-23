"""Convert IWA demo trajectory actions to operator `planned_actions` dicts for guided replay."""

from __future__ import annotations

from typing import Any

from autoppia_iwa.src.demo_webs.config import demo_web_projects
from autoppia_iwa.src.demo_webs.trajectory_registry import remap_url_to_frontend
from autoppia_iwa.src.execution.actions.base import BaseAction, Selector

# Ensure autoppia_iwa is on sys.path when used from operator tests.
import training._iwa_path  # noqa: F401

__all__ = [
    "frontend_url_for_project",
    "iwa_actions_to_planned_actions",
    "iwa_to_dict_planned",
]


def frontend_url_for_project(project_id: str) -> str:
    pid = str(project_id or "").strip()
    for project in demo_web_projects:
        if str(getattr(project, "id", "") or "").strip() == pid:
            return str(getattr(project, "frontend_url", "") or "").strip() or "http://localhost"
    return "http://localhost"


def _selector_to_dict(selector: Any) -> dict[str, Any] | None:
    if selector is None:
        return None
    if isinstance(selector, Selector):
        return selector.model_dump()
    if isinstance(selector, dict):
        return dict(selector)
    if hasattr(selector, "model_dump"):
        try:
            return dict(selector.model_dump())  # type: ignore[no-untyped-call]
        except Exception:
            return None
    return None


def iwa_to_dict_planned(
    action: Any,
    *,
    frontend_url: str | None = None,
) -> dict[str, Any]:
    """Map one IWA BaseAction (or dict) to a guided-replay `planned_action` dict."""
    if action is None:
        raise ValueError("action is required")
    if isinstance(action, BaseAction):
        raw = action.model_dump()
    elif isinstance(action, dict):
        raw = dict(action)
    else:
        raise TypeError(f"Unsupported action type: {type(action)}")

    action_type = str(raw.get("type") or "").strip()
    out: dict[str, Any] = {"type": action_type}

    if action_type == "NavigateAction":
        url = str(raw.get("url") or "").strip()
        if url and (frontend_url or "").strip():
            out["url"] = remap_url_to_frontend(url, str(frontend_url).strip())
        else:
            out["url"] = url
        out["go_back"] = bool(raw.get("go_back", False))
        out["go_forward"] = bool(raw.get("go_forward", False))
        return out

    if action_type == "WaitAction":
        out["time_seconds"] = raw.get("time_seconds")
        if raw.get("timeout_seconds") is not None:
            out["timeout_seconds"] = raw.get("timeout_seconds")
        sel = _selector_to_dict(raw.get("selector"))
        if sel:
            out["selector_candidates"] = [sel]
        if raw.get("field_name") is not None:
            out["field_name"] = str(raw.get("field_name") or "")
        return out

    if action_type in {"ClickAction", "TypeAction", "SelectAction", "HoverAction", "ScrollAction"}:
        sel = _selector_to_dict(raw.get("selector"))
        if sel:
            out["selector_candidates"] = [sel]
        if "text" in raw:
            out["text"] = str(raw.get("text") or "")
        if "value" in raw:
            out["value"] = str(raw.get("value") or "")
        if raw.get("field_name") is not None:
            out["field_name"] = str(raw.get("field_name") or "")
        return out

    for key, value in raw.items():
        if key in {"type", "selector"}:
            continue
        out[key] = value
    sel = _selector_to_dict(raw.get("selector"))
    if sel:
        out["selector_candidates"] = [sel]
    return out


def iwa_actions_to_planned_actions(
    actions: list[Any] | None,
    *,
    frontend_url: str | None = None,
) -> list[dict[str, Any]]:
    if not actions:
        return []
    return [iwa_to_dict_planned(a, frontend_url=frontend_url) for a in actions]
