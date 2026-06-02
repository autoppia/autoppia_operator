from __future__ import annotations

import json
import re
from typing import Any

from autoppia_harvester.models import ToolCall


ACTION_ALIASES = {
    "browser.navigate": "navigate",
    "browser.click": "click",
    "browser.input": "type",
    "browser.type": "type",
    "browser.select_option": "select_dropdown",
    "browser.select_dropdown": "select_dropdown",
    "browser.send_keys": "send_keys",
    "browser.wait": "wait",
    "browser.done": "done",
    "input": "type",
    "fill": "type",
    "select_option": "select_dropdown",
    "select": "select_dropdown",
}


def extract_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty Claude output")
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            if isinstance(parsed.get("structured_output"), dict):
                return parsed["structured_output"]
            if isinstance(parsed.get("result"), str):
                return extract_json_object(parsed["result"])
            return parsed
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
    if fenced:
        parsed = json.loads(fenced.group(1))
        if isinstance(parsed, dict):
            return parsed

    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(raw[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("could not parse JSON object from Claude output")


def normalize_tool_name(name: str) -> str:
    clean = str(name or "").strip()
    return ACTION_ALIASES.get(clean, clean.removeprefix("browser."))


def normalize_tool_call(raw: Any) -> ToolCall | None:
    if not isinstance(raw, dict):
        return None

    if isinstance(raw.get("function"), dict):
        function = raw["function"]
        raw = {"name": function.get("name"), "arguments": function.get("arguments")}

    name = raw.get("name") or raw.get("action") or raw.get("type")
    if not name:
        return None

    arguments = raw.get("arguments")
    if arguments is None:
        arguments = raw.get("args")
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
            arguments = parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}

    return ToolCall(
        name=normalize_tool_name(str(name)),
        arguments=arguments,
        reasoning=str(raw.get("reasoning") or "") or None,
    )


def normalize_trajectory(payload: dict[str, Any]) -> list[ToolCall]:
    raw_items = payload.get("trajectory")
    if raw_items is None:
        raw_items = payload.get("tool_calls")
    if raw_items is None:
        raw_items = payload.get("actions")
    if not isinstance(raw_items, list):
        return []
    calls: list[ToolCall] = []
    for item in raw_items:
        call = normalize_tool_call(item)
        if call:
            calls.append(call)
    return calls
