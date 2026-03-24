from __future__ import annotations

import os
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def candidate_text(value: Any) -> str | None:
    if isinstance(value, str):
        collapsed = " ".join(value.strip().split())
        return collapsed if collapsed else None
    if isinstance(value, int | float | bool):
        return str(value)
    if isinstance(value, list):
        for item in value:
            parsed = candidate_text(item)
            if parsed:
                return parsed
        return None
    if isinstance(value, dict):
        for key in (
            "content",
            "final_text",
            "final_answer",
            "summary",
            "answer",
            "result",
            "output",
            "text",
            "message",
        ):
            parsed = candidate_text(value.get(key))
            if parsed:
                return parsed
    return None


def normalize_demo_url(raw_url: str | None, *, force_localhost: bool | None = None) -> str:
    normalized = str(raw_url or "").strip()
    if not normalized:
        return normalized
    if force_localhost is None:
        force_localhost = env_bool("AGENT_FORCE_LOCALHOST_URLS", False)
    if not force_localhost:
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


def normalize_selector_payload(raw_selector: Any) -> dict[str, Any] | None:
    if not isinstance(raw_selector, dict):
        return None
    selector = dict(raw_selector)
    raw_type = str(selector.get("type") or "").strip()
    sel_type = raw_type.lower()
    case_sensitive = bool(selector.get("case_sensitive", False))

    def first_text(*keys: str) -> str:
        for key in keys:
            value = candidate_text(selector.get(key))
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
    if sel_type in {
        "text",
        "textselector",
        "textcontains",
        "textcontainsselector",
        "linktext",
        "partiallinktext",
    }:
        value = first_text("value", "text", "label", "name", "query")
        if not value:
            return None
        return {
            "type": "tagContainsSelector",
            "value": value,
            "case_sensitive": case_sensitive,
        }
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
        return {
            "type": "xpathSelector",
            "value": value,
            "case_sensitive": case_sensitive,
        }
    if sel_type == "tagcontainsselector":
        value = first_text("value", "text", "label")
        if not value:
            return None
        return {
            "type": "tagContainsSelector",
            "value": value,
            "case_sensitive": case_sensitive,
        }
    if sel_type in {
        "id",
        "class",
        "name",
        "href",
        "placeholder",
        "aria-label",
        "aria_label",
        "title",
        "role",
        "value",
        "type",
    }:
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
            return {
                "type": "tagContainsSelector",
                "value": value,
                "case_sensitive": case_sensitive,
            }
        return None
    return dict(selector)
