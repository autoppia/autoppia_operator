from __future__ import annotations

import copy
import re
from typing import Any

from .models import Skill, SkillRegistry


class SkillRunner:
    def __init__(self, registry: SkillRegistry) -> None:
        self.registry = registry

    def render_action(self, action: dict[str, Any], *, prompt: str) -> dict[str, Any]:
        rendered = copy.deepcopy(action)
        values = self._values_from_prompt(prompt)
        values.setdefault("base_url", self.registry.base_url.rstrip("/"))
        values.update({k: v for k, v in self.registry.defaults.items() if k not in values or not values[k]})
        return self._render_value(rendered, values)

    def action_at(self, skill: Skill, index: int, *, prompt: str) -> dict[str, Any] | None:
        if index < 0 or index >= len(skill.steps):
            return None
        return self.render_action(skill.steps[index].action, prompt=prompt)

    def _values_from_prompt(self, prompt: str) -> dict[str, str]:
        text = str(prompt or "")
        values: dict[str, str] = {}
        for key in ("username", "password", "email", "name", "subject", "message"):
            match = re.search(rf"{key}\s+(?:equals|is|=)\s+['\"]?([^'\".,;]+)", text, flags=re.IGNORECASE)
            if match:
                values[key] = str(match.group(1) or "").strip()
        if "username" not in values:
            match = re.search(r"username\s+([A-Za-z0-9_.@+-]{1,80})", text, flags=re.IGNORECASE)
            if match:
                values["username"] = str(match.group(1) or "").strip()
        if "password" not in values:
            match = re.search(r"password\s+([^\\s,.;]+)", text, flags=re.IGNORECASE)
            if match:
                values["password"] = str(match.group(1) or "").strip()
        search = re.search(r"(?:search|find|look for)\s+(?:for\s+)?(.+?)(?:\s+in\s+autocinema|\s+in\s+the\s+catalog|$)", text, flags=re.IGNORECASE)
        if search:
            values["search_term"] = str(search.group(1) or "").strip(" .'\"")
        return values

    def _render_value(self, value: Any, values: dict[str, str]) -> Any:
        if isinstance(value, str):
            out = value
            for key, replacement in values.items():
                out = out.replace("{" + key + "}", replacement)
            return out
        if isinstance(value, list):
            return [self._render_value(item, values) for item in value]
        if isinstance(value, dict):
            return {key: self._render_value(item, values) for key, item in value.items()}
        return value
