from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import inspect

from src.operator.support.utils import normalize_selector_payload


SUPPORTED_TRAJECTORY_ACTIONS = {
    "ClickAction",
    "TypeAction",
    "SendKeysAction",
    "NavigateAction",
}

_ATTRIBUTE_FORMATS = {
    "placeholder": "[placeholder='{value}']",
    "name": "[name='{value}']",
    "role": "[role='{value}']",
    "value": "[value='{value}']",
    "type": "[type='{value}']",
    "aria-label": "[aria-label='{value}']",
    "aria-labelledby": "[aria-labelledby='{value}']",
    "data-testid": "[data-testid='{value}']",
    "data-custom": "[data-custom='{value}']",
    "href": "a[href='{value}']",
    "title": "[title='{value}']",
}


class TrajectoryMapperError(ValueError):
    """Raised when a trajectory action cannot be mapped safely."""


class TrajectoryExecutionError(RuntimeError):
    """Raised when an already-mapped action fails at execution time."""

    def __init__(
        self,
        *,
        step_index: int,
        action: dict[str, Any],
        mapped_action: dict[str, Any],
        playwright_command: str,
        cause: Exception,
    ) -> None:
        self.step_index = int(step_index)
        self.action = dict(action)
        self.mapped_action = dict(mapped_action)
        self.playwright_command = str(playwright_command)
        self.cause = cause
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        return (
            f"step={self.step_index} type={self.mapped_action.get('type')} "
            f"command={self.playwright_command} "
            f"error={self.cause.__class__.__name__}: {self.cause}"
        )


@dataclass(slots=True)
class ExecutedTrajectoryStep:
    index: int
    action_type: str
    playwright_command: str
    ok: bool
    error: str | None = None


class TrajectoryExecutor:
    """
    Adapter that converts trajectory.py action dictionaries into:
    1) Playwright-executable commands.
    2) IWA action payloads compatible with BaseAction.create_action.
    """

    def __init__(self, *, timeout_ms: int = 4_000) -> None:
        self.timeout_ms = int(max(100, timeout_ms))

    def map_actions(self, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.map_action(action) for action in actions]

    def map_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(action, dict):
            raise TrajectoryMapperError(f"Action must be a dict. Received: {type(action).__name__}")

        action_type = str(action.get("type") or "").strip()
        if action_type not in SUPPORTED_TRAJECTORY_ACTIONS:
            raise TrajectoryMapperError(f"Unsupported trajectory action type: {action_type or '<missing>'}")

        mapped: dict[str, Any] = {"type": action_type}
        if action_type == "NavigateAction":
            mapped.update(self._map_navigate_action(action))
            return mapped

        if action_type == "SendKeysAction":
            mapped["keys"] = self._extract_keys(action)
            return mapped

        selector = normalize_selector_payload(action.get("selector"))
        if not isinstance(selector, dict):
            raise TrajectoryMapperError(f"{action_type} requires a valid selector payload.")
        mapped["selector"] = selector
        mapped["playwright_selector"] = self.selector_to_playwright(selector)

        if action_type == "TypeAction":
            text = self._extract_type_text(action)
            mapped["text"] = text

        return mapped

    def selector_to_playwright(self, selector: dict[str, Any]) -> str:
        selector_type = str(selector.get("type") or "").strip()
        value = str(selector.get("value") or "").strip()
        if not selector_type or not value:
            raise TrajectoryMapperError("Selector must include non-empty 'type' and 'value'.")

        case_sensitive = bool(selector.get("case_sensitive", False))
        selector_type_lower = selector_type.lower()

        if selector_type_lower == "attributevalueselector":
            attribute = str(selector.get("attribute") or "").strip()
            if not attribute:
                raise TrajectoryMapperError("attributeValueSelector requires 'attribute'.")
            if attribute == "id":
                return f"#{value.lstrip('#')}"
            if attribute == "class":
                classes = [chunk.lstrip(".") for chunk in value.split() if chunk.strip()]
                if not classes:
                    raise TrajectoryMapperError("class selector requires at least one class name.")
                return "".join(f".{class_name}" for class_name in classes)
            if attribute == "custom":
                return value
            if attribute in _ATTRIBUTE_FORMATS:
                return _ATTRIBUTE_FORMATS[attribute].format(value=value)
            return f"[{attribute}='{value}']"

        if selector_type_lower == "xpathselector":
            xpath = value[6:].strip() if value.lower().startswith("xpath=") else value
            if xpath.startswith("//") or xpath.startswith("(//"):
                return f"xpath={xpath}"
            return f"xpath=//{xpath}"

        if selector_type_lower == "tagcontainsselector":
            quoted = repr(value)
            return f"text={quoted}" if case_sensitive else f"text={quoted} i"

        raise TrajectoryMapperError(f"Unsupported selector type: {selector_type}")

    def to_iwa_action_payload(self, action: dict[str, Any]) -> dict[str, Any]:
        mapped = self.map_action(action)
        action_type = str(mapped["type"])

        if action_type == "SendKeysAction":
            keys = mapped.get("keys") if isinstance(mapped.get("keys"), list) else []
            if not keys:
                raise TrajectoryMapperError("SendKeysAction mapped without keys.")
            press_value = keys[0] if len(keys) == 1 else "+".join(keys)
            return {"type": "SendKeysIWAAction", "keys": press_value}

        if action_type == "NavigateAction":
            payload: dict[str, Any] = {"type": "NavigateAction"}
            payload.update({k: v for k, v in mapped.items() if k in {"url", "go_back", "go_forward"}})
            return payload

        payload = {"type": action_type, "selector": mapped["selector"]}
        if action_type == "TypeAction":
            payload["text"] = mapped["text"]
        return payload

    def to_iwa_action_payloads(self, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.to_iwa_action_payload(action) for action in actions]

    def to_iwa_actions(self, actions: list[dict[str, Any]]) -> list[Any]:
        """
        Build concrete autoppia_iwa BaseAction objects when autoppia_iwa is available.
        """
        try:
            from autoppia_iwa.src.execution.actions.base import BaseAction
            import autoppia_iwa.src.execution.actions.actions  # noqa: F401
        except Exception as exc:
            raise RuntimeError(f"Failed to import autoppia_iwa action classes: {exc}") from exc

        built_actions: list[Any] = []
        for payload in self.to_iwa_action_payloads(actions):
            built = BaseAction.create_action(payload)
            if built is None:
                raise TrajectoryMapperError(f"BaseAction.create_action returned None for payload: {payload}")
            built_actions.append(built)
        return built_actions

    async def execute_on_page(self, page: Any, actions: list[dict[str, Any]]) -> list[ExecutedTrajectoryStep]:
        """
        Execute mapped actions directly on a Playwright-like page object.
        The page only needs Playwright-compatible async methods (duck typing).
        """
        results: list[ExecutedTrajectoryStep] = []
        mapped_actions = self.map_actions(actions)

        for index, mapped in enumerate(mapped_actions):
            action_type = str(mapped.get("type") or "")
            command = ""
            try:
                command = await self._execute_mapped_action(page=page, mapped_action=mapped)
                results.append(
                    ExecutedTrajectoryStep(
                        index=index,
                        action_type=action_type,
                        playwright_command=command,
                        ok=True,
                    )
                )
            except Exception as exc:
                wrapped = TrajectoryExecutionError(
                    step_index=index,
                    action=actions[index],
                    mapped_action=mapped,
                    playwright_command=command or "<unavailable>",
                    cause=exc,
                )
                results.append(
                    ExecutedTrajectoryStep(
                        index=index,
                        action_type=action_type,
                        playwright_command=command or "<unavailable>",
                        ok=False,
                        error=str(wrapped),
                    )
                )
                raise wrapped from exc
        return results

    async def _execute_mapped_action(self, *, page: Any, mapped_action: dict[str, Any]) -> str:
        action_type = str(mapped_action.get("type") or "")
        if action_type == "NavigateAction":
            return await self._execute_navigate(page, mapped_action)

        if action_type == "ClickAction":
            playwright_selector = str(mapped_action.get("playwright_selector") or "")
            await self._ensure_selector_exists(page, playwright_selector)
            await _await_if_needed(page.click(playwright_selector, timeout=self.timeout_ms))
            return f"page.click({playwright_selector!r}, timeout={self.timeout_ms})"

        if action_type == "TypeAction":
            text = str(mapped_action.get("text") or "")
            playwright_selector = str(mapped_action.get("playwright_selector") or "")
            await self._ensure_selector_exists(page, playwright_selector)
            await _await_if_needed(page.fill(playwright_selector, text, timeout=self.timeout_ms))
            return f"page.fill({playwright_selector!r}, {text!r}, timeout={self.timeout_ms})"

        if action_type == "SendKeysAction":
            keys = mapped_action.get("keys") if isinstance(mapped_action.get("keys"), list) else []
            if not keys:
                raise TrajectoryMapperError("Mapped SendKeysAction has no keys.")
            for key in keys:
                keyboard = getattr(page, "keyboard", None)
                if keyboard is None or not hasattr(keyboard, "press"):
                    raise TrajectoryMapperError("Page object does not expose keyboard.press.")
                await _await_if_needed(keyboard.press(str(key)))
            joined = " | ".join(str(k) for k in keys)
            return f"page.keyboard.press({joined!r})"

        raise TrajectoryMapperError(f"Unsupported mapped action type: {action_type}")

    async def _execute_navigate(self, page: Any, mapped_action: dict[str, Any]) -> str:
        url = str(mapped_action.get("url") or "").strip()
        go_back = bool(mapped_action.get("go_back", False))
        go_forward = bool(mapped_action.get("go_forward", False))

        if url:
            await _await_if_needed(page.goto(url, timeout=self.timeout_ms))
            return f"page.goto({url!r}, timeout={self.timeout_ms})"
        if go_back:
            await _await_if_needed(page.go_back(timeout=self.timeout_ms))
            return f"page.go_back(timeout={self.timeout_ms})"
        if go_forward:
            await _await_if_needed(page.go_forward(timeout=self.timeout_ms))
            return f"page.go_forward(timeout={self.timeout_ms})"
        raise TrajectoryMapperError("NavigateAction requires 'url', 'go_back', or 'go_forward'.")

    def _map_navigate_action(self, action: dict[str, Any]) -> dict[str, Any]:
        url = str(action.get("url") or "").strip()
        go_back = bool(action.get("go_back", False))
        go_forward = bool(action.get("go_forward", False))
        if sum([bool(url), go_back, go_forward]) != 1:
            raise TrajectoryMapperError(
                "NavigateAction requires exactly one navigation target: "
                "'url', 'go_back=True', or 'go_forward=True'."
            )
        out: dict[str, Any] = {}
        if url:
            out["url"] = url
        if go_back:
            out["go_back"] = True
        if go_forward:
            out["go_forward"] = True
        return out

    def _extract_type_text(self, action: dict[str, Any]) -> str:
        text = action.get("text")
        value = action.get("value")
        resolved = text if text is not None else value
        if resolved is None:
            raise TrajectoryMapperError("TypeAction requires 'text' or 'value'.")
        return str(resolved)

    def _extract_keys(self, action: dict[str, Any]) -> list[str]:
        raw = action.get("keys")
        if isinstance(raw, str):
            keys = [raw.strip()]
        elif isinstance(raw, list):
            keys = [str(key).strip() for key in raw if str(key).strip()]
        else:
            keys = []
        if not keys:
            raise TrajectoryMapperError("SendKeysAction requires non-empty 'keys'.")
        return keys

    async def _ensure_selector_exists(self, page: Any, playwright_selector: str) -> None:
        if not playwright_selector:
            raise TrajectoryMapperError("Empty Playwright selector.")
        if not hasattr(page, "locator"):
            return

        locator = page.locator(playwright_selector)
        count_fn = getattr(locator, "count", None)
        if callable(count_fn):
            count = await _await_if_needed(count_fn())
            if int(count) <= 0:
                raise TimeoutError(
                    f"Selector not found in DOM (count=0): {playwright_selector}"
                )
            return

        wait_for_fn = getattr(locator, "wait_for", None)
        if callable(wait_for_fn):
            await _await_if_needed(wait_for_fn(state="attached", timeout=self.timeout_ms))


async def _await_if_needed(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value
