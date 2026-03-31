from __future__ import annotations

from typing import Any

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_trajectory_module = _load_module(
    "trajectory_validation_module",
    ROOT / "src" / "operator" / "agents" / "fsm" / "trajectory.py",
)
if not hasattr(_trajectory_module, "_TRAJECTORIES") and hasattr(_trajectory_module, "TRAJECTORIES"):
    _trajectory_module._TRAJECTORIES = _trajectory_module.TRAJECTORIES
_executor_module = _load_module(
    "trajectory_executor_validation_module",
    ROOT / "src" / "operator" / "runtime" / "trajectory_executor.py",
)

get_trajectory_bootstrap_actions = _trajectory_module.get_trajectory_bootstrap_actions
TrajectoryExecutor = _executor_module.TrajectoryExecutor
TrajectoryExecutionError = _executor_module.TrajectoryExecutionError


class _DummyLocator:
    def __init__(self, selector: str, *, available: set[str]) -> None:
        self.selector = selector
        self.available = available

    async def count(self) -> int:
        return 1 if self.selector in self.available else 0


class _DummyKeyboard:
    def __init__(self) -> None:
        self.pressed: list[str] = []

    async def press(self, key: str) -> None:
        self.pressed.append(str(key))


class _DummyPage:
    def __init__(self, *, available_selectors: set[str]) -> None:
        self.available_selectors = set(available_selectors)
        self.keyboard = _DummyKeyboard()
        self.clicked: list[str] = []
        self.filled: list[tuple[str, str]] = []
        self.gotos: list[str] = []

    def locator(self, selector: str) -> _DummyLocator:
        return _DummyLocator(selector, available=self.available_selectors)

    async def click(self, selector: str, timeout: int | None = None) -> None:
        if selector not in self.available_selectors:
            raise TimeoutError(f"Timeout while waiting for selector: {selector}")
        self.clicked.append(selector)

    async def fill(self, selector: str, text: str, timeout: int | None = None) -> None:
        if selector not in self.available_selectors:
            raise TimeoutError(f"Timeout while waiting for selector: {selector}")
        self.filled.append((selector, text))

    async def goto(self, url: str, timeout: int | None = None) -> None:
        self.gotos.append(url)

    async def go_back(self, timeout: int | None = None) -> None:
        return

    async def go_forward(self, timeout: int | None = None) -> None:
        return


def _load_bootstrap_actions() -> list[dict[str, Any]]:
    actions = get_trajectory_bootstrap_actions(
        web_project_id="p01_autocinema",
        use_case="SEARCH_FILM",
        prompt="Search for the movie 'La La Land'",
        max_actions=8,
    )
    assert actions, "Expected trajectory bootstrap actions for autocinema SEARCH_FILM"
    return actions


@pytest.mark.asyncio
async def test_trajectory_executor_executes_mapped_bootstrap_actions_on_dummy_page() -> None:
    executor = TrajectoryExecutor(timeout_ms=1200)
    actions = _load_bootstrap_actions()

    mapped = executor.map_actions(actions)
    selectors = {
        str(item.get("playwright_selector") or "")
        for item in mapped
        if str(item.get("playwright_selector") or "").strip()
    }
    page = _DummyPage(available_selectors={selector for selector in selectors if selector})

    results = await executor.execute_on_page(page, actions)
    assert results
    assert all(step.ok for step in results)


@pytest.mark.asyncio
async def test_trajectory_executor_prints_exact_selector_error_for_debugging(capsys: pytest.CaptureFixture[str]) -> None:
    executor = TrajectoryExecutor(timeout_ms=1200)
    actions = _load_bootstrap_actions()
    mapped = executor.map_actions(actions)
    selectors = [
        str(item.get("playwright_selector") or "")
        for item in mapped
        if str(item.get("playwright_selector") or "").strip()
    ]
    available = set(selectors)
    if selectors:
        available.discard(selectors[0])
    page = _DummyPage(available_selectors=available)

    with pytest.raises(TrajectoryExecutionError) as excinfo:
        await executor.execute_on_page(page, actions)

    cause = excinfo.value.cause
    print(f"{cause.__class__.__name__}: {cause}")
    captured = capsys.readouterr().out
    assert "TimeoutError:" in captured


def test_trajectory_executor_maps_send_keys_to_iwa_payload_schema() -> None:
    executor = TrajectoryExecutor()
    payload = executor.to_iwa_action_payload({"type": "SendKeysAction", "keys": ["Control", "K"]})
    assert payload == {"type": "SendKeysIWAAction", "keys": "Control+K"}
