#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

import argparse
import asyncio
import importlib.util
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)


def _load_module(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_trajectory_module = _load_module(
    "trajectory_debug_module",
    ROOT / "src" / "operator" / "agents" / "fsm" / "trajectory.py",
)
if not hasattr(_trajectory_module, "_TRAJECTORIES") and hasattr(_trajectory_module, "TRAJECTORIES"):
    _trajectory_module._TRAJECTORIES = _trajectory_module.TRAJECTORIES
_executor_module = _load_module(
    "trajectory_executor_debug_module",
    ROOT / "src" / "operator" / "runtime" / "trajectory_executor.py",
)

get_trajectory_bootstrap_actions = _trajectory_module.get_trajectory_bootstrap_actions
TrajectoryExecutor = _executor_module.TrajectoryExecutor
TrajectoryExecutionError = _executor_module.TrajectoryExecutionError


class DummyLocator:
    def __init__(self, selector: str, *, available: set[str]) -> None:
        self.selector = selector
        self.available = available

    async def count(self) -> int:
        return 1 if self.selector in self.available else 0


class DummyKeyboard:
    def __init__(self) -> None:
        self.pressed: list[str] = []

    async def press(self, key: str) -> None:
        self.pressed.append(str(key))


class DummyPage:
    def __init__(self, *, available_selectors: set[str]) -> None:
        self.available_selectors = set(available_selectors)
        self.keyboard = DummyKeyboard()
        self.clicked: list[str] = []
        self.filled: list[tuple[str, str]] = []
        self.gotos: list[str] = []

    def locator(self, selector: str) -> DummyLocator:
        return DummyLocator(selector, available=self.available_selectors)

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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Debug trajectory execution flow: "
            "get_trajectory_bootstrap_actions -> TrajectoryExecutor mapper -> execution"
        )
    )
    parser.add_argument("--web-project-id", default="p01_autocinema")
    parser.add_argument("--use-case", default="SEARCH_FILM")
    parser.add_argument("--prompt", default="Search for the movie 'La La Land'")
    parser.add_argument("--max-actions", type=int, default=8)
    parser.add_argument("--timeout-ms", type=int, default=4000)
    parser.add_argument(
        "--drop-first-selector",
        action="store_true",
        help="Force a missing selector to reproduce and print the exact TimeoutError.",
    )
    parser.add_argument(
        "--validate-iwa-schema",
        action="store_true",
        help="Also map actions to autoppia_iwa BaseAction objects to detect schema issues.",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    executor = TrajectoryExecutor(timeout_ms=int(args.timeout_ms))

    actions = get_trajectory_bootstrap_actions(
        web_project_id=str(args.web_project_id),
        use_case=str(args.use_case),
        prompt=str(args.prompt),
        max_actions=int(args.max_actions),
    )
    if not actions:
        print("No bootstrap trajectory actions found for the provided filters.")
        return 2

    print("Bootstrap actions:")
    print(json.dumps(actions, indent=2, ensure_ascii=False))

    mapped_actions = executor.map_actions(actions)
    print("\nMapped actions:")
    print(json.dumps(mapped_actions, indent=2, ensure_ascii=False))

    available_selectors = {
        str(action.get("playwright_selector") or "")
        for action in mapped_actions
        if str(action.get("playwright_selector") or "").strip()
    }
    available_selectors = {selector for selector in available_selectors if selector}

    if args.drop_first_selector and available_selectors:
        first_selector = sorted(available_selectors)[0]
        available_selectors.discard(first_selector)
        print(f"\nIntentionally removed selector to force error: {first_selector}")

    page = DummyPage(available_selectors=available_selectors)
    try:
        steps = await executor.execute_on_page(page, actions)
        print("\nDummy execution OK:")
        for step in steps:
            print(f"- step={step.index} type={step.action_type} command={step.playwright_command}")
    except TrajectoryExecutionError as exc:
        print("\nExecution failed with exact error:")
        print(f"{exc.cause.__class__.__name__}: {exc.cause}")
        print("Wrapped context:")
        print(str(exc))
        traceback.print_exc()
        return 1

    if args.validate_iwa_schema:
        try:
            iwa_actions = executor.to_iwa_actions(actions)
            print("\nIWA schema validation OK:")
            for idx, iwa_action in enumerate(iwa_actions):
                print(f"- step={idx} class={iwa_action.__class__.__name__} payload={iwa_action.model_dump()}")
        except Exception as exc:
            print("\nIWA schema validation failed with exact error:")
            print(f"{exc.__class__.__name__}: {exc}")
            traceback.print_exc()
            return 1

    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        return asyncio.run(_run(args))
    except Exception as exc:
        print("Fatal error:")
        print(f"{exc.__class__.__name__}: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
