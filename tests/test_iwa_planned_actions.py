from __future__ import annotations

from autoppia_iwa.src.execution.actions.actions import ClickAction, NavigateAction, TypeAction
from autoppia_iwa.src.execution.actions.base import Selector, SelectorType

import training._iwa_path  # noqa: F401  # must run before autoppia_iwa imports
from training.deterministic_harvester.iwa_planned_actions import (
    frontend_url_for_project,
    iwa_actions_to_planned_actions,
    iwa_to_dict_planned,
)


def test_frontend_url_for_autobooks_is_non_empty() -> None:
    url = frontend_url_for_project("autobooks")
    assert url.startswith("http://")


def test_iwa_to_dict_planned_navigate_remaps_host() -> None:
    nav = NavigateAction(url="http://localhost:8001/?seed=569")
    out = iwa_to_dict_planned(nav, frontend_url="http://127.0.0.1:9999")
    assert out["type"] == "NavigateAction"
    assert "127.0.0.1:9999" in out["url"]
    assert "seed=569" in out["url"]


def test_iwa_actions_slice_navigate_click_type_uses_selector_candidates() -> None:
    actions = [
        NavigateAction(url="http://localhost:8001/?seed=1"),
        ClickAction(selector=Selector(type=SelectorType.XPATH_SELECTOR, value="//a[normalize-space()='Login']", case_sensitive=False)),
        TypeAction(
            text="user1",
            selector=Selector(type=SelectorType.ATTRIBUTE_VALUE_SELECTOR, attribute="id", value="username-input", case_sensitive=False),
        ),
    ]
    planned = iwa_actions_to_planned_actions(actions, frontend_url="http://localhost:8001")
    assert len(planned) == 3
    assert planned[0]["type"] == "NavigateAction"
    assert planned[1]["type"] == "ClickAction"
    assert planned[1]["selector_candidates"]
    assert planned[1]["selector_candidates"][0]["type"] == "xpathSelector"
    assert planned[2]["type"] == "TypeAction"
    assert planned[2]["text"] == "user1"
    assert planned[2]["selector_candidates"][0]["value"] == "username-input"
