from __future__ import annotations

from training.deterministic_harvester.projects import project_config
from training.deterministic_harvester.selectors import (
    comment_submit_selectors,
    login_username_selectors,
    profile_tab_selectors,
    route_for_use_case,
    selector_candidates_for_ids,
)


def test_route_for_use_case_covers_movie_and_auth_flows() -> None:
    assert route_for_use_case("LOGIN") == "/login"
    assert route_for_use_case("CONTACT") == "/contact"
    assert route_for_use_case("ADD_TO_WATCHLIST") == "/movies"


def test_selector_candidates_for_ids_expands_dynamic_variants() -> None:
    selectors = selector_candidates_for_ids("contact-name-input")
    values = [selector.get("value") for selector in selectors]
    assert "contact-name-input" in values
    assert any(value != "contact-name-input" for value in values)


def test_login_username_selectors_include_id_candidates() -> None:
    selectors = login_username_selectors()
    assert selectors
    assert selectors[0]["attribute"] == "id"


def test_comment_submit_selectors_include_text_fallbacks() -> None:
    selectors = comment_submit_selectors()
    assert any(selector.get("value") == "Share Feedback" for selector in selectors)


def test_profile_tab_selectors_use_visible_text_fallbacks() -> None:
    selectors = profile_tab_selectors("add-movies")
    assert any(selector.get("value") == "Add Movies" for selector in selectors)


def test_project_config_derives_project_key_for_future_projects() -> None:
    config = project_config("autobooks")
    assert config.project_key == "web_2_autobooks"
