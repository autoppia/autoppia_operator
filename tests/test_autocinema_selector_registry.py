from __future__ import annotations

import json
from pathlib import Path

from training.deterministic_harvester.normalizer import route_for_web_project_use_case
from training.deterministic_harvester.projects import project_config
from training.deterministic_harvester.selectors import (
    comment_submit_selectors,
    login_submit_selectors,
    login_username_selectors,
    logout_selectors,
    profile_first_name_selectors,
    profile_tab_selectors,
    route_for_use_case,
    selector_candidates_for_ids,
    view_detail_selectors,
)
from training.deterministic_harvester.trajectory_selectors import (
    enrich_iwa_selector_candidates,
    list_iwa_use_cases,
    planned_actions_for_iwa_use_case,
    planned_actions_for_iwa_use_case_enriched,
    selector_candidates_from_iwa_trajectory,
)


def test_route_for_use_case_covers_movie_and_auth_flows() -> None:
    assert route_for_use_case("LOGIN") == "/login"
    assert route_for_use_case("CONTACT") == "/contact"
    assert route_for_use_case("ADD_TO_WATCHLIST") == "/movies"


def test_route_for_use_case_matches_normalizer_per_project() -> None:
    assert route_for_use_case("SEARCH_FILM", "autocinema") == "/search"
    assert route_for_use_case("SEARCH_BOOK", "autobooks") == "/search"
    assert route_for_use_case("CONTACT_BOOK", "autobooks") == "/contact"
    assert route_for_use_case("SEARCH_RESTAURANT", "autodining") == "/search"
    assert route_for_web_project_use_case(web_project_id="autobooks", use_case="VIEW_CART_BOOK") == "/cart"


def test_planned_actions_from_iwa_trajectory_non_cinema() -> None:
    if "SEARCH_RESTAURANT" not in list_iwa_use_cases("autodining"):
        return
    actions = planned_actions_for_iwa_use_case("autodining", "SEARCH_RESTAURANT")
    assert actions
    assert any("selector_candidates" in a for a in actions)
    cands = selector_candidates_from_iwa_trajectory("autodining", "SEARCH_RESTAURANT")
    assert cands
    assert all("attribute" in c or "type" in c for c in cands)


def test_iwa_enriched_expands_id_variants_per_project() -> None:
    # Exercise id/class/text expansion the same way as hand-built autocinema helpers,
    # independent of IWA using legacy vs canonical element ids in every step.
    base = [
        {
            "type": "attributeValueSelector",
            "attribute": "id",
            "value": "search-input",
            "case_sensitive": False,
        }
    ]
    out = enrich_iwa_selector_candidates(base, project_id="autobooks", seed=1)
    assert len(out) > 1
    id_vals = {c.get("value") for c in out if c.get("attribute") == "id"}
    assert "search-input" in id_vals
    assert len(id_vals) > 1

    if "LOGIN" in list_iwa_use_cases("autocinema"):
        raw = planned_actions_for_iwa_use_case("autocinema", "LOGIN")
        enriched = planned_actions_for_iwa_use_case_enriched("autocinema", "LOGIN", seed=3)
        assert len(enriched) == len(raw)
        for r, e in zip(raw, enriched, strict=True):
            rc, ec = r.get("selector_candidates"), e.get("selector_candidates")
            if not isinstance(rc, list) or not rc or not all(isinstance(s, dict) for s in rc):
                assert rc == ec
                continue
            st = str((rc[0] or {}).get("type") or "")
            if st == "attributeValueSelector" and str((rc[0] or {}).get("attribute") or "") == "id":
                assert len(ec) >= len(rc)


def test_selector_candidates_for_ids_expands_dynamic_variants() -> None:
    selectors = selector_candidates_for_ids("contact-name-input")
    values = [selector.get("value") for selector in selectors]
    assert "contact-name-input" in values
    assert any(value != "contact-name-input" for value in values)


def test_login_username_selectors_include_id_candidates() -> None:
    selectors = login_username_selectors()
    assert selectors
    assert selectors[0]["attribute"] == "id"
    assert any(selector.get("attribute") == "class" and selector.get("value") == "input-text" for selector in selectors)
    assert any(selector.get("attribute") == "placeholder" for selector in selectors)


def test_selector_candidates_for_ids_prefers_seeded_exact_variant() -> None:
    path = project_config("autocinema").id_variants_path
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    variants = payload["contact-name-input"]

    def hash_string(value: str) -> int:
        hash_value = 0
        for char in value:
            hash_value = ((hash_value << 5) - hash_value) + ord(char)
            hash_value &= 0xFFFFFFFF
            if hash_value >= 0x80000000:
                hash_value -= 0x100000000
        return abs(hash_value)

    seed = 23
    expected = variants[abs(hash_string(f"contact-name-input:{seed}")) % len(variants)]
    selectors = selector_candidates_for_ids("contact-name-input", seed=seed)
    values = [selector.get("value") for selector in selectors]
    assert values[0] == expected
    assert "contact-name-input" in values


def test_comment_submit_selectors_include_text_fallbacks() -> None:
    selectors = comment_submit_selectors()
    assert any(selector.get("value") == "Share Feedback" for selector in selectors)
    assert any(selector.get("attribute") == "class" and selector.get("value") == "button-primary" for selector in selectors)


def test_view_detail_selectors_include_seeded_featured_variant() -> None:
    seed = 9
    selectors = view_detail_selectors(seed=seed)
    values = [selector.get("value") for selector in selectors]
    local_variants = [
        "hero-view-details-btn",
        "view-details-button",
        "details-action",
        "view-movie-btn",
        "details-btn",
    ]

    def hash_string(value: str) -> int:
        hash_value = 0
        for char in value:
            hash_value = ((hash_value << 5) - hash_value) + ord(char)
            hash_value &= 0xFFFFFFFF
            if hash_value >= 0x80000000:
                hash_value -= 0x100000000
        return abs(hash_value)

    expected_local = local_variants[abs(hash_string(f"featured-view-details-button:{seed}")) % len(local_variants)]
    assert expected_local in values
    assert values[0] != ""


def test_profile_first_name_selectors_include_seeded_id_candidates() -> None:
    selectors = profile_first_name_selectors(seed=12)
    assert selectors
    assert selectors[0]["attribute"] == "id"


def test_login_submit_selectors_include_web_text_and_class_variants() -> None:
    selectors = login_submit_selectors(seed=9)
    assert selectors
    assert selectors[0]["attribute"] == "id"
    assert any(selector.get("attribute") == "class" and selector.get("value") == "button-primary" for selector in selectors)
    assert any(selector.get("type") == "tagContainsSelector" and selector.get("value") == "Sign in" for selector in selectors)


def test_profile_tab_selectors_use_visible_text_fallbacks() -> None:
    selectors = profile_tab_selectors("add-movies")
    assert any(selector.get("value") == "Add Movies" for selector in selectors)


def test_project_config_derives_project_key_for_future_projects() -> None:
    config = project_config("autobooks")
    assert config.project_key == "web_2_autobooks"


def test_selector_helpers_do_not_block_non_autocinema_projects() -> None:
    logout = logout_selectors("autobooks")
    tabs = profile_tab_selectors("add-movies", "autobooks")
    assert logout
    assert tabs
