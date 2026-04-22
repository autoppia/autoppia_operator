from __future__ import annotations

from training.deterministic_harvester.normalizer import normalize_task_row
from training.deterministic_harvester.planners import build_deterministic_plan


def _task_row(*, use_case: str, prompt: str, url: str, event_criteria: dict, relevant_data: dict | None = None) -> dict:
    return {
        "id": f"{use_case.lower()}-task",
        "url": url,
        "prompt": prompt,
        "specifications": {},
        "tests": [
            {
                "type": "CheckEventTest",
                "event_name": use_case,
                "event_criteria": event_criteria,
            }
        ],
        "relevant_data": relevant_data or {},
        "use_case": {
            "name": use_case,
            "constraints": [],
        },
    }


def test_build_deterministic_plan_for_add_to_watchlist_uses_login_and_movie_resolution() -> None:
    task_row = _task_row(
        use_case="ADD_TO_WATCHLIST",
        prompt="movie_name equals 'Inception'",
        url="http://example.test/movies?seed=23",
        event_criteria={"movie": {"name": "Inception"}},
        relevant_data={"user_for_login": {"username": "bob", "password": "letmein"}},
    )
    objective = normalize_task_row(task_row)
    plan = build_deterministic_plan(objective)
    action_types = [action["type"] for action in plan.actions]
    assert action_types[:4] == ["NavigateAction", "TypeAction", "TypeAction", "ClickAction"]
    assert "OpenMovieDetailAction" not in action_types
    assert action_types[-2:] == ["ClickAction", "ClickAction"]
    assert action_types[-1] == "ClickAction"
    watchlist_selectors = plan.actions[-1].get("selector_candidates") or []
    assert watchlist_selectors
    assert watchlist_selectors[0]["type"] == "attributeValueSelector"


def test_build_deterministic_plan_for_edit_user_targets_profile_form() -> None:
    task_row = _task_row(
        use_case="EDIT_USER",
        prompt="first_name equals 'Ben', website equals 'https://example.org'",
        url="http://example.test/profile?seed=9",
        event_criteria={"first_name": "Ben", "website": "https://example.org"},
        relevant_data={"user_for_login": {"username": "ben", "password": "secret123"}},
    )
    objective = normalize_task_row(task_row)
    plan = build_deterministic_plan(objective)
    assert plan.actions[0]["type"] == "NavigateAction"
    first_name_actions = [action for action in plan.actions if action["type"] == "TypeAction" and action.get("field_name") == "first name"]
    assert first_name_actions
    assert first_name_actions[0]["selector_candidates"][0]["type"] == "attributeValueSelector"
    assert plan.actions[-1]["selector_candidates"][0]["type"] in {
        "attributeValueSelector",
        "idSelector",
        "tagContainsSelector",
        "textSelector",
    }


def test_build_deterministic_plan_preserves_localhost_origin() -> None:
    task_row = _task_row(
        use_case="CONTACT",
        prompt="subject equals 'Local test'",
        url="http://localhost:3000/?seed=41",
        event_criteria={"subject": "Local test"},
    )
    task_row["web_project_id"] = "autocinema"

    objective = normalize_task_row(task_row)
    plan = build_deterministic_plan(objective)

    assert plan.actions[0]["url"] == "http://localhost:3000/contact?seed=41"


def test_build_deterministic_plan_filter_film_uses_seeded_dropdown_order() -> None:
    task_row = _task_row(
        use_case="FILTER_FILM",
        prompt="genres equals 'Sci-Fi' and year equals '1986'",
        url="http://localhost:3000/search?seed=415",
        event_criteria={"genres": "Sci-Fi", "year": 1986},
    )
    objective = normalize_task_row(task_row)
    plan = build_deterministic_plan(objective)
    select_actions = [action for action in plan.actions if action["type"] == "SelectAction"]
    assert len(select_actions) == 2
    year_selectors = select_actions[0]["selector_candidates"]
    genre_selectors = select_actions[1]["selector_candidates"]
    assert year_selectors[0]["value"] in {"section#library select:nth-of-type(1)", "section#library select:nth-of-type(2)"}
    assert genre_selectors[0]["value"] in {"section#library select:nth-of-type(1)", "section#library select:nth-of-type(2)"}
    assert year_selectors[0]["value"] != genre_selectors[0]["value"]


def test_build_deterministic_plan_remove_from_watchlist_uses_two_detail_toggle_clicks() -> None:
    task_row = _task_row(
        use_case="REMOVE_FROM_WATCHLIST",
        prompt="movie_name equals 'Dune'",
        url="http://localhost:3000/profile?seed=23",
        event_criteria={"movie": {"name": "Dune"}},
        relevant_data={"user_for_login": {"username": "bob", "password": "letmein"}},
    )
    objective = normalize_task_row(task_row)
    plan = build_deterministic_plan(objective)
    add_click = [action for action in plan.actions if action["type"] == "ClickAction" and action.get("field_name") == "watchlist"]
    remove_click = [action for action in plan.actions if action["type"] == "ClickAction" and action.get("field_name") == "remove from watchlist"]
    assert len(add_click) == 1
    assert len(remove_click) == 1
    assert add_click[0]["selector_candidates"] == remove_click[0]["selector_candidates"]


def test_build_deterministic_plan_add_film_uses_explicit_editor_selectors() -> None:
    task_row = _task_row(
        use_case="ADD_FILM",
        prompt="title equals 'New Film', director equals 'Jane Doe'",
        url="http://localhost:3000/profile?seed=23",
        event_criteria={"name": "New Film"},
        relevant_data={"user_for_login": {"username": "bob", "password": "letmein"}},
    )
    objective = normalize_task_row(task_row)
    plan = build_deterministic_plan(objective)
    type_actions = [action for action in plan.actions if action["type"] == "TypeAction"]
    assert type_actions
    assert all(action.get("selector_candidates") for action in type_actions)
    add_movies_clicks = [action for action in plan.actions if action["type"] == "ClickAction" and action.get("field_name") == "add movies"]
    assert len(add_movies_clicks) == 1
    assert add_movies_clicks[0]["selector_candidates"]
    first_selector = add_movies_clicks[0]["selector_candidates"][0]
    assert first_selector["type"] == "attributeValueSelector"
    assert first_selector["attribute"] == "custom"
    assert 'aria-controls*="add-movies"' in first_selector["value"]
