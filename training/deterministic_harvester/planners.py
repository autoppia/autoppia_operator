from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .normalizer import DeterministicTaskObjective
from .projects import seeded_url
from .resolvers import resolve_movie_detail_url
from .selectors import (
    comment_message_selectors,
    comment_name_selectors,
    comment_submit_selectors,
    contact_email_selectors,
    contact_message_selectors,
    contact_name_selectors,
    contact_subject_selectors,
    contact_submit_selectors,
    delete_movie_selectors,
    login_password_selectors,
    login_submit_selectors,
    login_username_selectors,
    logout_selectors,
    profile_save_selectors,
    profile_tab_selectors,
    register_confirm_password_selectors,
    register_email_selectors,
    register_password_selectors,
    register_submit_selectors,
    register_username_selectors,
    save_changes_selectors,
    search_submit_selectors,
    share_button_selectors,
    trailer_button_selectors,
    view_detail_selectors,
    watchlist_button_selectors,
)


@dataclass(frozen=True)
class DeterministicPlan:
    prompt_lines: tuple[str, ...]
    actions: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]


def _navigate(route: str, *, objective: DeterministicTaskObjective) -> dict[str, Any]:
    return {
        "type": "NavigateAction",
        "url": seeded_url(
            task_url=objective.task_url,
            project_id=objective.web_project_id,
            route=route,
            seed=objective.seed,
        ),
        "go_back": False,
        "go_forward": False,
    }


def _type(text: str, *, selectors: list[dict[str, Any]] | None = None, field_name: str = "") -> dict[str, Any]:
    payload: dict[str, Any] = {"type": "TypeAction", "text": str(text), "field_name": str(field_name or "").strip()}
    if selectors:
        payload["selector_candidates"] = selectors
    return payload


def _click(*, selectors: list[dict[str, Any]] | None = None, field_name: str = "") -> dict[str, Any]:
    payload: dict[str, Any] = {"type": "ClickAction", "field_name": str(field_name or "").strip()}
    if selectors:
        payload["selector_candidates"] = selectors
    return payload


def _movie_detail_action(objective: DeterministicTaskObjective) -> dict[str, Any]:
    filters = dict(objective.entity_filters)
    query = str(objective.field_values.get("query") or "")
    if query and "name_exact" not in filters and "name_contains" not in filters:
        filters["name_contains"] = query
    resolved_url = resolve_movie_detail_url(
        task_url=objective.task_url,
        filters=filters,
        web_project_id=objective.web_project_id,
    )
    route = resolved_url or objective.route_target
    if route.startswith("http://") or route.startswith("https://"):
        return {
            "type": "NavigateAction",
            "url": route,
            "go_back": False,
            "go_forward": False,
        }
    if resolved_url:
        return _navigate(route, objective=objective)
    return _click(selectors=view_detail_selectors(objective.web_project_id), field_name="movie detail")


def _login_actions(objective: DeterministicTaskObjective) -> list[dict[str, Any]]:
    username = str(objective.field_values.get("username") or "user1")
    password = str(objective.field_values.get("password") or "Passw0rd!")
    return [
        _navigate("/login", objective=objective),
        _type(username, selectors=login_username_selectors(objective.web_project_id), field_name="username"),
        _type(password, selectors=login_password_selectors(objective.web_project_id), field_name="password"),
        _click(selectors=login_submit_selectors(objective.web_project_id), field_name="submit"),
    ]


def _registration_actions(objective: DeterministicTaskObjective) -> list[dict[str, Any]]:
    username = str(objective.field_values.get("username") or f"newuser{objective.seed}")
    email = str(objective.field_values.get("email") or f"newuser{objective.seed}@gmail.com")
    password = str(objective.field_values.get("password") or "Passw0rd!")
    confirm_password = str(objective.field_values.get("confirm_password") or password)
    return [
        _navigate("/register", objective=objective),
        _type(username, selectors=register_username_selectors(objective.web_project_id), field_name="username"),
        _type(email, selectors=register_email_selectors(objective.web_project_id), field_name="email"),
        _type(password, selectors=register_password_selectors(objective.web_project_id), field_name="password"),
        _type(confirm_password, selectors=register_confirm_password_selectors(objective.web_project_id), field_name="confirm password"),
        _click(selectors=register_submit_selectors(objective.web_project_id), field_name="submit"),
    ]


def _contact_actions(objective: DeterministicTaskObjective) -> list[dict[str, Any]]:
    name = str(objective.field_values.get("name") or "Agent")
    email = str(objective.field_values.get("email") or "agent@example.com")
    subject = str(objective.field_values.get("subject") or "General inquiry")
    message = str(objective.field_values.get("message") or "Please help with my movie request.")
    return [
        _navigate("/contact", objective=objective),
        _type(name, selectors=contact_name_selectors(objective.web_project_id), field_name="name"),
        _type(email, selectors=contact_email_selectors(objective.web_project_id), field_name="email"),
        _type(subject, selectors=contact_subject_selectors(objective.web_project_id), field_name="subject"),
        _type(message, selectors=contact_message_selectors(objective.web_project_id), field_name="message"),
        _click(selectors=contact_submit_selectors(objective.web_project_id), field_name="submit"),
    ]


def _search_actions(objective: DeterministicTaskObjective) -> list[dict[str, Any]]:
    query = str(objective.field_values.get("query") or objective.field_values.get("movie_name") or "The Matrix")
    return [
        _navigate("/search", objective=objective),
        _type(query, field_name="search"),
        _click(selectors=search_submit_selectors(objective.web_project_id), field_name="submit"),
    ]


def _movie_target_actions(objective: DeterministicTaskObjective, *, include_search: bool = True) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if include_search and str(objective.field_values.get("query") or "").strip():
        actions.extend(_search_actions(objective))
    else:
        actions.append(_navigate(objective.route_target, objective=objective))
    actions.append(_movie_detail_action(objective))
    return actions


def _profile_fill_actions(objective: DeterministicTaskObjective) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for objective_key, field_name in (
        ("first_name", "first name"),
        ("last_name", "last name"),
        ("email", "email"),
        ("favorite_genres", "favorite genres"),
        ("location", "location"),
        ("website", "website"),
        ("bio", "bio"),
    ):
        value = objective.field_values.get(objective_key)
        if value:
            actions.append(_type(value, field_name=field_name))
    actions.append(_click(selectors=profile_save_selectors(objective.web_project_id), field_name="submit"))
    return actions


def _movie_editor_fill_actions(objective: DeterministicTaskObjective) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for objective_key, field_name in (
        ("title", "title"),
        ("director", "director"),
        ("year", "year"),
        ("duration", "duration"),
        ("rating", "rating"),
        ("trailer_url", "trailer"),
        ("genres", "genres"),
        ("cast", "cast"),
        ("synopsis", "synopsis"),
    ):
        value = str(objective.field_values.get(objective_key) or "").strip()
        if value:
            actions.append(_type(value, field_name=field_name))
    actions.append(_click(selectors=save_changes_selectors(objective.web_project_id), field_name="submit"))
    return actions


def build_deterministic_plan(objective: DeterministicTaskObjective) -> DeterministicPlan:
    use_case = objective.use_case
    prompt_lines = [f"Deterministic planner for {use_case}", f"seed={objective.seed}"]

    if use_case == "LOGIN":
        actions = _login_actions(objective)
    elif use_case == "REGISTRATION":
        actions = _registration_actions(objective)
    elif use_case == "CONTACT":
        actions = _contact_actions(objective)
    elif use_case == "SEARCH_FILM":
        actions = _search_actions(objective)
    elif use_case == "FILTER_FILM":
        genre = str(objective.field_values.get("genres") or objective.entity_filters.get("genre_exact") or "Action")
        actions = [
            _navigate("/search", objective=objective),
            _click(
                field_name="genre",
                selectors=[{"type": "tagContainsSelector", "value": genre, "case_sensitive": False}],
            ),
        ]
    elif use_case == "FILM_DETAIL":
        actions = _movie_target_actions(objective)
    elif use_case == "ADD_COMMENT":
        actions = _movie_target_actions(objective)
        actions.extend(
            [
                _type(str(objective.field_values.get("name") or "Agent"), selectors=comment_name_selectors(objective.web_project_id), field_name="name"),
                _type(
                    str(objective.field_values.get("content") or "Great movie"),
                    selectors=comment_message_selectors(objective.web_project_id),
                    field_name="comment",
                ),
                _click(selectors=comment_submit_selectors(objective.web_project_id), field_name="submit"),
            ]
        )
    elif use_case == "SHARE_MOVIE":
        actions = _movie_target_actions(objective)
        actions.append(_click(selectors=share_button_selectors(objective.web_project_id), field_name="share"))
    elif use_case == "WATCH_TRAILER":
        actions = _movie_target_actions(objective, include_search=False)
        actions.append(_click(selectors=trailer_button_selectors(objective.web_project_id), field_name="watch trailer"))
    elif use_case == "ADD_TO_WATCHLIST":
        actions = _login_actions(objective)
        actions.extend(_movie_target_actions(objective, include_search=False))
        actions.append(_click(selectors=watchlist_button_selectors(objective.web_project_id), field_name="watchlist"))
    elif use_case == "REMOVE_FROM_WATCHLIST":
        actions = _login_actions(objective)
        actions.extend(_movie_target_actions(objective, include_search=False))
        actions.append(_click(selectors=watchlist_button_selectors(objective.web_project_id), field_name="watchlist"))
        actions.append(_click(selectors=watchlist_button_selectors(objective.web_project_id), field_name="watchlist"))
    elif use_case == "EDIT_USER":
        actions = _login_actions(objective)
        actions.append(_navigate("/profile", objective=objective))
        actions.extend(_profile_fill_actions(objective))
    elif use_case == "LOGOUT":
        actions = _login_actions(objective)
        actions.append(_click(selectors=logout_selectors(objective.web_project_id), field_name="logout"))
    elif use_case == "ADD_FILM":
        actions = _login_actions(objective)
        actions.extend(
            [
                _navigate("/profile", objective=objective),
                _click(selectors=profile_tab_selectors("add-movies", objective.web_project_id), field_name="add movies"),
                *_movie_editor_fill_actions(objective),
            ]
        )
    elif use_case == "EDIT_FILM":
        actions = _login_actions(objective)
        actions.extend(
            [
                _navigate("/profile", objective=objective),
                _click(selectors=profile_tab_selectors("movies", objective.web_project_id), field_name="edit movies"),
                *_movie_editor_fill_actions(objective),
            ]
        )
    elif use_case == "DELETE_FILM":
        actions = _login_actions(objective)
        actions.extend(
            [
                _navigate("/profile", objective=objective),
                _click(selectors=profile_tab_selectors("movies", objective.web_project_id), field_name="edit movies"),
                _click(selectors=delete_movie_selectors(objective.web_project_id), field_name="delete movie"),
            ]
        )
    else:
        raise ValueError(f"Unsupported deterministic use case: {use_case}")

    return DeterministicPlan(
        prompt_lines=tuple(prompt_lines),
        actions=tuple(actions),
        metadata={
            "auth_required": bool(objective.auth_required),
            "entity_filters": dict(objective.entity_filters),
            "field_values": dict(objective.field_values),
            "route_target": objective.route_target,
            "web_project_id": objective.web_project_id,
        },
    )


__all__ = [
    "DeterministicPlan",
    "build_deterministic_plan",
]
