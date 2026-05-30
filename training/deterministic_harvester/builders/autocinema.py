from __future__ import annotations

import json
from typing import Any

from training.deterministic_harvester.builders.common import (
    DeterministicPlan,
    click,
    custom_selector,
    extract_path,
    generate_dynamic_order,
    navigate,
    select,
    type_text,
)
from training.deterministic_harvester.normalizer import DeterministicTaskObjective
from training.deterministic_harvester.resolvers import resolve_movie_detail_url
from training.deterministic_harvester.selectors import (
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
    profile_bio_selectors,
    profile_email_selectors,
    profile_favorite_genres_selectors,
    profile_first_name_selectors,
    profile_last_name_selectors,
    profile_location_selectors,
    profile_save_selectors,
    profile_tab_selectors,
    profile_website_selectors,
    register_confirm_password_selectors,
    register_email_selectors,
    register_password_selectors,
    register_submit_selectors,
    register_username_selectors,
    save_changes_selectors,
    search_input_selectors,
    search_submit_selectors,
    share_button_selectors,
    trailer_button_selectors,
    view_detail_selectors,
    watchlist_button_selectors,
)


def _resolve_target_movie_path(objective: DeterministicTaskObjective, *, require_trailer: bool = False) -> str:
    filters = dict(objective.entity_filters)
    query = str(objective.field_values.get("query") or "")
    if query and "name_exact" not in filters and "name_contains" not in filters:
        filters["name_contains"] = query
    if require_trailer:
        filters["requires_trailer"] = True
    resolved_url = resolve_movie_detail_url(
        task_url=objective.task_url,
        filters=filters,
        web_project_id=objective.web_project_id,
    )
    return extract_path(resolved_url or "")


def _movie_detail_action(objective: DeterministicTaskObjective, *, require_trailer: bool = False) -> dict[str, Any]:
    filters = dict(objective.entity_filters)
    query = str(objective.field_values.get("query") or "")
    if query and "name_exact" not in filters and "name_contains" not in filters:
        filters["name_contains"] = query
    if require_trailer:
        filters["requires_trailer"] = True
    resolved_url = resolve_movie_detail_url(
        task_url=objective.task_url,
        filters=filters,
        web_project_id=objective.web_project_id,
    )
    if not resolved_url:
        resolved_url = resolve_movie_detail_url(
            task_url=objective.task_url,
            filters={"requires_trailer": True} if require_trailer else {},
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
        return navigate(route, objective=objective)
    return click(selectors=view_detail_selectors(objective.web_project_id, objective.seed), field_name="movie detail")


def _login_actions(objective: DeterministicTaskObjective) -> list[dict[str, Any]]:
    username = str(objective.field_values.get("username") or "user1")
    password = str(objective.field_values.get("password") or "Passw0rd!")
    return [
        navigate("/login", objective=objective),
        type_text(username, selectors=login_username_selectors(objective.web_project_id, objective.seed), field_name="username"),
        type_text(password, selectors=login_password_selectors(objective.web_project_id, objective.seed), field_name="password"),
        click(selectors=login_submit_selectors(objective.web_project_id, objective.seed), field_name="submit"),
    ]


def _registration_actions(objective: DeterministicTaskObjective) -> list[dict[str, Any]]:
    username = str(objective.field_values.get("username") or f"newuser{objective.seed}")
    email = str(objective.field_values.get("email") or f"newuser{objective.seed}@gmail.com")
    password = str(objective.field_values.get("password") or "Passw0rd!")
    confirm_password = str(objective.field_values.get("confirm_password") or password)
    return [
        navigate("/register", objective=objective),
        type_text(username, selectors=register_username_selectors(objective.web_project_id, objective.seed), field_name="username"),
        type_text(email, selectors=register_email_selectors(objective.web_project_id, objective.seed), field_name="email"),
        type_text(password, selectors=register_password_selectors(objective.web_project_id, objective.seed), field_name="password"),
        type_text(confirm_password, selectors=register_confirm_password_selectors(objective.web_project_id, objective.seed), field_name="confirm password"),
        click(selectors=register_submit_selectors(objective.web_project_id, objective.seed), field_name="submit"),
    ]


def _contact_actions(objective: DeterministicTaskObjective) -> list[dict[str, Any]]:
    name = str(objective.field_values.get("name") or "Agent")
    email = str(objective.field_values.get("email") or "agent@example.com")
    subject = str(objective.field_values.get("subject") or "General inquiry")
    message = str(objective.field_values.get("message") or "Please help with my movie request.")
    return [
        navigate("/contact", objective=objective),
        type_text(name, selectors=contact_name_selectors(objective.web_project_id, objective.seed), field_name="name"),
        type_text(email, selectors=contact_email_selectors(objective.web_project_id, objective.seed), field_name="email"),
        type_text(subject, selectors=contact_subject_selectors(objective.web_project_id, objective.seed), field_name="subject"),
        type_text(message, selectors=contact_message_selectors(objective.web_project_id, objective.seed), field_name="message"),
        click(selectors=contact_submit_selectors(objective.web_project_id, objective.seed), field_name="submit"),
    ]


def _search_actions(objective: DeterministicTaskObjective) -> list[dict[str, Any]]:
    query = str(objective.field_values.get("query") or objective.field_values.get("movie_name") or "The Matrix")
    return [
        navigate("/search", objective=objective),
        type_text(query, selectors=search_input_selectors(objective.web_project_id, objective.seed), field_name="search"),
        click(selectors=search_submit_selectors(objective.web_project_id, objective.seed), field_name="submit"),
    ]


def _movie_target_actions(
    objective: DeterministicTaskObjective,
    *,
    include_search: bool = True,
    require_trailer: bool = False,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if include_search and str(objective.field_values.get("query") or "").strip():
        actions.extend(_search_actions(objective))
    else:
        actions.append(navigate(objective.route_target, objective=objective))
    actions.append(_movie_detail_action(objective, require_trailer=require_trailer))
    return actions


def _profile_fill_actions(objective: DeterministicTaskObjective) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    selectors_by_key = {
        "first_name": profile_first_name_selectors(objective.web_project_id, objective.seed),
        "last_name": profile_last_name_selectors(objective.web_project_id, objective.seed),
        "email": profile_email_selectors(objective.web_project_id, objective.seed),
        "favorite_genres": profile_favorite_genres_selectors(objective.web_project_id, objective.seed),
        "location": profile_location_selectors(objective.web_project_id, objective.seed),
        "website": profile_website_selectors(objective.web_project_id, objective.seed),
        "bio": profile_bio_selectors(objective.web_project_id, objective.seed),
    }
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
            actions.append(type_text(value, selectors=selectors_by_key.get(objective_key), field_name=field_name))
    actions.append(click(selectors=profile_save_selectors(objective.web_project_id, objective.seed), field_name="submit"))
    return actions


def _movie_editor_fill_actions(
    objective: DeterministicTaskObjective,
    *,
    target_movie_path: str = "",
    target_movie_title: str = "",
    editor_scope: str = "",
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    scope = ""
    if target_movie_path:
        scope = f'div.rounded-3xl:has(a[href*="{target_movie_path}"])'
    elif target_movie_title:
        scope = f"div.rounded-3xl:has(h3:has-text({json.dumps(target_movie_title)}))"
    elif editor_scope:
        scope = str(editor_scope).strip()

    scoped_selectors: dict[str, list[dict[str, Any]]] = {}
    if scope:
        scoped_selectors = {
            "title": [custom_selector(f'{scope} label:has-text("Title") input')],
            "director": [custom_selector(f'{scope} label:has-text("Director") input')],
            "year": [custom_selector(f'{scope} label:has-text("Year") input')],
            "duration": [custom_selector(f'{scope} label:has-text("Duration") input')],
            "rating": [custom_selector(f'{scope} label:has-text("Rating") input')],
            "trailer": [custom_selector(f'{scope} label:has-text("Trailer URL") input')],
            "genres": [custom_selector(f'{scope} div.space-y-2:has(p:has-text("Genres")) input')],
            "cast": [custom_selector(f'{scope} label:has-text("Cast") input')],
            "synopsis": [custom_selector(f'{scope} label:has-text("Synopsis") textarea')],
        }

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
            selectors = scoped_selectors.get(field_name)
            actions.append(type_text(value, selectors=selectors, field_name=field_name))
    save_selectors = list(save_changes_selectors(objective.web_project_id, objective.seed))
    if scope:
        save_selectors.insert(0, custom_selector(f"{scope} form button[type='submit']"))
    actions.append(click(selectors=save_selectors, field_name="submit"))
    return actions


def _build_login_plan(objective: DeterministicTaskObjective) -> list[dict[str, Any]]:
    return _login_actions(objective)


def _build_registration_plan(objective: DeterministicTaskObjective) -> list[dict[str, Any]]:
    return _registration_actions(objective)


def _build_contact_plan(objective: DeterministicTaskObjective) -> list[dict[str, Any]]:
    return _contact_actions(objective)


def _build_search_film_plan(objective: DeterministicTaskObjective) -> list[dict[str, Any]]:
    return _search_actions(objective)


def _build_filter_film_plan(objective: DeterministicTaskObjective) -> list[dict[str, Any]]:
    year = str(objective.field_values.get("year") or "").strip()
    genre = str(objective.field_values.get("genres") or objective.entity_filters.get("genre_exact") or "").strip()
    actions = [navigate("/search", objective=objective)]
    dropdown_order = generate_dynamic_order(objective.seed, "filter-dropdowns", 2)
    genre_slot = (dropdown_order.index(0) + 1) if 0 in dropdown_order else 1
    year_slot = (dropdown_order.index(1) + 1) if 1 in dropdown_order else 2

    def filter_selectors(slot: int, fallback_slot: int) -> list[dict[str, Any]]:
        selectors = [custom_selector(f"section#library select:nth-of-type({slot})")]
        if fallback_slot != slot:
            selectors.append(custom_selector(f"section#library select:nth-of-type({fallback_slot})"))
        return selectors

    if year:
        actions.append(
            select(
                year,
                field_name="year",
                selectors=filter_selectors(year_slot, 2 if year_slot == 1 else 1),
            )
        )
    if genre:
        actions.append(
            select(
                genre,
                field_name="genre",
                selectors=filter_selectors(genre_slot, 2 if genre_slot == 1 else 1),
            )
        )
    return actions


def _build_film_detail_plan(objective: DeterministicTaskObjective) -> list[dict[str, Any]]:
    return _movie_target_actions(objective)


def _build_add_comment_plan(objective: DeterministicTaskObjective) -> list[dict[str, Any]]:
    actions = _movie_target_actions(objective)
    actions.extend(
        [
            type_text(str(objective.field_values.get("name") or "Agent"), selectors=comment_name_selectors(objective.web_project_id, objective.seed), field_name="name"),
            type_text(
                str(objective.field_values.get("content") or "Great movie"),
                selectors=comment_message_selectors(objective.web_project_id, objective.seed),
                field_name="comment",
            ),
            click(selectors=comment_submit_selectors(objective.web_project_id, objective.seed), field_name="submit"),
        ]
    )
    return actions


def _build_share_movie_plan(objective: DeterministicTaskObjective) -> list[dict[str, Any]]:
    actions = _movie_target_actions(objective)
    actions.append(click(selectors=share_button_selectors(objective.web_project_id, objective.seed), field_name="share"))
    return actions


def _build_watch_trailer_plan(objective: DeterministicTaskObjective) -> list[dict[str, Any]]:
    actions = _movie_target_actions(objective, include_search=False, require_trailer=True)
    actions.append(click(selectors=trailer_button_selectors(objective.web_project_id, objective.seed), field_name="watch trailer"))
    return actions


def _build_add_to_watchlist_plan(objective: DeterministicTaskObjective) -> list[dict[str, Any]]:
    actions = _login_actions(objective)
    actions.extend(_movie_target_actions(objective, include_search=False))
    actions.append(click(selectors=watchlist_button_selectors(objective.web_project_id, objective.seed), field_name="watchlist"))
    return actions


def _build_remove_from_watchlist_plan(objective: DeterministicTaskObjective) -> list[dict[str, Any]]:
    actions = _login_actions(objective)
    actions.extend(_movie_target_actions(objective, include_search=False))
    watchlist_selectors = watchlist_button_selectors(objective.web_project_id, objective.seed)
    actions.append(click(selectors=watchlist_selectors, field_name="watchlist"))
    actions.append(click(selectors=watchlist_selectors, field_name="remove from watchlist"))
    return actions


def _build_edit_user_plan(objective: DeterministicTaskObjective) -> list[dict[str, Any]]:
    actions = _login_actions(objective)
    actions.append(navigate("/profile", objective=objective))
    actions.extend(_profile_fill_actions(objective))
    return actions


def _build_logout_plan(objective: DeterministicTaskObjective) -> list[dict[str, Any]]:
    actions = _login_actions(objective)
    actions.append(click(selectors=logout_selectors(objective.web_project_id, objective.seed), field_name="logout"))
    return actions


def _build_add_film_plan(objective: DeterministicTaskObjective) -> list[dict[str, Any]]:
    actions = _login_actions(objective)
    add_tab_selectors = list(profile_tab_selectors("add-movies", objective.web_project_id))
    add_tab_selectors.insert(0, custom_selector('button[role="tab"][aria-controls*="add-movies"]'))
    add_tab_selectors.insert(1, custom_selector('button[role="tab"][id*="trigger-add-movies"]'))
    add_tab_selectors.insert(2, custom_selector('button[role="tab"]:has-text("Movies")'))
    actions.extend(
        [
            navigate("/profile", objective=objective),
            click(selectors=add_tab_selectors, field_name="add movies"),
            *_movie_editor_fill_actions(
                objective,
                editor_scope='div[role="tabpanel"][data-state="active"] div.rounded-2xl:has(form)',
            ),
        ]
    )
    return actions


def _build_edit_film_plan(objective: DeterministicTaskObjective) -> list[dict[str, Any]]:
    target_path = _resolve_target_movie_path(objective)
    target_title = str(objective.entity_filters.get("name_exact") or objective.field_values.get("query") or "").strip()
    actions = _login_actions(objective)
    actions.extend(
        [
            navigate("/profile", objective=objective),
            click(selectors=profile_tab_selectors("movies", objective.web_project_id), field_name="edit movies"),
            *_movie_editor_fill_actions(objective, target_movie_path=target_path, target_movie_title=target_title),
        ]
    )
    return actions


def _build_delete_film_plan(objective: DeterministicTaskObjective) -> list[dict[str, Any]]:
    actions = _login_actions(objective)
    actions.extend(
        [
            navigate("/profile", objective=objective),
            click(selectors=profile_tab_selectors("movies", objective.web_project_id), field_name="edit movies"),
            click(selectors=delete_movie_selectors(objective.web_project_id, objective.seed), field_name="delete movie"),
        ]
    )
    return actions


AUTOCINEMA_PLAN_BUILDERS = {
    "LOGIN": _build_login_plan,
    "REGISTRATION": _build_registration_plan,
    "CONTACT": _build_contact_plan,
    "SEARCH_FILM": _build_search_film_plan,
    "FILTER_FILM": _build_filter_film_plan,
    "FILM_DETAIL": _build_film_detail_plan,
    "ADD_COMMENT": _build_add_comment_plan,
    "SHARE_MOVIE": _build_share_movie_plan,
    "WATCH_TRAILER": _build_watch_trailer_plan,
    "ADD_TO_WATCHLIST": _build_add_to_watchlist_plan,
    "REMOVE_FROM_WATCHLIST": _build_remove_from_watchlist_plan,
    "EDIT_USER": _build_edit_user_plan,
    "LOGOUT": _build_logout_plan,
    "ADD_FILM": _build_add_film_plan,
    "EDIT_FILM": _build_edit_film_plan,
    "DELETE_FILM": _build_delete_film_plan,
}


def build_autocinema_plan(objective: DeterministicTaskObjective) -> DeterministicPlan:
    use_case = str(objective.use_case or "").strip().upper()
    builder = AUTOCINEMA_PLAN_BUILDERS.get(use_case)
    if builder is None:
        raise ValueError(f"Unsupported deterministic use case: {use_case}")
    actions = builder(objective)
    return DeterministicPlan(
        prompt_lines=(f"Deterministic planner for {use_case}", f"seed={objective.seed}"),
        actions=tuple(actions),
        metadata={
            "auth_required": bool(objective.auth_required),
            "entity_filters": dict(objective.entity_filters),
            "field_values": dict(objective.field_values),
            "route_target": objective.route_target,
            "web_project_id": objective.web_project_id,
        },
    )


__all__ = ["AUTOCINEMA_PLAN_BUILDERS", "build_autocinema_plan"]
