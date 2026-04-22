from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

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
    search_submit_selectors,
    share_button_selectors,
    trailer_button_selectors,
    view_detail_selectors,
    watchlist_button_selectors,
    watchlist_remove_profile_selectors,
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


def _select(value: str, *, selectors: list[dict[str, Any]] | None = None, field_name: str = "") -> dict[str, Any]:
    payload: dict[str, Any] = {"type": "SelectAction", "value": str(value), "field_name": str(field_name or "").strip()}
    if selectors:
        payload["selector_candidates"] = selectors
    return payload


def _custom_selector(value: str) -> dict[str, Any]:
    return {
        "type": "attributeValueSelector",
        "attribute": "custom",
        "value": str(value),
        "case_sensitive": False,
    }


def _hash_string(value: str) -> int:
    hash_value = 0
    for char in str(value or ""):
        hash_value = ((hash_value << 5) - hash_value) + ord(char)
        hash_value &= 0xFFFFFFFF
        if hash_value >= 0x80000000:
            hash_value -= 0x100000000
    return abs(hash_value)


def _select_variant_index(seed: int, key: str, count: int) -> int:
    if int(count) <= 1:
        return 0
    return abs(_hash_string(f"{str(key or '').strip()}:{int(seed)}")) % int(count)


def _generate_hash_order(seed: int, key: str, count: int) -> list[int]:
    combined = f"{key}:{seed}"
    hash_value = 0
    for char in combined:
        hash_value = ((hash_value << 5) - hash_value) + ord(char)
        hash_value &= 0xFFFFFFFF
        if hash_value >= 0x80000000:
            hash_value -= 0x100000000
    order = list(range(count))
    for idx in range(count - 1, 0, -1):
        swap_idx = abs(hash_value + idx * 7919) % (idx + 1)
        order[idx], order[swap_idx] = order[swap_idx], order[idx]
    return order


def _generate_dynamic_order(seed: int, key: str, count: int) -> list[int]:
    if count <= 1:
        return [0]
    original = list(range(count))
    if int(seed) == 1:
        return original
    variants: list[list[int]] = []
    for offset in range(count):
        variants.append([(index + offset) % count for index in range(count)])
    for index in range(count - 1):
        swapped = [idx for idx in range(count)]
        swapped[index], swapped[index + 1] = swapped[index + 1], swapped[index]
        if swapped != original:
            variants.append(swapped)
    for split in range(1, count):
        reversed_part = [split - 1 - idx for idx in range(split)] + [split + idx for idx in range(count - split)]
        if reversed_part != original:
            variants.append(reversed_part)
    hash_variant = _generate_hash_order(seed, key, count)
    if hash_variant != original:
        variants.append(hash_variant)
    deduped: list[list[int]] = []
    seen: set[str] = set()
    for variant in variants:
        key_value = ",".join(str(item) for item in variant)
        if key_value in seen:
            continue
        seen.add(key_value)
        deduped.append(variant)
    if not deduped:
        return original
    variant_idx = _select_variant_index(seed, key, len(deduped))
    return deduped[variant_idx]


def _extract_path(route_or_url: str) -> str:
    raw = str(route_or_url or "").strip()
    if not raw:
        return ""
    if raw.startswith("http://") or raw.startswith("https://"):
        return str(urlparse(raw).path or "").strip()
    return raw.split("?", 1)[0].strip()


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
    return _extract_path(resolved_url or "")


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
        # If strict filters produce no candidate, still navigate to a concrete movie page
        # to avoid getting stuck on empty or 404 listing pages.
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
        return _navigate(route, objective=objective)
    return _click(selectors=view_detail_selectors(objective.web_project_id, objective.seed), field_name="movie detail")


def _login_actions(objective: DeterministicTaskObjective) -> list[dict[str, Any]]:
    username = str(objective.field_values.get("username") or "user1")
    password = str(objective.field_values.get("password") or "Passw0rd!")
    return [
        _navigate("/login", objective=objective),
        _type(username, selectors=login_username_selectors(objective.web_project_id, objective.seed), field_name="username"),
        _type(password, selectors=login_password_selectors(objective.web_project_id, objective.seed), field_name="password"),
        _click(selectors=login_submit_selectors(objective.web_project_id, objective.seed), field_name="submit"),
    ]


def _registration_actions(objective: DeterministicTaskObjective) -> list[dict[str, Any]]:
    username = str(objective.field_values.get("username") or f"newuser{objective.seed}")
    email = str(objective.field_values.get("email") or f"newuser{objective.seed}@gmail.com")
    password = str(objective.field_values.get("password") or "Passw0rd!")
    confirm_password = str(objective.field_values.get("confirm_password") or password)
    return [
        _navigate("/register", objective=objective),
        _type(username, selectors=register_username_selectors(objective.web_project_id, objective.seed), field_name="username"),
        _type(email, selectors=register_email_selectors(objective.web_project_id, objective.seed), field_name="email"),
        _type(password, selectors=register_password_selectors(objective.web_project_id, objective.seed), field_name="password"),
        _type(confirm_password, selectors=register_confirm_password_selectors(objective.web_project_id, objective.seed), field_name="confirm password"),
        _click(selectors=register_submit_selectors(objective.web_project_id, objective.seed), field_name="submit"),
    ]


def _contact_actions(objective: DeterministicTaskObjective) -> list[dict[str, Any]]:
    name = str(objective.field_values.get("name") or "Agent")
    email = str(objective.field_values.get("email") or "agent@example.com")
    subject = str(objective.field_values.get("subject") or "General inquiry")
    message = str(objective.field_values.get("message") or "Please help with my movie request.")
    return [
        _navigate("/contact", objective=objective),
        _type(name, selectors=contact_name_selectors(objective.web_project_id, objective.seed), field_name="name"),
        _type(email, selectors=contact_email_selectors(objective.web_project_id, objective.seed), field_name="email"),
        _type(subject, selectors=contact_subject_selectors(objective.web_project_id, objective.seed), field_name="subject"),
        _type(message, selectors=contact_message_selectors(objective.web_project_id, objective.seed), field_name="message"),
        _click(selectors=contact_submit_selectors(objective.web_project_id, objective.seed), field_name="submit"),
    ]


def _search_actions(objective: DeterministicTaskObjective) -> list[dict[str, Any]]:
    query = str(objective.field_values.get("query") or objective.field_values.get("movie_name") or "The Matrix")
    return [
        _navigate("/search", objective=objective),
        _type(query, field_name="search"),
        _click(selectors=search_submit_selectors(objective.web_project_id, objective.seed), field_name="submit"),
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
        actions.append(_navigate(objective.route_target, objective=objective))
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
            actions.append(_type(value, selectors=selectors_by_key.get(objective_key), field_name=field_name))
    actions.append(_click(selectors=profile_save_selectors(objective.web_project_id, objective.seed), field_name="submit"))
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
            "title": [_custom_selector(f'{scope} label:has-text("Title") input')],
            "director": [_custom_selector(f'{scope} label:has-text("Director") input')],
            "year": [_custom_selector(f'{scope} label:has-text("Year") input')],
            "duration": [_custom_selector(f'{scope} label:has-text("Duration") input')],
            "rating": [_custom_selector(f'{scope} label:has-text("Rating") input')],
            "trailer": [_custom_selector(f'{scope} label:has-text("Trailer URL") input')],
            "genres": [_custom_selector(f'{scope} div.space-y-2:has(p:has-text("Genres")) input')],
            "cast": [_custom_selector(f'{scope} label:has-text("Cast") input')],
            "synopsis": [_custom_selector(f'{scope} label:has-text("Synopsis") textarea')],
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
            actions.append(_type(value, selectors=selectors, field_name=field_name))
    save_selectors = list(save_changes_selectors(objective.web_project_id, objective.seed))
    if scope:
        save_selectors.insert(0, _custom_selector(f"{scope} form button[type='submit']"))
    actions.append(_click(selectors=save_selectors, field_name="submit"))
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
        year = str(objective.field_values.get("year") or "").strip()
        genre = str(objective.field_values.get("genres") or objective.entity_filters.get("genre_exact") or "").strip()
        actions = [_navigate("/search", objective=objective)]
        dropdown_order = _generate_dynamic_order(objective.seed, "filter-dropdowns", 2)
        genre_slot = (dropdown_order.index(0) + 1) if 0 in dropdown_order else 1
        year_slot = (dropdown_order.index(1) + 1) if 1 in dropdown_order else 2

        def _filter_selectors(slot: int, fallback_slot: int) -> list[dict[str, Any]]:
            selectors = [_custom_selector(f"section#library select:nth-of-type({slot})")]
            if fallback_slot != slot:
                selectors.append(_custom_selector(f"section#library select:nth-of-type({fallback_slot})"))
            return selectors

        if year:
            actions.append(
                _select(
                    year,
                    field_name="year",
                    selectors=_filter_selectors(year_slot, 2 if year_slot == 1 else 1),
                )
            )
        if genre:
            actions.append(
                _select(
                    genre,
                    field_name="genre",
                    selectors=_filter_selectors(genre_slot, 2 if genre_slot == 1 else 1),
                )
            )
    elif use_case == "FILM_DETAIL":
        actions = _movie_target_actions(objective)
    elif use_case == "ADD_COMMENT":
        actions = _movie_target_actions(objective)
        actions.extend(
            [
                _type(str(objective.field_values.get("name") or "Agent"), selectors=comment_name_selectors(objective.web_project_id, objective.seed), field_name="name"),
                _type(
                    str(objective.field_values.get("content") or "Great movie"),
                    selectors=comment_message_selectors(objective.web_project_id, objective.seed),
                    field_name="comment",
                ),
                _click(selectors=comment_submit_selectors(objective.web_project_id, objective.seed), field_name="submit"),
            ]
        )
    elif use_case == "SHARE_MOVIE":
        actions = _movie_target_actions(objective)
        actions.append(_click(selectors=share_button_selectors(objective.web_project_id, objective.seed), field_name="share"))
    elif use_case == "WATCH_TRAILER":
        actions = _movie_target_actions(objective, include_search=False, require_trailer=True)
        actions.append(_click(selectors=trailer_button_selectors(objective.web_project_id, objective.seed), field_name="watch trailer"))
    elif use_case == "ADD_TO_WATCHLIST":
        actions = _login_actions(objective)
        actions.extend(_movie_target_actions(objective, include_search=False))
        actions.append(_click(selectors=watchlist_button_selectors(objective.web_project_id, objective.seed), field_name="watchlist"))
    elif use_case == "REMOVE_FROM_WATCHLIST":
        target_path = _resolve_target_movie_path(objective)
        target_title = str(objective.entity_filters.get("name_exact") or objective.field_values.get("query") or "").strip()
        remove_selectors = list(watchlist_remove_profile_selectors(objective.web_project_id, objective.seed))
        if target_path:
            remove_selectors.insert(0, _custom_selector(f'div.rounded-3xl:has(a[href*="{target_path}"]) button:has-text("Remove from List")'))
        elif target_title:
            remove_selectors.insert(
                0,
                _custom_selector(f'div.rounded-3xl:has(h3:has-text({json.dumps(target_title)})) button:has-text("Remove from List")'),
            )
        actions = _login_actions(objective)
        actions.extend(
            [
                _navigate("/profile", objective=objective),
                _click(selectors=profile_tab_selectors("watchlist", objective.web_project_id), field_name="watchlist"),
                _click(selectors=remove_selectors, field_name="remove from watchlist"),
            ]
        )
    elif use_case == "EDIT_USER":
        actions = _login_actions(objective)
        actions.append(_navigate("/profile", objective=objective))
        actions.extend(_profile_fill_actions(objective))
    elif use_case == "LOGOUT":
        actions = _login_actions(objective)
        actions.append(_click(selectors=logout_selectors(objective.web_project_id, objective.seed), field_name="logout"))
    elif use_case == "ADD_FILM":
        actions = _login_actions(objective)
        add_tab_selectors = list(profile_tab_selectors("add-movies", objective.web_project_id))
        add_tab_selectors.insert(0, _custom_selector('button[role="tab"][aria-controls*="add-movies"]'))
        add_tab_selectors.insert(1, _custom_selector('button[role="tab"][id*="trigger-add-movies"]'))
        add_tab_selectors.insert(2, _custom_selector('button[role="tab"]:has-text("Movies")'))
        actions.extend(
            [
                _navigate("/profile", objective=objective),
                _click(selectors=add_tab_selectors, field_name="add movies"),
                *_movie_editor_fill_actions(
                    objective,
                    editor_scope='div[role="tabpanel"][data-state="active"] div.rounded-2xl:has(form)',
                ),
            ]
        )
    elif use_case == "EDIT_FILM":
        target_path = _resolve_target_movie_path(objective)
        target_title = str(objective.entity_filters.get("name_exact") or objective.field_values.get("query") or "").strip()
        actions = _login_actions(objective)
        actions.extend(
            [
                _navigate("/profile", objective=objective),
                _click(selectors=profile_tab_selectors("movies", objective.web_project_id), field_name="edit movies"),
                *_movie_editor_fill_actions(objective, target_movie_path=target_path, target_movie_title=target_title),
            ]
        )
    elif use_case == "DELETE_FILM":
        actions = _login_actions(objective)
        actions.extend(
            [
                _navigate("/profile", objective=objective),
                _click(selectors=profile_tab_selectors("movies", objective.web_project_id), field_name="edit movies"),
                _click(selectors=delete_movie_selectors(objective.web_project_id, objective.seed), field_name="delete movie"),
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
