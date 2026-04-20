from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from training.harvester_support import _selector_candidates

from .projects import project_config

_ROUTES = {
    "ADD_COMMENT": "/movies",
    "ADD_FILM": "/profile",
    "ADD_TO_WATCHLIST": "/movies",
    "CONTACT": "/contact",
    "DELETE_FILM": "/profile",
    "EDIT_FILM": "/profile",
    "EDIT_USER": "/profile",
    "FILM_DETAIL": "/movies",
    "FILTER_FILM": "/search",
    "LOGIN": "/login",
    "LOGOUT": "/profile",
    "REGISTRATION": "/register",
    "REMOVE_FROM_WATCHLIST": "/movies",
    "SEARCH_FILM": "/search",
    "SHARE_MOVIE": "/movies",
    "WATCH_TRAILER": "/movies",
}


def route_for_use_case(use_case: str, project_id: str = "autocinema") -> str:
    if str(project_id or "").strip() != "autocinema":
        raise ValueError(f"Project {project_id!r} is not yet supported by deterministic selectors")
    return _ROUTES.get(str(use_case or "").strip().upper(), "/")


@lru_cache(maxsize=32)
def _load_id_variants(project_id: str) -> dict[str, list[str]]:
    path = project_config(project_id).id_variants_path
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, list[str]] = {}
    if not isinstance(payload, dict):
        return out
    for key, value in payload.items():
        if isinstance(value, list):
            out[str(key).strip()] = [str(item).strip() for item in value if str(item).strip()]
    return out


def _expand_id_variants(project_id: str, ids: list[str] | None) -> list[str]:
    variants_map = _load_id_variants(project_id)
    out: list[str] = []
    seen: set[str] = set()
    for value in ids or []:
        candidate = str(value).strip()
        if not candidate:
            continue
        for item in [candidate, *variants_map.get(candidate, [])]:
            normalized = str(item).strip()
            lowered = normalized.lower()
            if normalized and lowered not in seen:
                seen.add(lowered)
                out.append(normalized)
    return out


def selector_candidates_for_ids(*ids: str, project_id: str = "autocinema") -> list[dict[str, Any]]:
    expanded = _expand_id_variants(project_id, [value for value in ids if str(value).strip()])
    return _selector_candidates(ids=expanded)


def selector_candidates_for_texts(*texts: str) -> list[dict[str, Any]]:
    return _selector_candidates(texts=[value for value in texts if str(value).strip()])


def _id_and_text_selectors(*, ids: list[str], texts: list[str], project_id: str = "autocinema") -> list[dict[str, Any]]:
    return _selector_candidates(ids=_expand_id_variants(project_id, ids), texts=texts)


def login_username_selectors(project_id: str = "autocinema") -> list[dict[str, Any]]:
    return selector_candidates_for_ids("login-username-input", project_id=project_id)


def login_password_selectors(project_id: str = "autocinema") -> list[dict[str, Any]]:
    return selector_candidates_for_ids("login-password-input", project_id=project_id)


def login_submit_selectors(project_id: str = "autocinema") -> list[dict[str, Any]]:
    return _id_and_text_selectors(ids=["login-sign-in-button"], texts=["Sign in", "Login"], project_id=project_id)


def register_username_selectors(project_id: str = "autocinema") -> list[dict[str, Any]]:
    return selector_candidates_for_ids("register-username-input", project_id=project_id)


def register_email_selectors(project_id: str = "autocinema") -> list[dict[str, Any]]:
    return selector_candidates_for_ids("register-email-input", project_id=project_id)


def register_password_selectors(project_id: str = "autocinema") -> list[dict[str, Any]]:
    return selector_candidates_for_ids("register-password-input", project_id=project_id)


def register_confirm_password_selectors(project_id: str = "autocinema") -> list[dict[str, Any]]:
    return selector_candidates_for_ids("register-confirm-password-input", project_id=project_id)


def register_submit_selectors(project_id: str = "autocinema") -> list[dict[str, Any]]:
    return _id_and_text_selectors(ids=["create-account-button"], texts=["Create account", "Register"], project_id=project_id)


def contact_name_selectors(project_id: str = "autocinema") -> list[dict[str, Any]]:
    return selector_candidates_for_ids("contact-name-input", project_id=project_id)


def contact_email_selectors(project_id: str = "autocinema") -> list[dict[str, Any]]:
    return selector_candidates_for_ids("contact-email-input", project_id=project_id)


def contact_subject_selectors(project_id: str = "autocinema") -> list[dict[str, Any]]:
    return selector_candidates_for_ids("contact-subject-input", project_id=project_id)


def contact_message_selectors(project_id: str = "autocinema") -> list[dict[str, Any]]:
    return selector_candidates_for_ids("contact-message-textarea", project_id=project_id)


def contact_submit_selectors(project_id: str = "autocinema") -> list[dict[str, Any]]:
    return _id_and_text_selectors(ids=["send-message-button"], texts=["Send Message"], project_id=project_id)


def comment_name_selectors(project_id: str = "autocinema") -> list[dict[str, Any]]:
    return selector_candidates_for_ids("comment-name-input", project_id=project_id)


def comment_message_selectors(project_id: str = "autocinema") -> list[dict[str, Any]]:
    return selector_candidates_for_ids("comment-message-textarea", project_id=project_id)


def comment_submit_selectors(project_id: str = "autocinema") -> list[dict[str, Any]]:
    return _id_and_text_selectors(ids=["share-feedback-button"], texts=["Share Feedback", "Post", "Comment"], project_id=project_id)


def watchlist_button_selectors(project_id: str = "autocinema") -> list[dict[str, Any]]:
    return _id_and_text_selectors(
        ids=["watchlist-button"],
        texts=["Add to watchlist", "Remove from watchlist", "Watchlist"],
        project_id=project_id,
    )


def share_button_selectors(project_id: str = "autocinema") -> list[dict[str, Any]]:
    return _id_and_text_selectors(ids=["share-button"], texts=["Share"], project_id=project_id)


def trailer_button_selectors(project_id: str = "autocinema") -> list[dict[str, Any]]:
    return _id_and_text_selectors(ids=["watch-trailer-button"], texts=["Watch trailer"], project_id=project_id)


def search_submit_selectors(project_id: str = "autocinema") -> list[dict[str, Any]]:
    return _id_and_text_selectors(ids=["search-submit-button"], texts=["Search"], project_id=project_id)


def view_detail_selectors(project_id: str = "autocinema") -> list[dict[str, Any]]:
    return _id_and_text_selectors(ids=["view-details-button"], texts=["View Details"], project_id=project_id)


def profile_save_selectors(project_id: str = "autocinema") -> list[dict[str, Any]]:
    return _id_and_text_selectors(ids=["save-profile-button"], texts=["Save Profile"], project_id=project_id)


def delete_movie_selectors(project_id: str = "autocinema") -> list[dict[str, Any]]:
    return _id_and_text_selectors(ids=["delete-movie-button"], texts=["Delete Movie", "Delete movie"], project_id=project_id)


def save_changes_selectors(project_id: str = "autocinema") -> list[dict[str, Any]]:
    return _id_and_text_selectors(ids=["save-changes-button"], texts=["Save changes", "Add Film", "Edit Film"], project_id=project_id)


def logout_selectors(project_id: str = "autocinema") -> list[dict[str, Any]]:
    if str(project_id or "").strip() != "autocinema":
        raise ValueError(f"Project {project_id!r} is not yet supported by deterministic selectors")
    return selector_candidates_for_texts("Logout")


def profile_tab_selectors(tab_name: str, project_id: str = "autocinema") -> list[dict[str, Any]]:
    if str(project_id or "").strip() != "autocinema":
        raise ValueError(f"Project {project_id!r} is not yet supported by deterministic selectors")
    normalized = str(tab_name or "").strip().lower()
    texts = {
        "movies": ["Edit Movies", "Movies"],
        "watchlist": ["Watchlist"],
        "add-movies": ["Add Movies"],
    }
    return selector_candidates_for_texts(*texts.get(normalized, [tab_name]))


__all__ = [
    "comment_message_selectors",
    "comment_name_selectors",
    "comment_submit_selectors",
    "contact_email_selectors",
    "contact_message_selectors",
    "contact_name_selectors",
    "contact_subject_selectors",
    "contact_submit_selectors",
    "delete_movie_selectors",
    "login_password_selectors",
    "login_submit_selectors",
    "login_username_selectors",
    "logout_selectors",
    "profile_save_selectors",
    "profile_tab_selectors",
    "register_confirm_password_selectors",
    "register_email_selectors",
    "register_password_selectors",
    "register_submit_selectors",
    "register_username_selectors",
    "route_for_use_case",
    "save_changes_selectors",
    "search_submit_selectors",
    "selector_candidates_for_ids",
    "selector_candidates_for_texts",
    "share_button_selectors",
    "trailer_button_selectors",
    "view_detail_selectors",
    "watchlist_button_selectors",
]
