from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from training.harvester_support import _selector_candidates

from .normalizer import route_for_web_project_use_case
from .projects import project_config

_AUTOCINEMA_LOCAL_ID_VARIANTS: dict[str, list[str]] = {
    "featured-view-details-button": [
        "hero-view-details-btn",
        "view-details-button",
        "details-action",
        "view-movie-btn",
        "details-btn",
    ],
}


def route_for_use_case(use_case: str, project_id: str = "autocinema") -> str:
    return route_for_web_project_use_case(
        web_project_id=project_id,
        use_case=str(use_case or "").strip().upper(),
    )


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


def _read_variants(path) -> dict[str, list[str]]:
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


@lru_cache(maxsize=32)
def _load_id_variants(project_id: str) -> dict[str, list[str]]:
    return _read_variants(project_config(project_id).id_variants_path)


@lru_cache(maxsize=32)
def _load_class_variants(project_id: str) -> dict[str, list[str]]:
    return _read_variants(project_config(project_id).class_variants_path)


@lru_cache(maxsize=32)
def _load_text_variants(project_id: str) -> dict[str, list[str]]:
    return _read_variants(project_config(project_id).text_variants_path)


def _expand_variants(variants_map: dict[str, list[str]], values: list[str] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values or []:
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


def _expand_id_variants(project_id: str, ids: list[str] | None) -> list[str]:
    return _expand_variants(_load_id_variants(project_id), ids)


def _seeded_variant(
    *,
    seed: int | None,
    key: str,
    variants: list[str],
    fallback: str = "",
) -> str:
    normalized_variants = [str(item).strip() for item in variants if str(item).strip()]
    if not normalized_variants:
        return str(fallback or "").strip()
    if seed is None or int(seed) == 1:
        return normalized_variants[0]
    index = _select_variant_index(int(seed), str(key or "").strip(), len(normalized_variants))
    return normalized_variants[index] if 0 <= index < len(normalized_variants) else normalized_variants[0]


def _seeded_selector_ids(
    *,
    project_id: str,
    seed: int | None,
    specs: list[dict[str, Any]],
) -> list[str]:
    if seed is None:
        return []
    variants_map = _load_id_variants(project_id)
    out: list[str] = []
    seen: set[str] = set()
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        key = str(spec.get("key") or "").strip()
        if not key:
            continue
        variants = spec.get("variants")
        variant_list = [str(item).strip() for item in variants] if isinstance(variants, list) else variants_map.get(key, [])
        fallback = str(spec.get("fallback") or "").strip()
        resolved = _seeded_variant(seed=seed, key=key, variants=variant_list, fallback=fallback)
        lowered = resolved.lower()
        if resolved and lowered not in seen:
            seen.add(lowered)
            out.append(resolved)
    return out


def _seeded_selector_values(
    *,
    seed: int | None,
    specs: list[dict[str, Any]],
    variants_map: dict[str, list[str]],
) -> list[str]:
    if seed is None:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        key = str(spec.get("key") or "").strip()
        if not key:
            continue
        variants = spec.get("variants")
        variant_list = [str(item).strip() for item in variants] if isinstance(variants, list) else variants_map.get(key, [])
        fallback = str(spec.get("fallback") or "").strip()
        resolved = _seeded_variant(seed=seed, key=key, variants=variant_list, fallback=fallback)
        lowered = resolved.lower()
        if resolved and lowered not in seen:
            seen.add(lowered)
            out.append(resolved)
    return out


def selector_candidates_for_ids(
    *ids: str,
    project_id: str = "autocinema",
    seed: int | None = None,
    seed_specs: list[dict[str, Any]] | None = None,
    include_expanded_variants: bool = True,
) -> list[dict[str, Any]]:
    canonical_ids = [value for value in ids if str(value).strip()]
    specs = seed_specs if isinstance(seed_specs, list) and seed_specs else [{"key": value, "fallback": value} for value in canonical_ids]
    exact_ids = _seeded_selector_ids(project_id=project_id, seed=seed, specs=specs)
    expanded = _expand_id_variants(project_id, canonical_ids) if include_expanded_variants else []
    ordered_ids = list(exact_ids)
    seen = {str(value).strip().lower() for value in ordered_ids}
    for value in expanded:
        lowered = str(value).strip().lower()
        if lowered and lowered not in seen:
            seen.add(lowered)
            ordered_ids.append(value)
    return _selector_candidates(ids=ordered_ids)


def selector_candidates_for_classes(
    *class_keys: str,
    project_id: str = "autocinema",
    seed: int | None = None,
    seed_specs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    canonical = [value for value in class_keys if str(value).strip()]
    specs = seed_specs if isinstance(seed_specs, list) and seed_specs else [{"key": value, "fallback": value} for value in canonical]
    exact = _seeded_selector_values(seed=seed, specs=specs, variants_map=_load_class_variants(project_id))
    expanded = _expand_variants(_load_class_variants(project_id), canonical)
    ordered = list(exact)
    seen = {str(value).strip().lower() for value in ordered}
    for value in expanded:
        lowered = str(value).strip().lower()
        if lowered and lowered not in seen:
            seen.add(lowered)
            ordered.append(value)
    return _selector_candidates(classes=ordered)


def selector_candidates_for_placeholders(
    *placeholder_keys: str,
    project_id: str = "autocinema",
    seed: int | None = None,
    seed_specs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    canonical = [value for value in placeholder_keys if str(value).strip()]
    specs = seed_specs if isinstance(seed_specs, list) and seed_specs else [{"key": value, "fallback": value} for value in canonical]
    exact = _seeded_selector_values(seed=seed, specs=specs, variants_map=_load_text_variants(project_id))
    expanded = _expand_variants(_load_text_variants(project_id), canonical)
    ordered = list(exact)
    seen = {str(value).strip().lower() for value in ordered}
    for value in expanded:
        lowered = str(value).strip().lower()
        if lowered and lowered not in seen:
            seen.add(lowered)
            ordered.append(value)
    return _selector_candidates(placeholders=ordered)


def selector_candidates_for_texts(*texts: str) -> list[dict[str, Any]]:
    return _selector_candidates(texts=[value for value in texts if str(value).strip()])


def selector_candidates_for_text_variant_keys(
    *text_keys: str,
    project_id: str = "autocinema",
    seed: int | None = None,
    seed_specs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    canonical = [value for value in text_keys if str(value).strip()]
    specs = seed_specs if isinstance(seed_specs, list) and seed_specs else [{"key": value, "fallback": value} for value in canonical]
    exact = _seeded_selector_values(seed=seed, specs=specs, variants_map=_load_text_variants(project_id))
    expanded = _expand_variants(_load_text_variants(project_id), canonical)
    ordered = list(exact)
    seen = {str(value).strip().lower() for value in ordered}
    for value in expanded:
        lowered = str(value).strip().lower()
        if lowered and lowered not in seen:
            seen.add(lowered)
            ordered.append(value)
    return _selector_candidates(texts=ordered)


def _combined_selectors(
    *,
    ids: list[str] | None = None,
    classes: list[str] | None = None,
    placeholders: list[str] | None = None,
    texts: list[str] | None = None,
    text_keys: list[str] | None = None,
    project_id: str = "autocinema",
    seed: int | None = None,
    id_seed_specs: list[dict[str, Any]] | None = None,
    class_seed_specs: list[dict[str, Any]] | None = None,
    placeholder_seed_specs: list[dict[str, Any]] | None = None,
    text_seed_specs: list[dict[str, Any]] | None = None,
    include_expanded_id_variants: bool = True,
) -> list[dict[str, Any]]:
    ordered_ids = [
        selector.get("value")
        for selector in selector_candidates_for_ids(
            *(ids or []),
            project_id=project_id,
            seed=seed,
            seed_specs=id_seed_specs,
            include_expanded_variants=include_expanded_id_variants,
        )
        if isinstance(selector, dict) and str(selector.get("attribute") or "") == "id"
    ]
    ordered_classes = [
        selector.get("value")
        for selector in selector_candidates_for_classes(
            *(classes or []),
            project_id=project_id,
            seed=seed,
            seed_specs=class_seed_specs,
        )
        if isinstance(selector, dict) and str(selector.get("attribute") or "") == "class"
    ]
    ordered_placeholders = [
        selector.get("value")
        for selector in selector_candidates_for_placeholders(
            *(placeholders or []),
            project_id=project_id,
            seed=seed,
            seed_specs=placeholder_seed_specs,
        )
        if isinstance(selector, dict) and str(selector.get("attribute") or "") == "placeholder"
    ]
    variant_texts = [
        selector.get("value")
        for selector in selector_candidates_for_text_variant_keys(
            *(text_keys or []),
            project_id=project_id,
            seed=seed,
            seed_specs=text_seed_specs,
        )
        if isinstance(selector, dict) and str(selector.get("type") or "") == "tagContainsSelector"
    ]
    return _selector_candidates(
        ids=[str(value) for value in ordered_ids if str(value).strip()],
        classes=[str(value) for value in ordered_classes if str(value).strip()],
        placeholders=[str(value) for value in ordered_placeholders if str(value).strip()],
        texts=[str(value) for value in [*(variant_texts or []), *(texts or [])] if str(value).strip()],
    )


def login_username_selectors(project_id: str = "autocinema", seed: int | None = None) -> list[dict[str, Any]]:
    return _combined_selectors(
        ids=["login-username-input"],
        classes=["input-text"],
        placeholders=["login_username_placeholder"],
        project_id=project_id,
        seed=seed,
    )


def login_password_selectors(project_id: str = "autocinema", seed: int | None = None) -> list[dict[str, Any]]:
    return _combined_selectors(
        ids=["login-password-input"],
        classes=["input-text"],
        placeholders=["login_password_placeholder"],
        project_id=project_id,
        seed=seed,
    )


def login_submit_selectors(project_id: str = "autocinema", seed: int | None = None) -> list[dict[str, Any]]:
    return _combined_selectors(
        ids=["login-sign-in-button"],
        classes=["button-primary"],
        text_keys=["sign_in"],
        texts=["Sign in", "Login"],
        project_id=project_id,
        seed=seed,
    )


def register_username_selectors(project_id: str = "autocinema", seed: int | None = None) -> list[dict[str, Any]]:
    return _combined_selectors(ids=["register-username-input"], classes=["input-text"], placeholders=["username_placeholder"], project_id=project_id, seed=seed)


def register_email_selectors(project_id: str = "autocinema", seed: int | None = None) -> list[dict[str, Any]]:
    return _combined_selectors(ids=["register-email-input"], classes=["input-text"], placeholders=["email_placeholder"], project_id=project_id, seed=seed)


def register_password_selectors(project_id: str = "autocinema", seed: int | None = None) -> list[dict[str, Any]]:
    return _combined_selectors(ids=["register-password-input"], classes=["input-text"], placeholders=["password_placeholder"], project_id=project_id, seed=seed)


def register_confirm_password_selectors(project_id: str = "autocinema", seed: int | None = None) -> list[dict[str, Any]]:
    return _combined_selectors(
        ids=["register-confirm-password-input"],
        classes=["input-text"],
        placeholders=["confirm_password_placeholder"],
        project_id=project_id,
        seed=seed,
    )


def register_submit_selectors(project_id: str = "autocinema", seed: int | None = None) -> list[dict[str, Any]]:
    return _combined_selectors(
        ids=["create-account-button"],
        classes=["button-primary"],
        text_keys=["create_account"],
        texts=["Create account", "Register"],
        project_id=project_id,
        seed=seed,
    )


def contact_name_selectors(project_id: str = "autocinema", seed: int | None = None) -> list[dict[str, Any]]:
    return _combined_selectors(ids=["contact-name-input"], classes=["input-text"], placeholders=["contact_name_placeholder"], project_id=project_id, seed=seed)


def contact_email_selectors(project_id: str = "autocinema", seed: int | None = None) -> list[dict[str, Any]]:
    return _combined_selectors(ids=["contact-email-input"], classes=["input-text"], placeholders=["contact_email_placeholder"], project_id=project_id, seed=seed)


def contact_subject_selectors(project_id: str = "autocinema", seed: int | None = None) -> list[dict[str, Any]]:
    return _combined_selectors(ids=["contact-subject-input"], classes=["input-text"], placeholders=["contact_subject_placeholder"], project_id=project_id, seed=seed)


def contact_message_selectors(project_id: str = "autocinema", seed: int | None = None) -> list[dict[str, Any]]:
    return _combined_selectors(ids=["contact-message-textarea"], classes=["input-text"], placeholders=["message_placeholder"], project_id=project_id, seed=seed)


def contact_submit_selectors(project_id: str = "autocinema", seed: int | None = None) -> list[dict[str, Any]]:
    return _combined_selectors(
        ids=["send-message-button"],
        classes=["button-primary"],
        text_keys=["send_message"],
        texts=["Send Message"],
        project_id=project_id,
        seed=seed,
    )


def comment_name_selectors(project_id: str = "autocinema", seed: int | None = None) -> list[dict[str, Any]]:
    return _combined_selectors(ids=["comment-name-input"], classes=["input-text"], placeholders=["name_placeholder"], project_id=project_id, seed=seed)


def comment_message_selectors(project_id: str = "autocinema", seed: int | None = None) -> list[dict[str, Any]]:
    return _combined_selectors(ids=["comment-message-textarea"], classes=["input-text"], placeholders=["message_placeholder"], project_id=project_id, seed=seed)


def comment_submit_selectors(project_id: str = "autocinema", seed: int | None = None) -> list[dict[str, Any]]:
    return _combined_selectors(
        ids=["share-feedback-button"],
        classes=["button-primary"],
        text_keys=["share_feedback"],
        texts=["Share Feedback", "Post", "Comment"],
        project_id=project_id,
        seed=seed,
    )


def watchlist_button_selectors(project_id: str = "autocinema", seed: int | None = None) -> list[dict[str, Any]]:
    return _combined_selectors(
        ids=["watchlist-button"],
        classes=["button-secondary"],
        text_keys=["add_to_watchlist", "remove_from_watchlist", "watchlist"],
        texts=["Add to watchlist", "Remove from watchlist", "Watchlist"],
        project_id=project_id,
        seed=seed,
    )


def share_button_selectors(project_id: str = "autocinema", seed: int | None = None) -> list[dict[str, Any]]:
    return _combined_selectors(ids=["share-button"], classes=["button-secondary"], text_keys=["share"], texts=["Share"], project_id=project_id, seed=seed)


def trailer_button_selectors(project_id: str = "autocinema", seed: int | None = None) -> list[dict[str, Any]]:
    selectors = _combined_selectors(
        ids=["watch-trailer-button"],
        classes=["button-primary"],
        text_keys=["watch_trailer"],
        texts=["Watch trailer"],
        project_id=project_id,
        seed=seed,
        include_expanded_id_variants=False,
    )
    custom = {
        "type": "attributeValueSelector",
        "attribute": "custom",
        "value": 'section button:has-text("Watch trailer")',
        "case_sensitive": False,
    }
    if custom not in selectors:
        selectors.append(custom)
    return selectors


def search_submit_selectors(project_id: str = "autocinema", seed: int | None = None) -> list[dict[str, Any]]:
    return _combined_selectors(ids=["search-submit-button"], classes=["search-button"], texts=["Search"], project_id=project_id, seed=seed)


def search_input_selectors(project_id: str = "autocinema", seed: int | None = None) -> list[dict[str, Any]]:
    # Search input has type="search" but no id — use attribute selector
    return [{"type": "attributeValueSelector", "attribute": "type", "value": "search", "case_sensitive": False}]


def view_detail_selectors(project_id: str = "autocinema", seed: int | None = None) -> list[dict[str, Any]]:
    return _combined_selectors(
        ids=["view-details-button", "featured-movie-view-details-btn"],
        classes=["button-primary"],
        text_keys=["view_details"],
        texts=["View Details"],
        project_id=project_id,
        seed=seed,
        id_seed_specs=[
            {"key": "view-details-button", "fallback": "view-details-button"},
            {"key": "featured-movie-view-details-btn", "fallback": "featured-movie-view-details-btn"},
            {
                "key": "featured-view-details-button",
                "variants": _AUTOCINEMA_LOCAL_ID_VARIANTS["featured-view-details-button"],
                "fallback": "hero-view-details-btn",
            },
        ],
    )


def profile_first_name_selectors(project_id: str = "autocinema", seed: int | None = None) -> list[dict[str, Any]]:
    return _combined_selectors(ids=["profile-first-name-input"], classes=["input-text"], placeholders=["first_name_placeholder"], project_id=project_id, seed=seed)


def profile_last_name_selectors(project_id: str = "autocinema", seed: int | None = None) -> list[dict[str, Any]]:
    return _combined_selectors(ids=["profile-last-name-input"], classes=["input-text"], placeholders=["last_name_placeholder"], project_id=project_id, seed=seed)


def profile_email_selectors(project_id: str = "autocinema", seed: int | None = None) -> list[dict[str, Any]]:
    return _combined_selectors(ids=["profile-email-input"], classes=["input-text"], placeholders=["email_placeholder_profile"], project_id=project_id, seed=seed)


def profile_favorite_genres_selectors(project_id: str = "autocinema", seed: int | None = None) -> list[dict[str, Any]]:
    return _combined_selectors(ids=["profile-favorite-genres-input"], classes=["input-text"], placeholders=["favorite_genres_placeholder"], project_id=project_id, seed=seed)


def profile_location_selectors(project_id: str = "autocinema", seed: int | None = None) -> list[dict[str, Any]]:
    return _combined_selectors(ids=["profile-location-input"], classes=["input-text"], placeholders=["location_placeholder"], project_id=project_id, seed=seed)


def profile_website_selectors(project_id: str = "autocinema", seed: int | None = None) -> list[dict[str, Any]]:
    return _combined_selectors(ids=["profile-website-input"], classes=["input-text"], placeholders=["website_placeholder"], project_id=project_id, seed=seed)


def profile_bio_selectors(project_id: str = "autocinema", seed: int | None = None) -> list[dict[str, Any]]:
    return _combined_selectors(ids=["profile-bio-textarea"], classes=["input-text"], placeholders=["bio_placeholder"], project_id=project_id, seed=seed)


def profile_save_selectors(project_id: str = "autocinema", seed: int | None = None) -> list[dict[str, Any]]:
    return _combined_selectors(ids=["save-profile-button"], classes=["button-primary"], text_keys=["save_profile"], texts=["Save Profile"], project_id=project_id, seed=seed)


def delete_movie_selectors(project_id: str = "autocinema", seed: int | None = None) -> list[dict[str, Any]]:
    return _combined_selectors(
        ids=["delete-movie-button"],
        classes=["button-secondary"],
        text_keys=["delete_movie"],
        texts=["Delete Movie", "Delete movie"],
        project_id=project_id,
        seed=seed,
    )


def save_changes_selectors(project_id: str = "autocinema", seed: int | None = None) -> list[dict[str, Any]]:
    return _combined_selectors(
        ids=["save-changes-button"],
        classes=["button-primary"],
        text_keys=["save_changes", "add_film"],
        texts=["Save changes", "Add Film", "Edit Film"],
        project_id=project_id,
        seed=seed,
    )


def logout_selectors(project_id: str = "autocinema", seed: int | None = None) -> list[dict[str, Any]]:
    selectors = _combined_selectors(
        text_keys=["logout"],
        texts=["Logout", "Log out", "Sign out"],
        project_id=project_id,
        seed=seed,
    )
    header_selector = {
        "type": "attributeValueSelector",
        "attribute": "custom",
        "value": 'header nav button:has-text("Logout")',
        "case_sensitive": False,
    }
    if header_selector not in selectors:
        selectors.insert(0, header_selector)
    return selectors


def watchlist_remove_profile_selectors(project_id: str = "autocinema", seed: int | None = None) -> list[dict[str, Any]]:
    selectors = _combined_selectors(
        classes=["button-secondary"],
        texts=["Remove from List", "Remove from watchlist"],
        text_keys=["remove_from_watchlist"],
        project_id=project_id,
        seed=seed,
    )
    custom = {
        "type": "attributeValueSelector",
        "attribute": "custom",
        "value": 'button:has-text("Remove from List")',
        "case_sensitive": False,
    }
    if custom not in selectors:
        selectors.insert(0, custom)
    return selectors


def profile_tab_selectors(tab_name: str, project_id: str = "autocinema") -> list[dict[str, Any]]:
    normalized = str(tab_name or "").strip().lower()
    texts = {
        "movies": ["Edit Movies", "Movies"],
        "watchlist": ["Watchlist"],
        "add-movies": ["Add Movies", "Add Film", "Add movie"],
    }
    text_keys = {
        "movies": ["edit_movies"],
        "watchlist": ["watchlist"],
        "add-movies": ["add_movies", "add_film"],
    }
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for selector in [
        *selector_candidates_for_text_variant_keys(*text_keys.get(normalized, []), project_id=project_id),
        *selector_candidates_for_texts(*texts.get(normalized, [tab_name, normalized.replace("-", " ")])),
    ]:
        if not isinstance(selector, dict):
            continue
        key = json.dumps(selector, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(selector)
    return out


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
    "profile_bio_selectors",
    "profile_email_selectors",
    "profile_favorite_genres_selectors",
    "profile_first_name_selectors",
    "profile_last_name_selectors",
    "profile_location_selectors",
    "profile_save_selectors",
    "profile_tab_selectors",
    "profile_website_selectors",
    "register_confirm_password_selectors",
    "register_email_selectors",
    "register_password_selectors",
    "register_submit_selectors",
    "register_username_selectors",
    "route_for_use_case",
    "save_changes_selectors",
    "search_input_selectors",
    "search_submit_selectors",
    "selector_candidates_for_classes",
    "selector_candidates_for_ids",
    "selector_candidates_for_placeholders",
    "selector_candidates_for_text_variant_keys",
    "selector_candidates_for_texts",
    "share_button_selectors",
    "trailer_button_selectors",
    "view_detail_selectors",
    "watchlist_button_selectors",
    "watchlist_remove_profile_selectors",
]
