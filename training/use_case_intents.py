from __future__ import annotations

from collections.abc import Iterable

_INTENT_VARIANTS: dict[str, tuple[str, ...]] = {
    "LOGIN": ("LOGIN", "LOGIN_BOOK", "AUTOZONE_LOGIN"),
    "REGISTRATION": ("REGISTRATION", "REGISTRATION_BOOK", "AUTOZONE_REGISTER"),
    "LOGOUT": ("LOGOUT", "LOGOUT_BOOK", "AUTOZONE_LOGOUT"),
    "CONTACT": ("CONTACT", "CONTACT_BOOK"),
    "DETAIL": ("FILM_DETAIL", "BOOK_DETAIL", "VIEW_DETAIL"),
    "SEARCH": ("SEARCH_FILM", "SEARCH_BOOK", "SEARCH_PRODUCT"),
    "FILTER": ("FILTER_FILM", "FILTER_BOOK", "CATEGORY_FILTER"),
    "ADD_COMMENT": ("ADD_COMMENT", "ADD_COMMENT_BOOK", "REVIEW_CREATED"),
    "EDIT_COMMENT": ("EDIT_COMMENT_BOOK", "REVIEW_UPDATED"),
    "DELETE_COMMENT": ("DELETE_COMMENT_BOOK", "REVIEW_DELETED"),
    "ADD_ENTITY": ("ADD_FILM", "ADD_BOOK"),
    "EDIT_ENTITY": ("EDIT_FILM", "EDIT_BOOK"),
    "DELETE_ENTITY": ("DELETE_FILM", "DELETE_BOOK"),
    "EDIT_USER": ("EDIT_USER", "EDIT_USER_BOOK"),
    "ADD_TO_LIST": ("ADD_TO_WATCHLIST", "ADD_TO_READING_LIST", "ADD_TO_WISHLIST"),
    "REMOVE_FROM_LIST": ("REMOVE_FROM_WATCHLIST", "REMOVE_FROM_READING_LIST"),
    "SHARE": ("SHARE_MOVIE", "SHARE_BOOK", "SHARE_PRODUCT", "SHARE_COMPLETED"),
    "MEDIA_PREVIEW": ("WATCH_TRAILER", "OPEN_PREVIEW"),
    "VIEW_CART": ("VIEW_CART_BOOK", "VIEW_CART"),
    "ADD_TO_CART": ("ADD_TO_CART_BOOK", "ADD_TO_CART"),
    "REMOVE_FROM_CART": ("REMOVE_FROM_CART_BOOK",),
    "PURCHASE": ("PURCHASE_BOOK", "PROCEED_TO_CHECKOUT", "CHECKOUT_STARTED", "ORDER_COMPLETED"),
}

_VARIANT_TO_INTENT: dict[str, str] = {}
for intent_name, variants in _INTENT_VARIANTS.items():
    for variant in variants:
        normalized = str(variant).strip().upper()
        if normalized:
            _VARIANT_TO_INTENT[normalized] = intent_name


def canonical_intent(use_case: str) -> str:
    normalized = str(use_case or "").strip().upper()
    if not normalized:
        return "UNKNOWN"
    return _VARIANT_TO_INTENT.get(normalized, normalized)


def variants_for_intent(intent: str) -> tuple[str, ...]:
    normalized = str(intent or "").strip().upper()
    if not normalized:
        return ()
    return _INTENT_VARIANTS.get(normalized, ())


def resolve_by_intent(use_case: str, known_keys: Iterable[str]) -> str:
    normalized = str(use_case or "").strip().upper()
    keys = {str(key).strip().upper() for key in known_keys if str(key).strip()}
    if normalized in keys:
        return normalized
    intent = canonical_intent(normalized)
    for variant in variants_for_intent(intent):
        if variant in keys:
            return variant
    return normalized


__all__ = [
    "canonical_intent",
    "resolve_by_intent",
    "variants_for_intent",
]
