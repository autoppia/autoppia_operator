from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .projects import resolve_project_id


@dataclass(frozen=True)
class ConstraintHint:
    field: str
    operator: str
    value: Any
    source: str


@dataclass(frozen=True)
class DeterministicTaskObjective:
    web_project_id: str
    use_case: str
    seed: int
    task_url: str
    prompt: str
    route_target: str
    field_values: dict[str, str]
    entity_filters: dict[str, Any]
    auth_required: bool
    success_expectations: dict[str, Any]
    constraints: tuple[ConstraintHint, ...]
    relevant_data: dict[str, Any]
    raw_task: dict[str, Any]


_FIELD_ALIASES: dict[str, str] = {
    "author": "name",
    "bio": "bio",
    "cast": "cast",
    "comment_content": "content",
    "commenter_name": "name",
    "comment_name": "name",
    "confirm_password": "confirm_password",
    "content": "content",
    "director": "director",
    "duration": "duration",
    "email": "email",
    "favorite_genres": "favorite_genres",
    "first_name": "first_name",
    "genre_name": "genres",
    "genres": "genres",
    "location": "location",
    "message": "message",
    "movie_name": "movie_name",
    "movie_duration": "duration",
    "movie_rating": "rating",
    "movie_year": "year",
    "name": "name",
    "password": "password",
    "query": "query",
    "rating": "rating",
    "subject": "subject",
    "signup_email": "email",
    "signup_password": "password",
    "signup_username": "username",
    "synopsis": "synopsis",
    "title": "title",
    "trailer_url": "trailer_url",
    "url": "website",
    "user": "username",
    "username": "username",
    "website": "website",
    "year": "year",
}

_ROUTE_BY_USE_CASE: dict[str, str] = {
    "ADD_COMMENT": "/",
    "ADD_FILM": "/profile",
    "ADD_TO_WATCHLIST": "/",
    "CONTACT": "/contact",
    "DELETE_FILM": "/profile",
    "EDIT_FILM": "/profile",
    "EDIT_USER": "/profile",
    "FILM_DETAIL": "/",
    "FILTER_FILM": "/search",
    "LOGIN": "/login",
    "LOGOUT": "/profile",
    "REGISTRATION": "/register",
    "REMOVE_FROM_WATCHLIST": "/",
    "SEARCH_FILM": "/search",
    "SHARE_MOVIE": "/",
    "WATCH_TRAILER": "/",
}

# autobooks (IWA p02) — primary route hints for `route_target` / success expectations
_ROUTE_BY_USE_CASE_AUTOBOOKS: dict[str, str] = {
    "ADD_BOOK": "/profile",
    "ADD_COMMENT_BOOK": "/books",
    "ADD_TO_CART_BOOK": "/search",
    "ADD_TO_READING_LIST": "/search",
    "BOOK_DETAIL": "/books",
    "CONTACT_BOOK": "/contact",
    "DELETE_BOOK": "/profile",
    "EDIT_BOOK": "/profile",
    "EDIT_USER_BOOK": "/profile",
    "FILTER_BOOK": "/search",
    "LOGIN_BOOK": "/login",
    "LOGOUT_BOOK": "/profile",
    "OPEN_PREVIEW": "/search",
    "PURCHASE_BOOK": "/books",
    "REGISTRATION_BOOK": "/register",
    "REMOVE_FROM_CART_BOOK": "/cart",
    "REMOVE_FROM_READING_LIST": "/search",
    "SEARCH_BOOK": "/search",
    "SHARE_BOOK": "/books",
    "VIEW_CART_BOOK": "/cart",
}

# autozone (IWA p03) — primary route hints for `route_target`
_ROUTE_BY_USE_CASE_AUTOZONE: dict[str, str] = {
    "ADD_TO_CART": "/",
    "ADD_TO_WISHLIST": "/",
    "CAROUSEL_SCROLL": "/",
    "CATEGORY_FILTER": "/search",
    "CHECKOUT_STARTED": "/checkout",
    "DETAILS_TOGGLE": "/",
    "ORDER_COMPLETED": "/checkout",
    "PROCEED_TO_CHECKOUT": "/checkout",
    "QUANTITY_CHANGED": "/cart",
    "SEARCH_PRODUCT": "/search",
    "SHARE_PRODUCT": "/",
    "VIEW_CART": "/cart",
    "VIEW_DETAIL": "/",
    "VIEW_WISHLIST": "/wishlist",
}

# autodining (IWA p04) — primary route hints for `route_target` / success expectations
_ROUTE_BY_USE_CASE_AUTODINING: dict[str, str] = {
    "ABOUT_FEATURE_CLICK": "/about",
    "ABOUT_PAGE_VIEW": "/about",
    "BOOK_RESTAURANT": "/",
    "COLLAPSE_MENU": "/",
    "CONTACT_CARD_CLICK": "/contact",
    "CONTACT_FORM_SUBMIT": "/contact",
    "CONTACT_PAGE_VIEW": "/contact",
    "COUNTRY_SELECTED": "/",
    "DATE_DROPDOWN_OPENED": "/",
    "HELP_CATEGORY_SELECTED": "/help",
    "HELP_FAQ_TOGGLED": "/help",
    "HELP_PAGE_VIEW": "/help",
    "OCCASION_SELECTED": "/",
    "PEOPLE_DROPDOWN_OPENED": "/",
    "RESERVATION_COMPLETE": "/",
    "SCROLL_VIEW": "/",
    "SEARCH_RESTAURANT": "/search",
    "TIME_DROPDOWN_OPENED": "/",
    "VIEW_FULL_MENU": "/",
    "VIEW_RESTAURANT": "/",
}

# autocrm (IWA p05)
_ROUTE_BY_USE_CASE_AUTOCRM: dict[str, str] = {
    "ADD_CLIENT": "/clients",
    "ADD_NEW_MATTER": "/matters",
    "ARCHIVE_MATTER": "/matters",
    "BILLING_SEARCH": "/billing",
    "CHANGE_USER_NAME": "/settings",
    "DELETE_CLIENT": "/clients",
    "DELETE_MATTER": "/matters",
    "DOCUMENT_DELETED": "/matters",
    "DOCUMENT_RENAMED": "/matters",
    "FILTER_CLIENTS": "/clients",
    "FILTER_MATTER_STATUS": "/matters",
    "HELP_VIEWED": "/help",
    "LOG_DELETE": "/matters",
    "LOG_EDITED": "/matters",
    "NEW_CALENDAR_EVENT_ADDED": "/calendar",
    "NEW_LOG_ADDED": "/matters",
    "SEARCH_CLIENT": "/search",
    "SEARCH_MATTER": "/search",
    "SORT_MATTER_BY_CREATED_AT": "/matters",
    "UPDATE_MATTER": "/matters",
    "VIEW_CLIENT_DETAILS": "/clients",
    "VIEW_MATTER_DETAILS": "/matters",
    "VIEW_PENDING_EVENTS": "/calendar",
}

# automail (IWA p06)
_ROUTE_BY_USE_CASE_AUTOMAIL: dict[str, str] = {
    "ADD_LABEL": "/inbox",
    "ARCHIVE_EMAIL": "/inbox",
    "CLEAR_SELECTION": "/inbox",
    "CREATE_LABEL": "/inbox",
    "DELETE_EMAIL": "/inbox",
    "EDIT_DRAFT_EMAIL": "/drafts",
    "EMAILS_NEXT_PAGE": "/inbox",
    "EMAILS_PREV_PAGE": "/inbox",
    "EMAIL_SAVE_AS_DRAFT": "/compose",
    "FORWARD_EMAIL": "/inbox",
    "MARK_AS_SPAM": "/inbox",
    "MARK_AS_UNREAD": "/inbox",
    "MARK_EMAIL_AS_IMPORTANT": "/inbox",
    "REPLY_EMAIL": "/inbox",
    "SEARCH_EMAIL": "/search",
    "SEND_EMAIL": "/compose",
    "STAR_AN_EMAIL": "/inbox",
    "TEMPLATE_BODY_EDITED": "/templates",
    "TEMPLATE_CANCELED": "/templates",
    "TEMPLATE_SAVED_DRAFT": "/templates",
    "TEMPLATE_SELECTED": "/templates",
    "TEMPLATE_SENT": "/templates",
    "THEME_CHANGED": "/settings",
    "VIEW_EMAIL": "/inbox",
    "VIEW_TEMPLATES": "/templates",
}

# autolodge (IWA p08)
_ROUTE_BY_USE_CASE_AUTOLODGE: dict[str, str] = {
    "ADD_TO_WISHLIST": "/stay",
    "APPLY_FILTERS": "/",
    "BACK_TO_ALL_HOTELS": "/",
    "BOOK_FROM_WISHLIST": "/wishlist",
    "CONFIRM_AND_PAY": "/confirm",
    "EDIT_CHECK_IN_OUT_DATES": "/stay",
    "EDIT_NUMBER_OF_GUESTS": "/stay",
    "FAQ_OPENED": "/help",
    "HELP_VIEWED": "/help",
    "MESSAGE_HOST": "/stay",
    "PAYMENT_METHOD_SELECTED": "/confirm",
    "POPULAR_HOTELS_VIEWED": "/",
    "REMOVE_FROM_WISHLIST": "/wishlist",
    "RESERVE_HOTEL": "/stay",
    "SEARCH_HOTEL": "/search",
    "SHARE_HOTEL": "/",
    "SUBMIT_REVIEW": "/stay",
    "VIEW_HOTEL": "/stay",
    "WISHLIST_OPENED": "/wishlist",
}

# autodelivery (IWA p07)
_ROUTE_BY_USE_CASE_AUTODELIVERY: dict[str, str] = {
    "ADDRESS_ADDED": "/checkout",
    "ADD_TO_CART_MENU_ITEM": "/restaurant",
    "ADD_TO_CART_MODAL_OPEN": "/restaurant",
    "BACK_TO_ALL_RESTAURANTS": "/",
    "DELETE_REVIEW": "/restaurant",
    "DELIVERY_PRIORITY_SELECTED": "/checkout",
    "DROPOFF_PREFERENCE": "/checkout",
    "EDIT_CART_ITEM": "/cart",
    "EMPTY_CART": "/cart",
    "ITEM_INCREMENTED": "/cart",
    "OPEN_CHECKOUT_PAGE": "/checkout",
    "PLACE_ORDER": "/checkout",
    "QUICK_ORDER_STARTED": "/cart",
    "RESTAURANT_FILTER": "/search",
    "RESTAURANT_NEXT_PAGE": "/",
    "RESTAURANT_PREV_PAGE": "/",
    "REVIEW_SUBMITTED": "/restaurant",
    "SEARCH_DELIVERY_RESTAURANT": "/search",
    "VIEW_ALL_RESTAURANTS": "/",
    "VIEW_DELIVERY_RESTAURANT": "/restaurant",
}

# autowork (IWA p10)
_ROUTE_BY_USE_CASE_AUTOWORK: dict[str, str] = {
    "ADD_SKILL": "/jobs",
    "BOOK_A_CONSULTATION": "/experts",
    "BROWSE_FAVORITE_EXPERT": "/favorites",
    "CANCEL_HIRE": "/hires",
    "CHOOSE_BUDGET_TYPE": "/jobs",
    "CHOOSE_PROJECT_SIZE": "/jobs",
    "CHOOSE_PROJECT_TIMELINE": "/jobs",
    "CLOSE_POST_A_JOB_WINDOW": "/jobs",
    "CONTACT_EXPERT_MESSAGE_SENT": "/experts",
    "CONTACT_EXPERT_OPENED": "/experts",
    "EDIT_ABOUT": "/profile",
    "EDIT_PROFILE_EMAIL": "/profile",
    "EDIT_PROFILE_LOCATION": "/profile",
    "EDIT_PROFILE_NAME": "/profile",
    "EDIT_PROFILE_TITLE": "/profile",
    "FAVORITE_EXPERT_REMOVED": "/favorites",
    "FAVORITE_EXPERT_SELECTED": "/favorites",
    "HIRE_BTN_CLICKED": "/experts",
    "HIRE_CONSULTANT": "/hires",
    "HIRE_LATER_ADDED": "/hire-later",
    "HIRE_LATER_REMOVED": "/hire-later",
    "HIRE_LATER_START": "/hire-later",
    "NAVBAR_EXPERTS_CLICK": "/experts",
    "NAVBAR_FAVORITES_CLICK": "/favorites",
    "NAVBAR_HIRES_CLICK": "/hires",
    "NAVBAR_HIRE_LATER_CLICK": "/hire-later",
    "NAVBAR_JOBS_CLICK": "/jobs",
    "NAVBAR_PROFILE_CLICK": "/profile",
    "POST_A_JOB": "/jobs",
    "QUICK_HIRE": "/experts",
    "SEARCH_SKILL": "/jobs",
    "SELECT_HIRING_TEAM": "/hires",
    "SET_RATE_RANGE": "/jobs",
    "SUBMIT_JOB": "/jobs",
    "WRITE_JOB_DESCRIPTION": "/jobs",
    "WRITE_JOB_TITLE": "/jobs",
}

# autoconnect (IWA p09)
_ROUTE_BY_USE_CASE_AUTOCONNECT: dict[str, str] = {
    "ADD_EXPERIENCE": "/profile",
    "APPLY_FOR_JOB": "/jobs",
    "BACK_TO_ALL_JOBS": "/jobs",
    "CANCEL_APPLICATION": "/jobs",
    "COMMENT_ON_POST": "/",
    "CONNECT_WITH_USER": "/",
    "EDIT_EXPERIENCE": "/profile",
    "EDIT_PROFILE": "/profile",
    "FILTER_JOBS": "/jobs",
    "FOLLOW_PAGE": "/",
    "HIDE_POST": "/",
    "HOME_NAVBAR": "/",
    "JOBS_NAVBAR": "/jobs",
    "LIKE_POST": "/",
    "POST_STATUS": "/",
    "REMOVE_POST": "/",
    "SAVE_POST": "/",
    "SEARCH_JOBS": "/jobs",
    "SEARCH_USERS": "/search",
    "UNFOLLOW_PAGE": "/",
    "UNHIDE_POST": "/",
    "VIEW_APPLIED_JOBS": "/jobs",
    "VIEW_HIDDEN_POSTS": "/",
    "VIEW_JOB": "/jobs",
    "VIEW_SAVED_POSTS": "/",
    "VIEW_USER_PROFILE": "/profile",
}

# autocalendar (IWA p11)
_ROUTE_BY_USE_CASE_AUTOCALENDAR: dict[str, str] = {
    "ADD_EVENT": "/",
    "ADD_NEW_CALENDAR": "/",
    "CANCEL_ADD_EVENT": "/",
    "CELL_CLICKED": "/",
    "CREATE_CALENDAR": "/",
    "DELETE_ADDED_EVENT": "/",
    "EVENT_ADD_ATTENDEE": "/",
    "EVENT_ADD_REMINDER": "/",
    "EVENT_REMOVE_ATTENDEE": "/",
    "EVENT_REMOVE_REMINDER": "/",
    "EVENT_WIZARD_OPEN": "/",
    "SEARCH_SUBMIT": "/search",
    "SELECT_CALENDAR": "/",
    "SELECT_DAY": "/",
    "SELECT_FIVE_DAYS": "/",
    "SELECT_MONTH": "/",
    "SELECT_TODAY": "/",
    "SELECT_WEEK": "/",
    "UNSELECT_CALENDAR": "/",
}

# autolist (IWA p12)
_ROUTE_BY_USE_CASE_AUTOLIST: dict[str, str] = {
    "AUTOLIST_ADD_TASK_CLICKED": "/tasks",
    "AUTOLIST_ADD_TEAM_CLICKED": "/teams",
    "AUTOLIST_CANCEL_TASK_CREATION": "/tasks",
    "AUTOLIST_COMPLETE_TASK": "/tasks",
    "AUTOLIST_DELETE_TASK": "/tasks",
    "AUTOLIST_EDIT_TASK_MODAL_OPENED": "/tasks",
    "AUTOLIST_SELECT_DATE_FOR_TASK": "/tasks",
    "AUTOLIST_SELECT_TASK_PRIORITY": "/tasks",
    "AUTOLIST_TASK_ADDED": "/tasks",
    "AUTOLIST_TEAM_CREATED": "/teams",
    "AUTOLIST_TEAM_MEMBERS_ADDED": "/teams",
    "AUTOLIST_TEAM_ROLE_ASSIGNED": "/teams",
}

# autohealth (IWA p14)
_ROUTE_BY_USE_CASE_AUTOHEALTH: dict[str, str] = {
    "OPEN_APPOINTMENT_FORM": "/appointments",
    "APPOINTMENT_BOOKED_SUCCESSFULLY": "/appointments",
    "REQUEST_QUICK_APPOINTMENT": "/",
    "SEARCH_APPOINTMENT": "/appointments",
    "SEARCH_DOCTORS": "/doctors",
    "SEARCH_PRESCRIPTION": "/prescriptions",
    "REFILL_PRESCRIPTION": "/prescriptions",
    "VIEW_PRESCRIPTION": "/prescriptions",
    "SEARCH_MEDICAL_ANALYSIS": "/medical-records",
    "VIEW_MEDICAL_ANALYSIS": "/medical-records",
    "VIEW_DOCTOR_PROFILE": "/doctors",
    "VIEW_DOCTOR_EDUCATION": "/doctors",
    "VIEW_DOCTOR_AVAILABILITY": "/doctors",
    "FILTER_DOCTOR_REVIEWS": "/doctors",
    "OPEN_CONTACT_DOCTOR_FORM": "/doctors",
    "CONTACT_DOCTOR": "/doctors",
}


def _route_target(*, web_project_id: str, use_case: str) -> str:
    pid = str(web_project_id or "").strip().lower() or "autocinema"
    if pid == "autobooks":
        return _ROUTE_BY_USE_CASE_AUTOBOOKS.get(use_case, "/")
    if pid == "autocalendar":
        return _ROUTE_BY_USE_CASE_AUTOCALENDAR.get(use_case, "/")
    if pid == "autoconnect":
        return _ROUTE_BY_USE_CASE_AUTOCONNECT.get(use_case, "/")
    if pid == "autodining":
        return _ROUTE_BY_USE_CASE_AUTODINING.get(use_case, "/")
    if pid == "autodelivery":
        return _ROUTE_BY_USE_CASE_AUTODELIVERY.get(use_case, "/")
    if pid == "autocrm":
        return _ROUTE_BY_USE_CASE_AUTOCRM.get(use_case, "/")
    if pid == "automail":
        return _ROUTE_BY_USE_CASE_AUTOMAIL.get(use_case, "/")
    if pid == "autolist":
        return _ROUTE_BY_USE_CASE_AUTOLIST.get(use_case, "/")
    if pid == "autohealth":
        return _ROUTE_BY_USE_CASE_AUTOHEALTH.get(use_case, "/")
    if pid == "autolodge":
        return _ROUTE_BY_USE_CASE_AUTOLODGE.get(use_case, "/")
    if pid == "autowork":
        return _ROUTE_BY_USE_CASE_AUTOWORK.get(use_case, "/")
    if pid == "autozone":
        return _ROUTE_BY_USE_CASE_AUTOZONE.get(use_case, "/")
    return _ROUTE_BY_USE_CASE.get(use_case, "/")


def route_for_web_project_use_case(*, web_project_id: str, use_case: str) -> str:
    """URL path target for a demo web project and IWA use case (see `_route_target`)."""
    return _route_target(
        web_project_id=web_project_id,
        use_case=str(use_case or "").strip().upper(),
    )


_DEFAULT_VALUES: dict[str, str] = {
    "bio": "films lover",
    "cast": "John,Roy",
    "confirm_password": "Passw0rd!",
    "content": "Great movie",
    "director": "Anthony Russo",
    "duration": "120",
    "email": "agent@example.com",
    "favorite_genres": "Drama, Action",
    "first_name": "Benjamin",
    "genres": "Action",
    "last_name": "Viewer",
    "location": "Madrid",
    "message": "Please help with my movie request.",
    "name": "Agent",
    "password": "Passw0rd!",
    "query": "The Matrix",
    "rating": "5.0",
    "subject": "General inquiry",
    "synopsis": "A thrilling movie generated by the deterministic harvester.",
    "title": "New Film",
    "trailer_url": "https://example.org/trailer",
    "username": "user1",
    "website": "https://example.org",
    "year": "2021",
    # autohealth defaults
    "appointment_request": "true",
    "consultation_fee": "180",
    "date": "2025-10-15",
    "doctor_name": "Dr. Olivia Carter",
    "doctor_filter": "Dr.",
    "emergency_contact": "Jordan Cole",
    "filter_rating": "5",
    "insurance_provider": "Aetna",
    "language": "English",
    "medicine_name": "Vitamin D",
    "notes": "Please review my latest symptoms.",
    "patient_email": "patient@example.com",
    "patient_name": "Maria Anderson",
    "patient_phone": "+1-202-555-0153",
    "preferred_contact_method": "phone",
    "reason_for_visit": "Chronic migraine management",
    "record_date": "2025-10-12",
    "record_title": "Follow-up report",
    "record_type": "Lab results",
    "sort_order": "Newest First",
    "speciality": "Cardiology",
    "start_date": "2024-07-05",
    "status": "Active",
    "time": "01:15 PM",
    "urgency": "Low - General inquiry",
}

_MOVIE_FILTER_FIELDS = {
    "movie_name",
    "title",
    "director",
    "genres",
    "year",
    "duration",
    "rating",
}

_AUTH_EVENT_PRIORITY_FIELDS: dict[str, tuple[str, ...]] = {
    "LOGIN": ("username", "email"),
    "LOGOUT": ("username", "email"),
    "REGISTRATION": ("username", "email"),
}

_KNOWN_CREDENTIAL_PLACEHOLDERS = {
    "<username>",
    "<password>",
    "<signup_username>",
    "<signup_email>",
    "<signup_password>",
    "<web_agent_id>",
}

_AUTOHEALTH_FIELD_ALIASES: dict[str, str] = {
    "doctor": "doctor_name",
    "doctor_name": "doctor_name",
    "specialty": "speciality",
    "speciality": "speciality",
    "patient_name": "patient_name",
    "patient_email": "patient_email",
    "patient_phone": "patient_phone",
    "reason_for_visit": "reason_for_visit",
    "insurance_provider": "insurance_provider",
    "emergency_contact": "emergency_contact",
    "notes": "notes",
    "date": "date",
    "time": "time",
    "medicine_name": "medicine_name",
    "record_title": "record_title",
    "record_type": "record_type",
    "record_date": "record_date",
    "language": "language",
    "consultation_fee": "consultation_fee",
    "filter_rating": "filter_rating",
    "urgency": "urgency",
    "preferred_contact_method": "preferred_contact_method",
    "appointment_request": "appointment_request",
}

_PROJECT_FIELD_ALLOWLIST: dict[str, set[str]] = {
    "autohealth": set(_AUTOHEALTH_FIELD_ALIASES.values()),
}


def _norm_ws(text: str) -> str:
    return " ".join(str(text or "").split())


def _prompt_constraints(task: str) -> dict[str, str]:
    text = str(task or "")
    out: dict[str, str] = {}
    pattern = re.compile(
        r"\b([a-z][a-z0-9 _-]{1,40})\b\s*(?:equals|=|is|:)\s*(?:'([^']+)'|\"([^\"]+)\"|(<[^>]+>)|([0-9]+(?:\.[0-9]+)?)|([^\s,;]+))",
        flags=re.I,
    )
    for match in pattern.finditer(text):
        raw_key = _norm_ws(match.group(1) or "").lower()
        key = re.sub(r"^(?:the|a|an)\s+", "", raw_key).strip().replace(" ", "_")
        value = next((group for group in match.groups()[1:] if group), "")
        value = _norm_ws(value).strip(" \t\r\n'\"`.,;:!?")
        if not key or not value:
            continue
        out[key[:40]] = value[:120]
        if len(out) >= 16:
            break
    return out


def _normalize_field(raw: str) -> str:
    key = str(raw or "").strip().lower().replace(" ", "_").replace(".", "_")
    while "__" in key:
        key = key.replace("__", "_")
    if key in _AUTOHEALTH_FIELD_ALIASES:
        return _AUTOHEALTH_FIELD_ALIASES[key]
    return _FIELD_ALIASES.get(key, key)


def extract_seed_from_task_url(task_url: str) -> int:
    try:
        parsed = urlparse(str(task_url))
        query = parse_qs(parsed.query or "")
        values = query.get("seed") or []
        if values:
            return max(1, min(int(values[0]), 999))
    except Exception:
        return 1
    return 1


def _extract_seed(task_url: str) -> int:
    return extract_seed_from_task_url(task_url)


def _load_raw_task_rows(cache_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    rows: Any
    if isinstance(payload, dict) and isinstance(payload.get("tasks"), list):
        rows = payload["tasks"]
    elif isinstance(payload, dict):
        nested_rows: list[dict[str, Any]] = []
        for value in payload.values():
            if isinstance(value, dict) and isinstance(value.get("tasks"), list):
                nested_rows.extend(row for row in value["tasks"] if isinstance(row, dict))
        rows = nested_rows if nested_rows else payload
    else:
        rows = payload
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def load_task_rows(*, cache_path: Path, use_case: str, web_project_id: str | None = None) -> list[dict[str, Any]]:
    normalized = str(use_case or "").strip().upper()
    normalized_project = str(web_project_id or "").strip()
    matched: list[dict[str, Any]] = []
    for row in _load_raw_task_rows(cache_path):
        row_project = str(row.get("web_project_id") or "").strip()
        if normalized_project and row_project and row_project != normalized_project:
            continue
        use_case_payload = row.get("use_case")
        name = str(use_case_payload.get("name") or "") if isinstance(use_case_payload, dict) else ""
        if name.strip().upper() == normalized:
            matched.append(row)
    return matched


def load_task_row(*, cache_path: Path, use_case: str, seed: int | None = None, web_project_id: str | None = None) -> dict[str, Any]:
    matched = load_task_rows(cache_path=cache_path, use_case=use_case, web_project_id=web_project_id)
    if not matched:
        raise ValueError(f"No task found for use_case={use_case} in {cache_path}")
    if seed is None:
        return matched[0]
    target_seed = int(seed)
    for row in matched:
        if extract_seed_from_task_url(str(row.get("url") or "")) == target_seed:
            return row
    return matched[0]


def task_seeds_for_use_case(*, cache_path: Path, use_case: str, web_project_id: str | None = None) -> list[int]:
    seeds: list[int] = []
    seen: set[int] = set()
    for row in load_task_rows(cache_path=cache_path, use_case=use_case, web_project_id=web_project_id):
        seed = extract_seed_from_task_url(str(row.get("url") or ""))
        if seed in seen:
            continue
        seen.add(seed)
        seeds.append(seed)
    return seeds


def _flatten_criteria(prefix: str, payload: Any) -> list[tuple[str, Any]]:
    if isinstance(payload, dict) and "operator" in payload and "value" in payload:
        return [(prefix, payload)]
    if isinstance(payload, dict):
        flattened: list[tuple[str, Any]] = []
        for key, value in payload.items():
            next_prefix = f"{prefix}_{key}" if prefix else str(key)
            flattened.extend(_flatten_criteria(next_prefix, value))
        return flattened
    return [(prefix, payload)]


def _constraint_hints_from_task(task_row: dict[str, Any]) -> list[ConstraintHint]:
    hints: list[ConstraintHint] = []
    use_case_payload = task_row.get("use_case")
    if isinstance(use_case_payload, dict):
        for constraint in use_case_payload.get("constraints") or []:
            if not isinstance(constraint, dict):
                continue
            hints.append(
                ConstraintHint(
                    field=_normalize_field(str(constraint.get("field") or "")),
                    operator=str(constraint.get("operator") or "equals").strip().lower(),
                    value=constraint.get("value"),
                    source="use_case.constraints",
                )
            )
    for test in task_row.get("tests") or []:
        if not isinstance(test, dict) or str(test.get("type") or "") != "CheckEventTest":
            continue
        criteria = test.get("event_criteria")
        if not isinstance(criteria, dict):
            continue
        for field, value in _flatten_criteria("", criteria):
            normalized_field = _normalize_field(field)
            if isinstance(value, dict) and "operator" in value and "value" in value:
                hints.append(
                    ConstraintHint(
                        field=normalized_field,
                        operator=str(value.get("operator") or "equals").strip().lower(),
                        value=value.get("value"),
                        source="tests.event_criteria",
                    )
                )
            else:
                hints.append(
                    ConstraintHint(
                        field=normalized_field,
                        operator="equals",
                        value=value,
                        source="tests.event_criteria",
                    )
                )
    for field, value in _prompt_constraints(str(task_row.get("prompt") or "")).items():
        hints.append(
            ConstraintHint(
                field=_normalize_field(field),
                operator="equals",
                value=value,
                source="prompt",
            )
        )
    return hints


def _auth_event_values_from_tests(task_row: dict[str, Any], *, use_case: str) -> dict[str, str]:
    allowed_fields = _AUTH_EVENT_PRIORITY_FIELDS.get(use_case, ())
    if not allowed_fields:
        return {}
    out: dict[str, str] = {}
    tests = task_row.get("tests")
    if isinstance(tests, list):
        for test in tests:
            if not isinstance(test, dict):
                continue
            if str(test.get("type") or "").strip() != "CheckEventTest":
                continue
            if str(test.get("event_name") or "").strip().upper() != use_case:
                continue
            criteria = test.get("event_criteria")
            if not isinstance(criteria, dict):
                continue
            for field in allowed_fields:
                if field in out:
                    continue
                if field not in criteria:
                    continue
                value = criteria.get(field)
                if isinstance(value, dict) and "value" in value:
                    value = value.get("value")
                scalar = _coerce_scalar(value)
                if scalar:
                    out[field] = scalar
    return out


def _coerce_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return str(value[0]).strip() if value else ""
    return str(value).strip()


def _coerce_numeric(value: Any) -> float | None:
    raw = _coerce_scalar(value)
    if not raw:
        return None
    try:
        return float(raw)
    except Exception:
        match = re.search(r"-?\d+(?:\.\d+)?", raw)
        if not match:
            return None
        try:
            return float(match.group(0))
        except Exception:
            return None


def _is_placeholder_value(value: str) -> bool:
    return "<" in str(value or "") and ">" in str(value or "")


def _seed_to_web_agent_id(seed: int) -> str:
    seed_i = int(seed)
    if 1 <= seed_i <= 255:
        return str(seed_i)
    normalized = ((seed_i - 1) % 255) + 1
    return str(normalized)


def _replace_credential_placeholders(value: str, *, web_agent_id: str) -> str:
    rendered = str(value or "")
    rendered = rendered.replace("<username>", f"user{web_agent_id}")
    rendered = rendered.replace("<password>", _DEFAULT_VALUES["password"])
    rendered = rendered.replace("<signup_username>", f"newuser{web_agent_id}")
    rendered = rendered.replace("<signup_email>", f"newuser{web_agent_id}@gmail.com")
    rendered = rendered.replace("<signup_password>", _DEFAULT_VALUES["password"])
    rendered = rendered.replace("<web_agent_id>", web_agent_id)
    return rendered


def _looks_like_credential_placeholder(value: str) -> bool:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return False
    if normalized in _KNOWN_CREDENTIAL_PLACEHOLDERS:
        return True
    return _is_placeholder_value(normalized)


def _resolve_auth_field_values(
    *,
    use_case: str,
    seed: int,
    field_values: dict[str, str],
    relevant_data: dict[str, Any],
) -> None:
    web_agent_id = _seed_to_web_agent_id(seed)
    for key in ("username", "password", "email", "confirm_password"):
        raw = _coerce_scalar(field_values.get(key))
        if not raw:
            continue
        field_values[key] = _replace_credential_placeholders(raw, web_agent_id=web_agent_id)

    login_user = relevant_data.get("user_for_login")
    if isinstance(login_user, dict):
        for key in ("username", "password"):
            candidate = _replace_credential_placeholders(_coerce_scalar(login_user.get(key)), web_agent_id=web_agent_id)
            if not candidate:
                continue
            if key not in field_values or _looks_like_credential_placeholder(field_values.get(key, "")):
                field_values[key] = candidate

    if use_case == "REGISTRATION":
        registration_defaults = {
            "username": f"newuser{web_agent_id}",
            "email": f"newuser{web_agent_id}@gmail.com",
            "password": _DEFAULT_VALUES["password"],
        }
        for key, fallback in registration_defaults.items():
            current = _coerce_scalar(field_values.get(key))
            if not current or _looks_like_credential_placeholder(current):
                field_values[key] = fallback
        confirm = _coerce_scalar(field_values.get("confirm_password"))
        if not confirm or _looks_like_credential_placeholder(confirm):
            field_values["confirm_password"] = str(field_values.get("password") or _DEFAULT_VALUES["confirm_password"])
        return

    for key in ("username", "password"):
        current = _coerce_scalar(field_values.get(key))
        if not current or _looks_like_credential_placeholder(current):
            field_values[key] = _DEFAULT_VALUES[key]


def _pick_value(field: str, operator: str, value: Any, seed: int) -> str:
    raw = _coerce_scalar(value)
    if operator in {"equals", "contains", "in_list"} and raw:
        return raw
    if field in {"rating", "year", "duration"} and raw:
        try:
            numeric = float(raw)
        except Exception:
            return raw
        if operator in {"greater_than", "greater_equal"}:
            return str(int(numeric) if numeric.is_integer() else numeric)
        if operator in {"less_than", "less_equal"}:
            adjusted = max(1.0, numeric - 1.0) if operator == "less_than" else numeric
            return str(int(adjusted) if adjusted.is_integer() else adjusted)
        return raw
    candidate = _DEFAULT_VALUES.get(field, f"Agent {seed}")
    if operator in {"not_equals", "not_contains", "not_in_list"} and raw:
        lowered = raw.lower()
        if lowered not in candidate.lower():
            return candidate
        return f"{candidate}-{seed}"
    return raw or candidate


def _maybe_store_field_value(
    field_values: dict[str, str],
    hint: ConstraintHint,
    seed: int,
    *,
    web_project_id: str,
) -> None:
    if not hint.field:
        return
    if hint.field in _MOVIE_FILTER_FIELDS and hint.field not in {"title", "director", "genres", "year", "duration", "rating"}:
        return
    project_allowed = _PROJECT_FIELD_ALLOWLIST.get(str(web_project_id or "").strip().lower(), set())
    if (
        hint.field not in _DEFAULT_VALUES
        and hint.field not in {"title", "director", "genres", "year", "duration", "rating", "movie_name"}
        and hint.field not in project_allowed
    ):
        return
    if hint.field in field_values and hint.operator not in {"equals", "contains"}:
        return
    chosen = _pick_value(hint.field, hint.operator, hint.value, seed)
    if chosen:
        field_values.setdefault(hint.field, chosen)
        if hint.operator in {"equals", "contains"}:
            field_values[hint.field] = chosen


def _update_entity_filters(filters: dict[str, Any], hint: ConstraintHint) -> None:
    raw = _coerce_scalar(hint.value)
    if not raw:
        return
    if hint.field in {"movie_name", "title", "name"}:
        if hint.operator == "equals":
            filters["name_exact"] = raw
        elif hint.operator == "contains":
            filters["name_contains"] = raw
        elif hint.operator == "not_contains":
            filters["name_not_contains"] = raw
    elif hint.field == "director":
        if hint.operator == "equals":
            filters["director_exact"] = raw
        elif hint.operator == "contains":
            filters["director_contains"] = raw
    elif hint.field == "genres":
        if hint.operator == "equals":
            filters["genre_exact"] = raw
        elif hint.operator in {"contains", "not_contains"}:
            filters["genre_contains" if hint.operator == "contains" else "genre_not_contains"] = raw
        elif hint.operator == "in_list":
            values = [str(item).strip() for item in hint.value] if isinstance(hint.value, list) else [raw]
            normalized = [value for value in values if value]
            if normalized:
                filters["genre_any_of"] = normalized
        elif hint.operator == "not_in_list":
            values = [str(item).strip() for item in hint.value] if isinstance(hint.value, list) else [raw]
            normalized = [value for value in values if value]
            if normalized:
                filters["genre_none_of"] = normalized
    elif hint.field == "duration":
        numeric = _coerce_numeric(raw)
        if numeric is None:
            return
        if hint.operator == "equals":
            value = int(numeric)
            filters["duration_gte"] = value
            filters["duration_lte"] = value
        if hint.operator in {"greater_than", "greater_equal"}:
            filters["duration_gte"] = int(numeric)
        elif hint.operator in {"less_than", "less_equal"}:
            filters["duration_lte"] = int(numeric)
    elif hint.field == "rating":
        numeric = _coerce_numeric(raw)
        if numeric is None:
            return
        if hint.operator == "equals":
            value = float(numeric)
            filters["rating_gte"] = value
            filters["rating_lte"] = value
        if hint.operator in {"greater_than", "greater_equal"}:
            filters["rating_gte"] = float(numeric)
        elif hint.operator in {"less_than", "less_equal"}:
            filters["rating_lte"] = float(numeric)
    elif hint.field == "year":
        numeric = _coerce_numeric(raw)
        if numeric is None:
            return
        if hint.operator == "equals":
            value = int(numeric)
            filters["year_gte"] = value
            filters["year_lte"] = value
        if hint.operator in {"greater_than", "greater_equal"}:
            filters["year_gte"] = int(numeric)
        elif hint.operator in {"less_than", "less_equal"}:
            filters["year_lte"] = int(numeric)


def _relevant_data_payload(task_row: dict[str, Any]) -> dict[str, Any]:
    payload = task_row.get("relevant_data")
    return dict(payload) if isinstance(payload, dict) else {}


def _auth_required(use_case: str, field_values: dict[str, str]) -> bool:
    public_cases = {
        "ADD_COMMENT",
        "CONTACT",
        "FILM_DETAIL",
        "FILTER_FILM",
        "LOGIN",
        "REGISTRATION",
        "SEARCH_FILM",
        "SHARE_MOVIE",
        "WATCH_TRAILER",
    }
    if use_case in public_cases:
        return use_case in {"LOGIN", "REGISTRATION"}
    protected_cases = {"ADD_FILM", "ADD_TO_WATCHLIST", "DELETE_FILM", "EDIT_FILM", "EDIT_USER", "LOGOUT", "REMOVE_FROM_WATCHLIST"}
    return bool(field_values.get("username") or field_values.get("password") or use_case in protected_cases)


def normalize_task_row(task_row: dict[str, Any], *, seed: int | None = None) -> DeterministicTaskObjective:
    use_case_payload = task_row.get("use_case")
    web_project_id = resolve_project_id(task_row)
    use_case = str((use_case_payload or {}).get("name") or "").strip().upper()
    task_url = str(task_row.get("url") or "")
    task_seed = int(seed or _extract_seed(task_url))
    field_values: dict[str, str] = {}
    entity_filters: dict[str, Any] = {}
    constraints = _constraint_hints_from_task(task_row)

    for hint in constraints:
        _maybe_store_field_value(field_values, hint, task_seed, web_project_id=web_project_id)
        _update_entity_filters(entity_filters, hint)

    if use_case in _AUTH_EVENT_PRIORITY_FIELDS:
        # Event criteria for auth use cases should come only from task tests.
        prioritized_event_values = _auth_event_values_from_tests(task_row, use_case=use_case)
        for field in ("username", "email"):
            value = _coerce_scalar(prioritized_event_values.get(field))
            if value:
                field_values[field] = value

    relevant_data = _relevant_data_payload(task_row)
    _resolve_auth_field_values(
        use_case=use_case,
        seed=task_seed,
        field_values=field_values,
        relevant_data=relevant_data,
    )

    if "movie_name" in field_values and "query" not in field_values:
        field_values["query"] = field_values["movie_name"]
    if "name_exact" in entity_filters and "query" not in field_values:
        field_values["query"] = str(entity_filters["name_exact"])
    if "name_contains" in entity_filters and "query" not in field_values:
        field_values["query"] = str(entity_filters["name_contains"])
    if use_case in {"ADD_FILM", "EDIT_FILM"} and not str(field_values.get("title") or "").strip():
        # Task caches often express film names as `name` criteria; map them into
        # the editor's `title` field so planned edits can satisfy name checks.
        name_value = str(field_values.get("name") or "").strip()
        if name_value:
            field_values["title"] = name_value

    success_expectations = {
        "event_names": [str(test.get("event_name") or "").strip() for test in (task_row.get("tests") or []) if isinstance(test, dict) and str(test.get("event_name") or "").strip()],
        "texts": [],
        "url_contains": [],
    }
    route_target = _route_target(web_project_id=web_project_id, use_case=use_case)
    if str(web_project_id or "").strip().lower() == "autobooks":
        if use_case == "CONTACT_BOOK":
            success_expectations["url_contains"] = ["/contact"]
        elif use_case in {"SEARCH_BOOK", "FILTER_BOOK", "OPEN_PREVIEW"}:
            success_expectations["url_contains"] = ["/search"]
        elif use_case in {"VIEW_CART_BOOK", "REMOVE_FROM_CART_BOOK", "ADD_TO_CART_BOOK"}:
            success_expectations["url_contains"] = ["/cart"]
        elif use_case in {
            "BOOK_DETAIL",
            "SHARE_BOOK",
            "PURCHASE_BOOK",
            "ADD_COMMENT_BOOK",
            "ADD_TO_READING_LIST",
            "REMOVE_FROM_READING_LIST",
        }:
            success_expectations["url_contains"] = ["/books"]
        elif use_case in {"LOGIN_BOOK", "REGISTRATION_BOOK"}:
            success_expectations["url_contains"] = ["/", "/login", "/register"]
        elif use_case in {"DELETE_BOOK", "ADD_BOOK", "EDIT_BOOK", "EDIT_USER_BOOK", "LOGOUT_BOOK"}:
            success_expectations["url_contains"] = ["/profile"]
    elif str(web_project_id or "").strip().lower() == "autozone":
        if use_case in {"SEARCH_PRODUCT", "CATEGORY_FILTER"}:
            success_expectations["url_contains"] = ["/search", "/"]
        elif use_case in {"VIEW_CART", "QUANTITY_CHANGED"}:
            success_expectations["url_contains"] = ["/cart"]
        elif use_case in {"VIEW_WISHLIST", "ADD_TO_WISHLIST"}:
            success_expectations["url_contains"] = ["/wishlist", "/"]
        elif use_case in {"PROCEED_TO_CHECKOUT", "CHECKOUT_STARTED", "ORDER_COMPLETED"}:
            success_expectations["url_contains"] = ["/checkout", "/cart"]
        elif use_case in {"VIEW_DETAIL", "DETAILS_TOGGLE", "ADD_TO_CART", "SHARE_PRODUCT", "CAROUSEL_SCROLL"}:
            success_expectations["url_contains"] = ["/product", "/", "/p/"]
    elif str(web_project_id or "").strip().lower() == "autodining":
        if use_case in {"CONTACT_FORM_SUBMIT", "CONTACT_PAGE_VIEW", "CONTACT_CARD_CLICK"}:
            success_expectations["url_contains"] = ["/contact", "/"]
        elif use_case in {"ABOUT_PAGE_VIEW", "ABOUT_FEATURE_CLICK"}:
            success_expectations["url_contains"] = ["/about", "/"]
        elif use_case in {"HELP_PAGE_VIEW", "HELP_CATEGORY_SELECTED", "HELP_FAQ_TOGGLED"}:
            success_expectations["url_contains"] = ["/help", "/"]
        elif use_case == "SEARCH_RESTAURANT":
            success_expectations["url_contains"] = ["/search", "/"]
        elif use_case in {
            "VIEW_RESTAURANT",
            "VIEW_FULL_MENU",
            "COLLAPSE_MENU",
            "DATE_DROPDOWN_OPENED",
            "TIME_DROPDOWN_OPENED",
            "PEOPLE_DROPDOWN_OPENED",
            "SCROLL_VIEW",
            "BOOK_RESTAURANT",
            "COUNTRY_SELECTED",
            "OCCASION_SELECTED",
            "RESERVATION_COMPLETE",
        }:
            success_expectations["url_contains"] = ["/", "/restaurant", "/dining", "/r/"]
    elif str(web_project_id or "").strip().lower() == "autocrm":
        if use_case == "HELP_VIEWED":
            success_expectations["url_contains"] = ["/help", "/"]
        elif use_case == "BILLING_SEARCH":
            success_expectations["url_contains"] = ["/billing", "/search", "/"]
        elif use_case in {"SEARCH_MATTER", "SEARCH_CLIENT", "FILTER_CLIENTS", "FILTER_MATTER_STATUS", "SORT_MATTER_BY_CREATED_AT"}:
            success_expectations["url_contains"] = ["/search", "/"]
        elif use_case in {"VIEW_CLIENT_DETAILS", "ADD_CLIENT", "DELETE_CLIENT"}:
            success_expectations["url_contains"] = ["/clients", "/"]
        elif use_case in {
            "UPDATE_MATTER",
            "ADD_NEW_MATTER",
            "VIEW_MATTER_DETAILS",
            "ARCHIVE_MATTER",
            "DELETE_MATTER",
            "NEW_LOG_ADDED",
            "LOG_EDITED",
            "LOG_DELETE",
            "DOCUMENT_RENAMED",
            "DOCUMENT_DELETED",
        }:
            success_expectations["url_contains"] = ["/matters", "/"]
        elif use_case in {"VIEW_PENDING_EVENTS", "NEW_CALENDAR_EVENT_ADDED"}:
            success_expectations["url_contains"] = ["/calendar", "/"]
        elif use_case == "CHANGE_USER_NAME":
            success_expectations["url_contains"] = ["/settings", "/profile", "/"]
        else:
            success_expectations["url_contains"] = ["/"]
    elif str(web_project_id or "").strip().lower() == "automail":
        if use_case == "SEARCH_EMAIL":
            success_expectations["url_contains"] = ["/search", "/"]
        elif use_case in {
            "VIEW_TEMPLATES",
            "TEMPLATE_SELECTED",
            "TEMPLATE_BODY_EDITED",
            "TEMPLATE_SENT",
            "TEMPLATE_SAVED_DRAFT",
            "TEMPLATE_CANCELED",
        }:
            success_expectations["url_contains"] = ["/templates", "/"]
        elif use_case in {"THEME_CHANGED", "CREATE_LABEL", "ADD_LABEL", "CLEAR_SELECTION"}:
            success_expectations["url_contains"] = ["/settings", "/inbox", "/"]
        elif use_case in {"SEND_EMAIL", "EMAIL_SAVE_AS_DRAFT", "EDIT_DRAFT_EMAIL"}:
            success_expectations["url_contains"] = ["/compose", "/drafts", "/"]
        else:
            success_expectations["url_contains"] = ["/inbox", "/", "/mail", "/email"]
    elif str(web_project_id or "").strip().lower() == "autolodge":
        if use_case in {"HELP_VIEWED", "FAQ_OPENED"}:
            success_expectations["url_contains"] = ["/help", "/"]
        elif use_case == "SEARCH_HOTEL":
            success_expectations["url_contains"] = ["/search", "/"]
        elif use_case in {
            "ADD_TO_WISHLIST",
            "REMOVE_FROM_WISHLIST",
            "WISHLIST_OPENED",
            "BOOK_FROM_WISHLIST",
        }:
            success_expectations["url_contains"] = ["/wishlist", "/"]
        elif use_case in {"CONFIRM_AND_PAY", "PAYMENT_METHOD_SELECTED"}:
            success_expectations["url_contains"] = ["/confirm", "/stay", "/"]
        elif use_case in {
            "VIEW_HOTEL",
            "RESERVE_HOTEL",
            "EDIT_NUMBER_OF_GUESTS",
            "EDIT_CHECK_IN_OUT_DATES",
            "MESSAGE_HOST",
            "SUBMIT_REVIEW",
        }:
            success_expectations["url_contains"] = ["/stay", "/", "/hotel", "/h/"]
        else:
            success_expectations["url_contains"] = ["/", "/stay", "/search"]
    elif str(web_project_id or "").strip().lower() == "autodelivery":
        if use_case in {"SEARCH_DELIVERY_RESTAURANT", "RESTAURANT_FILTER"}:
            success_expectations["url_contains"] = ["/search", "/"]
        elif use_case in {
            "QUICK_ORDER_STARTED",
            "EMPTY_CART",
            "EDIT_CART_ITEM",
            "ITEM_INCREMENTED",
        }:
            success_expectations["url_contains"] = ["/cart", "/"]
        elif use_case in {
            "OPEN_CHECKOUT_PAGE",
            "DROPOFF_PREFERENCE",
            "ADDRESS_ADDED",
            "PLACE_ORDER",
            "DELIVERY_PRIORITY_SELECTED",
        }:
            success_expectations["url_contains"] = ["/checkout", "/cart", "/"]
        elif use_case in {
            "VIEW_DELIVERY_RESTAURANT",
            "ADD_TO_CART_MODAL_OPEN",
            "ADD_TO_CART_MENU_ITEM",
            "REVIEW_SUBMITTED",
            "DELETE_REVIEW",
        }:
            success_expectations["url_contains"] = ["/restaurant", "/r/", "/"]
        else:
            success_expectations["url_contains"] = ["/", "/restaurant", "/search"]
    elif str(web_project_id or "").strip().lower() == "autowork":
        if use_case in {
            "POST_A_JOB",
            "WRITE_JOB_TITLE",
            "SEARCH_SKILL",
            "ADD_SKILL",
            "CHOOSE_BUDGET_TYPE",
            "CHOOSE_PROJECT_SIZE",
            "CHOOSE_PROJECT_TIMELINE",
            "SET_RATE_RANGE",
            "WRITE_JOB_DESCRIPTION",
            "SUBMIT_JOB",
            "CLOSE_POST_A_JOB_WINDOW",
        }:
            success_expectations["url_contains"] = ["/jobs", "/post", "/"]
        elif use_case in {
            "NAVBAR_PROFILE_CLICK",
            "EDIT_PROFILE_NAME",
            "EDIT_ABOUT",
            "EDIT_PROFILE_EMAIL",
            "EDIT_PROFILE_TITLE",
            "EDIT_PROFILE_LOCATION",
        }:
            success_expectations["url_contains"] = ["/profile", "/"]
        elif use_case in {
            "BOOK_A_CONSULTATION",
            "HIRE_BTN_CLICKED",
            "QUICK_HIRE",
            "SELECT_HIRING_TEAM",
            "HIRE_CONSULTANT",
            "CANCEL_HIRE",
            "CONTACT_EXPERT_OPENED",
            "CONTACT_EXPERT_MESSAGE_SENT",
        }:
            success_expectations["url_contains"] = ["/experts", "/hire", "/"]
        elif use_case in {
            "HIRE_LATER_ADDED",
            "HIRE_LATER_REMOVED",
            "HIRE_LATER_START",
            "NAVBAR_HIRE_LATER_CLICK",
        }:
            success_expectations["url_contains"] = ["/hire-later", "/"]
        elif use_case in {
            "BROWSE_FAVORITE_EXPERT",
            "FAVORITE_EXPERT_SELECTED",
            "FAVORITE_EXPERT_REMOVED",
            "NAVBAR_FAVORITES_CLICK",
        }:
            success_expectations["url_contains"] = ["/favorites", "/"]
        elif use_case in {"NAVBAR_HIRES_CLICK"}:
            success_expectations["url_contains"] = ["/hires", "/"]
        else:
            success_expectations["url_contains"] = ["/", "/jobs", "/experts"]
    elif str(web_project_id or "").strip().lower() == "autoconnect":
        if use_case in {"SEARCH_JOBS", "FILTER_JOBS", "BACK_TO_ALL_JOBS", "VIEW_APPLIED_JOBS", "APPLY_FOR_JOB", "VIEW_JOB", "CANCEL_APPLICATION", "JOBS_NAVBAR"}:
            success_expectations["url_contains"] = ["/jobs", "/"]
        elif use_case in {"SEARCH_USERS", "FOLLOW_PAGE", "UNFOLLOW_PAGE", "VIEW_USER_PROFILE", "CONNECT_WITH_USER"}:
            success_expectations["url_contains"] = ["/", "/search", "/profile", "/u/"]
        elif use_case in {
            "POST_STATUS",
            "LIKE_POST",
            "COMMENT_ON_POST",
            "SAVE_POST",
            "HIDE_POST",
            "REMOVE_POST",
            "VIEW_SAVED_POSTS",
            "VIEW_HIDDEN_POSTS",
            "UNHIDE_POST",
            "HOME_NAVBAR",
        }:
            success_expectations["url_contains"] = ["/", "/feed", "/in"]
        elif use_case in {"EDIT_PROFILE", "EDIT_EXPERIENCE", "ADD_EXPERIENCE"}:
            success_expectations["url_contains"] = ["/profile", "/"]
        else:
            success_expectations["url_contains"] = ["/", "/jobs", "/profile"]
    elif str(web_project_id or "").strip().lower() == "autocalendar":
        if use_case == "SEARCH_SUBMIT":
            success_expectations["url_contains"] = ["/search", "/"]
        elif use_case in {
            "ADD_NEW_CALENDAR",
            "CREATE_CALENDAR",
            "UNSELECT_CALENDAR",
            "SELECT_CALENDAR",
        }:
            success_expectations["url_contains"] = ["/", "/calendar", "/cal"]
        elif use_case in {
            "EVENT_WIZARD_OPEN",
            "ADD_EVENT",
            "CANCEL_ADD_EVENT",
            "DELETE_ADDED_EVENT",
            "CELL_CLICKED",
            "EVENT_ADD_REMINDER",
            "EVENT_REMOVE_REMINDER",
            "EVENT_ADD_ATTENDEE",
            "EVENT_REMOVE_ATTENDEE",
        }:
            success_expectations["url_contains"] = ["/", "/event", "/e/"]
        else:
            success_expectations["url_contains"] = ["/", "/calendar", "/day", "/week", "/month"]
    elif str(web_project_id or "").strip().lower() == "autolist":
        if use_case in {
            "AUTOLIST_ADD_TEAM_CLICKED",
            "AUTOLIST_TEAM_MEMBERS_ADDED",
            "AUTOLIST_TEAM_ROLE_ASSIGNED",
            "AUTOLIST_TEAM_CREATED",
        }:
            success_expectations["url_contains"] = ["/teams", "/"]
        else:
            success_expectations["url_contains"] = ["/tasks", "/", "/task"]
    elif str(web_project_id or "").strip().lower() == "autohealth":
        if use_case in {
            "SEARCH_DOCTORS",
            "VIEW_DOCTOR_PROFILE",
            "VIEW_DOCTOR_EDUCATION",
            "VIEW_DOCTOR_AVAILABILITY",
            "FILTER_DOCTOR_REVIEWS",
            "OPEN_CONTACT_DOCTOR_FORM",
            "CONTACT_DOCTOR",
        }:
            success_expectations["url_contains"] = ["/doctors", "/", "/doctors/"]
        elif use_case in {
            "OPEN_APPOINTMENT_FORM",
            "APPOINTMENT_BOOKED_SUCCESSFULLY",
            "REQUEST_QUICK_APPOINTMENT",
            "SEARCH_APPOINTMENT",
        }:
            success_expectations["url_contains"] = ["/appointments", "/"]
        elif use_case in {"SEARCH_PRESCRIPTION", "REFILL_PRESCRIPTION", "VIEW_PRESCRIPTION"}:
            success_expectations["url_contains"] = ["/prescriptions", "/"]
        elif use_case in {"SEARCH_MEDICAL_ANALYSIS", "VIEW_MEDICAL_ANALYSIS"}:
            success_expectations["url_contains"] = ["/medical-records", "/"]
        else:
            success_expectations["url_contains"] = ["/"]
    elif use_case == "CONTACT":
        success_expectations["texts"] = ["Message Sent!"]
        success_expectations["url_contains"] = ["/contact"]
    elif use_case in {"LOGIN", "REGISTRATION", "EDIT_USER", "ADD_FILM", "EDIT_FILM", "DELETE_FILM", "LOGOUT"}:
        success_expectations["url_contains"] = ["/profile", "/"]
    elif use_case in {"ADD_TO_WATCHLIST", "REMOVE_FROM_WATCHLIST", "SHARE_MOVIE", "WATCH_TRAILER", "ADD_COMMENT", "FILM_DETAIL"}:
        success_expectations["url_contains"] = ["/movies/"]
    elif use_case in {"SEARCH_FILM", "FILTER_FILM"}:
        success_expectations["url_contains"] = ["/search"]

    return DeterministicTaskObjective(
        web_project_id=web_project_id,
        use_case=use_case,
        seed=task_seed,
        task_url=task_url,
        prompt=str(task_row.get("prompt") or ""),
        route_target=route_target,
        field_values=field_values,
        entity_filters=entity_filters,
        auth_required=_auth_required(use_case, field_values),
        success_expectations=success_expectations,
        constraints=tuple(constraints),
        relevant_data=relevant_data,
        raw_task=task_row,
    )


def load_task_objective(*, cache_path: Path, use_case: str, seed: int | None = None, web_project_id: str | None = None) -> DeterministicTaskObjective:
    row = load_task_row(cache_path=cache_path, use_case=use_case, seed=seed, web_project_id=web_project_id)
    if seed is None:
        return normalize_task_row(dict(row))
    row_seed = extract_seed_from_task_url(str(row.get("url") or ""))
    if row_seed == int(seed):
        return normalize_task_row(dict(row), seed=int(seed))
    normalized = dict(row)
    normalized["url"] = str(row.get("url") or "").split("?")[0] + f"?seed={int(seed)}"
    return normalize_task_row(normalized, seed=seed)


__all__ = [
    "ConstraintHint",
    "DeterministicTaskObjective",
    "extract_seed_from_task_url",
    "load_task_objective",
    "load_task_row",
    "load_task_rows",
    "normalize_task_row",
    "route_for_web_project_use_case",
    "task_seeds_for_use_case",
]
