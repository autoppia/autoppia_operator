"""Seed-aware action adaptor for deterministic plan builders.

Some IWA trajectories hardcode values (e.g. a dropdown specialty) that only exist in
the dataset for a specific seed. When the trajectory is replayed at a different seed,
those values may be absent, causing the action to fail silently and producing no gold.

This module probes the web project's dataset API for the target seed and replaces
hardcoded values with ones that actually exist in that seed's data.

Adaptors are registered per (project_id, use_case). When no adaptor is registered,
the actions are returned unchanged.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Project → API project_key mapping
# ---------------------------------------------------------------------------

_PROJECT_KEY: dict[str, str] = {
    "autocinema": "web_01_autocinema",
    "autobooks": "web_02_autobooks",
    "autozone": "web_03_autozone",
    "autodining": "web_04_autodining",
    "autocrm": "web_05_autocrm",
    "automail": "web_06_automail",
    "autodelivery": "web_07_autodelivery",
    "autolodge": "web_08_autolodge",
    "autoconnect": "web_09_autoconnect",
    "autowork": "web_10_autowork",
    "autocalendar": "web_11_autocalendar",
    "autolist": "web_12_autolist",
    "autodrive": "web_13_autodrive",
    "autohealth": "web_14_autohealth",
    "autostats": "web_15_autostats",
    "autodiscord": "web_16_autodiscord",
}


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _base_url_from_task_url(task_url: str) -> str:
    parsed = urlparse(str(task_url or ""))
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return ""


def _fetch_rows(base_url: str, project_key: str, entity_type: str, seed: int) -> list[dict[str, Any]]:
    url = (
        f"{base_url.rstrip('/')}/api/datasets/load"
        f"?project_key={project_key}&entity_type={entity_type}&seed_value={seed}&limit=500"
    )
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
            data = json.loads(resp.read())
        if isinstance(data, list):
            return data
        return data.get("data", data.get("rows", []))
    except Exception:
        return []


def _unique_values(rows: list[dict[str, Any]], field: str) -> list[str]:
    """Extract unique non-empty scalar values for `field`, flattening list fields."""
    seen: list[str] = []
    for row in rows:
        val = row.get(field)
        items: list[Any] = val if isinstance(val, list) else [val]
        for item in items:
            s = str(item).strip()
            if s and s not in seen:
                seen.append(s)
    return seen


def _get_testid(action: dict[str, Any]) -> str:
    """Return the data-testid value from a selector_candidates list, or ''."""
    for cand in action.get("selector_candidates") or []:
        if isinstance(cand, dict) and str(cand.get("attribute", "")).lower() == "data-testid":
            return str(cand.get("value", ""))
    return ""


# ---------------------------------------------------------------------------
# Per-use-case adaptors
# ---------------------------------------------------------------------------

_TESTID_TO_FIELD = {
    "doctor-specialty-filter": "specialty",
    "doctor-language-filter": "language",
    "doctors-sort-by": "sort_by",
    "doctors-sort-order": "sort_order",
}

# V2 dynamic mode hides the page behind DataReadyGate while it reloads data for the
# new seed. This wait gives the DB fetch + React re-render time to complete so the
# <select> options are populated before we attempt to select them.
_V2_DATA_READY_WAIT = {"type": "WaitAction", "time_seconds": 3.0}

# TypeAction to type a partial doctor name — guarantees hasSearch=true in handleSearch
# even if the SelectAction doesn't update React state in time before the click.
_DOCTOR_NAME_SEARCH_CANDIDATES = [
    {"type": "attributeValueSelector", "attribute": "data-testid", "value": "doctor-name-search", "case_sensitive": False},
    {"type": "attributeValueSelector", "attribute": "id", "value": "doctor-name-search", "case_sensitive": False},
]

# V4 dynamic popups (data-v4="true") appear after a seed-dependent delay and cover the
# entire viewport with z-index 99999, intercepting all pointer events.  Clicking the
# Close button (aria-label="Close") dismisses the popup so the search button becomes
# clickable.  If no popup is visible the click fails silently and execution continues.
_V4_POPUP_DISMISS = {
    "type": "ClickAction",
    "field_name": "dismiss popup",
    "selector_candidates": [
        {"type": "attributeValueSelector", "attribute": "aria-label", "value": "Close", "case_sensitive": False},
    ],
}


def _adapt_doctors_search_flow(
    actions: list[dict[str, Any]],
    seed: int,
    base_url: str,
    project_key: str,
) -> list[dict[str, Any]]:
    """Generic adaptor for any autohealth use case that goes through the /doctors search flow.

    Handles all UCs that:
      1. Use SelectDropDownOptionAction on doctor-specialty-filter / doctor-language-filter
      2. Click doctors-search-button (covered by the V4 popup overlay)

    Strategy:
      - Fetch doctors from the dataset API at the target seed.
      - Replace hardcoded specialty/language values with ones that exist at that seed.
      - Sort-by / sort-order dropdowns carry static option values — convert type only.
      - Inject WaitAction(3s) before first select (V2 DataReadyGate delay).
      - Inject TypeAction(partial name) before first select (guarantees hasSearch=true).
      - Inject V4 popup dismiss before doctors-search-button click.
      - All other SelectDropDownOptionAction (e.g. form fields) are converted to
        SelectAction keeping their original text value unchanged.
    """
    rows = _fetch_rows(base_url, project_key, "doctors", seed)
    specialties = _unique_values(rows, "specialty")
    languages = _unique_values(rows, "languages")  # list field per doctor

    doctor_name_fragment = ""
    if rows:
        full_name = str(rows[0].get("name") or "").strip()
        doctor_name_fragment = full_name[:4] if full_name else ""

    adapted: list[dict[str, Any]] = []
    injected_wait = False
    injected_type = False
    for action in actions:
        a = dict(action)
        if a.get("type") == "SelectDropDownOptionAction":
            if not injected_wait:
                adapted.append(dict(_V2_DATA_READY_WAIT))
                injected_wait = True
            if not injected_type and doctor_name_fragment:
                adapted.append({
                    "type": "TypeAction",
                    "text": doctor_name_fragment,
                    "field_name": "doctor search",
                    "selector_candidates": list(_DOCTOR_NAME_SEARCH_CANDIDATES),
                })
                injected_type = True
            testid = _get_testid(a)
            if testid == "doctor-specialty-filter":
                value = specialties[0] if specialties else a.get("text", "")
            elif testid == "doctor-language-filter":
                value = languages[0] if languages else a.get("text", "")
            else:
                # Static dropdown (sort-by, sort-order, form fields) — keep original value.
                value = a.get("text", "")
            a = {
                "type": "SelectAction",
                "value": value,
                "field_name": _TESTID_TO_FIELD.get(testid, "filter"),
                "selector_candidates": action.get("selector_candidates", []),
            }
        elif a.get("type") == "ClickAction" and _get_testid(a) == "doctors-search-button":
            adapted.append(dict(_V4_POPUP_DISMISS))
        adapted.append(a)
    return adapted


# ---------------------------------------------------------------------------
# Registry and dispatch
# ---------------------------------------------------------------------------

_AdaptorFn = Any  # (actions, seed, base_url, project_key) -> list[dict]

_DOCTORS_SEARCH_USE_CASES = {
    "SEARCH_DOCTORS",
    "CONTACT_DOCTOR",
    "FILTER_DOCTOR_REVIEWS",
    "OPEN_CONTACT_DOCTOR_FORM",
    "VIEW_DOCTOR_AVAILABILITY",
    "VIEW_DOCTOR_PROFILE",
}

_ADAPTORS: dict[tuple[str, str], _AdaptorFn] = {
    ("autohealth", uc): _adapt_doctors_search_flow
    for uc in _DOCTORS_SEARCH_USE_CASES
}


def adapt_actions_for_seed(
    project_id: str,
    use_case: str,
    actions: list[dict[str, Any]],
    seed: int,
    task_url: str,
) -> list[dict[str, Any]]:
    """Return actions adapted to data available at `seed`, or unchanged if no adaptor registered."""
    pid = str(project_id or "").strip().lower()
    uc = str(use_case or "").strip().upper()
    adaptor_fn = _ADAPTORS.get((pid, uc))
    if adaptor_fn is None:
        return actions
    base_url = _base_url_from_task_url(task_url)
    if not base_url:
        return actions
    project_key = _PROJECT_KEY.get(pid, pid)
    return adaptor_fn(actions, seed, base_url, project_key)


__all__ = ["adapt_actions_for_seed"]
