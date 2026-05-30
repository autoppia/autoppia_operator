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


def _fetch_rows_page(base_url: str, project_key: str, entity_type: str, seed: int, filter_key: str) -> list[dict[str, Any]]:
    """Fetch the same 50 rows the web page loads (distribute method, matching the frontend's fetchSeededSelection call).

    When V2 is disabled the frontend always uses seed=1. When V2 is enabled it uses the URL seed.
    Calling with BOTH seeds and merging the results makes adaptors robust to either V2 mode.
    """
    url = (
        f"{base_url.rstrip('/')}/api/datasets/load"
        f"?project_key={project_key}&entity_type={entity_type}"
        f"&seed_value={seed}&limit=50&method=distribute&filter_key={filter_key}"
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


def _fetch_page_rows_v2safe(
    base_url: str,
    project_key: str,
    entity_type: str,
    seed: int,
    filter_key: str,
) -> list[dict[str, Any]]:
    """Return rows present in BOTH seed=1 and the actual seed's distribute-50 sets.

    The web frontend fetches 50 rows via the distribute method. When V2 is disabled
    it always uses seed=1 regardless of the URL seed; when V2 is enabled it uses the
    URL seed.  Taking the intersection guarantees the rows we select are visible on
    the page in either V2 mode.  Seed=1 rows are used as the base so that fields
    reflect what V2-off browsers actually show.
    """
    rows_s1 = _fetch_rows_page(base_url, project_key, entity_type, 1, filter_key)
    rows_seed = _fetch_rows_page(base_url, project_key, entity_type, seed, filter_key)
    if seed == 1:
        return rows_s1
    seed_ids: set[str] = {str(r.get("id") or "") for r in rows_seed if r.get("id")}
    # Keep seed=1 rows that are also present at the actual seed (safe for V2 on/off)
    intersected = [r for r in rows_s1 if str(r.get("id") or "") in seed_ids]
    if intersected:
        return intersected
    # No intersection: fall back to seed=1 subset (works when V2 is disabled)
    return rows_s1


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


def _get_id_attr(action: dict[str, Any]) -> str:
    """Return the id attribute value from selector_candidates, or ''."""
    for cand in action.get("selector_candidates") or []:
        if isinstance(cand, dict) and str(cand.get("attribute", "")).lower() == "id":
            return str(cand.get("value", ""))
    return ""


def _replace_in_xpath(action: dict[str, Any], old: str, new: str) -> dict[str, Any]:
    """Return a copy of action with old replaced by new in all xpathSelector values."""
    cands = action.get("selector_candidates")
    if not cands:
        return action
    changed = False
    new_cands = []
    for cand in cands:
        if isinstance(cand, dict) and cand.get("type") == "xpathSelector":
            v = str(cand.get("value", ""))
            if old in v:
                cand = {**cand, "value": v.replace(old, new)}
                changed = True
        new_cands.append(cand)
    if not changed:
        return action
    return {**action, "selector_candidates": new_cands}


def _replace_xpath_containing(action: dict[str, Any], contains_substr: str, new_xpath: str) -> dict[str, Any]:
    """Replace all xpathSelector values that contain `contains_substr` with `new_xpath`."""
    cands = action.get("selector_candidates")
    if not cands:
        return action
    changed = False
    new_cands = []
    for cand in cands:
        if isinstance(cand, dict) and cand.get("type") == "xpathSelector":
            v = str(cand.get("value", ""))
            if contains_substr in v:
                cand = {**cand, "value": new_xpath}
                changed = True
        new_cands.append(cand)
    if not changed:
        return action
    return {**action, "selector_candidates": new_cands}


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


def _adapt_search_appointment(
    actions: list[dict[str, Any]],
    seed: int,
    base_url: str,
    project_key: str,
) -> list[dict[str, Any]]:
    """SEARCH_APPOINTMENT: replace hardcoded specialty and date with page-visible appointment data."""
    rows = _fetch_page_rows_v2safe(base_url, project_key, "appointments", seed, "specialty")
    if not rows:
        return actions
    specialty = str(rows[0].get("specialty") or "").strip()
    date = str(rows[0].get("date") or "").strip()
    adapted = []
    for action in actions:
        a = dict(action)
        if a.get("type") == "TypeAction":
            if _get_id_attr(a) == "specialty-filter":
                a["text"] = specialty
            elif _get_id_attr(a) == "date-filter":
                a["text"] = date
        elif a.get("type") == "ClickAction":
            a = _replace_in_xpath(a, "Cardiology", specialty)
        adapted.append(a)
    return adapted


def _adapt_open_appointment_form(
    actions: list[dict[str, Any]],
    seed: int,
    base_url: str,
    project_key: str,
) -> list[dict[str, Any]]:
    """OPEN_APPOINTMENT_FORM: use first page-visible appointment's date and click first row."""
    rows = _fetch_page_rows_v2safe(base_url, project_key, "appointments", seed, "specialty")
    if not rows:
        return actions
    date = str(rows[0].get("date") or "").strip()
    adapted = []
    for action in actions:
        a = dict(action)
        if a.get("type") == "TypeAction" and _get_id_attr(a) == "date-filter":
            a["text"] = date
        elif a.get("type") == "ClickAction":
            a = _replace_xpath_containing(a, "tbody/tr", "(//tbody//tr//button)[1]")
        adapted.append(a)
    return adapted


def _adapt_appointment_booked(
    actions: list[dict[str, Any]],
    seed: int,
    base_url: str,
    project_key: str,
) -> list[dict[str, Any]]:
    """APPOINTMENT_BOOKED_SUCCESSFULLY: use first page-visible appointment's date, click first row."""
    rows = _fetch_page_rows_v2safe(base_url, project_key, "appointments", seed, "specialty")
    if not rows:
        return actions
    date = str(rows[0].get("date") or "").strip()
    adapted = []
    for action in actions:
        a = dict(action)
        if a.get("type") == "TypeAction" and _get_id_attr(a) == "date-filter":
            a["text"] = date
        elif a.get("type") == "ClickAction":
            a = _replace_xpath_containing(a, "tbody/tr", "(//tbody//tr//button)[1]")
        adapted.append(a)
    return adapted


def _adapt_request_quick_appointment(
    actions: list[dict[str, Any]],
    seed: int,
    base_url: str,
    project_key: str,
) -> list[dict[str, Any]]:
    """REQUEST_QUICK_APPOINTMENT: replace hardcoded specialty with one visible in the doctors page."""
    rows = _fetch_page_rows_v2safe(base_url, project_key, "doctors", seed, "specialty")
    specialties = _unique_values(rows, "specialty")
    specialty = specialties[0] if specialties else "General"
    adapted = []
    for action in actions:
        a = dict(action)
        if a.get("type") == "TypeAction":
            cands = a.get("selector_candidates") or []
            if any(c.get("type") == "xpathSelector" and "Speciality" in str(c.get("value", "")) for c in cands):
                a["text"] = specialty
        elif a.get("type") == "ClickAction":
            a = _replace_in_xpath(a, "Pediatrics", specialty)
        adapted.append(a)
    return adapted


def _adapt_search_prescription(
    actions: list[dict[str, Any]],
    seed: int,
    base_url: str,
    project_key: str,
) -> list[dict[str, Any]]:
    """SEARCH_PRESCRIPTION: replace hardcoded medicine and doctor fragments.

    Uses the same 50-row distribute set the page loads so the search term matches
    visible rows regardless of whether V2 is enabled or disabled.
    """
    rows = _fetch_page_rows_v2safe(base_url, project_key, "prescriptions", seed, "category")
    if not rows:
        return actions
    medicine = str(rows[0].get("medicineName") or "").strip()
    doctor = str(rows[0].get("doctorName") or "").strip()
    med_frag = medicine[:6] if medicine else "med"
    doc_parts = doctor.replace("Dr. ", "").split()
    doc_frag = doc_parts[0][:5] if doc_parts else "Dr."
    adapted = []
    for action in actions:
        a = dict(action)
        if a.get("type") == "TypeAction":
            testid = _get_testid(a)
            if testid == "search-prescription-medicine":
                a["text"] = med_frag
            elif testid == "search-prescription-doctor":
                a["text"] = doc_frag
        adapted.append(a)
    return adapted


def _pick_refillable_prescription(rows: list[dict[str, Any]]) -> str:
    """Return the doctor_name of the best refillable prescription from `rows`.

    Prefers doctors whose EVERY prescription in `rows` is refillable (guarantees the
    refill button is enabled no matter which row is clicked). Falls back to any
    refillable prescription if no such doctor exists.
    """
    by_doctor: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        doc = str(r.get("doctorName") or "").strip()
        if doc:
            by_doctor.setdefault(doc, []).append(r)
    candidates = [
        (doc, prs) for doc, prs in by_doctor.items()
        if all(int(p.get("refillsRemaining") or 0) > 0 for p in prs)
    ]
    if candidates:
        candidates.sort(key=lambda x: len(x[1]))
        return candidates[0][0]
    refillable = [r for r in rows if int(r.get("refillsRemaining") or 0) > 0]
    return str((refillable or rows or [{}])[0].get("doctorName") or "").strip()


def _adapt_refill_prescription(
    actions: list[dict[str, Any]],
    seed: int,
    base_url: str,
    project_key: str,
) -> list[dict[str, Any]]:
    """REFILL_PRESCRIPTION: find a refillable prescription visible on the page.

    Uses the seed=1 distribute-50 rows (what the page shows when V2 is disabled) as the
    source of truth for which rows are visible.  Picks a doctor where ALL their seed=1
    prescriptions have refillsRemaining > 0 so the refill button is enabled regardless
    of which row the XPath clicks first.  Also verifies the doctor exists in the
    actual-seed page for V2-on compatibility.
    """
    rows_s1 = _fetch_rows_page(base_url, project_key, "prescriptions", 1, "category")
    if not rows_s1:
        return actions

    by_doctor: dict[str, list[dict[str, Any]]] = {}
    for r in rows_s1:
        doc = str(r.get("doctorName") or "").strip()
        if doc:
            by_doctor.setdefault(doc, []).append(r)

    if seed != 1:
        rows_seed = _fetch_rows_page(base_url, project_key, "prescriptions", seed, "category")
        seed_doctors: set[str] = {str(r.get("doctorName") or "").strip() for r in rows_seed}
    else:
        seed_doctors = set(by_doctor)

    # Doctor must have ALL seed=1 prescriptions refillable AND appear in seed-N page (V2-on compat)
    candidates = [
        (doc, prs)
        for doc, prs in by_doctor.items()
        if all(int(p.get("refillsRemaining") or 0) > 0 for p in prs) and doc in seed_doctors
    ]
    if not candidates:
        candidates = [
            (doc, prs)
            for doc, prs in by_doctor.items()
            if all(int(p.get("refillsRemaining") or 0) > 0 for p in prs)
        ]

    if candidates:
        candidates.sort(key=lambda x: (len(x[1]), x[0]))
        doctor_name = candidates[0][0]
    else:
        refillable = [r for r in rows_s1 if int(r.get("refillsRemaining") or 0) > 0]
        doctor_name = str((refillable or rows_s1 or [{}])[0].get("doctorName") or "").strip()

    frag = doctor_name.replace("Dr. ", "").strip().lower()
    adapted = []
    for action in actions:
        a = dict(action)
        if a.get("type") == "TypeAction" and _get_testid(a) == "search-prescription-doctor":
            a["text"] = frag
        elif a.get("type") == "ClickAction":
            a = _replace_xpath_containing(
                a,
                "view-prescription-btn",
                f"(//tr[contains(.,'{doctor_name}')]//button[@data-testid='view-prescription-btn'])[1]",
            )
        adapted.append(a)
    return adapted


def _adapt_view_prescription(
    actions: list[dict[str, Any]],
    seed: int,
    base_url: str,
    project_key: str,
) -> list[dict[str, Any]]:
    """VIEW_PRESCRIPTION: use first prescription visible on the page to search and click."""
    rows = _fetch_page_rows_v2safe(base_url, project_key, "prescriptions", seed, "category")
    if not rows:
        return actions
    doctor_name = str(rows[0].get("doctorName") or "").strip()
    doctor_search = doctor_name.replace("Dr. ", "").strip()
    adapted = []
    for action in actions:
        a = dict(action)
        if a.get("type") == "TypeAction" and _get_testid(a) == "search-prescription-doctor":
            a["text"] = doctor_search
        elif a.get("type") == "ClickAction":
            a = _replace_xpath_containing(
                a,
                "view-prescription-btn",
                f"(//tr[contains(.,'{doctor_name}')]//button[@data-testid='view-prescription-btn'])[1]",
            )
        adapted.append(a)
    return adapted


_VIEW_RECORD_BTN_XPATH = "(//div[contains(@class,'rounded-lg')][contains(@class,'border')]//button[@data-testid='view-record-btn'])[1]"


def _adapt_view_medical_analysis(
    actions: list[dict[str, Any]],
    seed: int,
    base_url: str,
    project_key: str,
) -> list[dict[str, Any]]:
    """VIEW_MEDICAL_ANALYSIS: no records API — use generic 'a' search and simplified click."""
    adapted = []
    for action in actions:
        a = dict(action)
        if a.get("type") == "TypeAction" and _get_testid(a) == "search-record-title":
            a["text"] = "a"
        elif a.get("type") == "ClickAction":
            a = _replace_xpath_containing(a, "view-record-btn", _VIEW_RECORD_BTN_XPATH)
        adapted.append(a)
    return adapted


def _adapt_search_medical_analysis(
    actions: list[dict[str, Any]],
    seed: int,
    base_url: str,
    project_key: str,
) -> list[dict[str, Any]]:
    """SEARCH_MEDICAL_ANALYSIS: no records API — use generic search terms."""
    adapted = []
    for action in actions:
        a = dict(action)
        if a.get("type") == "TypeAction":
            testid = _get_testid(a)
            if testid == "search-record-title":
                a["text"] = "a"
            elif testid == "search-record-doctor":
                a["text"] = "Dr."
        adapted.append(a)
    return adapted


def _adapt_view_doctor_education(
    actions: list[dict[str, Any]],
    seed: int,
    base_url: str,
    project_key: str,
) -> list[dict[str, Any]]:
    """VIEW_DOCTOR_EDUCATION: replace hardcoded doctor name in TypeAction and XPath."""
    rows = _fetch_rows(base_url, project_key, "doctors", seed)
    if not rows:
        return actions
    doctor_name = str(rows[0].get("name") or "").strip()
    adapted = []
    for action in actions:
        a = dict(action)
        if a.get("type") == "TypeAction" and _get_testid(a) == "doctor-name-search":
            a["text"] = doctor_name
        a = _replace_in_xpath(a, "Dr. Thomas Thomas", doctor_name)
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
    **{("autohealth", uc): _adapt_doctors_search_flow for uc in _DOCTORS_SEARCH_USE_CASES},
    **{
        ("autohealth", "SEARCH_APPOINTMENT"): _adapt_search_appointment,
        ("autohealth", "OPEN_APPOINTMENT_FORM"): _adapt_open_appointment_form,
        ("autohealth", "APPOINTMENT_BOOKED_SUCCESSFULLY"): _adapt_appointment_booked,
        ("autohealth", "REQUEST_QUICK_APPOINTMENT"): _adapt_request_quick_appointment,
        ("autohealth", "SEARCH_PRESCRIPTION"): _adapt_search_prescription,
        ("autohealth", "REFILL_PRESCRIPTION"): _adapt_refill_prescription,
        ("autohealth", "VIEW_PRESCRIPTION"): _adapt_view_prescription,
        ("autohealth", "VIEW_MEDICAL_ANALYSIS"): _adapt_view_medical_analysis,
        ("autohealth", "SEARCH_MEDICAL_ANALYSIS"): _adapt_search_medical_analysis,
        ("autohealth", "VIEW_DOCTOR_EDUCATION"): _adapt_view_doctor_education,
    },
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
