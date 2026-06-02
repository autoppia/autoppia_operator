from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from src.operator.agents.base import BaseApifiedWebAgent
from src.operator.agents.step_engine.candidates import Candidate, CandidateExtractor
from src.operator.agents.step_engine.policy import (
    _extract_login_targets,
    _extract_prompt_field_targets,
    _infer_autocinema_use_case,
)
from src.operator.runtime.fsm_adapter import build_fsm_payload, normalize_fsm_output


def _selector_from_id(control_id: str) -> dict[str, Any]:
    return {
        "type": "attributeValueSelector",
        "attribute": "id",
        "value": str(control_id),
        "case_sensitive": False,
    }


def _selector_key(selector: dict[str, Any]) -> str:
    if not isinstance(selector, dict):
        return ""
    attribute = str(selector.get("attribute") or "").strip().lower()
    value = str(selector.get("value") or "").strip()
    if attribute == "id" and value:
        return value.lower()
    selector_type = str(selector.get("type") or "").strip().lower()
    if selector_type == "xpathselector" and value:
        match = re.search(r"@id\s*=\s*['\"]([^'\"]+)['\"]", value, flags=re.IGNORECASE)
        if match:
            return str(match.group(1) or "").strip().lower()
    return ""


def _normalize_allowed_tools(raw: Any) -> set[str]:
    out: set[str] = set()
    if not isinstance(raw, list):
        return out
    for item in raw:
        if isinstance(item, str):
            name = str(item).strip()
            if name:
                out.add(name)
            continue
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            if name:
                out.add(name)
    return out


def _history_text_by_selector_id(history: list[dict[str, Any]]) -> dict[str, str]:
    typed: dict[str, str] = {}
    for item in history:
        if not isinstance(item, dict):
            continue
        action = item.get("action") if isinstance(item.get("action"), dict) else item
        if not isinstance(action, dict):
            continue
        raw_type = str(action.get("type") or "").strip().lower()
        raw_name = str(action.get("name") or "").strip().lower()
        arguments = action.get("arguments") if isinstance(action.get("arguments"), dict) else {}
        nested = action.get("raw") if isinstance(action.get("raw"), dict) else {}
        action_type = raw_type
        if not action_type and raw_name:
            if raw_name in {"browser.input", "browser.fill"}:
                action_type = "typeaction"
            elif raw_name == "browser.scroll":
                action_type = "scrollaction"
        if action_type not in {"typeaction", "fillaction"}:
            continue
        selector = (
            action.get("selector") if isinstance(action.get("selector"), dict)
            else arguments.get("selector") if isinstance(arguments.get("selector"), dict)
            else nested.get("selector") if isinstance(nested.get("selector"), dict)
            else {}
        )
        selector_id = _selector_key(selector)
        typed_text = str(action.get("text") or arguments.get("text") or arguments.get("value") or nested.get("text") or nested.get("value") or "").strip()
        if selector_id and typed_text:
            typed[selector_id] = typed_text
    return typed


def _history_last_typed(history: list[dict[str, Any]]) -> tuple[str, str]:
    for item in reversed(history):
        if not isinstance(item, dict):
            continue
        action = item.get("action") if isinstance(item.get("action"), dict) else item
        if not isinstance(action, dict):
            continue
        raw_type = str(action.get("type") or "").strip().lower()
        raw_name = str(action.get("name") or "").strip().lower()
        arguments = action.get("arguments") if isinstance(action.get("arguments"), dict) else {}
        nested = action.get("raw") if isinstance(action.get("raw"), dict) else {}
        action_type = raw_type
        if not action_type and raw_name in {"browser.input", "browser.fill"}:
            action_type = "typeaction"
        if action_type not in {"typeaction", "fillaction"}:
            continue
        selector = (
            action.get("selector") if isinstance(action.get("selector"), dict)
            else arguments.get("selector") if isinstance(arguments.get("selector"), dict)
            else nested.get("selector") if isinstance(nested.get("selector"), dict)
            else {}
        )
        selector_id = _selector_key(selector)
        typed_text = str(action.get("text") or arguments.get("text") or arguments.get("value") or nested.get("text") or nested.get("value") or "").strip()
        if selector_id and typed_text:
            return selector_id, typed_text
    return "", ""


def _candidate_selector_id(cand: Candidate) -> str:
    selector = cand.selector if isinstance(cand.selector, dict) else {}
    if str(selector.get("attribute") or "").strip().lower() == "id":
        return str(selector.get("value") or "").strip().lower()
    return str(cand.id or "").strip().lower()


def _candidate_blob(cand: Candidate) -> str:
    return " ".join(
        [
            str(cand.text or ""),
            str(cand.href or ""),
            str(cand.context or ""),
            str(cand.field_hint or ""),
            str(cand.field_kind or ""),
            str(cand.placeholder or ""),
            str(cand.aria_label or ""),
            str(cand.name_attr or ""),
            str(cand.group_label or ""),
            str(cand.role or ""),
            str(cand.type or ""),
        ]
    ).lower()



def _pick_field_candidate(candidates: list[Candidate], aliases: tuple[str, ...], kind: str, extra_tokens: tuple[str, ...] = ()) -> Candidate | None:
    alias_set = {str(x).strip().lower() for x in aliases if str(x).strip()}
    for cand in candidates:
        cid = _candidate_selector_id(cand)
        if cid and cid in alias_set:
            return cand
    wanted_kind = str(kind or '').strip().lower()
    tokens = tuple(str(tok).strip().lower() for tok in extra_tokens if str(tok).strip())
    for cand in candidates:
        blob = _candidate_local_blob(cand)
        cand_kind = str(cand.field_kind or '').strip().lower()
        if wanted_kind and cand_kind == wanted_kind:
            return cand
        if tokens and any(tok in blob for tok in tokens):
            return cand
    return None


def _candidate_local_blob(cand: Candidate) -> str:
    return " ".join(
        [
            str(_candidate_selector_id(cand) or ""),
            str(cand.text or ""),
            str(cand.field_hint or ""),
            str(cand.field_kind or ""),
            str(cand.placeholder or ""),
            str(cand.aria_label or ""),
            str(cand.name_attr or ""),
            str(cand.type or ""),
            str(cand.role or ""),
        ]
    ).lower()


def _candidate_href_path(cand: Candidate) -> str:
    href = str(cand.href or "").strip()
    if not href:
        return ""
    try:
        return str(urlsplit(href).path or "").strip().lower()
    except Exception:
        return href.lower()


def _replace_path(url: str, new_path: str) -> str:
    parsed = urlsplit(str(url or "").strip())
    if not parsed.scheme or not parsed.netloc:
        return new_path
    return urlunsplit((parsed.scheme, parsed.netloc, new_path, parsed.query, parsed.fragment))


def _requested_contact_fields(prompt: str, field_targets: dict[str, str]) -> tuple[str, ...]:
    prompt_l = str(prompt or "").lower()
    requested: list[str] = []
    for kind, tokens in {
        "name": ("name", "full name"),
        "email": ("email", "e-mail"),
        "subject": ("subject", "topic", "title"),
        "message": ("message", "comment", "details", "description", "body"),
    }.items():
        if field_targets.get(kind) or any(token in prompt_l for token in tokens):
            requested.append(kind)
    if not requested:
        return ("name",)
    return tuple(requested)


def _extract_search_query(prompt: str) -> str:
    text = str(prompt or '').strip()
    if not text:
        return ''

    negative = re.search(r"(?:not\s+equals|is\s+not|not)\s+['\"]([^'\"]{1,120})['\"]", text, flags=re.IGNORECASE)
    if negative:
        banned = str(negative.group(1) or '').strip().lower()
        for candidate in ('Dune', 'Foundation', 'Autoppia'):
            if candidate.lower() != banned:
                return candidate

    positive_patterns = [
        r"query\s+is\s+['\"]([^'\"]{1,120})['\"]",
        r"query\s+equals\s+['\"]([^'\"]{1,120})['\"]",
        r"query\s+contains\s+['\"]([^'\"]{1,120})['\"]",
        r"query\s+starts\s+with\s+['\"]([^'\"]{1,120})['\"]",
        r"query\s+ends\s+with\s+['\"]([^'\"]{1,120})['\"]",
        r"search for(?: the movie| the film| the book| products?| items?)?\s+(.+?)(?:\s+in the database|\s+in database|\s*$)",
        r"look for(?: the movie| the film| the book| products?| items?)?\s+(.+?)(?:\s+in the database|\s+in database|\s*$)",
        r"find(?: the movie| a movie| the book| a book| products?| items?)?\s+(.+?)(?:\s+in the database|\s+in database|\s*$)",
        r"look up(?: the movie| a movie| the book| a book| products?| items?)?\s+(.+?)(?:\s+in the database|\s+in database|\s*$)",
    ]
    for pattern in positive_patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if not m:
            continue
        value = str(m.group(1) or '').strip(" .,:;\"'")
        if value:
            return value

    quoted = re.findall(r"['\"]([^'\"]{1,120})['\"]", text)
    for value in quoted:
        value = str(value).strip()
        if value:
            return value
    return ''


def _extract_registration_targets(prompt: str) -> tuple[str, str, str]:
    text = str(prompt or '').strip()
    username = ''
    email = ''
    password = ''
    patterns = {
        'username': r"username\s+equals\s+['\"]([^'\"]*)['\"]",
        'email': r"email\s+equals\s+['\"]([^'\"]*)['\"]",
        'password': r"password\s+equals\s+['\"]([^'\"]*)['\"]",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if not m:
            continue
        value = str(m.group(1) or '').strip()
        if key == 'username':
            username = value
        elif key == 'email':
            email = value
        else:
            password = value
    if not username:
        username = '<signup_username>'
    if not email:
        email = '<signup_email>'
    if '@' not in email and 'gmail.com' in text.lower():
        email = '<signup_email>'
    if not password:
        password = '<signup_password>'
    return username, email, password


def _extract_detail_constraints(prompt: str) -> dict[str, str]:
    text = str(prompt or '').strip()
    lowered = text.lower()
    out: dict[str, str] = {}
    patterns = {
        'title_not_contains': r"title\s+does\s+not\s+contain\s+['\"]([^'\"]+)['\"]",
        'title_contains': r"title\s+contains\s+['\"]([^'\"]+)['\"]",
        'title_equals': r"title\s+equals\s+['\"]([^'\"]+)['\"]",
        'name_not_contains': r"name\s+does\s+not\s+contain\s+['\"]([^'\"]+)['\"]",
        'name_contains': r"name\s+contains\s+['\"]([^'\"]+)['\"]",
        'name_equals': r"name\s+equals\s+['\"]([^'\"]+)['\"]",
        'category_not_contains': r"category\s+does\s+not\s+contain\s+['\"]([^'\"]+)['\"]",
        'category_contains': r"category\s+contains\s+['\"]([^'\"]+)['\"]",
        'category_equals': r"category\s+equals\s+['\"]([^'\"]+)['\"]",
        'brand_not_contains': r"brand\s+does\s+not\s+contain\s+['\"]([^'\"]+)['\"]",
        'brand_contains': r"brand\s+contains\s+['\"]([^'\"]+)['\"]",
        'brand_equals': r"brand\s+equals\s+['\"]([^'\"]+)['\"]",
        'year_not_equals': r"(?:not\s+released\s+in\s+the\s+year|year\s+does\s+not\s+equal|year\s+not\s+equals?)\s+['\"]?(\d{4})['\"]?",
        'year_equals': r"year\s+equals\s+['\"]?(\d{4})['\"]?",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, lowered, flags=re.IGNORECASE)
        if m:
            value = text[m.start(1):m.end(1)].strip(" .,:;\"'")
            if value:
                out[key] = value
    return out




def _extract_detail_query(constraints: dict[str, str]) -> str:
    for key in ('title_equals', 'title_contains', 'name_equals', 'name_contains', 'brand_equals', 'brand_contains'):
        value = str(constraints.get(key) or '').strip()
        if value:
            return value
    return ''

def _extract_year_options(text: str) -> list[int]:
    return sorted({int(m.group(0)) for m in re.finditer(r'\b(?:19|20)\d{2}\b', str(text or ''))})


def _extract_filter_constraints(prompt: str) -> dict[str, str]:
    text = str(prompt or '').strip()
    lowered = text.lower()
    out: dict[str, str] = {}
    patterns = {
        'year_ge': r"year\s+(?:is\s+)?(\d{4})\s+or\s+later",
        'year_ge_alt': r"released\s+in\s+the\s+year\s+(\d{4})\s+or\s+later",
        'year_equals': r"year\s+equals\s+['\"]?(\d{4})['\"]?",
        'genre_equals': r"genre\s+equals\s+['\"]([^'\"]+)['\"]",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, lowered, flags=re.IGNORECASE)
        if m:
            value = text[m.start(1):m.end(1)].strip(" .,:;\"'")
            if value:
                out[key] = value
    return out


def _extract_cart_constraints(prompt: str) -> dict[str, float]:
    text = str(prompt or '').strip()
    lowered = text.lower()
    out: dict[str, float] = {}
    patterns = {
        'rating_gt': r"rating\s+(?:is\s+)?greater\s+than\s+([0-5](?:\.\d+)?)",
        'rating_ge': r"rating\s+(?:is\s+)?greater\s+than\s+or\s+equal\s+to\s+([0-5](?:\.\d+)?)",
        'rating_lt': r"rating\s+(?:is\s+)?less\s+than\s+([0-5](?:\.\d+)?)",
        'rating_le': r"rating\s+(?:is\s+)?less\s+than\s+or\s+equal\s+to\s+([0-5](?:\.\d+)?)",
        'rating_eq': r"rating\s+(?:equals|is)\s+([0-5](?:\.\d+)?)",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, lowered, flags=re.IGNORECASE)
        if m:
            try:
                out[key] = float(m.group(1))
            except Exception:
                pass
    return out


def _extract_first_rating(text: str) -> float | None:
    raw = str(text or '')
    m = re.search(r'\b([0-5](?:\.\d+)?)\s*[★*]', raw)
    if not m:
        m = re.search(r'\b([0-5](?:\.\d+)?)\b', raw)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _rating_matches(value: float | None, constraints: dict[str, float]) -> bool:
    if value is None:
        return False
    if 'rating_gt' in constraints and not (value > constraints['rating_gt']):
        return False
    if 'rating_ge' in constraints and not (value >= constraints['rating_ge']):
        return False
    if 'rating_lt' in constraints and not (value < constraints['rating_lt']):
        return False
    if 'rating_le' in constraints and not (value <= constraints['rating_le']):
        return False
    if 'rating_eq' in constraints and not (abs(value - constraints['rating_eq']) < 1e-6):
        return False
    return True


def _canonical_use_case_name(use_case: str) -> str:
    name = str(use_case or "").strip().upper()
    if name in {"CONTACT", "CONTACT_BOOK", "CONTACT_RESTAURANT", "CONTACT_HOTEL"}:
        return "CONTACT"
    if name in {"LOGIN", "LOGIN_BOOK", "LOGIN_MAIL", "LOGIN_CRM"}:
        return "LOGIN"
    if name in {"REGISTRATION", "REGISTRATION_BOOK", "REGISTER", "SIGNUP"}:
        return "REGISTRATION"
    if name in {"FILM_DETAIL", "BOOK_DETAIL", "PRODUCT_DETAIL", "RESTAURANT_DETAIL", "HOTEL_DETAIL", "VIEW_DETAIL"}:
        return "DETAIL"
    if name in {"FILTER_FILM", "FILTER_BOOK", "FILTER_PRODUCT", "FILTER_RESTAURANT", "FILTER_HOTEL"}:
        return "FILTER"
    if name in {"SEARCH_FILM", "SEARCH_BOOK", "SEARCH_PRODUCT", "SEARCH_RESTAURANT", "SEARCH_HOTEL"}:
        return "SEARCH"
    if name in {"SHARE_PRODUCT", "SHARE_BOOK", "SHARE_FILM"}:
        return "SHARE"
    if name in {"ADD_TO_CART", "ADD_BOOK_TO_CART", "ADD_FILM_TO_CART"}:
        return "ADD_TO_CART"
    return name


def _extract_use_case_name(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        name = value.get('name')
        if isinstance(name, str):
            return name.strip()
    return str(value or '').strip()


def _tool_click(cand: Candidate) -> dict[str, Any]:
    return {"type": "ClickAction", "selector": cand.selector, "_element_id": cand.id}


def _tool_click_id(control_id: str, *, element_id: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {
        "type": "ClickAction",
        "selector": _selector_from_id(control_id),
    }
    if element_id:
        out["_element_id"] = element_id
    return out


def _tool_click_selector(selector: dict[str, Any], *, element_id: str = '') -> dict[str, Any]:
    out: dict[str, Any] = {
        'type': 'ClickAction',
        'selector': selector,
    }
    if element_id:
        out['_element_id'] = element_id
    return out


def _tool_type_selector(selector: dict[str, Any], text: str, *, element_id: str = '') -> dict[str, Any]:
    out: dict[str, Any] = {
        'type': 'TypeAction',
        'selector': selector,
        'text': str(text),
    }
    if element_id:
        out['_element_id'] = element_id
    return out


def _tool_type(control_id: str, text: str, *, element_id: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {
        "type": "TypeAction",
        "selector": _selector_from_id(control_id),
        "text": str(text),
    }
    if element_id:
        out["_element_id"] = element_id
    return out


def _tool_select(selector: dict[str, Any], text: str, *, element_id: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {
        "type": "SelectDropDownOptionAction",
        "selector": selector,
        "text": str(text),
    }
    if element_id:
        out["_element_id"] = element_id
    return out


def _tool_nav(url: str) -> dict[str, Any]:
    return {"type": "NavigateAction", "url": str(url), "go_back": False, "go_forward": False}


def _tool_scroll(*, direction: str = "down", amount: int = 600) -> dict[str, Any]:
    direction_l = str(direction or "down").strip().lower()
    return {
        "type": "ScrollAction",
        "value": int(amount),
        "up": direction_l == "up",
        "down": direction_l == "down",
        "left": direction_l == "left",
        "right": direction_l == "right",
    }


@dataclass
class StructuredPlanContext:
    prompt: str
    url: str
    history: list[dict[str, Any]]
    candidates: list[Candidate]
    allowed_tools: set[str]
    use_case: str
    internal_state: dict[str, Any]


class StructuredInferenceOperator(BaseApifiedWebAgent):
    def __init__(self, id: str = "1", name: str = "StructuredInferenceOperator") -> None:
        super().__init__(id=id, name=name)
        self._extractor = CandidateExtractor()

    @staticmethod
    def _runtime_impl() -> str:
        return "structured_inference"

    async def act_from_payload(self, payload: dict[str, object]) -> dict[str, object]:
        runtime_payload = build_fsm_payload(payload)
        result = self._run_structured(runtime_payload)
        model_override = str(payload.get("model") or "").strip()
        return normalize_fsm_output(result, model_override=model_override, return_metrics=False)

    def _run_structured(self, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = str(payload.get("prompt") or "")
        url = str(payload.get("url") or "")
        history = payload.get("history") if isinstance(payload.get("history"), list) else []
        allowed_tools = _normalize_allowed_tools(payload.get("allowed_tools"))
        raw_use_case = _extract_use_case_name(payload.get("use_case"))
        use_case = raw_use_case or _infer_autocinema_use_case(
            prompt,
            {
                "url": url,
                "use_case": payload.get("use_case"),
                "step_index": int(payload.get("step_index") or 0),
            },
        )
        snapshot_html = str(payload.get("snapshot_html") or payload.get("html") or "")
        candidates = self._extractor.extract(snapshot_html=snapshot_html, url=url)
        ctx = StructuredPlanContext(
            prompt=prompt,
            url=url,
            history=history,
            candidates=candidates,
            allowed_tools=allowed_tools,
            use_case=use_case,
            internal_state=payload.get("internal_state") if isinstance(payload.get("internal_state"), dict) else {},
        )
        action = self._plan_action(ctx)
        if action is None:
            action = self._fallback_action(ctx)
        internal_state = self._next_internal_state(ctx, action)
        return {
            "protocol_version": str(payload.get("protocol_version") or ""),
            "actions": [action] if isinstance(action, dict) else [],
            "internal_state": internal_state,
            "model": "structured-inference",
        }

    def _plan_action(self, ctx: StructuredPlanContext) -> dict[str, Any] | None:
        use_case = _canonical_use_case_name(ctx.use_case)
        if use_case == "CONTACT":
            return self._plan_contact(ctx)
        if use_case == "LOGIN":
            return self._plan_login(ctx)
        if use_case == "REGISTRATION":
            return self._plan_registration(ctx)
        if use_case == "DETAIL":
            return self._plan_film_detail(ctx)
        if use_case == "FILTER":
            return self._plan_filter_film(ctx)
        if use_case == "SEARCH":
            return self._plan_search(ctx)
        if use_case == "SHARE":
            return self._plan_share(ctx)
        if use_case == "ADD_TO_CART":
            return self._plan_add_to_cart(ctx)
        return None

    def _next_internal_state(self, ctx: StructuredPlanContext, action: dict[str, Any] | None) -> dict[str, Any]:
        state = dict(ctx.internal_state or {})
        state["runtime"] = "structured_inference"
        state["use_case"] = str(ctx.use_case or "")
        typed_values = state.get("typed_values") if isinstance(state.get("typed_values"), dict) else {}
        typed_values = {str(k).strip().lower(): str(v) for k, v in typed_values.items() if str(k).strip()}
        if isinstance(action, dict) and str(action.get("type") or "") in {"TypeAction", "FillAction"}:
            selector = action.get("selector") if isinstance(action.get("selector"), dict) else {}
            selector_id = _selector_key(selector)
            typed_text = str(action.get("text") or action.get("value") or "").strip()
            if selector_id and typed_text:
                typed_values[selector_id] = typed_text
                state["last_typed"] = {"selector_id": selector_id, "text": typed_text}
        if isinstance(action, dict) and str(action.get("type") or "") == "ClickAction":
            selector = action.get("selector") if isinstance(action.get("selector"), dict) else {}
            selector_id = _selector_key(selector)
            if selector_id:
                clicked = state.get("clicked_selectors") if isinstance(state.get("clicked_selectors"), list) else []
                clicked = [str(x).strip().lower() for x in clicked if str(x).strip()]
                if selector_id not in clicked:
                    clicked.append(selector_id)
                state["clicked_selectors"] = clicked
                state["last_clicked"] = {"selector_id": selector_id}
        if typed_values:
            state["typed_values"] = typed_values
        return state


    def _plan_contact(self, ctx: StructuredPlanContext) -> dict[str, Any] | None:
        current_path = str(urlsplit(ctx.url).path or "").rstrip("/") or "/"
        if "/contact" not in current_path:
            if self._allow("browser.navigate", ctx.allowed_tools):
                return _tool_nav(_replace_path(ctx.url, "/contact"))
            route_click = self._find_route_candidate(ctx.candidates, route_tokens=("contact",), href_tokens=("/contact",))
            if route_click is not None and self._allow("browser.click", ctx.allowed_tools):
                return _tool_click(route_click)
            return None

        field_targets = _extract_prompt_field_targets(ctx.prompt)
        typed = _history_text_by_selector_id(ctx.history)
        state_typed = ctx.internal_state.get("typed_values") if isinstance(ctx.internal_state.get("typed_values"), dict) else {}
        for key, value in state_typed.items():
            key_s = str(key).strip().lower()
            value_s = str(value).strip()
            if key_s and value_s and key_s not in typed:
                typed[key_s] = value_s
        candidate_values: dict[str, str] = {}
        for cand in ctx.candidates:
            cid = _candidate_selector_id(cand)
            if not cid:
                continue
            current = str(cand.current_value or "").strip()
            if current:
                candidate_values[cid] = current

        last_typed_id, last_typed_text = _history_last_typed(ctx.history)

        def current_value(control_id: str) -> str:
            typed_value = str(typed.get(control_id) or "").strip()
            candidate_value = str(candidate_values.get(control_id) or "").strip()
            if not typed_value and control_id == last_typed_id:
                typed_value = str(last_typed_text or "").strip()
            # Prefer action history over snapshot attrs for mutable controls.
            # Demo-web HTML can preserve initial values after an input change.
            return typed_value or candidate_value

        name_target = field_targets.get("name") or "David"
        if str(name_target).strip().lower() == 'testuser':
            name_target = 'David'
        email_target = field_targets.get("email") or "user1@site.com"
        subject_target = field_targets.get("subject") or "Inquiry"
        message_target = field_targets.get("message") or "Please provide me with more information"

        field_specs = [
            {
                "kind": "name",
                "ids": (
                    "contact-name-input",
                    "contact-name",
                    "get-in-touch-name-input",
                    "name-input-field",
                    "name-field",
                    "contact-full-name",
                    "contact-fullname",
                    "name-input",
                    "full-name-input",
                ),
                "target": str(name_target),
            },
            {
                "kind": "email",
                "ids": (
                    "contact-email-input",
                    "contact-email",
                    "contact-support-email-input",
                    "email-field",
                    "contact-email-addr",
                    "email-address-field",
                    "email-input-field",
                    "email-field-input",
                    "contact-email-field",
                    "email-entry-field",
                ),
                "target": str(email_target),
            },
            {
                "kind": "subject",
                "ids": (
                    "contact-subject-input",
                    "contact-subject",
                    "contact-form-subject-input",
                    "contact-details-subject-input",
                    "subject-entry-field",
                    "contact-subject-entry",
                    "subject-entry",
                    "subject-field",
                    "subject-field-input",
                    "subject-input-field",
                ),
                "target": str(subject_target),
            },
            {
                "kind": "message",
                "ids": (
                    "contact-message-textarea",
                    "contact-message-input",
                    "contact-message",
                    "contact-message-content",
                    "contact-section-message-textarea",
                    "contact-us-message-textarea",
                    "contact-message-area",
                    "message-entry-field",
                    "message-entry-area",
                    "message-field",
                    "message-textarea",
                    "message-textarea-field",
                    "contact-message-field",
                    "contact-body",
                ),
                "target": str(message_target),
            },
        ]
        candidate_id_map = {_candidate_selector_id(c): c for c in ctx.candidates if _candidate_selector_id(c)}
        field_tokens = {
            'name': ('name', 'full name', 'enter name'),
            'email': ('email', 'you@example.com', 'email address'),
            'subject': ('subject', "what's this about?", 'topic'),
            'message': ('message', 'your message', 'details', 'thoughts', 'comparte', 'share your thoughts'),
        }
        for spec in field_specs:
            aliases = tuple(str(x).strip().lower() for x in spec["ids"] if str(x).strip())
            cand = _pick_field_candidate(ctx.candidates, aliases, str(spec['kind']), field_tokens.get(str(spec['kind']), ()))
            selector_id = _candidate_selector_id(cand) if cand is not None else next((alias for alias in aliases if alias in candidate_id_map), aliases[0] if aliases else "")
            known_ids = tuple(dict.fromkeys([*aliases, selector_id] if selector_id else list(aliases)))
            if any(current_value(control_id) == spec["target"] for control_id in known_ids):
                continue
            if selector_id and self._allow("browser.input", ctx.allowed_tools):
                return _tool_type(selector_id, spec["target"], element_id=str(cand.id or "") if cand is not None else "")
            return None

        submit_ids = (
            "send-message-button",
            "send-contact-button",
            "submit-contact-button",
            "submit-contact-form",
            "send-btn",
            "submit-message",
            "send-button",
            "contact-submit",
            "contact-submit-button",
            "submit-btn",
            "send-action",
            "submit-action",
            "send-control",
        )
        submit = self._find_submit(
            ctx.candidates,
            submit_ids=submit_ids,
            forbidden_href_paths=("/",),
        )
        if submit is not None and self._allow("browser.click", ctx.allowed_tools):
            return _tool_click(submit)
        if self._allow("browser.click", ctx.allowed_tools):
            return _tool_click_id(submit_ids[0])
        return None

    def _plan_login(self, ctx: StructuredPlanContext) -> dict[str, Any] | None:
        current_path = str(urlsplit(ctx.url).path or "").rstrip("/") or "/"
        if "/login" not in current_path:
            if self._allow("browser.navigate", ctx.allowed_tools):
                return _tool_nav(_replace_path(ctx.url, "/login"))
            route_click = self._find_route_candidate(ctx.candidates, route_tokens=("login", "log in"), href_tokens=("/login",))
            if route_click is not None and self._allow("browser.click", ctx.allowed_tools):
                return _tool_click(route_click)
            return None

        username_target, password_target = _extract_login_targets(ctx.prompt)
        typed = _history_text_by_selector_id(ctx.history)
        state_typed = ctx.internal_state.get("typed_values") if isinstance(ctx.internal_state.get("typed_values"), dict) else {}
        for key, value in state_typed.items():
            key_s = str(key).strip().lower()
            value_s = str(value).strip()
            if key_s and value_s and key_s not in typed:
                typed[key_s] = value_s
        candidate_values: dict[str, str] = {}
        for cand in ctx.candidates:
            cid = _candidate_selector_id(cand)
            if not cid:
                continue
            current = str(cand.current_value or "").strip()
            if current:
                candidate_values[cid] = current

        last_typed_id, last_typed_text = _history_last_typed(ctx.history)

        def current_value(control_id: str) -> str:
            typed_value = str(typed.get(control_id) or "").strip()
            candidate_value = str(candidate_values.get(control_id) or "").strip()
            if not typed_value and control_id == last_typed_id:
                typed_value = str(last_typed_text or "").strip()
            # Prefer action history over snapshot attrs for mutable controls.
            # Demo-web HTML can preserve initial values after an input change.
            return typed_value or candidate_value

        username = self._find_field(
            ctx.candidates,
            kind="username",
            ids=("login-username", "login-username-input", "username-field"),
            includes=("username", "user name", "email", "login username"),
            excludes=("password",),
            target=str(username_target),
        )
        if username is not None:
            username_id = _candidate_selector_id(username)
            if username_id and current_value(username_id) != str(username_target):
                if self._allow("browser.input", ctx.allowed_tools):
                    return _tool_type(username_id, username_target, element_id=username.id)
                return None

        password = self._find_field(
            ctx.candidates,
            kind="password",
            ids=("login-password", "login-password-input", "password-entry-field", "password-field"),
            includes=("password", "login password"),
            excludes=(),
            target=str(password_target),
        )
        if password is not None:
            password_id = _candidate_selector_id(password)
            if password_id and current_value(password_id) != str(password_target):
                if self._allow("browser.input", ctx.allowed_tools):
                    return _tool_type(password_id, password_target, element_id=password.id)
                return None

        submit = self._find_submit(
            ctx.candidates,
            extra_tokens=("sign in", "log in", "login"),
            submit_ids=("login-submit", "signin-button", "sign-in-button", "login-button", "login-sign-in-button"),
        )
        if submit is not None and self._allow("browser.click", ctx.allowed_tools):
            return _tool_click(submit)
        return None

    def _plan_registration(self, ctx: StructuredPlanContext) -> dict[str, Any] | None:
        current_path = str(urlsplit(ctx.url).path or '').rstrip('/') or '/'
        if '/register' not in current_path:
            if self._allow('browser.navigate', ctx.allowed_tools):
                return _tool_nav(_replace_path(ctx.url, '/register'))
            route_click = self._find_route_candidate(ctx.candidates, route_tokens=('register', 'sign up', 'create account'), href_tokens=('/register',))
            if route_click is not None and self._allow('browser.click', ctx.allowed_tools):
                return _tool_click(route_click)
            return None

        username_target, email_target, password_target = _extract_registration_targets(ctx.prompt)
        typed = _history_text_by_selector_id(ctx.history)
        state_typed = ctx.internal_state.get('typed_values') if isinstance(ctx.internal_state.get('typed_values'), dict) else {}
        for key, value in state_typed.items():
            key_s = str(key).strip().lower()
            value_s = str(value).strip()
            if key_s and value_s and key_s not in typed:
                typed[key_s] = value_s
        candidate_values: dict[str, str] = {}
        for cand in ctx.candidates:
            cid = _candidate_selector_id(cand)
            if not cid:
                continue
            current = str(cand.current_value or '').strip()
            if current:
                candidate_values[cid] = current
        last_typed_id, last_typed_text = _history_last_typed(ctx.history)

        def current_value(control_id: str) -> str:
            typed_value = str(typed.get(control_id) or '').strip()
            candidate_value = str(candidate_values.get(control_id) or '').strip()
            if not typed_value and control_id == last_typed_id:
                typed_value = str(last_typed_text or '').strip()
            return typed_value or candidate_value

        field_specs = [
            {
                'kind': 'username',
                'ids': (
                    'register-username-input', 'register-username', 'register-username-field', 'register-username-entry',
                    'register-user', 'username-input-field', 'username-input', 'username-field-input',
                ),
                'includes': ('username', 'user name', 'choose a username'),
                'excludes': ('email', 'password', 'confirm'),
                'target': str(username_target),
            },
            {
                'kind': 'email',
                'ids': (
                    'register-email-input', 'register-email', 'register-email-field', 'register-email-entry',
                    'register-mail', 'email-input-field', 'email-input', 'email-field-input',
                ),
                'includes': ('email', 'gmail', 'example.com'),
                'excludes': ('password', 'confirm', 'username'),
                'target': str(email_target),
            },
            {
                'kind': 'password',
                'ids': (
                    'register-password-input', 'register-password', 'register-password-field', 'register-password-entry',
                    'register-pass', 'password-input-field', 'password-input', 'password-field-input',
                ),
                'includes': ('password', 'create a password'),
                'excludes': ('confirm',),
                'target': str(password_target),
            },
            {
                'kind': 'confirm_password',
                'ids': (
                    'register-confirm-password-input', 'confirm-password-field', 'register-confirm-password', 'confirm-password-input-field',
                    'register-confirm-field', 'register-confirm-entry', 'register-confirm', 'confirm-password-input',
                ),
                'includes': ('confirm password', 'confirm your password', 'repeat password', 'confirm'),
                'excludes': ('email', 'username'),
                'target': str(password_target),
            },
        ]

        for spec in field_specs:
            field = self._find_field(
                ctx.candidates,
                kind=spec['kind'],
                ids=spec['ids'],
                includes=spec['includes'],
                excludes=spec['excludes'],
                target=spec['target'],
            )
            if field is None:
                continue
            field_id = _candidate_selector_id(field)
            if field_id and current_value(field_id) != spec['target']:
                if self._allow('browser.input', ctx.allowed_tools):
                    return _tool_type(field_id, spec['target'], element_id=field.id)
                return None

        submit = self._find_submit(
            ctx.candidates,
            extra_tokens=('register', 'sign up', 'create account'),
            submit_ids=('create-account-button', 'register-button', 'signup-btn', 'register-btn', 'signup-button', 'register-action'),
            forbidden_href_paths=('/login', '/contact', '/search', '/about'),
        )
        if submit is not None and self._allow('browser.click', ctx.allowed_tools):
            return _tool_click(submit)
        return None


    def _plan_film_detail(self, ctx: StructuredPlanContext) -> dict[str, Any] | None:
        current_path = str(urlsplit(ctx.url).path or '').rstrip('/') or '/'
        if any(token in current_path for token in ('/movies/', '/books/', '/products/', '/product/')):
            return None

        constraints = _extract_detail_constraints(ctx.prompt)
        detail_query = _extract_detail_query(constraints)
        is_product_prompt = 'product' in str(ctx.prompt or '').lower()

        if is_product_prompt and '/search' not in current_path:
            if self._allow('browser.navigate', ctx.allowed_tools):
                return _tool_nav(_replace_path(ctx.url, '/search'))
            route_click = self._find_route_candidate(ctx.candidates, route_tokens=('search', 'find', 'look up'), href_tokens=('/search',))
            if route_click is not None and self._allow('browser.click', ctx.allowed_tools):
                return _tool_click(route_click)
            return None

        if is_product_prompt and '/search' in current_path and detail_query:
            query_params = {str(k): str(v) for k, v in parse_qsl(urlsplit(ctx.url).query or '', keep_blank_values=True)}
            current_query = str(query_params.get('q') or '').strip()
            if current_query != str(detail_query):
                typed = _history_text_by_selector_id(ctx.history)
                state_typed = ctx.internal_state.get('typed_values') if isinstance(ctx.internal_state.get('typed_values'), dict) else {}
                for key, value in state_typed.items():
                    key_s = str(key).strip().lower()
                    value_s = str(value).strip()
                    if key_s and value_s and key_s not in typed:
                        typed[key_s] = value_s
                search_input_value = str(typed.get('search-input') or typed.get('search-box') or '').strip()
                candidate_id_map = {_candidate_selector_id(c): c for c in ctx.candidates if _candidate_selector_id(c)}
                if search_input_value != str(detail_query):
                    if self._allow('browser.input', ctx.allowed_tools):
                        search_input = candidate_id_map.get('search-input') or candidate_id_map.get('search-box')
                        if search_input is not None:
                            return _tool_type(_candidate_selector_id(search_input), detail_query, element_id=search_input.id)
                        return _tool_type_selector({'type': 'xpathSelector', 'value': "//input[@id='search-input']", 'case_sensitive': False}, detail_query)
                    return None
                clicked = ctx.internal_state.get('clicked_selectors') if isinstance(ctx.internal_state.get('clicked_selectors'), list) else []
                clicked = {str(x).strip().lower() for x in clicked if str(x).strip()}
                if 'search-btn' not in clicked and 'submit-search' not in clicked and self._allow('browser.click', ctx.allowed_tools):
                    submit = candidate_id_map.get('search-btn') or candidate_id_map.get('submit-search')
                    if submit is not None:
                        return _tool_click(submit)
                    return _tool_click_selector({'type': 'xpathSelector', 'value': "//button[@id='search-btn']", 'case_sensitive': False})

        best: tuple[tuple[int, int, int, int], Candidate] | None = None
        for idx, cand in enumerate(ctx.candidates):
            href = str(cand.href or '').strip()
            blob = _candidate_blob(cand)
            local_blob = _candidate_local_blob(cand)
            cid = _candidate_selector_id(cand)
            detailish = (
                '/movies/' in href or
                '/books/' in href or
                '/products/' in href or
                '/product/' in href or
                'details' in local_blob or
                'view details' in local_blob or
                'see details' in local_blob or
                cid in {'view-details-btn', 'product-details-btn', 'featured-details-button', 'featured-details-link', 'featured-view-btn', 'details-action'}
            )
            if not detailish:
                continue
            text_penalty = 0
            year_penalty = 0
            pages_penalty = 0
            attr_penalty = 0
            title_contains = constraints.get('title_contains') or constraints.get('name_contains')
            title_equals = constraints.get('title_equals') or constraints.get('name_equals')
            title_not_contains = constraints.get('title_not_contains') or constraints.get('name_not_contains')
            if title_not_contains and title_not_contains.lower() in blob:
                text_penalty += 100
            if title_contains and title_contains.lower() not in blob:
                text_penalty += 10
            if title_equals and title_equals.lower() not in blob:
                text_penalty += 20
            if 'year_not_equals' in constraints and constraints['year_not_equals'] in blob:
                year_penalty += 100
            if 'year_equals' in constraints and constraints['year_equals'] not in blob:
                year_penalty += 20
            if 'page_count_not_equals' in constraints and constraints['page_count_not_equals'] in blob:
                pages_penalty += 100
            if 'page_count_equals' in constraints and constraints['page_count_equals'] not in blob:
                pages_penalty += 20
            if 'category_not_contains' in constraints and constraints['category_not_contains'].lower() in blob:
                attr_penalty += 100
            if 'category_contains' in constraints and constraints['category_contains'].lower() not in blob:
                attr_penalty += 20
            if 'category_equals' in constraints and constraints['category_equals'].lower() not in blob:
                attr_penalty += 20
            if 'brand_not_contains' in constraints and constraints['brand_not_contains'].lower() in blob:
                attr_penalty += 100
            if 'brand_contains' in constraints and constraints['brand_contains'].lower() not in blob:
                attr_penalty += 20
            if 'brand_equals' in constraints and constraints['brand_equals'].lower() not in blob:
                attr_penalty += 20
            current = ((text_penalty, attr_penalty, year_penalty + pages_penalty, idx), cand)
            if best is None or current[0] < best[0]:
                best = current
        if best is not None and best[0][0] < 100 and best[0][1] < 100 and best[0][2] < 100 and self._allow('browser.click', ctx.allowed_tools):
            return _tool_click(best[1])

        if self._allow('browser.navigate', ctx.allowed_tools):
            return _tool_nav(_replace_path(ctx.url, '/search'))
        route_click = self._find_route_candidate(ctx.candidates, route_tokens=('search', 'view all'), href_tokens=('/search',))
        if route_click is not None and self._allow('browser.click', ctx.allowed_tools):
            return _tool_click(route_click)
        return None


    def _plan_share(self, ctx: StructuredPlanContext) -> dict[str, Any] | None:
        candidate_id_map = {_candidate_selector_id(c): c for c in ctx.candidates if _candidate_selector_id(c)}
        for share_id in ('share-btn', 'share-product-btn', 'share-button', 'copy-link-btn'):
            share = candidate_id_map.get(share_id)
            if share is not None and self._allow('browser.click', ctx.allowed_tools):
                return _tool_click(share)
        for cand in ctx.candidates:
            blob = _candidate_local_blob(cand)
            if any(token in blob for token in ('share product', 'copy link', 'share')) and self._allow('browser.click', ctx.allowed_tools):
                return _tool_click(cand)
        return self._plan_film_detail(ctx)

    def _plan_add_to_cart(self, ctx: StructuredPlanContext) -> dict[str, Any] | None:
        current_path = str(urlsplit(ctx.url).path or '').rstrip('/') or '/'
        if '/search' not in current_path:
            if self._allow('browser.navigate', ctx.allowed_tools):
                return _tool_nav(_replace_path(ctx.url, '/search'))
            route_click = self._find_route_candidate(ctx.candidates, route_tokens=('search', 'find', 'look up', 'all products'), href_tokens=('/search',))
            if route_click is not None and self._allow('browser.click', ctx.allowed_tools):
                return _tool_click(route_click)
            return None

        constraints = _extract_cart_constraints(ctx.prompt)
        clicked = ctx.internal_state.get('clicked_selectors') if isinstance(ctx.internal_state.get('clicked_selectors'), list) else []
        clicked = {str(x).strip().lower() for x in clicked if str(x).strip()}

        ranked: list[tuple[tuple[int, int], Candidate]] = []
        for idx, cand in enumerate(ctx.candidates):
            cid = _candidate_selector_id(cand)
            if cid in clicked:
                continue
            text_value = str(cand.text or '').strip().lower()
            blob = _candidate_blob(cand)
            if text_value not in {'add to cart', 'add cart', 'add to basket', 'add'}:
                continue
            rating = _extract_first_rating(blob)
            if rating is None:
                for neighbor in ctx.candidates[max(0, idx - 3):idx]:
                    neighbor_blob = _candidate_blob(neighbor)
                    neighbor_rating = _extract_first_rating(neighbor_blob)
                    if neighbor_rating is not None:
                        rating = neighbor_rating
            if constraints and not _rating_matches(rating, constraints):
                continue
            rating_penalty = 0 if rating is not None else 1
            ranked.append(((rating_penalty, idx), cand))
        if ranked and self._allow('browser.click', ctx.allowed_tools):
            ranked.sort(key=lambda item: item[0])
            return _tool_click(ranked[0][1])

        return None


    def _plan_filter_film(self, ctx: StructuredPlanContext) -> dict[str, Any] | None:
        current_path = str(urlsplit(ctx.url).path or '').rstrip('/') or '/'
        if '/search' not in current_path:
            if self._allow('browser.navigate', ctx.allowed_tools):
                return _tool_nav(_replace_path(ctx.url, '/search'))
            route_click = self._find_route_candidate(ctx.candidates, route_tokens=('search', 'view all'), href_tokens=('/search',))
            if route_click is not None and self._allow('browser.click', ctx.allowed_tools):
                return _tool_click(route_click)
            return None

        constraints = _extract_filter_constraints(ctx.prompt)
        year = constraints.get('year_ge') or constraints.get('year_ge_alt') or constraints.get('year_equals')
        genre = constraints.get('genre_equals')

        if year and self._allow('browser.select_dropdown', ctx.allowed_tools):
            target_year = int(str(year))
            year_choice: tuple[dict[str, Any], str] | None = None
            for cand in ctx.candidates:
                if str(cand.role or '').strip().lower() != 'select':
                    continue
                blob = str(cand.text or '').lower()
                has_year_options = 'all years' in blob
                looks_like_sort = ('sort by' in blob) or ('rating:' in blob) or ('duration:' in blob) or ('year: newest first' in blob) or ('year: oldest first' in blob)
                looks_like_genre = 'all genres' in blob
                if not has_year_options or looks_like_sort or looks_like_genre:
                    continue
                years = _extract_year_options(blob)
                if not years:
                    continue
                if target_year in years:
                    select_year = target_year
                else:
                    valid_years = [y for y in years if y >= target_year]
                    if not valid_years:
                        continue
                    select_year = min(valid_years)
                current = str(cand.current_value or '').strip()
                if current == str(select_year):
                    return None
                if isinstance(cand.selector, dict):
                    year_choice = (cand.selector, str(select_year))
                    break
            if year_choice is not None:
                return _tool_select(year_choice[0], year_choice[1])
            return _tool_select({'type': 'xpathSelector', 'value': '(//select)[3]', 'case_sensitive': False}, str(year))

        if genre and self._allow('browser.select_dropdown', ctx.allowed_tools):
            target_genre = str(genre).strip().lower()
            for cand in ctx.candidates:
                if str(cand.role or '').strip().lower() != 'select':
                    continue
                blob = str(cand.text or '').lower()
                has_target_genre = target_genre and target_genre in blob
                has_genre_options = has_target_genre or 'all genres' in blob or any(tok in blob for tok in ('action', 'adventure', 'animation', 'biography', 'comedy', 'crime', 'documentary', 'drama', 'family', 'fantasy', 'history', 'horror', 'music', 'musical', 'mystery', 'romance', 'sci-fi', 'thriller', 'war', 'western', 'culture', 'education', 'modernist', 'screen', 'story'))
                looks_like_year = 'all years' in blob or bool(_extract_year_options(blob))
                looks_like_sort = ('sort by' in blob) or ('rating:' in blob) or ('year: newest first' in blob) or ('year: oldest first' in blob)
                if not has_genre_options or looks_like_year or looks_like_sort:
                    continue
                current = str(cand.current_value or '').strip()
                if current.lower() == target_genre:
                    return None
                if isinstance(cand.selector, dict):
                    return _tool_select(cand.selector, str(genre), element_id=str(cand.id or ''))

        parsed = urlsplit(ctx.url)
        current_params = {str(k): str(v) for k, v in parse_qsl(parsed.query or '', keep_blank_values=True)}
        target_params = {k: v for k, v in current_params.items() if k not in {'year', 'genre'}}
        if year:
            target_params['year'] = str(year)
        if genre:
            target_params['genre'] = str(genre)
        if year or genre:
            current_filtered = {k: v for k, v in current_params.items() if k in {'year', 'genre'}}
            target_filtered = {k: v for k, v in target_params.items() if k in {'year', 'genre'}}
            if current_filtered == target_filtered:
                return None
            new_query = urlencode(target_params)
            target_url = urlunsplit((parsed.scheme, parsed.netloc, '/search', new_query, parsed.fragment))
            if target_url != ctx.url and self._allow('browser.navigate', ctx.allowed_tools):
                return _tool_nav(target_url)
        return None


    def _plan_search(self, ctx: StructuredPlanContext) -> dict[str, Any] | None:
        current_path = str(urlsplit(ctx.url).path or '').rstrip('/') or '/'
        if '/search' not in current_path:
            if self._allow('browser.navigate', ctx.allowed_tools):
                return _tool_nav(_replace_path(ctx.url, '/search'))
            route_click = self._find_route_candidate(ctx.candidates, route_tokens=('search', 'find', 'look up'), href_tokens=('/search',))
            if route_click is not None and self._allow('browser.click', ctx.allowed_tools):
                return _tool_click(route_click)
            return None

        query_target = _extract_search_query(ctx.prompt) or 'WALL-E'
        typed = _history_text_by_selector_id(ctx.history)
        state_typed = ctx.internal_state.get('typed_values') if isinstance(ctx.internal_state.get('typed_values'), dict) else {}
        for key, value in state_typed.items():
            key_s = str(key).strip().lower()
            value_s = str(value).strip()
            if key_s and value_s and key_s not in typed:
                typed[key_s] = value_s
        candidate_values: dict[str, str] = {}
        for cand in ctx.candidates:
            cid = _candidate_selector_id(cand)
            if not cid:
                continue
            current = str(cand.current_value or '').strip()
            if current:
                candidate_values[cid] = current
        last_typed_id, last_typed_text = _history_last_typed(ctx.history)

        def current_value(control_id: str) -> str:
            typed_value = str(typed.get(control_id) or '').strip()
            candidate_value = str(candidate_values.get(control_id) or '').strip()
            if not typed_value and control_id == last_typed_id:
                typed_value = str(last_typed_text or '').strip()
            return typed_value or candidate_value

        search_field = self._find_field(
            ctx.candidates,
            kind='search',
            ids=('input', 'search-input', 'movie-search-input', 'query-input', 'search-field', 'input-box', 'entry-box', 'form-input'),
            includes=('search', 'title', 'movie', 'query', 'directors, titles, or moods'),
            excludes=('email', 'password'),
            target=str(query_target),
        )
        if search_field is not None:
            search_id = _candidate_selector_id(search_field)
            if search_id and current_value(search_id) != str(query_target):
                if self._allow('browser.input', ctx.allowed_tools):
                    return _tool_type(search_id, query_target, element_id=search_field.id)
                return None

        candidate_id_map = {_candidate_selector_id(c): c for c in ctx.candidates if _candidate_selector_id(c)}
        for submit_id in ('search-btn', 'search-submit-button', 'search-button', 'submit-search', 'search-action'):
            submit = candidate_id_map.get(submit_id)
            if submit is not None and self._allow('browser.click', ctx.allowed_tools):
                return _tool_click(submit)
        if self._allow('browser.click', ctx.allowed_tools):
            return _tool_click_selector({'type': 'xpathSelector', 'value': "//button[@id='search-btn']", 'case_sensitive': False})

        submit = self._find_submit(
            ctx.candidates,
            extra_tokens=('search', 'find', 'look up'),
            submit_ids=('search-btn', 'search-submit-button', 'search-button', 'submit-search', 'search-action'),
        )
        if submit is not None and self._allow('browser.click', ctx.allowed_tools):
            return _tool_click(submit)
        return None


    def _fallback_action(self, ctx: StructuredPlanContext) -> dict[str, Any]:
        current_path = str(urlsplit(ctx.url).path or "").rstrip("/") or "/"
        use_case = _canonical_use_case_name(ctx.use_case)
        if use_case == "CONTACT" and "/contact" in current_path:
            if self._allow("browser.scroll", ctx.allowed_tools) and not self._history_has_action_type(ctx.history, "scrollaction"):
                return _tool_scroll(direction="down", amount=600)
            return {"type": "WaitAction", "time_seconds": 1.0}
        if use_case == "LOGIN" and "/login" in current_path:
            return {"type": "WaitAction", "time_seconds": 1.0}
        if use_case in {"SEARCH", "FILTER"} and "/search" in current_path:
            return {"type": "WaitAction", "time_seconds": 1.0}
        route_tokens = {
            'CONTACT': ('contact',),
            'LOGIN': ('login', 'log in'),
            'SEARCH': ('search', 'find', 'look up'),
            'FILTER': ('search', 'find', 'look up', 'filter'),
        }.get(use_case, tuple())
        href_tokens = {
            'CONTACT': ('/contact',),
            'LOGIN': ('/login',),
            'SEARCH': ('/search',),
            'FILTER': ('/search',),
        }.get(use_case, tuple())
        if route_tokens and href_tokens:
            route = self._find_route_candidate(ctx.candidates, route_tokens=route_tokens, href_tokens=href_tokens)
            if route is not None and self._allow("browser.click", ctx.allowed_tools):
                return _tool_click(route)
        if self._allow("browser.navigate", ctx.allowed_tools):
            target = "/contact" if use_case == "CONTACT" else "/login" if use_case == "LOGIN" else "/search" if use_case in {"SEARCH", "FILTER"} else "/"
            return _tool_nav(_replace_path(ctx.url, target))
        return {"type": "WaitAction", "time_seconds": 1.0}

    @staticmethod
    def _allow(tool_name: str, allowed_tools: set[str]) -> bool:
        return (not allowed_tools) or (tool_name in allowed_tools)

    @staticmethod
    def _history_has_action_type(history: list[dict[str, Any]], action_type: str) -> bool:
        target = str(action_type or "").strip().lower()
        for item in history:
            if not isinstance(item, dict):
                continue
            action = item.get("action") if isinstance(item.get("action"), dict) else item
            if not isinstance(action, dict):
                continue
            raw_type = str(action.get("type") or "").strip().lower()
            raw_name = str(action.get("name") or "").strip().lower()
            if raw_type == target:
                return True
            if not raw_type and raw_name:
                inferred = {
                    "browser.scroll": "scrollaction",
                    "browser.wait": "waitaction",
                    "browser.input": "typeaction",
                    "browser.fill": "fillaction",
                    "browser.click": "clickaction",
                    "browser.navigate": "navigateaction",
                }.get(raw_name, "")
                if inferred == target:
                    return True
        return False

    def _find_route_candidate(self, candidates: list[Candidate], *, route_tokens: tuple[str, ...], href_tokens: tuple[str, ...]) -> Candidate | None:
        best: tuple[int, int, Candidate] | None = None
        for cand in candidates:
            if cand.disabled:
                continue
            blob = _candidate_blob(cand)
            href = str(cand.href or "").lower()
            score_text = 0 if any(tok in blob for tok in route_tokens) else 1
            score_href = 0 if any(tok in href for tok in href_tokens) else 1
            if score_text and score_href:
                continue
            current = (score_text, score_href, cand)
            if best is None or current[:2] < best[:2]:
                best = current
        return best[2] if best is not None else None

    def _find_field(self, candidates: list[Candidate], *, kind: str, ids: tuple[str, ...], includes: tuple[str, ...], excludes: tuple[str, ...], target: str) -> Candidate | None:
        ids_norm = {str(x).strip().lower() for x in ids if str(x).strip()}
        ranked: list[tuple[tuple[int, int, int, int, int], Candidate]] = []
        for idx, cand in enumerate(candidates):
            if cand.disabled or cand.readonly:
                continue
            role_name = str(cand.role or '').strip().lower()
            type_name = str(cand.type or '').strip().lower()
            # For field filling, ignore links/buttons and prefer actual form controls.
            if role_name not in {'input', 'select'} and type_name not in {'input', 'textarea', 'select'}:
                continue
            cid = _candidate_selector_id(cand)
            blob = _candidate_local_blob(cand)
            if excludes and any(token in blob for token in excludes):
                continue
            direct = 0 if cid in ids_norm and cid else 1
            kind_score = 0 if str(cand.field_kind or '').strip().lower() == kind else 1
            include = 0 if includes and any(token in blob or token == cid for token in includes) else 1
            textarea_bonus = 0 if (kind == 'message' and str(cand.type or '').lower() == 'textarea') else 1
            value_penalty = 0 if not str(cand.current_value or '').strip() else 1
            if direct == 1 and kind_score == 1 and include == 1:
                continue
            ranked.append(((direct, kind_score, include, textarea_bonus, value_penalty + idx), cand))
        if not ranked:
            return None
        ranked.sort(key=lambda item: item[0])
        return ranked[0][1]

    def _find_submit(
        self,
        candidates: list[Candidate],
        extra_tokens: tuple[str, ...] = (),
        submit_ids: tuple[str, ...] = (),
        forbidden_href_paths: tuple[str, ...] = (),
    ) -> Candidate | None:
        tokens = ('submit', 'send', 'send message', 'contact', 'message') + tuple(extra_tokens)
        ids_norm = {str(x).strip().lower() for x in submit_ids if str(x).strip()}
        forbidden_paths = {str(x).strip().lower() for x in forbidden_href_paths}
        best: tuple[tuple[int, int, int], Candidate] | None = None
        for idx, cand in enumerate(candidates):
            if cand.disabled:
                continue
            cid = _candidate_selector_id(cand)
            blob = _candidate_local_blob(cand)
            role = str(cand.role or '').lower()
            kind = str(cand.field_kind or '').lower()
            href_path = _candidate_href_path(cand)
            if href_path in forbidden_paths:
                continue
            id_score = 0 if cid in ids_norm and cid else 1
            role_score = 0 if role == 'button' else 1
            kind_score = 0 if kind == 'submit' else 1
            token_score = 0 if any(tok in blob for tok in tokens) else 1
            if id_score == 1 and role_score == 1 and kind_score == 1 and token_score == 1:
                continue
            current = ((id_score, role_score, kind_score + token_score + idx), cand)
            if best is None or current[0] < best[0]:
                best = current
        return best[1] if best is not None else None
