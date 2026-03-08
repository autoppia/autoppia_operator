from __future__ import annotations

from dataclasses import dataclass, field
import ast
import base64
import hashlib
import json
import os
import re
from typing import Any, Callable, Dict, List, Literal
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from pathlib import Path
from datetime import datetime

from pydantic import BaseModel, Field

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover
    BeautifulSoup = None


FSM_MODES = {
    "BOOTSTRAP",
    "POPUP",
    "PLAN",
    "NAV",
    "EXTRACT",
    "SYNTH",
    "REPORT",
    "STUCK",
    "DONE",
}

OBS_META_TOOLS = {
    "META.EXTRACT_LINKS",
    "META.EXTRACT_FACTS",
    "META.SEARCH_TEXT",
    "META.FIND_ELEMENTS",
    "META.VISION_QA",
}

CONTROL_META_TOOLS = {
    "META.SOLVE_POPUPS",
    "META.REPLAN",
    "META.SELECT_NEXT_TARGET",
    "META.ESCALATE",
    "META.SET_MODE",
    "META.MARK_PROGRESS",
}

META_TOOLS = OBS_META_TOOLS | CONTROL_META_TOOLS

SUPPORTED_BROWSER_TOOL_NAMES = {
    "browser.navigate",
    "browser.click",
    "browser.type",
    "browser.scroll",
    "browser.wait",
    "browser.back",
    "browser.end",
    "browser.select",
    "browser.send_keys",
    "browser.hold_key",
}

MAX_INTERNAL_META_STEPS = 3
MAX_FACTS = 32
MAX_CHECKPOINTS = 20
MAX_PENDING_URLS = 30
MAX_PENDING_ELEMENTS = 50
MAX_VISITED_URLS = 80
MAX_PAGE_HASHES = 80
MAX_STR = 400
MAX_HISTORY_SUMMARY_CHARS = 2400
HISTORY_RECENT_LIMIT = 12
MAX_VISUAL_NOTES = 12
MAX_VISUAL_HINTS = 20

_TASK_TERM_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "then",
    "open",
    "click",
    "visit",
    "please",
    "show",
    "website",
    "page",
    "current",
    "value",
    "finish",
    "first",
    "your",
    "using",
    "account",
    "successfully",
    "continue",
    "task",
    "next",
    "value",
    "total",
    "price",
    "count",
    "number",
    "amount",
    "many",
    "much",
    "tell",
    "show",
    "find",
    "what",
    "when",
    "which",
    "balance",
}


def _utc_now() -> str:
    try:
        return datetime.utcnow().isoformat() + "Z"
    except Exception:
        return ""


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _norm_ws(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _merge_usage_dicts(base: Dict[str, Any] | None, extra: Dict[str, Any] | None) -> Dict[str, int]:
    out = {
        "prompt_tokens": int(((base or {}).get("prompt_tokens") or 0)),
        "completion_tokens": int(((base or {}).get("completion_tokens") or 0)),
        "total_tokens": int(((base or {}).get("total_tokens") or 0)),
    }
    out["prompt_tokens"] += int(((extra or {}).get("prompt_tokens") or 0))
    out["completion_tokens"] += int(((extra or {}).get("completion_tokens") or 0))
    out["total_tokens"] += int(((extra or {}).get("total_tokens") or 0))
    return out


def _dom_digest(html: str) -> str:
    try:
        data = str(html or "").encode("utf-8", errors="ignore")
        return hashlib.sha256(data).hexdigest()[:16]
    except Exception:
        return ""


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]{2,}", str(text or "").lower())}


def _focus_terms(text: str, *, max_terms: int = 18) -> set[str]:
    tokens = [t for t in re.findall(r"[a-z0-9]{3,}", str(text or "").lower()) if t not in _TASK_TERM_STOPWORDS]
    freq: Dict[str, int] = {}
    for token in tokens:
        freq[token] = int(freq.get(token) or 0) + 1
    ranked = sorted(freq.items(), key=lambda item: (item[1], len(item[0]), item[0]), reverse=True)
    return {token for token, _ in ranked[: max(1, int(max_terms))]}


def _looks_like_informational_task(prompt: str) -> bool:
    text = _norm_ws(prompt).lower()
    if not text:
        return False
    info_patterns = [
        r"\bwhat(?:'s| is)\b",
        r"\btell me\b",
        r"\bhow much\b",
        r"\bhow many\b",
        r"\bvalue\b",
        r"\bprice\b",
        r"\btotal\b",
        r"\bbalance\b",
        r"\bcount\b",
        r"\bnumber of\b",
        r"\bwhich\b",
        r"\bwhen\b",
        r"\bfind\b",
        r"\bshow me\b",
    ]
    return any(re.search(pattern, text) for pattern in info_patterns)


def _valueish_text(text: str) -> bool:
    value = _norm_ws(text)
    if not value:
        return False
    if len(value) > 96:
        return False
    if len(re.findall(r"\d", value)) >= 1:
        return True
    if re.search(r"\b(?:yes|no|active|inactive|online|offline|open|closed)\b", value.lower()):
        return True
    if re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", value.lower()):
        return True
    return False


def _labelish_text(text: str) -> bool:
    label = _norm_ws(text)
    if not label:
        return False
    if len(label) > 64:
        return False
    if len(label.split()) > 8:
        return False
    if len(re.findall(r"\d", label)) > 0:
        return False
    if re.search(r"^[^a-zA-Z]*$", label):
        return False
    return True


def _value_line_text(text: str) -> bool:
    line = _norm_ws(text)
    if not line:
        return False
    if len(line) > 140:
        return False
    if len(re.findall(r"\d", line)) >= 1:
        return True
    if re.search(r"\b(?:yes|no|active|inactive|online|offline|open|closed)\b", line.lower()):
        return True
    return False


def _fact_overlap_score(prompt: str, fact: str) -> int:
    p_terms = _focus_terms(prompt, max_terms=16)
    if not p_terms:
        return 0
    f_terms = _tokenize(fact)
    return len(p_terms.intersection(f_terms))


def _anchor_overlap_score(prompt: str, fact: str) -> int:
    anchors = _focus_terms(prompt, max_terms=12)
    if not anchors:
        return 0
    return len(anchors.intersection(_tokenize(fact)))


def _digit_tokens(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:[.,]\d+)?", str(text or "")))


def _best_page_evidence(prompt: str, text_ir: Dict[str, Any]) -> str:
    text_ir = text_ir if isinstance(text_ir, dict) else {}
    page_facts = text_ir.get("page_facts") if isinstance(text_ir.get("page_facts"), list) else []
    value_lines = text_ir.get("value_lines") if isinstance(text_ir.get("value_lines"), list) else []
    relevant_lines = text_ir.get("relevant_lines") if isinstance(text_ir.get("relevant_lines"), list) else []
    candidates = _dedupe_keep_order(
        [str(x) for x in page_facts[:12] + relevant_lines[:12] + value_lines[:16] if _norm_ws(x)],
        24,
    )
    if not candidates:
        return ""
    ranked: List[tuple[int, str]] = []
    for fact in candidates:
        clean = _norm_ws(fact)
        if not clean:
            continue
        raw_overlap = _fact_overlap_score(prompt, clean)
        anchor_overlap = _anchor_overlap_score(prompt, clean)
        score = raw_overlap + (anchor_overlap * 2)
        if _valueish_text(clean):
            score += 1
        if ":" in clean or "-" in clean:
            score += 1
        if clean in page_facts:
            score += 2
        if clean in relevant_lines:
            score += 1
        ranked.append((score, f"{raw_overlap}:{anchor_overlap}::{clean}"))
    ranked.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    if not ranked:
        return ""
    best_score, tagged_fact = ranked[0]
    overlap_blob, _, best_fact = tagged_fact.partition("::")
    try:
        raw_overlap_str, anchor_overlap_str = overlap_blob.split(":", 1)
        raw_overlap = int(raw_overlap_str)
        anchor_overlap = int(anchor_overlap_str)
    except Exception:
        raw_overlap = 0
        anchor_overlap = 0
    if raw_overlap <= 0 and anchor_overlap <= 0:
        return ""
    return best_fact[:220]


def _content_supported_by_page_evidence(prompt: str, content: str, text_ir: Dict[str, Any]) -> bool:
    answer = _norm_ws(content)
    if not answer:
        return False
    page_facts = text_ir.get("page_facts") if isinstance(text_ir.get("page_facts"), list) else []
    likely = []
    if page_facts:
        likely = sorted(
            [_norm_ws(fact) for fact in page_facts if _norm_ws(fact)],
            key=lambda fact: (_fact_overlap_score(prompt, fact), len(fact)),
            reverse=True,
        )[:8]
    if not likely:
        return False
    answer_terms = _tokenize(answer)
    answer_digits = _digit_tokens(answer)
    for fact in likely:
        if answer.lower() in fact.lower() or fact.lower() in answer.lower():
            if _fact_overlap_score(prompt, fact) >= 1:
                return True
        fact_terms = _tokenize(fact)
        fact_digits = _digit_tokens(fact)
        if answer_digits and fact_digits and answer_digits.intersection(fact_digits):
            if answer_terms.intersection(fact_terms) and _fact_overlap_score(prompt, fact) >= 1:
                return True
        if _fact_overlap_score(answer, fact) >= 2:
            if _fact_overlap_score(prompt, fact) >= 1:
                return True
    return False


def _looks_like_vague_informational_answer(text: str) -> bool:
    answer = _norm_ws(text).lower()
    if not answer:
        return True
    vague_patterns = [
        r"\blikely\b",
        r"\bprobably\b",
        r"\bappears to\b",
        r"\bseems to\b",
        r"\bmay be\b",
        r"\bmight be\b",
        r"\bshould\b",
        r"\buser is on\b",
        r"\bpage .*display",
        r"\bpage .*show",
    ]
    return any(re.search(pattern, answer) for pattern in vague_patterns)


def _page_context_overlap(prompt: str, url: str, text_ir: Dict[str, Any]) -> int:
    terms = _focus_terms(prompt, max_terms=16)
    if not terms:
        return 0
    text_ir = text_ir if isinstance(text_ir, dict) else {}
    headings = text_ir.get("headings") if isinstance(text_ir.get("headings"), list) else []
    try:
        parsed = urlsplit(str(url or ""))
        url_context = " ".join([str(parsed.path or ""), str(parsed.query or "")])
    except Exception:
        url_context = str(url or "")
    haystack = " ".join(
        [
            url_context,
            str(text_ir.get("title") or ""),
            " ".join(str(x) for x in headings[:8]),
        ]
    ).lower()
    hay_terms = _tokenize(haystack)
    return len(terms.intersection(hay_terms))


def _page_context_ready_for_informational_answer(prompt: str, url: str, text_ir: Dict[str, Any]) -> bool:
    overlap = _page_context_overlap(prompt, url, text_ir)
    try:
        path = str(urlsplit(str(url or "")).path or "/").strip() or "/"
    except Exception:
        path = "/"
    non_root_path = path not in {"", "/"}
    return bool(non_root_path or overlap >= 2)


def _runtime_page_evidence_ready(prompt: str, url: str, text_ir: Dict[str, Any], *, step_index: int) -> bool:
    if _page_context_ready_for_informational_answer(prompt, url, text_ir):
        return True
    if int(step_index) < 1:
        return False
    best_fact = _best_page_evidence(prompt, text_ir)
    if not best_fact:
        return False
    return bool(
        _fact_overlap_score(prompt, best_fact) >= 1
        and _anchor_overlap_score(prompt, best_fact) >= 1
    )


def _env_str(name: str, default: str = "") -> str:
    raw = os.getenv(name)
    if raw is None:
        return str(default)
    return str(raw).strip()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_screenshot_data_url(value: Any) -> str:
    if isinstance(value, bytes):
        encoded = base64.b64encode(value).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    raw = str(value or "").strip()
    if not raw:
        return ""
    if raw.startswith("data:image/"):
        return raw
    return f"data:image/png;base64,{raw}"


def _vision_signature(*, screenshot: Any, question: str, url: str) -> str:
    data_url = _normalize_screenshot_data_url(screenshot)
    if not data_url:
        return ""
    payload = json.dumps(
        {
            "url": str(url or "")[:300],
            "question": str(question or "")[:600],
            "image_hash": hashlib.sha1(data_url.encode("utf-8", errors="ignore")).hexdigest()[:16],
        },
        ensure_ascii=True,
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _task_constraints(task: str) -> Dict[str, str]:
    text = str(task or "")
    out: Dict[str, str] = {}
    pattern = re.compile(
        r"\b([a-z][a-z0-9 _-]{1,40})\b\s*(?:equals|=|is|:)\s*(?:'([^']+)'|\"([^\"]+)\"|(<[^>]+>)|([0-9]+(?:\.[0-9]+)?)|([^\s,;]+))",
        flags=re.I,
    )
    for match in pattern.finditer(text):
        raw_key = _norm_ws(match.group(1) or "").lower()
        key = re.sub(r"^(?:the|a|an)\s+", "", raw_key).strip().replace(" ", "_")
        value = next((g for g in match.groups()[1:] if g), "")
        value = _norm_ws(value).strip(" \t\r\n'\"`.,;:!?")
        if not key or not value:
            continue
        if len(key) > 40:
            key = key[:40]
        out[key] = value[:120]
        if len(out) >= 16:
            break
    return out


def _constraint_key_tokens(key: str) -> set[str]:
    return _tokenize(str(key or "").replace("_", " "))


def _constraint_value_matches(expected: str, actual: str) -> bool:
    exp = _norm_ws(expected).lower()
    act = _norm_ws(actual).lower()
    if not exp or not act:
        return False
    if exp == act:
        return True
    if exp in act or act in exp:
        return True
    exp_tokens = _tokenize(exp)
    act_tokens = _tokenize(act)
    if exp_tokens and act_tokens and exp_tokens.issubset(act_tokens):
        return True
    return False


def _safe_url(raw: str, base: str = "") -> str:
    txt = str(raw or "").strip()
    if not txt:
        return ""
    if txt.startswith("http://") or txt.startswith("https://"):
        return txt
    if txt.startswith("/"):
        return urljoin(base or "", txt)
    # Relative path support (e.g. "search", "products/list") under current host.
    if base and " " not in txt and not re.match(r"^[a-z]+://", txt, flags=re.I):
        if txt.startswith("./"):
            txt = txt[2:]
        if txt:
            return urljoin(base or "", txt)
    if "." in txt and " " not in txt and "/" not in txt:
        return f"https://{txt}"
    return txt


def _query_map(url: str) -> Dict[str, str]:
    try:
        parsed = urlsplit(str(url or "").strip())
        out: Dict[str, str] = {}
        for k, v in parse_qsl(str(parsed.query or ""), keep_blank_values=True):
            key = str(k or "").strip()
            if not key:
                continue
            out[key[:80]] = str(v or "")[:120]
            if len(out) >= 16:
                break
        return out
    except Exception:
        return {}


def _with_query(url: str, query: Dict[str, str]) -> str:
    try:
        parsed = urlsplit(str(url or "").strip())
        if not parsed.scheme or not parsed.netloc:
            return str(url or "")
        q_items = list(query.items())[:16]
        q_encoded = urlencode(q_items, doseq=False)
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path or "/", q_encoded, parsed.fragment or ""))
    except Exception:
        return str(url or "")


def _candidate_text(*parts: Any) -> str:
    for item in parts:
        if isinstance(item, str):
            cleaned = _norm_ws(item)
            if cleaned:
                return cleaned[:MAX_STR]
    return ""


def _sanitize_selector(selector: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not isinstance(selector, dict):
        return None
    out = dict(selector)
    sel_type = str(out.get("type") or "").strip()
    if sel_type == "xpathSelector":
        value = str(out.get("value") or "").strip()
        if value.lower().startswith("xpath="):
            value = value[6:].strip()
        if value.startswith("///"):
            value = "//" + value.lstrip("/")
        elif value.startswith("/") and not value.startswith("//"):
            value = value.lstrip("/")
        value = re.sub(r"\s+", " ", value).strip()
        if not value:
            return None
        out["value"] = value
    return out


def _dedupe_keep_order(values: List[str], limit: int) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for raw in values:
        value = _norm_ws(raw)[:MAX_STR]
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
        if len(out) >= limit:
            break
    return out


def _action_type_for_tool(tool_name: str) -> str | None:
    t = str(tool_name or "").strip().lower()
    mapping = {
        "browser.navigate": "NavigateAction",
        "browser.click": "ClickAction",
        "browser.type": "TypeAction",
        "browser.scroll": "ScrollAction",
        "browser.wait": "WaitAction",
        "browser.back": "NavigateAction",
        "browser.select": "SelectDropDownOptionAction",
        "browser.send_keys": "SendKeysIWAAction",
        "browser.hold_key": "HoldKeyAction",
    }
    return mapping.get(t)


def _root_url(value: str) -> str:
    try:
        parsed = urlsplit(str(value or "").strip())
        if parsed.scheme and parsed.netloc:
            return urlunsplit((parsed.scheme, parsed.netloc, "/", "", ""))
    except Exception:
        pass
    return ""


def _canonical_allowed_tool_name(raw: str) -> str:
    name = str(raw or "").strip().lower()
    if not name:
        return ""
    if name.startswith("browser.") or name.startswith("user."):
        return name
    if name.startswith("meta."):
        return name.upper()
    if name == "request_user_input":
        return "user.request_input"
    alias = {
        "navigate": "browser.navigate",
        "back": "browser.back",
        "click": "browser.click",
        "double_click": "browser.click",
        "right_click": "browser.click",
        "middle_click": "browser.click",
        "triple_click": "browser.click",
        "hover": "browser.click",
        "type": "browser.type",
        "select": "browser.select",
        "select_drop_down_option": "browser.select",
        "scroll": "browser.scroll",
        "wait": "browser.wait",
        "send_keys_i_w_a": "browser.send_keys",
        "send_keys": "browser.send_keys",
        "hold_key": "browser.hold_key",
        "end": "browser.end",
    }
    if name in alias:
        return alias[name]
    # Keep permissive for custom browser tools.
    return f"browser.{name}"


class Subgoal(BaseModel):
    id: str
    text: str
    status: Literal["pending", "active", "done", "blocked"] = "pending"


class AgentPlan(BaseModel):
    subgoals: List[Subgoal] = Field(default_factory=list)
    active_id: str = ""


class AgentFrontier(BaseModel):
    pending_urls: List[str] = Field(default_factory=list)
    pending_elements: List[str] = Field(default_factory=list)


class AgentVisited(BaseModel):
    urls: List[str] = Field(default_factory=list)
    page_hashes: Dict[str, str] = Field(default_factory=dict)


class AgentMemory(BaseModel):
    facts: List[str] = Field(default_factory=list)
    checkpoints: List[str] = Field(default_factory=list)
    visual_notes: List[str] = Field(default_factory=list)
    visual_element_hints: List[str] = Field(default_factory=list)
    last_vision_signature: str = ""
    history_summary: str = ""
    strategy_summary: str = ""
    prev_page_summary: str = ""
    prev_page_ir_text: str = ""
    prev_candidate_sigs: List[str] = Field(default_factory=list)
    obs_extract_dom_hash: str = ""
    obs_extract_payload: Dict[str, Any] = Field(default_factory=dict)
    obs_candidate_hints: List[str] = Field(default_factory=list)


class AgentFormProgress(BaseModel):
    typed_selector_sigs: List[str] = Field(default_factory=list)
    typed_candidate_ids: List[str] = Field(default_factory=list)
    typed_values_by_selector: Dict[str, str] = Field(default_factory=dict)
    typed_values_by_candidate: Dict[str, str] = Field(default_factory=dict)
    submit_attempt_sigs: List[str] = Field(default_factory=list)
    active_group_id: str = ""
    active_group_label: str = ""
    active_group_context: str = ""
    active_group_candidate_ids: List[str] = Field(default_factory=list)


class AgentFocusRegion(BaseModel):
    region_id: str = ""
    region_kind: str = ""
    region_label: str = ""
    region_context: str = ""
    candidate_ids: List[str] = Field(default_factory=list)
    recent_region_ids: List[str] = Field(default_factory=list)


class ProgressEffect(BaseModel):
    step_index: int = 0
    label: str = ""
    action_type: str = ""
    target_id: str = ""
    region_id: str = ""
    expected_effect: str = ""
    expected_effect_met: bool = False
    region_changed: bool = False
    repeated_target: bool = False
    url_changed: bool = False
    dom_changed: bool = False
    exec_ok: bool = True
    error: str = ""


class AgentProgressLedger(BaseModel):
    recent_effects: List[ProgressEffect] = Field(default_factory=list)
    region_attempts: Dict[str, int] = Field(default_factory=dict)
    blocked_regions: List[str] = Field(default_factory=list)
    successful_patterns: List[str] = Field(default_factory=list)
    failed_patterns: List[str] = Field(default_factory=list)
    satisfied_constraints: List[str] = Field(default_factory=list)
    attempted_constraints: Dict[str, int] = Field(default_factory=dict)
    last_effect: str = ""
    no_progress_score: int = 0
    consecutive_no_effect_steps: int = 0
    pending_expected_effect: str = ""
    pending_expected_target_id: str = ""
    pending_expected_region_id: str = ""
    pending_expected_action_type: str = ""


class AgentCounters(BaseModel):
    stall_count: int = 0
    repeat_action_count: int = 0
    meta_steps_used: int = 0


class AgentBlocklist(BaseModel):
    element_ids: List[str] = Field(default_factory=list)
    until_step: int = 0


class AgentState(BaseModel):
    mode: str = "BOOTSTRAP"
    plan: AgentPlan = Field(default_factory=AgentPlan)
    frontier: AgentFrontier = Field(default_factory=AgentFrontier)
    visited: AgentVisited = Field(default_factory=AgentVisited)
    memory: AgentMemory = Field(default_factory=AgentMemory)
    form_progress: AgentFormProgress = Field(default_factory=AgentFormProgress)
    focus_region: AgentFocusRegion = Field(default_factory=AgentFocusRegion)
    progress: AgentProgressLedger = Field(default_factory=AgentProgressLedger)
    counters: AgentCounters = Field(default_factory=AgentCounters)
    blocklist: AgentBlocklist = Field(default_factory=AgentBlocklist)
    last_url: str = ""
    last_dom_hash: str = ""
    last_action_sig: str = ""
    last_action_element_id: str = ""
    escalated_once: bool = False
    session_query: Dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_state_in(cls, state_in: Any, prompt: str) -> "AgentState":
        if isinstance(state_in, dict):
            try:
                st = cls.model_validate(state_in)
            except Exception:
                st = cls()
        else:
            st = cls()
        if st.mode not in FSM_MODES:
            st.mode = "BOOTSTRAP"
        if not st.plan.subgoals:
            st.plan = AgentPlan(subgoals=_split_prompt_subgoals(prompt), active_id="")
            if st.plan.subgoals:
                st.plan.subgoals[0].status = "active"
                st.plan.active_id = st.plan.subgoals[0].id
        return st._sanitize()

    def _sanitize(self) -> "AgentState":
        self.mode = self.mode if self.mode in FSM_MODES else "BOOTSTRAP"
        self.frontier.pending_urls = _dedupe_keep_order(self.frontier.pending_urls, MAX_PENDING_URLS)
        self.frontier.pending_elements = _dedupe_keep_order(self.frontier.pending_elements, MAX_PENDING_ELEMENTS)
        self.visited.urls = _dedupe_keep_order(self.visited.urls, MAX_VISITED_URLS)
        # Keep deterministic dict size by insertion order.
        if len(self.visited.page_hashes) > MAX_PAGE_HASHES:
            trimmed: Dict[str, str] = {}
            for key in list(self.visited.page_hashes.keys())[-MAX_PAGE_HASHES:]:
                trimmed[str(key)[:MAX_STR]] = str(self.visited.page_hashes.get(key) or "")[:64]
            self.visited.page_hashes = trimmed
        else:
            self.visited.page_hashes = {
                str(k)[:MAX_STR]: str(v)[:64] for k, v in self.visited.page_hashes.items()
            }
        self.memory.facts = _dedupe_keep_order(self.memory.facts, MAX_FACTS)
        self.memory.checkpoints = _dedupe_keep_order(self.memory.checkpoints, MAX_CHECKPOINTS)
        self.memory.visual_notes = _dedupe_keep_order(
            [str(x)[:260] for x in self.memory.visual_notes if str(x).strip()],
            MAX_VISUAL_NOTES,
        )
        self.memory.visual_element_hints = _dedupe_keep_order(
            [str(x)[:120] for x in self.memory.visual_element_hints if str(x).strip()],
            MAX_VISUAL_HINTS,
        )
        self.memory.last_vision_signature = str(self.memory.last_vision_signature or "")[:64]
        self.memory.history_summary = _norm_ws(self.memory.history_summary)[:MAX_HISTORY_SUMMARY_CHARS]
        self.memory.strategy_summary = _norm_ws(self.memory.strategy_summary)[:320]
        self.memory.prev_page_summary = _norm_ws(self.memory.prev_page_summary)[:800]
        self.memory.prev_page_ir_text = _norm_ws(self.memory.prev_page_ir_text)[:2000]
        self.memory.prev_candidate_sigs = _dedupe_keep_order(
            [str(x)[:220] for x in self.memory.prev_candidate_sigs if str(x).strip()],
            80,
        )
        self.memory.obs_extract_dom_hash = str(self.memory.obs_extract_dom_hash or "")[:64]
        self.memory.obs_candidate_hints = _dedupe_keep_order(
            [str(x)[:120] for x in self.memory.obs_candidate_hints if str(x).strip()],
            MAX_VISUAL_HINTS,
        )
        obs_payload = self.memory.obs_extract_payload if isinstance(self.memory.obs_extract_payload, dict) else {}
        self.memory.obs_extract_payload = {
            "page_kind": str(obs_payload.get("page_kind") or "")[:80],
            "summary": str(obs_payload.get("summary") or "")[:600],
            "regions": [
                {
                    "kind": str((item or {}).get("kind") or "")[:40],
                    "label": str((item or {}).get("label") or "")[:160],
                    "candidate_ids": [str(x)[:120] for x in list((item or {}).get("candidate_ids") or [])[:8] if str(x).strip()],
                }
                for item in list(obs_payload.get("regions") or [])[:8]
                if isinstance(item, dict)
            ],
            "forms": [
                {
                    "label": str((item or {}).get("label") or "")[:160],
                    "fields": [str(x)[:80] for x in list((item or {}).get("fields") or [])[:8] if str(x).strip()],
                    "commit_ids": [str(x)[:120] for x in list((item or {}).get("commit_ids") or [])[:6] if str(x).strip()],
                }
                for item in list(obs_payload.get("forms") or [])[:8]
                if isinstance(item, dict)
            ],
            "facts": [str(x)[:180] for x in list(obs_payload.get("facts") or [])[:8] if str(x).strip()],
            "primary_candidate_ids": [str(x)[:120] for x in list(obs_payload.get("primary_candidate_ids") or [])[:12] if str(x).strip()],
        }
        self.form_progress.typed_selector_sigs = _dedupe_keep_order(
            [str(x)[:220] for x in self.form_progress.typed_selector_sigs if str(x).strip()],
            MAX_PENDING_ELEMENTS * 2,
        )
        self.form_progress.typed_candidate_ids = _dedupe_keep_order(
            [str(x)[:120] for x in self.form_progress.typed_candidate_ids if str(x).strip()],
            MAX_PENDING_ELEMENTS * 2,
        )
        self.form_progress.typed_values_by_selector = {
            str(k)[:220]: str(v)[:220]
            for k, v in self.form_progress.typed_values_by_selector.items()
            if str(k).strip() and str(v).strip()
        }
        self.form_progress.typed_values_by_candidate = {
            str(k)[:120]: str(v)[:220]
            for k, v in self.form_progress.typed_values_by_candidate.items()
            if str(k).strip() and str(v).strip()
        }
        self.form_progress.submit_attempt_sigs = _dedupe_keep_order(
            [str(x)[:220] for x in self.form_progress.submit_attempt_sigs if str(x).strip()],
            MAX_PENDING_ELEMENTS * 2,
        )
        self.form_progress.active_group_id = str(self.form_progress.active_group_id or "")[:120]
        self.form_progress.active_group_label = str(self.form_progress.active_group_label or "")[:160]
        self.form_progress.active_group_context = str(self.form_progress.active_group_context or "")[:320]
        self.form_progress.active_group_candidate_ids = _dedupe_keep_order(
            [str(x)[:120] for x in self.form_progress.active_group_candidate_ids if str(x).strip()],
            MAX_PENDING_ELEMENTS,
        )
        self.focus_region.region_id = str(self.focus_region.region_id or "")[:120]
        self.focus_region.region_kind = str(self.focus_region.region_kind or "")[:40]
        self.focus_region.region_label = str(self.focus_region.region_label or "")[:160]
        self.focus_region.region_context = str(self.focus_region.region_context or "")[:320]
        self.focus_region.candidate_ids = _dedupe_keep_order(
            [str(x)[:120] for x in self.focus_region.candidate_ids if str(x).strip()],
            MAX_PENDING_ELEMENTS,
        )
        self.focus_region.recent_region_ids = _dedupe_keep_order(
            [str(x)[:120] for x in self.focus_region.recent_region_ids if str(x).strip()],
            MAX_PENDING_ELEMENTS,
        )
        self.progress.recent_effects = self.progress.recent_effects[-16:]
        for effect in self.progress.recent_effects:
            effect.step_index = max(0, int(effect.step_index or 0))
            effect.label = str(effect.label or "")[:40]
            effect.action_type = str(effect.action_type or "")[:80]
            effect.target_id = str(effect.target_id or "")[:120]
            effect.region_id = str(effect.region_id or "")[:120]
            effect.expected_effect = str(effect.expected_effect or "")[:40]
            effect.expected_effect_met = bool(effect.expected_effect_met)
            effect.error = str(effect.error or "")[:220]
        self.progress.region_attempts = {
            str(k)[:120]: max(0, int(v or 0))
            for k, v in self.progress.region_attempts.items()
            if str(k).strip()
        }
        self.progress.blocked_regions = _dedupe_keep_order(
            [str(x)[:120] for x in self.progress.blocked_regions if str(x).strip()],
            MAX_PENDING_ELEMENTS,
        )
        self.progress.successful_patterns = _dedupe_keep_order(
            [str(x)[:80] for x in self.progress.successful_patterns if str(x).strip()],
            32,
        )
        self.progress.failed_patterns = _dedupe_keep_order(
            [str(x)[:80] for x in self.progress.failed_patterns if str(x).strip()],
            32,
        )
        self.progress.satisfied_constraints = _dedupe_keep_order(
            [str(x)[:80] for x in self.progress.satisfied_constraints if str(x).strip()],
            32,
        )
        self.progress.attempted_constraints = {
            str(k)[:80]: max(0, int(v or 0))
            for k, v in self.progress.attempted_constraints.items()
            if str(k).strip()
        }
        self.progress.last_effect = str(self.progress.last_effect or "")[:40]
        self.progress.no_progress_score = max(0, int(self.progress.no_progress_score or 0))
        self.progress.consecutive_no_effect_steps = max(0, int(self.progress.consecutive_no_effect_steps or 0))
        self.progress.pending_expected_effect = str(self.progress.pending_expected_effect or "")[:40]
        self.progress.pending_expected_target_id = str(self.progress.pending_expected_target_id or "")[:120]
        self.progress.pending_expected_region_id = str(self.progress.pending_expected_region_id or "")[:120]
        self.progress.pending_expected_action_type = str(self.progress.pending_expected_action_type or "")[:80]
        self.blocklist.element_ids = _dedupe_keep_order(self.blocklist.element_ids, MAX_PENDING_ELEMENTS)
        self.last_url = str(self.last_url or "")[:MAX_STR]
        self.last_dom_hash = str(self.last_dom_hash or "")[:64]
        self.last_action_sig = str(self.last_action_sig or "")[:MAX_STR]
        self.last_action_element_id = str(self.last_action_element_id or "")[:120]
        if len(self.session_query) > 16:
            trimmed_q: Dict[str, str] = {}
            for key in list(self.session_query.keys())[:16]:
                trimmed_q[str(key)[:80]] = str(self.session_query.get(key) or "")[:120]
            self.session_query = trimmed_q
        else:
            self.session_query = {
                str(k)[:80]: str(v)[:120]
                for k, v in self.session_query.items()
                if str(k).strip()
            }
        self.counters.stall_count = max(0, int(self.counters.stall_count or 0))
        self.counters.repeat_action_count = max(0, int(self.counters.repeat_action_count or 0))
        self.counters.meta_steps_used = max(0, int(self.counters.meta_steps_used or 0))
        self.blocklist.until_step = max(0, int(self.blocklist.until_step or 0))
        self.plan.subgoals = self.plan.subgoals[:8]
        for sg in self.plan.subgoals:
            sg.id = str(sg.id or "")[:40]
            sg.text = _norm_ws(sg.text)[:MAX_STR]
            if sg.status not in {"pending", "active", "done", "blocked"}:
                sg.status = "pending"
        if self.plan.active_id and not any(sg.id == self.plan.active_id for sg in self.plan.subgoals):
            self.plan.active_id = ""
        return self

    def to_state_out(self) -> Dict[str, Any]:
        self._sanitize()
        return self.model_dump(mode="json", exclude_none=True)


def _split_prompt_subgoals(prompt: str) -> List[Subgoal]:
    raw = _norm_ws(prompt)
    if not raw:
        return []
    # Do not split on dots to avoid breaking hostnames like autoppia.com.
    parts = [p.strip() for p in re.split(r"\bthen\b|;|,|\band\b", raw, flags=re.I) if p.strip()]
    out: List[Subgoal] = []
    for idx, part in enumerate(parts[:8]):
        out.append(Subgoal(id=f"sg_{idx+1}", text=part[:MAX_STR], status="pending"))
    return out or [Subgoal(id="sg_1", text=raw[:MAX_STR], status="pending")]


class FlagDetector:
    def detect(
        self,
        *,
        snapshot_html: str,
        url: str,
        history: List[Dict[str, Any]],
        state: AgentState,
    ) -> Dict[str, Any]:
        html = str(snapshot_html or "")
        lower = html.lower()
        text = self._visible_text(html).lower()
        digest = _dom_digest(html)
        cookie_banner = any(token in lower for token in ("cookie", "cookies", "gdpr", "consent"))
        modal_dialog = any(
            token in lower
            for token in ('role="dialog"', "aria-modal", "<dialog", "modal", "popup", "pop-up")
        )
        interactive_modal_form = self._interactive_modal_form(html)
        captcha_suspected = any(token in lower for token in ("captcha", "recaptcha", "hcaptcha", "cloudflare challenge"))
        login_form = ("type=\"password\"" in lower) or ("signin" in text) or ("log in" in text) or ("sign in" in text)
        search_box = (
            ("type=\"search\"" in lower)
            or ("placeholder=\"search" in lower)
            or bool(re.search(r"(?:id|name|aria-label)=['\"][^'\"]*search", lower))
        )
        product_cards = (
            text.count("add to cart") >= 1
            or text.count("watch trailer") >= 2
            or text.count("add to watchlist") >= 1
            or lower.count("movie-card") >= 2
            or lower.count("product-card") >= 2
        )
        results_list = (
            ("search results" in text)
            or ("results for" in text)
            or ("result-item" in lower)
            or ("results-list" in lower)
            or (search_box and text.count("result") >= 2)
        )
        pricing_table = ("pricing" in text and "$" in text) or ("<table" in lower and "price" in text)
        hard_error_tokens = (
            "error 404",
            "404 page",
            "error 500",
            "internal server error",
            "access denied",
            "site can’t be reached",
            "site can't be reached",
            "temporarily unavailable",
            "service unavailable",
            "bad gateway",
        )
        error_page = any(token in text for token in hard_error_tokens)
        if not error_page and "not found" in text:
            error_page = any(token in text for token in ("page not found", "resource not found", "this page could not be found", "404"))
        url_changed = bool(state.last_url and str(url or "") != state.last_url)
        dom_changed = bool(state.last_dom_hash and digest != state.last_dom_hash)
        repeat_hint = int(state.counters.repeat_action_count or 0)
        stall_suggested = int(state.counters.stall_count or 0)
        last_action_type = str(state.last_action_sig or "").split("|", 1)[0].strip().lower()
        no_visual_progress = (not url_changed) and (not dom_changed)
        # A click/type/select can legitimately trigger async backend events without visible DOM changes.
        if no_visual_progress and last_action_type not in {"clickaction", "typeaction", "selectdropdownoptionaction"}:
            stall_suggested += 1
        loop_level = "none"
        if repeat_hint >= 1 or stall_suggested >= 1:
            loop_level = "low"
        if repeat_hint >= 2 or stall_suggested >= 3:
            loop_level = "high"
        return {
            "cookie_banner": bool(cookie_banner),
            "modal_dialog": bool(modal_dialog and not interactive_modal_form),
            "interactive_modal_form": bool(interactive_modal_form),
            "captcha_suspected": bool(captcha_suspected),
            "login_form": bool(login_form),
            "search_box": bool(search_box),
            "product_cards": bool(product_cards),
            "results_list": bool(results_list),
            "pricing_table": bool(pricing_table),
            "error_page": bool(error_page),
            "url_changed": bool(url_changed),
            "dom_changed": bool(dom_changed),
            "no_visual_progress": bool(no_visual_progress),
            "loop_level": loop_level,
            "stall_count_suggested": max(0, int(stall_suggested)),
            "dom_hash": digest,
        }

    def _visible_text(self, html: str) -> str:
        if not html:
            return ""
        if BeautifulSoup is None:
            return _norm_ws(re.sub(r"<[^>]+>", " ", html))[:4000]
        try:
            soup = BeautifulSoup(html, "lxml")
            for tag in soup(["script", "style", "noscript"]):
                try:
                    tag.decompose()
                except Exception:
                    pass
            return _norm_ws(soup.get_text(" ", strip=True))[:6000]
        except Exception:
            return _norm_ws(re.sub(r"<[^>]+>", " ", html))[:4000]

    def _interactive_modal_form(self, html: str) -> bool:
        if not html:
            return False
        if BeautifulSoup is None:
            lower = str(html or "").lower()
            if not any(token in lower for token in ('role="dialog"', "aria-modal", "<dialog", "modal")):
                return False
            if re.search(r"<form\b", lower) and len(re.findall(r"<(?:input|select|textarea)\b", lower)) >= 2:
                return True
            if re.search(r"\b(sign in|log in|sign up|signup|create account)\b", lower):
                return True
            return bool(re.search(r"type=['\"](?:password|email)['\"]", lower))
        try:
            soup = BeautifulSoup(html, "lxml")
            dialog_nodes = soup.select("[role='dialog'], dialog, [aria-modal='true'], .modal, .popup")
            for node in dialog_nodes[:8]:
                try:
                    node_text = _norm_ws(node.get_text(" ", strip=True)).lower()
                    if re.search(r"\b(sign in|log in|sign up|signup|create account)\b", node_text):
                        return True
                    inputs = node.select("input, select, textarea")
                    if not inputs:
                        continue
                    if node.find("form") is not None and len(inputs) >= 2:
                        return True
                    for input_node in inputs[:10]:
                        attrs = input_node.attrs if isinstance(getattr(input_node, "attrs", None), dict) else {}
                        input_type = str(attrs.get("type") or "").strip().lower()
                        blob = " ".join(
                            [
                                str(attrs.get("name") or ""),
                                str(attrs.get("id") or ""),
                                str(attrs.get("placeholder") or ""),
                                str(attrs.get("aria-label") or ""),
                            ]
                        ).lower()
                        if input_type in {"password", "email"}:
                            return True
                        if any(token in blob for token in ("email", "password", "username", "user name", "sign in", "log in")):
                            return True
                except Exception:
                    continue
        except Exception:
            return False
        return False


@dataclass
class Candidate:
    id: str
    role: str
    type: str
    text: str
    href: str
    context: str
    selector: Dict[str, Any]
    dom_path: str
    field_hint: str = ""
    field_kind: str = ""
    input_type: str = ""
    ui_state: str = ""
    region_id: str = ""
    region_kind: str = ""
    region_label: str = ""
    parent_region_id: str = ""
    region_ancestor_ids: List[str] = field(default_factory=list)
    group_id: str = ""
    group_label: str = ""
    disabled: bool = False
    bbox: Dict[str, float] | None = None

    def as_obs(self) -> Dict[str, Any]:
        out = {
            "id": self.id,
            "role": self.role,
            "type": self.type,
            "text": self.text[:260],
            "href": self.href[:420] if self.href else "",
            "context": self.context[:600] if self.context else "",
            "selector": self.selector,
            "dom_path": self.dom_path[:260],
            "field_hint": self.field_hint[:180] if self.field_hint else "",
            "field_kind": self.field_kind[:80] if self.field_kind else "",
            "input_type": self.input_type[:40] if self.input_type else "",
            "ui_state": self.ui_state[:40] if self.ui_state else "",
            "region_id": self.region_id[:120] if self.region_id else "",
            "region_kind": self.region_kind[:40] if self.region_kind else "",
            "region_label": self.region_label[:180] if self.region_label else "",
            "parent_region_id": self.parent_region_id[:120] if self.parent_region_id else "",
            "region_ancestor_ids": self.region_ancestor_ids[:8] if self.region_ancestor_ids else [],
            "group_id": self.group_id[:120] if self.group_id else "",
            "group_label": self.group_label[:180] if self.group_label else "",
            "disabled": bool(self.disabled),
            "bbox": self.bbox,
        }
        return out


class CandidateExtractor:
    def extract(self, *, snapshot_html: str, url: str) -> List[Candidate]:
        html = str(snapshot_html or "")
        if not html or BeautifulSoup is None:
            return []
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            return []
        nodes = list(
            soup.select(
                "a,button,input,select,textarea,"
                "[role='button'],[role='link'],[role='tab'],[role='menuitem'],"
                "[role='checkbox'],[role='radio'],[role='switch'],[role='combobox']"
            )
        )
        id_counts: Dict[str, int] = {}
        for node in nodes:
            attrs = node.attrs if isinstance(getattr(node, "attrs", None), dict) else {}
            node_id = _norm_ws(attrs.get("id"))
            if node_id:
                id_counts[node_id] = int(id_counts.get(node_id) or 0) + 1
        out: List[Candidate] = []
        for node in nodes:
            try:
                attrs = node.attrs if isinstance(getattr(node, "attrs", None), dict) else {}
                tag = str(getattr(node, "name", "") or "").lower()
                role = str(attrs.get("role") or "").strip().lower()
                role_name = self._role_name(tag, role, attrs)
                if not role_name:
                    continue
                field_hint = self._field_hint(node)
                current_value = _norm_ws(attrs.get("value"))
                if tag == "select":
                    selected_texts: List[str] = []
                    for option in node.find_all("option", limit=12):
                        opt_text = _norm_ws(option.get_text(" ", strip=True))
                        opt_value = _norm_ws(option.get("value"))
                        selected_attr = str(option.get("selected") or "").strip().lower()
                        if selected_attr in {"selected", "true"} or option.has_attr("selected"):
                            selected_blob = _candidate_text(opt_text, opt_value)
                            if selected_blob:
                                selected_texts.append(selected_blob[:80])
                    if selected_texts:
                        current_value = _candidate_text(*selected_texts[:2])[:80]
                text = _norm_ws(node.get_text(" ", strip=True))
                if not text:
                    text = _candidate_text(
                        attrs.get("aria-label"),
                        attrs.get("placeholder"),
                        current_value,
                        field_hint,
                    )
                elif current_value and tag == "select":
                    text = _norm_ws(f"{text} current={current_value}")[:260]
                raw_href = _norm_ws(attrs.get("href"))
                href = _safe_url(str(raw_href or ""), base=str(url or ""))
                dom_path = self._dom_path(node)
                selector = self._selector_for(
                    tag=tag,
                    attrs=attrs,
                    text=text,
                    href=href,
                    raw_href=raw_href,
                    dom_path=dom_path,
                    id_counts=id_counts,
                )
                if not isinstance(selector, dict):
                    continue
                stable_id = self._stable_id(attrs=attrs, selector=selector, text=text, href=href, dom_path=dom_path)
                context = self._context(node)
                input_type = str(attrs.get("type") or "").strip().lower()
                ui_state = str(attrs.get("data-state") or "").strip().lower()
                aria_selected = str(attrs.get("aria-selected") or "").strip().lower()
                if not ui_state and aria_selected in {"true", "false"}:
                    ui_state = "active" if aria_selected == "true" else "inactive"
                disabled = ("disabled" in attrs) or (str(attrs.get("aria-disabled") or "").strip().lower() == "true")
                group_id = self._group_id(node)
                group_label = self._group_label(node)
                region_kind = self._region_kind(node)
                region_id = group_id
                region_label = group_label
                parent_region_id, region_ancestor_ids = self._region_lineage(node)
                field_kind = self._field_kind(
                    tag=tag,
                    attrs=attrs,
                    role_name=role_name,
                    text=text,
                    field_hint=field_hint,
                    context=context,
                )
                out.append(
                    Candidate(
                        id=stable_id,
                        role=role_name,
                        type=tag or role_name,
                        text=text[:260],
                        href=href[:320],
                        context=context[:700],
                        selector=selector,
                        dom_path=dom_path[:260],
                        field_hint=field_hint[:160],
                        field_kind=field_kind[:80],
                        input_type=input_type[:40],
                        ui_state=ui_state[:40],
                        region_id=region_id[:120],
                        region_kind=region_kind[:40],
                        region_label=region_label[:180],
                        parent_region_id=parent_region_id[:120],
                        region_ancestor_ids=region_ancestor_ids[:8],
                        group_id=group_id[:120],
                        group_label=group_label[:180],
                        disabled=bool(disabled),
                        bbox=None,
                    )
                )
            except Exception:
                continue
        dedup: Dict[str, Candidate] = {}
        for cand in out:
            dedup[cand.id] = cand
        return list(dedup.values())[:220]

    def _role_name(self, tag: str, role: str, attrs: Dict[str, Any]) -> str:
        if tag == "a" or role == "link":
            return "link"
        if tag == "button" or role in {"button", "tab", "menuitem", "checkbox", "radio", "switch"}:
            return "button"
        input_type = str(attrs.get("type") or "").strip().lower()
        if tag == "input" and input_type in {"submit", "button", "reset", "image"}:
            return "button"
        if tag in {"input", "textarea"}:
            return "input"
        if tag == "select" or role == "combobox":
            return "select"
        return ""

    def _dom_path(self, node: Any) -> str:
        parts: List[str] = []
        cur = node
        guard = 0
        while cur is not None and guard < 12:
            guard += 1
            tag = str(getattr(cur, "name", "") or "").lower()
            if not tag:
                break
            parent = getattr(cur, "parent", None)
            idx = 1
            try:
                if parent is not None and hasattr(parent, "find_all"):
                    siblings = [x for x in parent.find_all(tag, recursive=False)]
                    for s_idx, sibling in enumerate(siblings, start=1):
                        if sibling is cur:
                            idx = s_idx
                            break
            except Exception:
                idx = 1
            parts.append(f"{tag}[{idx}]")
            if tag == "html":
                break
            cur = parent
        parts.reverse()
        return "/".join(parts)

    def _selector_for(
        self,
        *,
        tag: str,
        attrs: Dict[str, Any],
        text: str,
        href: str,
        raw_href: str,
        dom_path: str,
        id_counts: Dict[str, int],
    ) -> Dict[str, Any]:
        node_id = _norm_ws(attrs.get("id"))
        if node_id and int(id_counts.get(node_id) or 0) <= 1:
            return {"type": "attributeValueSelector", "attribute": "id", "value": node_id, "case_sensitive": False}
        node_name = _norm_ws(attrs.get("name"))
        if node_name and tag in {"input", "textarea", "select"}:
            return {"type": "attributeValueSelector", "attribute": "name", "value": node_name, "case_sensitive": False}
        if raw_href and tag == "a":
            return {
                "type": "attributeValueSelector",
                "attribute": "href",
                "value": raw_href,
                "case_sensitive": False,
            }
        if text and tag in {"button", "a"}:
            clean = text.replace('"', "'")[:120]
            xpath = f"//{tag}[contains(normalize-space(.), \"{clean}\")]"
            return {"type": "xpathSelector", "value": xpath, "case_sensitive": False}
        if dom_path:
            # IWA Selector prepends '//' for xpath values that do not start with '//'.
            # Keep this as a raw DOM path to avoid generating invalid triple-slash selectors.
            return {"type": "xpathSelector", "value": dom_path, "case_sensitive": False}
        return {"type": "xpathSelector", "value": f"//{tag}[1]", "case_sensitive": False}

    def _stable_id(self, *, attrs: Dict[str, Any], selector: Dict[str, Any], text: str, href: str, dom_path: str) -> str:
        existing = _norm_ws(attrs.get("data-element-id"))
        if existing:
            return existing[:80]
        payload = {
            "selector": selector,
            "text": text[:120],
            "href": href[:220],
            "dom_path": dom_path[:180],
        }
        raw = json.dumps(payload, ensure_ascii=True, sort_keys=True)
        return f"el_{hashlib.sha1(raw.encode('utf-8', errors='ignore')).hexdigest()[:12]}"

    def _context(self, node: Any) -> str:
        try:
            container = node
            for _ in range(3):
                parent = getattr(container, "parent", None)
                if parent is None:
                    break
                container = parent
                if str(getattr(container, "name", "") or "").lower() in {"section", "article", "div", "main", "li"}:
                    break
            return _norm_ws(container.get_text(" ", strip=True))
        except Exception:
            return ""

    def _group_container(self, node: Any) -> Any:
        cur = node
        for _ in range(6):
            cur = getattr(cur, "parent", None)
            if cur is None:
                break
            tag = str(getattr(cur, "name", "") or "").lower()
            if tag in {"form", "fieldset"}:
                return cur
            if tag in {"section", "article", "nav", "aside", "div", "li"}:
                try:
                    interactive = cur.select("a,button,input,select,textarea,[role='button'],[role='link']")
                except Exception:
                    interactive = []
                if len(interactive) >= 2:
                    return cur
        return getattr(node, "parent", None)

    def _group_id(self, node: Any) -> str:
        container = self._group_container(node)
        if container is None:
            return ""
        try:
            dom_path = self._dom_path(container)
            if dom_path:
                return hashlib.sha1(dom_path.encode("utf-8", errors="ignore")).hexdigest()[:12]
        except Exception:
            return ""
        return ""

    def _group_label(self, node: Any) -> str:
        container = self._group_container(node)
        if container is None:
            return ""
        try:
            for selector in ("legend", "h1", "h2", "h3", "h4", "label"):
                found = container.find(selector)
                if found is not None:
                    txt = _norm_ws(found.get_text(" ", strip=True))
                    if txt:
                        return txt[:180]
            txt = _norm_ws(container.get_text(" ", strip=True))
            return txt[:180]
        except Exception:
            return ""

    def _region_kind(self, node: Any) -> str:
        container = self._group_container(node)
        if container is None:
            return "page"
        tag = str(getattr(container, "name", "") or "").lower()
        attrs = container.attrs if isinstance(getattr(container, "attrs", None), dict) else {}
        role = str(attrs.get("role") or "").strip().lower()
        classes = " ".join(str(x) for x in (attrs.get("class") or [] if isinstance(attrs.get("class"), list) else [attrs.get("class")])) .lower()
        if tag in {"form", "fieldset"}:
            return "form"
        if tag == "dialog" or role == "dialog" or str(attrs.get("aria-modal") or "").strip().lower() == "true":
            return "dialog"
        if role in {"tabpanel", "tablist", "tab"} or "tab" in classes:
            return "tab"
        if role in {"row", "gridcell", "grid"} or tag in {"tr", "table"}:
            return "grid"
        if tag in {"li", "article"}:
            return "item"
        if tag in {"section", "aside", "nav"}:
            return tag
        return "group"

    def _region_lineage(self, node: Any) -> tuple[str, List[str]]:
        parent_region_id = ""
        ancestor_ids: List[str] = []
        cur = self._group_container(node)
        hops = 0
        while cur is not None and hops < 6:
            hops += 1
            cur = self._group_container(cur)
            if cur is None:
                break
            try:
                dom_path = self._dom_path(cur)
            except Exception:
                dom_path = ""
            if not dom_path:
                continue
            region_id = hashlib.sha1(dom_path.encode("utf-8", errors="ignore")).hexdigest()[:12]
            if not parent_region_id:
                parent_region_id = region_id
            ancestor_ids.append(region_id)
        return parent_region_id[:120], ancestor_ids[:8]

    def _field_kind(
        self,
        *,
        tag: str,
        attrs: Dict[str, Any],
        role_name: str,
        text: str,
        field_hint: str,
        context: str,
    ) -> str:
        input_type = str(attrs.get("type") or "").strip().lower()
        local_blob = " ".join(
            [
                text,
                str(attrs.get("name") or ""),
                str(attrs.get("id") or ""),
                str(attrs.get("placeholder") or ""),
                str(attrs.get("aria-label") or ""),
                str(attrs.get("href") or ""),
            ]
        ).lower()
        blob = " ".join(
            [
                text,
                field_hint,
                context,
                str(attrs.get("name") or ""),
                str(attrs.get("id") or ""),
                str(attrs.get("placeholder") or ""),
                str(attrs.get("aria-label") or ""),
            ]
        ).lower()
        if role_name == "button":
            if any(k in local_blob for k in ("register", "sign up", "signup", "create account")):
                return "account_create"
            if any(k in local_blob for k in ("login", "log in", "sign in", "signin", "authenticate")):
                return "auth_entry"
            if any(k in local_blob for k in ("next", "prev", "previous", "page ", "pagination")):
                return "pager"
            if any(k in local_blob for k in ("submit", "save", "apply", "search", "find", "continue", "go")):
                return "submit"
            return "button"
        if role_name == "link":
            if any(k in local_blob for k in ("register", "sign up", "signup", "create account")):
                return "account_create"
            if any(k in local_blob for k in ("login", "log in", "sign in", "signin", "authenticate")):
                return "auth_entry"
            if any(k in local_blob for k in ("next", "prev", "previous", "page ", "pagination")):
                return "pager"
            return "link"
        if role_name == "select":
            if any(k in blob for k in ("sort", "order", "newest", "oldest", "rating")):
                return "sort"
            if any(k in blob for k in ("genre", "genres", "category", "categories", "tag", "tags")):
                return "genre"
            if any(k in blob for k in ("year", "release", "released", "date")) or re.search(r"\b(19|20)\d{2}\b", blob):
                return "year"
            return "select"
        if input_type == "password" or any(k in blob for k in ("password", "passcode", "pwd")):
            if any(k in blob for k in ("confirm", "repeat", "again")):
                return "confirm_password"
            return "password"
        if input_type == "email" or any(k in blob for k in ("email", "e-mail")):
            return "email"
        if any(k in blob for k in ("username", "user name", "login", "sign in")):
            return "username"
        if any(k in blob for k in ("search", "query", "find")):
            return "search"
        if any(k in blob for k in ("year", "release year", "released", "date")):
            return "year"
        if any(k in blob for k in ("name", "full name")):
            return "name"
        return "text"

    def _field_hint(self, node: Any) -> str:
        try:
            attrs = node.attrs if isinstance(getattr(node, "attrs", None), dict) else {}
            node_id = _norm_ws(attrs.get("id"))
            if node_id and hasattr(node, "find_parent"):
                root = node.find_parent("html")
                if root is not None:
                    label = root.find("label", attrs={"for": node_id})
                    if label is not None:
                        txt = _norm_ws(label.get_text(" ", strip=True))
                        if txt:
                            return txt[:120]
            parent = getattr(node, "parent", None)
            if parent is not None and str(getattr(parent, "name", "") or "").lower() == "label":
                txt = _norm_ws(parent.get_text(" ", strip=True))
                if txt:
                    return txt[:120]
            cur = node
            for _ in range(3):
                cur = getattr(cur, "parent", None)
                if cur is None:
                    break
                txt = _norm_ws(cur.get_text(" ", strip=True))
                if txt and any(
                    k in txt.lower()
                    for k in (
                        "name",
                        "email",
                        "password",
                        "username",
                        "title",
                        "description",
                        "rating",
                        "year",
                        "duration",
                        "cast",
                        "search",
                    )
                ):
                    return txt[:120]
        except Exception:
            return ""
        return ""


class CandidateRanker:
    def _candidate_constraint_match(
        self,
        *,
        cand: Candidate,
        task_constraints: Dict[str, str],
    ) -> Dict[str, str]:
        if not task_constraints:
            return {}
        blob = " ".join([cand.text, cand.field_hint, cand.context, cand.group_label, cand.region_label, cand.href]).lower()
        matches: Dict[str, str] = {}
        for key, value in task_constraints.items():
            key_tokens = _constraint_key_tokens(key)
            key_match = bool(key_tokens and key_tokens.intersection(_tokenize(blob)))
            value_match = _constraint_value_matches(str(value), cand.text)
            if value_match:
                matches[str(key)] = "value"
            elif key_match:
                matches[str(key)] = "field"
        return matches

    def _candidate_constraint_keys(self, *, cand: Candidate, task_constraints: Dict[str, str]) -> set[str]:
        return set(self._candidate_constraint_match(cand=cand, task_constraints=task_constraints).keys())

    def _candidate_group_key(self, cand: Candidate) -> str:
        return str(cand.group_id or _norm_ws(cand.context).lower() or cand.id)

    def _prompt_field_needs(self, task: str) -> set[str]:
        text = str(task or "")
        needs: set[str] = set()
        if re.search(r"\busername\b|\buser\b", text, flags=re.I):
            needs.add("username")
        if re.search(r"\bemail\b|\be-mail\b", text, flags=re.I):
            needs.add("email")
        if re.search(r"\bpassword\b|\bpass\b|\bpwd\b", text, flags=re.I):
            needs.add("password")
        if re.search(r"\bconfirm password\b|\brepeat password\b", text, flags=re.I):
            needs.add("confirm_password")
        if re.search(r"\b(19|20|21)\d{2}\b", text):
            needs.add("year")
        if re.search(r"\bgenre\b|\bgenres\b|\bcategory\b|\bcategories\b|\btag\b|\btags\b", text, flags=re.I):
            needs.add("genre")
        if re.search(r"\bname\b", text, flags=re.I):
            needs.add("name")
        if re.search(r"\bsearch\b|\bfind\b|\bquery\b", text, flags=re.I):
            needs.add("search")
        return needs

    def _task_operation_hints(self, task: str) -> set[str]:
        text = str(task or "").lower()
        ops: set[str] = set()
        if re.search(r"\b(add|create|new|insert)\b", text):
            ops.add("create")
        if re.search(r"\b(edit|update|modify|change)\b", text):
            ops.add("update")
        if re.search(r"\b(delete|remove|erase|discard)\b", text):
            ops.add("delete")
        if re.search(r"\b(log ?in|sign in|authenticate)\b", text):
            ops.add("auth_login")
        if re.search(r"\b(register|sign up|signup|create account)\b", text):
            ops.add("auth_register")
        return ops

    def _task_has_explicit_credentials(self, task: str) -> bool:
        prompt = str(task or "")
        constraints = _task_constraints(prompt)
        for key in ("username", "email", "password", "confirm_password"):
            if str(constraints.get(key) or "").strip():
                return True
        if re.search(r"<\s*(username|email|password|signup_email|signup_password)\s*>", prompt, flags=re.I):
            return True
        if re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", prompt):
            return True
        return False

    def _candidate_action_tags(self, cand: Candidate) -> set[str]:
        blob = " ".join([cand.text, cand.href, cand.field_hint, cand.field_kind, cand.group_label]).lower()
        tags: set[str] = set()
        if re.search(r"\b(delete|remove|erase|discard)\b", blob):
            tags.add("delete")
        if re.search(r"\b(edit|update|modify|change)\b", blob):
            tags.add("update")
        if re.search(r"\b(add|create|new)\b", blob):
            tags.add("create")
        if re.search(r"\b(log ?in|sign in|authenticate)\b", blob):
            tags.add("auth_login")
        if re.search(r"\b(register|sign up|signup|create account)\b", blob):
            tags.add("auth_register")
        if re.search(r"\b(profile|account|manage|dashboard)\b", blob):
            tags.add("manage_nav")
        return tags

    def _is_section_switch_candidate(self, cand: Candidate) -> bool:
        if cand.role not in {"button", "link"}:
            return False
        selector_attr = str((cand.selector or {}).get("attribute") or "").strip().lower()
        selector_value = str((cand.selector or {}).get("value") or "").strip().lower()
        blob = " ".join([cand.text, cand.field_hint, cand.group_label, cand.context]).lower()
        if selector_attr == "id" and ("trigger-" in selector_value or selector_value.startswith("tab-")):
            return True
        if "tabpanel" in blob or "tablist" in blob:
            return True
        if re.search(r"\b(tab|section|panel|view)\b", blob) and cand.role == "button":
            return True
        return False

    def _looks_submit_like(self, cand: Candidate) -> bool:
        if cand.field_kind in {"submit", "account_create", "auth_entry"}:
            return True
        if cand.role == "input" and str(cand.input_type or "").strip().lower() in {"submit", "button", "image", "reset"}:
            return True
        blob = " ".join([cand.text, cand.field_hint, cand.group_label]).lower()
        return bool(
            cand.role in {"button", "input"}
            and any(
                k in blob
                for k in (
                    "submit",
                    "save",
                    "apply",
                    "find",
                    "search",
                    "filter",
                    "go",
                    "continue",
                    "send",
                    "sign up",
                    "signup",
                    "register",
                    "login",
                    "sign in",
                    "create account",
                )
            )
        )

    def _selector_signature(self, selector: Dict[str, Any] | None) -> str:
        selector = _sanitize_selector(selector)
        if not isinstance(selector, dict):
            return ""
        try:
            return json.dumps(selector, ensure_ascii=True, sort_keys=True)
        except Exception:
            return ""

    def rank(
        self,
        *,
        task: str,
        mode: str,
        flags: Dict[str, Any],
        candidates: List[Candidate],
        state: AgentState,
        current_url: str = "",
        top_k: int = 30,
    ) -> List[Candidate]:
        task_tokens = _focus_terms(task)
        task_constraints = _task_constraints(task)
        satisfied_constraints = set(state.progress.satisfied_constraints or [])
        prompt_needs = self._prompt_field_needs(task)
        task_ops = self._task_operation_hints(task)
        task_has_credentials = self._task_has_explicit_credentials(task)
        mutation_ops = task_ops.intersection({"create", "update", "delete"})
        delete_only_task = mutation_ops == {"delete"}
        blocked = set(state.blocklist.element_ids if state.blocklist.until_step > 0 else [])
        visual_hints = set(state.memory.visual_element_hints)
        obs_hints = set(state.memory.obs_candidate_hints)
        active_group_ids = set(state.form_progress.active_group_candidate_ids)
        active_group_id = str(state.form_progress.active_group_id or "").strip()
        active_group_context = _norm_ws(state.form_progress.active_group_context).lower()
        typed_candidate_ids = set(state.form_progress.typed_candidate_ids or [])
        typed_selector_sigs = set(state.form_progress.typed_selector_sigs or [])
        focus_region_id = str(state.focus_region.region_id or "").strip()
        focus_region_context = _norm_ws(state.focus_region.region_context).lower()
        focus_region_ids = set(state.focus_region.candidate_ids)
        focus_recent_regions = set(state.focus_region.recent_region_ids)
        blocked_regions = set(state.progress.blocked_regions)
        current_path = str(urlsplit(str(current_url or "")).path or "/").rstrip("/") or "/"
        last_action_type = str(state.last_action_sig or "").split("|", 1)[0].strip().lower()
        candidate_tags: Dict[str, set[str]] = {cand.id: self._candidate_action_tags(cand) for cand in candidates}
        group_stats: Dict[str, Dict[str, Any]] = {}
        exact_value_match_keys: Dict[str, int] = {}
        for cand in candidates:
            group_key = self._candidate_group_key(cand)
            stats = group_stats.setdefault(
                group_key,
                {
                    "roles": set(),
                    "field_kinds": set(),
                    "input_like": 0,
                    "constraint_hits": 0,
                    "context_len": 0,
                    "typed_field_kinds": set(),
                    "typed_inputs": 0,
                    "submit_like": 0,
                },
            )
            stats["roles"].add(cand.role)
            if cand.role in {"input", "select", "button"}:
                stats["input_like"] += 1
            if self._looks_submit_like(cand) and not cand.disabled:
                stats["submit_like"] += 1
            stats["context_len"] = max(int(stats.get("context_len") or 0), len(str(cand.context or "")))
            if cand.field_kind:
                stats["field_kinds"].add(cand.field_kind)
            sel_sig = self._selector_signature(cand.selector)
            if (cand.id and cand.id in typed_candidate_ids) or (sel_sig and sel_sig in typed_selector_sigs):
                stats["typed_inputs"] = int(stats.get("typed_inputs") or 0) + 1
                if cand.field_kind:
                    stats["typed_field_kinds"].add(cand.field_kind)
            if cand.field_kind and cand.field_kind in prompt_needs:
                stats["constraint_hits"] += 1
            if task_constraints:
                blob = " ".join([cand.text, cand.href, cand.context, cand.field_hint]).lower()
                for key in task_constraints.keys():
                    key_blob = str(key or "").replace("_", " ").lower()
                    if key_blob and any(tok in blob for tok in _tokenize(key_blob)):
                        stats["constraint_hits"] += 1
                        break
            for key, match_type in self._candidate_constraint_match(cand=cand, task_constraints=task_constraints).items():
                if match_type == "value":
                    exact_value_match_keys[key] = int(exact_value_match_keys.get(key) or 0) + 1
        relevant_groups = {
            key
            for key, stats in group_stats.items()
            if int(stats.get("constraint_hits") or 0) > 0
            or (prompt_needs.intersection(set(stats.get("field_kinds") or set())) and int(stats.get("input_like") or 0) > 0)
        }
        has_visible_constraint_match = False
        for cand in candidates:
            blob = " ".join([cand.text, cand.href, cand.context, cand.field_hint]).lower()
            constraint_matches = self._candidate_constraint_match(cand=cand, task_constraints=task_constraints)
            if (
                cand.role in {"select", "input"}
                and any(match == "value" for match in constraint_matches.values())
                and ("current=" in blob or " value=" in blob)
            ):
                has_visible_constraint_match = True
                break
        credential_needs = {"username", "email", "password", "confirm_password"}.intersection(prompt_needs)
        has_relevant_form_group = any(
            key in relevant_groups and int(stats.get("input_like") or 0) > 0
            for key, stats in group_stats.items()
        )
        has_direct_constraint_controls = any(
            cand.role in {"input", "select"} and cand.field_kind and cand.field_kind in prompt_needs
            for cand in candidates
        )
        has_password_input_visible = any(
            cand.role == "input" and cand.field_kind in {"password", "confirm_password"}
            for cand in candidates
        )
        has_non_form_controls = any(cand.role in {"button", "link"} for cand in candidates)
        has_same_region_commit_controls = any(
            self._looks_submit_like(cand) and not cand.disabled
            for cand in candidates
        )
        has_mutation_controls = any(
            candidate_tags.get(cand.id, set()).intersection(mutation_ops)
            for cand in candidates
            if cand.role in {"button", "link", "input"}
        )
        scored: List[tuple[float, Candidate]] = []
        for cand in candidates:
            if cand.id in blocked:
                continue
            blob = " ".join([cand.text, cand.href, cand.context, cand.field_hint]).lower()
            cand_context = _norm_ws(cand.context).lower()
            group_key = self._candidate_group_key(cand)
            group_kinds = set((group_stats.get(group_key) or {}).get("field_kinds") or set())
            group_typed_kinds = set((group_stats.get(group_key) or {}).get("typed_field_kinds") or set())
            group_typed_inputs = int((group_stats.get(group_key) or {}).get("typed_inputs") or 0)
            action_tags = candidate_tags.get(cand.id, set())
            constraint_matches = self._candidate_constraint_match(cand=cand, task_constraints=task_constraints)
            constraint_keys = set(constraint_matches.keys())
            unmet_constraint_keys = {key for key in constraint_keys if key not in satisfied_constraints}
            current_value_already_matches = (
                cand.role in {"select", "input"}
                and any(constraint_matches.get(key) == "value" for key in constraint_keys)
                and ("current=" in blob or " value=" in blob)
            )
            is_section_switch = self._is_section_switch_candidate(cand)
            score = 0.0
            if cand.role == "input":
                score += 3.6
            elif cand.role == "button":
                score += 3.0
            elif cand.role == "link":
                score += 2.0
            elif cand.role == "select":
                score += 2.6
            if cand.href:
                score += 0.5
            overlap = len(task_tokens.intersection(_focus_terms(blob)))
            score += min(6.0, overlap * 1.1)
            if cand.selector.get("type") == "attributeValueSelector":
                attr = str(cand.selector.get("attribute") or "")
                if attr == "id":
                    score += 2.0
                elif attr in {"name", "href", "aria-label", "placeholder"}:
                    score += 1.1
            elif cand.selector.get("type") == "xpathSelector":
                score -= 0.6
            if cand.text:
                score += 0.5
            if cand.context:
                score += 0.4
            if cand.field_hint:
                score += 0.8
            if cand.group_label:
                score += 0.3
            if cand.field_kind and cand.field_kind in prompt_needs:
                score += 4.0
            if credential_needs and cand.field_kind == "auth_entry":
                score += 5.8
            if cand.field_kind == "account_create":
                if "email" in prompt_needs:
                    score += 5.8
                elif credential_needs:
                    score += 1.2
            if cand.field_kind == "year" and "year" in prompt_needs:
                score += 2.2
            if cand.field_kind == "genre" and "genre" in prompt_needs:
                score += 2.2
            if cand.field_kind == "confirm_password" and "confirm_password" not in prompt_needs:
                score -= 0.6
            if cand.field_kind == "search" and (prompt_needs - {"search"}):
                score -= 3.8
            if cand.field_kind == "sort" and "year" in prompt_needs and any(
                "year" in set(stats.get("field_kinds") or set()) for stats in group_stats.values()
            ):
                score -= 4.4
            if {"username", "email", "password", "confirm_password"}.intersection(prompt_needs):
                if cand.role == "input" and cand.field_kind in {"text", "name", "search"} and not (
                    {"username", "email", "password", "confirm_password"} & group_kinds
                ):
                    score -= 4.6
            if group_key in relevant_groups:
                score += 3.8
                if cand.role in {"input", "select", "button"}:
                    score += 1.0
            if {"username", "password"}.issubset(group_kinds) or {"email", "password"}.issubset(group_kinds):
                score += 2.6
            if {"username", "email", "password"}.intersection(group_kinds) and cand.role in {"input", "button"}:
                score += 1.4
            if cand.id in visual_hints:
                score += 4.5
            if cand.id in obs_hints:
                score += 3.2
            same_focus_region = False
            if focus_region_id and cand.region_id and cand.region_id == focus_region_id:
                same_focus_region = True
            elif focus_region_context and cand_context and cand_context == focus_region_context:
                same_focus_region = True
            elif focus_region_ids and cand.id in focus_region_ids:
                same_focus_region = True
            if same_focus_region:
                score += 4.4
                if cand.role in {"button", "input", "select"}:
                    score += 1.8
                if last_action_type in {"typeaction", "selectdropdownoptionaction", "clickaction"}:
                    score += 1.2
                if cand.field_kind == "pager":
                    score -= 4.0
                    if has_same_region_commit_controls:
                        score -= 2.0
            elif focus_region_id and focus_region_id in set(cand.region_ancestor_ids or []):
                score += 2.6
                if cand.role in {"button", "input", "select"}:
                    score += 0.8
            elif focus_recent_regions and cand.region_id and cand.region_id in focus_recent_regions:
                score += 0.8
            elif focus_region_id or focus_region_context or focus_region_ids:
                if cand.role in {"link", "button"}:
                    score -= 1.8
                if cand.role in {"input", "select"} and not prompt_needs.intersection({cand.field_kind}):
                    score -= 1.2
            if cand.region_id and cand.region_id in blocked_regions:
                score -= 3.4
            same_active_group = False
            if active_group_id and cand.group_id and cand.group_id == active_group_id:
                same_active_group = True
            elif active_group_context and cand_context and cand_context == active_group_context:
                same_active_group = True
            elif active_group_ids and cand.id in active_group_ids:
                same_active_group = True
            if same_active_group:
                score += 2.4
                if cand.role in {"button", "input", "select"}:
                    score += 1.2
                if last_action_type in {"typeaction", "selectdropdownoptionaction"} and cand.role == "button":
                    score += 2.8
                if last_action_type == "selectdropdownoptionaction" and cand.role == "select":
                    score -= 1.6
                if cand.role == "link":
                    score -= 1.0
            relevant_form_field_kinds = {
                kind
                for kind in group_kinds
                if kind in {"username", "email", "password", "confirm_password"}
                or kind in prompt_needs
            }
            has_local_multifield_form = int((group_stats.get(group_key) or {}).get("input_like") or 0) >= 2
            remaining_relevant_kinds = {
                kind for kind in relevant_form_field_kinds if kind not in group_typed_kinds
            }
            input_already_typed = (
                cand.role == "input"
                and (
                    (cand.id and cand.id in typed_candidate_ids)
                    or (sel_sig and sel_sig in typed_selector_sigs)
                )
            )
            if input_already_typed:
                score -= 18.0
                if cand.field_kind in remaining_relevant_kinds:
                    score += 2.0
            if has_local_multifield_form and relevant_form_field_kinds:
                if cand.role == "input" and cand.field_kind in remaining_relevant_kinds:
                    score += 7.0
                    if same_active_group:
                        score += 2.0
                if self._looks_submit_like(cand):
                    if remaining_relevant_kinds:
                        score -= 18.0
                        if same_active_group:
                            score -= 3.0
                    elif group_typed_inputs > 0:
                        score += 6.5
                        if same_active_group:
                            score += 2.0
                if cand.role == "link" and remaining_relevant_kinds:
                    score -= 4.0
            if has_local_multifield_form and credential_needs:
                if cand.role == "input" and cand.field_kind in {"username", "email", "password", "confirm_password"}:
                    if cand.field_kind in remaining_relevant_kinds:
                        score += 6.0
                if cand.role in {"link", "button"} and not self._looks_submit_like(cand) and not action_tags.intersection({"auth_login", "auth_register"}):
                    score -= 4.0
            cand_path = str(urlsplit(str(cand.href or "")).path or "").rstrip("/") or "/"
            if cand.role == "link" and cand.href and cand_path == current_path:
                score -= 2.6
            if cand.role == "input" and task_constraints:
                for key, value in task_constraints.items():
                    key_blob = str(key or "").replace("_", " ").lower()
                    if key_blob and any(tok in blob for tok in _tokenize(key_blob)):
                        score += 3.2
                    value_norm = str(value or "").strip().lower()
                    if value_norm and len(value_norm) >= 2 and value_norm in blob:
                        score += 1.2
            if unmet_constraint_keys:
                score += 3.0 + min(3.0, float(len(unmet_constraint_keys)))
                if any(constraint_matches.get(key) == "value" for key in unmet_constraint_keys):
                    score += 4.2
                elif any(
                    constraint_matches.get(key) == "field" and int(exact_value_match_keys.get(key) or 0) > 0
                    for key in unmet_constraint_keys
                ):
                    if cand.role in {"button", "link"}:
                        score -= 3.6
            elif constraint_keys:
                if cand.role in {"button", "link"} and cand.field_kind not in {"submit", "auth_entry", "account_create"}:
                    score -= 3.2
                elif cand.role == "select":
                    score -= 2.0
            if current_value_already_matches:
                score -= 22.0
                if cand.role == "select":
                    score -= 8.0
                if cand.field_kind in prompt_needs:
                    score -= 4.0
            if has_visible_constraint_match and cand.role in {"link", "button"} and overlap:
                score += 4.0
            if task_constraints or prompt_needs:
                if cand.role in {"input", "select", "button"} and not cand.href and (not has_relevant_form_group or group_key in relevant_groups):
                    score += 1.8
                elif cand.role == "link" and cand.href and has_relevant_form_group:
                    score -= 1.2
                if cand.role == "link" and cand.href and has_direct_constraint_controls and not constraint_keys:
                    score -= 5.0
                elif cand.role in {"link", "button"} and overlap:
                    score += 1.5
                if not has_relevant_form_group:
                    if cand.role == "button" and cand.field_kind == "submit":
                        if credential_needs:
                            score -= 2.8
                        else:
                            score += 2.4
                    if cand.role == "input" and credential_needs and cand.field_kind == "search":
                        score -= 2.0
                    if cand.role == "link" and credential_needs and overlap == 0:
                        score -= 1.6
                    if cand.role == "button" and credential_needs and overlap == 0:
                        score -= 1.2
                    if cand.role == "link" and cand.field_kind == "link" and overlap == 0:
                        score -= 2.6
                    if cand.role == "link" and int((group_stats.get(group_key) or {}).get("input_like") or 0) == 0:
                        if int((group_stats.get(group_key) or {}).get("context_len") or 0) >= 220:
                            score -= 2.2
            if mode == "POPUP":
                if any(k in blob for k in ("accept", "reject", "agree", "close", "dismiss", "continue")):
                    score += 6.0
            if mode == "EXTRACT" and cand.role == "link":
                score += 0.8
            if cand.field_kind == "pager":
                score -= 1.6
                if "page" in prompt_needs:
                    score += 0.8
            if bool(flags.get("search_box")) and cand.role == "input" and "search" in blob:
                score += 3.0 if not (prompt_needs - {"search"}) else -2.0
            if mutation_ops and not has_mutation_controls:
                if "auth_register" in action_tags:
                    score += 14.0 if not task_has_credentials else 5.5
                elif "auth_login" in action_tags:
                    score += 2.4 if not task_has_credentials else 10.0
                elif "manage_nav" in action_tags:
                    score += 4.8
                if cand.field_kind == "search" or (cand.role in {"link", "button", "input"} and "search" in blob):
                    score -= 5.2
                if cand.role == "link" and overlap and not action_tags:
                    score -= 1.8
                elif cand.role == "button" and overlap == 0:
                    score -= 2.2
            if mutation_ops and has_mutation_controls:
                if action_tags.intersection(mutation_ops):
                    score += 11.0
                elif cand.role in {"input", "select"} and cand.field_kind not in prompt_needs and not action_tags:
                    score -= 6.0
                elif cand.field_kind in {"search", "sort"}:
                    score -= 5.0
            if delete_only_task and has_non_form_controls and not has_password_input_visible:
                if cand.role in {"input", "select"} and not action_tags:
                    score -= 7.5
                elif cand.role in {"button", "link"}:
                    score += 1.8
                if is_section_switch:
                    score += 6.0
                    if cand.ui_state == "inactive":
                        score += 2.5
                    elif cand.ui_state == "active":
                        score -= 4.0
                if self._looks_submit_like(cand) and not action_tags and not is_section_switch:
                    score -= 5.0
            scored.append((score, cand))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [cand for _, cand in scored[: max(1, int(top_k or 30))]]


class ObsBuilder:
    def _group_label_for_candidate(self, cand: Candidate) -> str:
        return self._short_group_label(cand.context, default=cand.id)

    def _task_operation_hints(self, task: str) -> set[str]:
        text = str(task or "").lower()
        ops: set[str] = set()
        if re.search(r"\b(add|create|new|insert)\b", text):
            ops.add("create")
        if re.search(r"\b(edit|update|modify|change)\b", text):
            ops.add("update")
        if re.search(r"\b(delete|remove|erase|discard)\b", text):
            ops.add("delete")
        if re.search(r"\b(log ?in|sign in|authenticate)\b", text):
            ops.add("auth_login")
        if re.search(r"\b(register|sign up|signup|create account)\b", text):
            ops.add("auth_register")
        return ops

    def _task_has_explicit_credentials(self, task: str) -> bool:
        prompt = str(task or "")
        constraints = _task_constraints(prompt)
        for key in ("username", "email", "password", "confirm_password"):
            value = str(constraints.get(key) or "").strip()
            if value:
                return True
        if re.search(r"<\s*(username|email|password|signup_email|signup_password)\s*>", prompt, flags=re.I):
            return True
        if re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", prompt):
            return True
        return False

    def _candidate_action_tags(self, cand: Candidate) -> set[str]:
        blob = " ".join([cand.text, cand.href, cand.field_hint, cand.field_kind, cand.group_label]).lower()
        tags: set[str] = set()
        if re.search(r"\b(delete|remove|erase|discard)\b", blob):
            tags.add("delete")
        if re.search(r"\b(edit|update|modify|change)\b", blob):
            tags.add("update")
        if re.search(r"\b(add|create|new)\b", blob):
            tags.add("create")
        if re.search(r"\b(log ?in|sign in|authenticate)\b", blob):
            tags.add("auth_login")
        if re.search(r"\b(register|sign up|signup|create account)\b", blob):
            tags.add("auth_register")
        if re.search(r"\b(profile|account|manage|dashboard)\b", blob):
            tags.add("manage_nav")
        return tags

    def _capability_gap_summary(
        self,
        *,
        prompt: str,
        state: AgentState,
        candidates: List[Candidate],
    ) -> Dict[str, Any]:
        task_ops = self._task_operation_hints(prompt)
        mutation_ops = sorted(task_ops.intersection({"create", "update", "delete"}))
        available_ops: set[str] = set()
        auth_entry_available = False
        register_available = False
        login_available = False
        management_entry_available = False
        typed_or_active_ids = set(state.form_progress.typed_candidate_ids) | set(state.form_progress.active_group_candidate_ids)
        for cand in candidates:
            tags = self._candidate_action_tags(cand)
            available_ops.update(tags.intersection({"create", "update", "delete"}))
            if "auth_register" in tags:
                auth_entry_available = True
                register_available = True
            if "auth_login" in tags:
                auth_entry_available = True
                login_available = True
            if "manage_nav" in tags:
                management_entry_available = True
            if cand.role in {"input", "select", "textarea"} and cand.id in typed_or_active_ids:
                management_entry_available = management_entry_available or bool(tags.intersection({"create", "update", "delete"}))
        missing_ops = [op for op in mutation_ops if op not in available_ops]
        read_only_for_task = bool(mutation_ops) and bool(missing_ops)
        task_has_credentials = self._task_has_explicit_credentials(prompt)
        preferred_transition = ""
        if read_only_for_task:
            if not task_has_credentials and register_available:
                preferred_transition = "register"
            elif task_has_credentials and login_available:
                preferred_transition = "login"
            elif management_entry_available:
                preferred_transition = "manage"
            elif register_available:
                preferred_transition = "register"
            elif login_available:
                preferred_transition = "login"
        local_mutation_controls_visible = bool(mutation_ops) and any(op in available_ops for op in mutation_ops)
        active_manage_context = bool(local_mutation_controls_visible) and (
            bool(state.form_progress.active_group_candidate_ids) or bool(state.form_progress.active_group_label)
        )
        strategy_parts: List[str] = []
        if read_only_for_task:
            strategy_parts.append(
                "Current page appears read-only for the requested operation: missing "
                + ", ".join(missing_ops[:3])
                + "."
            )
            if preferred_transition == "register":
                strategy_parts.append("Prefer registration to reach an authenticated management context.")
            elif preferred_transition == "login":
                strategy_parts.append("Prefer sign-in to reach an authenticated management context.")
            elif preferred_transition == "manage":
                strategy_parts.append("Move toward a profile/account/manage area before acting locally.")
        elif local_mutation_controls_visible:
            strategy_parts.append("Relevant mutation controls are visible on this page.")
            strategy_parts.append("Prefer those local controls over unrelated profile or search fields.")
        elif active_manage_context:
            strategy_parts.append("Stay within the active management context instead of rediscovering the page.")
        strategy_summary = _norm_ws(" ".join(strategy_parts))[:320]
        return {
            "task_operations": sorted(task_ops),
            "available_operations": sorted(available_ops),
            "missing_task_operations": missing_ops,
            "auth_entry_available": bool(auth_entry_available),
            "management_entry_available": bool(management_entry_available),
            "task_has_explicit_credentials": bool(task_has_credentials),
            "read_only_for_task": bool(read_only_for_task),
            "preferred_transition": preferred_transition,
            "local_mutation_controls_visible": bool(local_mutation_controls_visible),
            "active_manage_context": bool(active_manage_context),
            "strategy_summary": strategy_summary,
        }

    def build_text_ir(self, snapshot_html: str) -> Dict[str, Any]:
        html = str(snapshot_html or "")
        if not html:
            return {
                "visible_text": "",
                "visible_lines": [],
                "headings": [],
                "title": "",
                "forms": [],
                "control_groups": [],
                "cards": [],
                "page_facts": [],
                "html_excerpt": "",
            }
        if BeautifulSoup is None:
            cleaned = _norm_ws(re.sub(r"<[^>]+>", " ", html))
            return {
                "visible_text": cleaned[:10000],
                "visible_lines": [cleaned[:240]] if cleaned else [],
                "headings": [],
                "title": "",
                "forms": [],
                "control_groups": [],
                "cards": [],
                "page_facts": [],
                "html_excerpt": str(html)[:12000],
            }
        try:
            soup = BeautifulSoup(html, "lxml")
            for node in soup(["script", "style", "noscript"]):
                try:
                    node.decompose()
                except Exception:
                    pass
            title = ""
            try:
                title_tag = soup.find("title")
                title = _norm_ws(title_tag.get_text(" ", strip=True) if title_tag else "")
            except Exception:
                title = ""
            headings: List[str] = []
            for h in soup.find_all(["h1", "h2", "h3"], limit=25):
                txt = _norm_ws(h.get_text(" ", strip=True))
                if txt:
                    headings.append(txt[:220])
            forms = self._extract_forms(soup)
            control_panels = self._extract_control_panels(soup)
            visible_lines = self._visible_lines(soup)
            page_facts = self._extract_page_facts(soup=soup, visible_lines=visible_lines)
            value_lines = self._extract_visible_value_lines(visible_lines=visible_lines)
            html_excerpt = ""
            try:
                html_excerpt = str(soup.body or soup)[:12000]
            except Exception:
                html_excerpt = str(html)[:12000]
            text = _norm_ws(soup.get_text(" ", strip=True))
            return {
                "visible_text": text[:10000],
                "visible_lines": visible_lines[:120],
                "headings": _dedupe_keep_order(headings, 20),
                "title": title[:260],
                "forms": forms[:8],
                "control_groups": (self._control_groups_from_forms(forms) + control_panels)[:8],
                "cards": [],
                "page_facts": page_facts[:12],
                "value_lines": value_lines[:16],
                "relevant_lines": [],
                "html_excerpt": html_excerpt,
            }
        except Exception:
            cleaned = _norm_ws(re.sub(r"<[^>]+>", " ", html))
            return {
                "visible_text": cleaned[:10000],
                "visible_lines": [cleaned[:240]] if cleaned else [],
                "headings": [],
                "title": "",
                "forms": [],
                "control_groups": [],
                "cards": [],
                "page_facts": [],
                "value_lines": [],
                "relevant_lines": [],
                "html_excerpt": str(html)[:12000],
            }

    def _visible_lines(self, soup: Any) -> List[str]:
        try:
            raw = str((soup.body or soup).get_text("\n", strip=True))
        except Exception:
            raw = ""
        lines = [_norm_ws(line)[:220] for line in raw.splitlines() if _norm_ws(line)]
        return _dedupe_keep_order(lines, 160)

    def _extract_page_facts(self, *, soup: Any, visible_lines: List[str]) -> List[str]:
        facts: List[str] = []

        def section_context(idx: int, current_label: str) -> str:
            best = ""
            best_score = -1
            current = _norm_ws(current_label).lower()
            for back in range(max(0, idx - 12), idx):
                candidate = _norm_ws(visible_lines[back])
                if not _labelish_text(candidate):
                    continue
                if _valueish_text(candidate):
                    continue
                lowered = candidate.lower()
                if lowered == current:
                    continue
                score = 0
                if len(candidate.split()) <= 3:
                    score += 3
                if len(candidate) <= 24:
                    score += 2
                if candidate.istitle() or candidate.isupper():
                    score += 1
                if lowered.startswith("view "):
                    score -= 2
                if "toggle" in lowered or "search" == lowered:
                    score -= 3
                if score > best_score:
                    best = candidate
                    best_score = score
            return best

        # Table / row pairs
        try:
            for row in soup.find_all(["tr"], limit=80):
                cells = [
                    _norm_ws(cell.get_text(" ", strip=True))
                    for cell in row.find_all(["th", "td"], limit=4)
                    if _norm_ws(cell.get_text(" ", strip=True))
                ]
                if len(cells) >= 2 and _labelish_text(cells[0]) and _valueish_text(cells[1]):
                    facts.append(f"{cells[0]}: {cells[1]}"[:180])
        except Exception:
            pass

        # Definition lists
        try:
            for dt in soup.find_all("dt", limit=60):
                label = _norm_ws(dt.get_text(" ", strip=True))
                dd = dt.find_next_sibling("dd")
                value = _norm_ws(dd.get_text(" ", strip=True)) if dd is not None else ""
                if _labelish_text(label) and _valueish_text(value):
                    facts.append(f"{label}: {value}"[:180])
        except Exception:
            pass

        # Metric cards / generic block pairs
        try:
            for node in soup.find_all(["div", "section", "article", "li"], limit=160):
                strings = []
                for piece in node.stripped_strings:
                    txt = _norm_ws(piece)
                    if txt:
                        strings.append(txt[:120])
                    if len(strings) >= 4:
                        break
                if len(strings) < 2:
                    continue
                label = strings[0]
                value = strings[1]
                if _labelish_text(label) and _valueish_text(value):
                    facts.append(f"{label}: {value}"[:180])
        except Exception:
            pass

        # Adjacent visible lines as label/value pairs
        for idx in range(max(0, len(visible_lines) - 1)):
            label = str(visible_lines[idx] or "")
            value = str(visible_lines[idx + 1] or "")
            if _labelish_text(label) and _valueish_text(value):
                facts.append(f"{label}: {value}"[:180])
                section_label = section_context(idx, label)
                if section_label:
                    facts.append(f"{section_label} - {label}: {value}"[:180])

        return _dedupe_keep_order(facts, 16)

    def _extract_visible_value_lines(self, *, visible_lines: List[str]) -> List[str]:
        lines: List[str] = []
        total = len(visible_lines)

        def section_context(idx: int, current_label: str) -> str:
            best = ""
            best_score = -1
            current = _norm_ws(current_label).lower()
            for back in range(max(0, idx - 12), max(0, idx - 1)):
                candidate = _norm_ws(visible_lines[back])
                if not _labelish_text(candidate):
                    continue
                if _valueish_text(candidate):
                    continue
                lowered = candidate.lower()
                if lowered == current:
                    continue
                score = 0
                if len(candidate.split()) <= 3:
                    score += 3
                if len(candidate) <= 24:
                    score += 2
                if candidate.istitle() or candidate.isupper():
                    score += 1
                if lowered.startswith("view "):
                    score -= 2
                if "toggle" in lowered or "search" == lowered:
                    score -= 3
                if score > best_score:
                    best = candidate
                    best_score = score
            return best

        for idx, raw in enumerate(visible_lines):
            line = _norm_ws(raw)
            if not _value_line_text(line):
                continue
            label = _norm_ws(visible_lines[idx - 1]) if idx > 0 else ""
            if _labelish_text(label):
                lines.append(f"{label}: {line}"[:180])
                section_label = section_context(idx, label)
                if section_label:
                    lines.append(f"{section_label} - {label}: {line}"[:180])
            lines.append(line[:180])
            next_line = _norm_ws(visible_lines[idx + 1]) if idx + 1 < total else ""
            if _labelish_text(line) and _valueish_text(next_line):
                lines.append(f"{line}: {next_line}"[:180])
        return _dedupe_keep_order(lines, 24)

    def _relevant_visible_lines(
        self,
        *,
        prompt: str,
        visible_lines: List[str],
        page_facts: List[str],
        value_lines: List[str],
    ) -> List[str]:
        ranked: List[tuple[int, str]] = []
        seen: set[str] = set()
        for source in [page_facts[:12], value_lines[:16], visible_lines[:80]]:
            for item in source:
                clean = _norm_ws(item)
                if not clean or clean in seen:
                    continue
                seen.add(clean)
                score = _fact_overlap_score(prompt, clean)
                if clean in value_lines:
                    score += 2
                if clean in page_facts:
                    score += 2
                if _value_line_text(clean):
                    score += 1
                if ":" in clean:
                    score += 1
                ranked.append((score, clean[:180]))
        ranked.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
        selected = [text for score, text in ranked if score > 0][:16]
        if selected:
            return selected
        fallback: List[str] = []
        for item in (value_lines[:12] + visible_lines[:12]):
            clean = _norm_ws(item)
            if clean and clean not in fallback:
                fallback.append(clean[:180])
        return fallback[:16]

    def _likely_answers(self, *, prompt: str, page_facts: List[str]) -> List[str]:
        scored: List[tuple[int, str]] = []
        for fact in page_facts:
            clean = _norm_ws(fact)
            if not clean:
                continue
            score = _fact_overlap_score(prompt, clean) + (_anchor_overlap_score(prompt, clean) * 2)
            if ":" in clean:
                score += 1
            if _valueish_text(clean.split(":", 1)[-1] if ":" in clean else clean):
                score += 1
            scored.append((score, clean[:180]))
        scored.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
        return [fact for score, fact in scored[:6] if score > 0] or [fact for _, fact in scored[:3]]

    def _extract_forms(self, soup: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for idx, form in enumerate(soup.find_all("form", limit=12), start=1):
            try:
                attrs = form.attrs if isinstance(getattr(form, "attrs", None), dict) else {}
                form_text = _norm_ws(form.get_text(" ", strip=True))
                controls: List[Dict[str, Any]] = []
                commit_controls: List[str] = []
                for control in form.find_all(["input", "textarea", "select", "button"], limit=20):
                    c_attrs = control.attrs if isinstance(getattr(control, "attrs", None), dict) else {}
                    tag = str(getattr(control, "name", "") or "").lower()
                    label = self._field_hint_from_node(control)
                    required = ("required" in c_attrs) or (str(c_attrs.get("aria-required") or "").strip().lower() == "true")
                    options: List[str] = []
                    current_value = _norm_ws(c_attrs.get("value"))
                    if tag == "select":
                        selected_texts: List[str] = []
                        for option in control.find_all("option", limit=12):
                            opt_text = _norm_ws(option.get_text(" ", strip=True))
                            opt_value = _norm_ws(option.get("value"))
                            opt_blob = _candidate_text(opt_text, opt_value)
                            if opt_blob:
                                options.append(opt_blob[:80])
                            selected_attr = str(option.get("selected") or "").strip().lower()
                            if selected_attr in {"selected", "true"} or option.has_attr("selected"):
                                selected_blob = _candidate_text(opt_text, opt_value)
                                if selected_blob:
                                    selected_texts.append(selected_blob[:80])
                        if selected_texts:
                            current_value = _candidate_text(*selected_texts[:2])[:80]
                    controls.append(
                        {
                            "tag": tag,
                            "type": _norm_ws(c_attrs.get("type")).lower(),
                            "name": _norm_ws(c_attrs.get("name"))[:80],
                            "id": _norm_ws(c_attrs.get("id"))[:80],
                            "label": label[:120],
                            "placeholder": _norm_ws(c_attrs.get("placeholder"))[:120],
                            "aria_label": _norm_ws(c_attrs.get("aria-label"))[:120],
                            "text": _norm_ws(control.get_text(" ", strip=True))[:120],
                            "value": current_value[:80],
                            "required": bool(required),
                            "options": _dedupe_keep_order(options, 8),
                        }
                    )
                    role_blob = " ".join(
                        [
                            tag,
                            str(c_attrs.get("type") or ""),
                            label,
                            _norm_ws(c_attrs.get("placeholder")),
                            _norm_ws(c_attrs.get("aria-label")),
                            _norm_ws(control.get_text(" ", strip=True)),
                        ]
                    ).lower()
                    if tag == "button" or str(c_attrs.get("type") or "").strip().lower() in {"submit", "button"} or any(
                        token in role_blob for token in ("submit", "save", "apply", "search", "find", "continue", "register", "sign up", "log in", "sign in")
                    ):
                        commit_label = _candidate_text(
                            label,
                            c_attrs.get("aria-label"),
                            c_attrs.get("placeholder"),
                            control.get_text(" ", strip=True),
                            c_attrs.get("value"),
                        )
                        if commit_label:
                            commit_controls.append(commit_label[:120])
                if not controls and not form_text:
                    continue
                out.append(
                    {
                        "id": _candidate_text(attrs.get("id"), f"form_{idx}")[:80],
                        "name": _norm_ws(attrs.get("name"))[:80],
                        "method": _norm_ws(attrs.get("method")).upper()[:20],
                        "action": _norm_ws(attrs.get("action"))[:180],
                        "text": form_text[:360],
                        "controls": controls[:12],
                        "commit_controls": _dedupe_keep_order(commit_controls, 6),
                    }
                )
            except Exception:
                continue
        return out

    def _field_hint_from_node(self, node: Any) -> str:
        try:
            attrs = node.attrs if isinstance(getattr(node, "attrs", None), dict) else {}
            node_id = _norm_ws(attrs.get("id"))
            if node_id and hasattr(node, "find_parent"):
                root = node.find_parent("html")
                if root is not None:
                    label = root.find("label", attrs={"for": node_id})
                    if label is not None:
                        txt = _norm_ws(label.get_text(" ", strip=True))
                        if txt:
                            return txt[:120]
            parent = getattr(node, "parent", None)
            if parent is not None and str(getattr(parent, "name", "") or "").lower() == "label":
                txt = _norm_ws(parent.get_text(" ", strip=True))
                if txt:
                    return txt[:120]
            cur = node
            for _ in range(3):
                cur = getattr(cur, "parent", None)
                if cur is None:
                    break
                txt = _norm_ws(cur.get_text(" ", strip=True))
                if txt:
                    return txt[:120]
        except Exception:
            return ""
        return ""

    def _short_group_label(self, text: str, *, default: str) -> str:
        clean = _norm_ws(text)
        if not clean:
            return default
        words = clean.split()
        return " ".join(words[:16])[:160]

    def _control_groups_from_forms(self, forms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        groups: List[Dict[str, Any]] = []
        for idx, form in enumerate(forms[:8], start=1):
            if not isinstance(form, dict):
                continue
            controls = form.get("controls") if isinstance(form.get("controls"), list) else []
            if not controls:
                continue
            summary: List[str] = []
            for control in controls[:10]:
                if not isinstance(control, dict):
                    continue
                bits = [
                    str(control.get("tag") or ""),
                    str(control.get("type") or ""),
                    _candidate_text(
                        control.get("label"),
                        control.get("placeholder"),
                        control.get("aria_label"),
                        control.get("name"),
                        control.get("text"),
                    ),
                ]
                blob = _norm_ws(" ".join(x for x in bits if x))
                if blob:
                    if control.get("options"):
                        blob += " options=" + ",".join(str(x)[:40] for x in (control.get("options") or [])[:5])
                    current_value = _candidate_text(control.get("value"))
                    if current_value:
                        blob += f" current={current_value[:40]}"
                    summary.append(blob[:180])
            if not summary:
                continue
            label = self._short_group_label(
                _candidate_text(form.get("name"), form.get("text"), form.get("id")),
                default=f"group_{idx}",
            )
            groups.append(
                {
                    "group_id": f"form_group_{idx}",
                    "kind": "form",
                    "label": label,
                    "control_count": len(controls),
                    "controls": summary[:8],
                }
            )
        return groups[:8]

    def _extract_control_panels(self, soup: Any) -> List[Dict[str, Any]]:
        groups: List[Dict[str, Any]] = []
        seen_keys: set[str] = set()
        for idx, node in enumerate(soup.find_all(["section", "div", "aside", "nav"], limit=80), start=1):
            try:
                if node.find("form") is not None:
                    continue
                controls = node.find_all(["input", "textarea", "select", "button"], limit=12)
                if len(controls) < 2:
                    continue
                node_text = _norm_ws(node.get_text(" ", strip=True))
                if not node_text:
                    continue
                node_id = _norm_ws((node.attrs or {}).get("id")) if isinstance(getattr(node, "attrs", None), dict) else ""
                key = node_id or node_text[:200]
                if not key or key in seen_keys:
                    continue
                summary: List[str] = []
                for control in controls[:8]:
                    c_attrs = control.attrs if isinstance(getattr(control, "attrs", None), dict) else {}
                    tag = str(getattr(control, "name", "") or "").lower()
                    label = self._field_hint_from_node(control)
                    options: List[str] = []
                    current_value = _norm_ws(c_attrs.get("value"))
                    if tag == "select":
                        selected_texts: List[str] = []
                        for option in control.find_all("option", limit=12):
                            opt_text = _norm_ws(option.get_text(" ", strip=True))
                            opt_value = _norm_ws(option.get("value"))
                            opt_blob = _candidate_text(opt_text, opt_value)
                            if opt_blob:
                                options.append(opt_blob[:80])
                            selected_attr = str(option.get("selected") or "").strip().lower()
                            if selected_attr in {"selected", "true"} or option.has_attr("selected"):
                                selected_blob = _candidate_text(opt_text, opt_value)
                                if selected_blob:
                                    selected_texts.append(selected_blob[:80])
                        if selected_texts:
                            current_value = _candidate_text(*selected_texts[:2])[:80]
                    bits = [
                        tag,
                        _norm_ws(c_attrs.get("type")).lower(),
                        _candidate_text(
                            label,
                            c_attrs.get("aria-label"),
                            c_attrs.get("placeholder"),
                            c_attrs.get("name"),
                            control.get_text(" ", strip=True),
                        ),
                    ]
                    blob = _norm_ws(" ".join(x for x in bits if x))
                    if not blob:
                        continue
                    if options:
                        blob += " options=" + ",".join(str(x)[:40] for x in options[:5])
                    if current_value:
                        blob += f" current={current_value[:40]}"
                    summary.append(blob[:180])
                if len(summary) < 2:
                    continue
                seen_keys.add(key)
                groups.append(
                    {
                        "group_id": f"panel_group_{idx}",
                        "kind": "controls",
                        "label": self._short_group_label(node_text, default=f"controls_{idx}"),
                        "control_count": len(summary),
                        "controls": summary[:8],
                    }
                )
            except Exception:
                continue
        groups.sort(key=lambda item: int(item.get("control_count") or 0), reverse=True)
        return groups[:8]

    def _candidate_groups(self, candidates: List[Candidate]) -> List[Dict[str, Any]]:
        groups: Dict[str, Dict[str, Any]] = {}
        for cand in candidates:
            context = _norm_ws(cand.context)
            key = context[:320] if context else ""
            if not key:
                continue
            group = groups.get(key)
            if group is None:
                group = {
                    "context": key,
                    "candidate_ids": [],
                    "roles": {},
                    "controls": [],
                    "links": [],
                }
                groups[key] = group
            group["candidate_ids"].append(cand.id)
            group["roles"][cand.role] = int(group["roles"].get(cand.role) or 0) + 1
            item = {
                "id": cand.id,
                "role": cand.role,
                "text": cand.text[:120],
                "field_hint": cand.field_hint[:80],
                "href": cand.href[:140],
            }
            if cand.role in {"input", "select", "button"}:
                group["controls"].append(item)
            elif cand.role == "link":
                group["links"].append(item)
        ranked: List[Dict[str, Any]] = []
        for key, group in groups.items():
            roles = group["roles"] if isinstance(group.get("roles"), dict) else {}
            n_controls = len(group["controls"])
            n_links = len(group["links"])
            total = len(group["candidate_ids"])
            if total < 2:
                continue
            label = self._short_group_label(key, default="group")
            if n_controls >= max(2, n_links):
                kind = "controls"
                sample = group["controls"][:6]
            else:
                kind = "items"
                sample = group["links"][:4] + group["controls"][:2]
            ranked.append(
                {
                    "group_id": f"group_{hashlib.sha1(key.encode('utf-8', errors='ignore')).hexdigest()[:10]}",
                    "kind": kind,
                    "label": label,
                    "context": key[:280],
                    "candidate_ids": group["candidate_ids"][:10],
                    "role_counts": roles,
                    "items": sample,
                    "_sort": (min(8, total) + min(4, n_controls if kind == "controls" else n_links)),
                }
            )
        ranked.sort(key=lambda item: int(item.get("_sort") or 0), reverse=True)
        for item in ranked:
            item.pop("_sort", None)
        return ranked[:12]

    def _card_summaries(self, groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cards: List[Dict[str, Any]] = []
        for group in groups:
            if not isinstance(group, dict) or str(group.get("kind") or "") != "items":
                continue
            items = group.get("items") if isinstance(group.get("items"), list) else []
            actions: List[Dict[str, Any]] = []
            facts: List[str] = []
            for item in items[:6]:
                if not isinstance(item, dict):
                    continue
                blob = _candidate_text(item.get("text"), item.get("field_hint"), item.get("href"))
                if blob:
                    facts.append(blob[:90])
                actions.append(
                    {
                        "id": str(item.get("id") or "")[:80],
                        "role": str(item.get("role") or "")[:20],
                        "text": str(item.get("text") or "")[:100],
                        "href": str(item.get("href") or "")[:120],
                    }
                )
            if actions:
                cards.append(
                    {
                        "label": str(group.get("label") or "")[:160],
                        "facts": _dedupe_keep_order(facts, 4),
                        "actions": actions[:4],
                    }
                )
        return cards[:8]

    def _page_observations(
        self,
        *,
        prompt: str,
        flags: Dict[str, Any],
        state: AgentState,
        text_ir: Dict[str, Any],
        candidates: List[Candidate],
        history_recent: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        text_ir = text_ir if isinstance(text_ir, dict) else {}
        role_counts: Dict[str, int] = {}
        type_counts: Dict[str, int] = {}
        for cand in candidates:
            role_counts[cand.role] = int(role_counts.get(cand.role) or 0) + 1
            type_counts[cand.type] = int(type_counts.get(cand.type) or 0) + 1
        focus_region_id = str(state.focus_region.region_id or "").strip()
        focus_region_context = _norm_ws(state.focus_region.region_context)
        focus_candidate_ids = set(state.focus_region.candidate_ids)
        local_candidates = 0
        for cand in candidates:
            if (
                (focus_region_id and cand.region_id and cand.region_id == focus_region_id)
                or (focus_region_context and _norm_ws(cand.context) == focus_region_context)
                or (focus_candidate_ids and cand.id in focus_candidate_ids)
            ):
                local_candidates += 1
        recent_failures = sum(
            1
            for item in history_recent
            if isinstance(item, dict) and ((not bool(item.get("exec_ok", True))) or bool(_candidate_text(item.get("error"))))
        )
        capability_gap = self._capability_gap_summary(prompt=prompt, state=state, candidates=candidates)
        page_facts = text_ir.get("page_facts") if isinstance(text_ir.get("page_facts"), list) else []
        value_lines = text_ir.get("value_lines") if isinstance(text_ir.get("value_lines"), list) else []
        relevant_lines = self._relevant_visible_lines(
            prompt=prompt,
            visible_lines=(text_ir.get("visible_lines") if isinstance(text_ir.get("visible_lines"), list) else []),
            page_facts=page_facts,
            value_lines=value_lines,
        )
        likely_answers = self._likely_answers(prompt=prompt, page_facts=page_facts)
        headings = text_ir.get("headings") if isinstance(text_ir.get("headings"), list) else []
        visible_text = str(text_ir.get("visible_text") or "")
        return {
            "candidate_count": len(candidates),
            "local_candidate_count": int(local_candidates),
            "global_candidate_count": max(0, int(len(candidates) - local_candidates)),
            "role_counts": role_counts,
            "type_counts": type_counts,
            "heading_count": len(text_ir.get("headings") if isinstance(text_ir.get("headings"), list) else []),
            "form_count": len(text_ir.get("forms") if isinstance(text_ir.get("forms"), list) else []),
            "control_group_count": len(text_ir.get("control_groups") if isinstance(text_ir.get("control_groups"), list) else []),
            "card_count": len(text_ir.get("cards") if isinstance(text_ir.get("cards"), list) else []),
            "visible_text_chars": len(str(text_ir.get("visible_text") or "")),
            "visual_note_count": len(state.memory.visual_notes),
            "visual_hint_count": len(state.memory.visual_element_hints),
            "active_group_candidate_count": len(state.form_progress.active_group_candidate_ids),
            "url_changed": bool(flags.get("url_changed")),
            "dom_changed": bool(flags.get("dom_changed")),
            "error_page": bool(flags.get("error_page")),
            "challenge_present": bool(flags.get("captcha_suspected")),
            "popup_present": bool(flags.get("cookie_banner")) or bool(flags.get("modal_dialog")),
            "stall_count": int(state.counters.stall_count or 0),
            "repeat_action_count": int(state.counters.repeat_action_count or 0),
            "recent_failures": recent_failures,
            "frontier_url_count": len(state.frontier.pending_urls),
            "frontier_element_count": len(state.frontier.pending_elements),
            "page_fact_count": len(page_facts),
            "value_line_count": len(value_lines),
            "likely_answers": likely_answers[:6],
            "relevant_lines": relevant_lines[:10],
            "informational_task": bool(_looks_like_informational_task(prompt)),
            "page_stats": {
                "title": str(text_ir.get("title") or "")[:200],
                "heading_count": len(headings),
                "visible_text_chars": len(visible_text),
                "links": int(role_counts.get("link") or 0),
                "controls": int(sum(int(role_counts.get(role) or 0) for role in ("button", "input", "select"))),
                "forms": len(text_ir.get("forms") if isinstance(text_ir.get("forms"), list) else []),
                "control_groups": len(text_ir.get("control_groups") if isinstance(text_ir.get("control_groups"), list) else []),
                "cards": len(text_ir.get("cards") if isinstance(text_ir.get("cards"), list) else []),
            },
            "last_effect": str(state.progress.last_effect or "")[:40],
            "no_progress_score": int(state.progress.no_progress_score or 0),
            "consecutive_no_effect_steps": int(state.progress.consecutive_no_effect_steps or 0),
            "capability_gap": capability_gap,
        }

    def _page_ir_text(self, *, prompt: str, text_ir: Dict[str, Any], candidates: List[Candidate]) -> str:
        text_ir = text_ir if isinstance(text_ir, dict) else {}
        headings = text_ir.get("headings") if isinstance(text_ir.get("headings"), list) else []
        forms = text_ir.get("forms") if isinstance(text_ir.get("forms"), list) else []
        control_groups = text_ir.get("control_groups") if isinstance(text_ir.get("control_groups"), list) else []
        cards = text_ir.get("cards") if isinstance(text_ir.get("cards"), list) else []
        active_region = text_ir.get("active_region") if isinstance(text_ir.get("active_region"), dict) else {}
        active_group = text_ir.get("active_group") if isinstance(text_ir.get("active_group"), dict) else {}
        capability_gap = text_ir.get("capability_gap") if isinstance(text_ir.get("capability_gap"), dict) else {}
        llm_extract = text_ir.get("llm_extract") if isinstance(text_ir.get("llm_extract"), dict) else {}
        page_facts = text_ir.get("page_facts") if isinstance(text_ir.get("page_facts"), list) else []
        value_lines = text_ir.get("value_lines") if isinstance(text_ir.get("value_lines"), list) else []
        relevant_lines = self._relevant_visible_lines(
            prompt=prompt,
            visible_lines=(text_ir.get("visible_lines") if isinstance(text_ir.get("visible_lines"), list) else []),
            page_facts=page_facts,
            value_lines=value_lines,
        )
        likely_answers = self._likely_answers(prompt=prompt, page_facts=page_facts)
        visible_text = str(text_ir.get("visible_text") or "")
        parts: List[str] = []
        title = _candidate_text(text_ir.get("title"))
        if title:
            parts.append(f"TITLE: {title[:260]}")
        strategy_summary = _candidate_text(capability_gap.get("strategy_summary"))
        if strategy_summary:
            parts.append("STRATEGY STATE: " + strategy_summary)
        if llm_extract:
            extract_summary = _candidate_text(llm_extract.get("summary"))
            if extract_summary:
                parts.append("OBS EXTRACT SUMMARY: " + extract_summary)
            extract_facts = llm_extract.get("facts") if isinstance(llm_extract.get("facts"), list) else []
            if extract_facts:
                parts.append("OBS EXTRACT FACTS: " + " | ".join(str(x)[:140] for x in extract_facts[:6]))
        if page_facts:
            parts.append("PAGE FACTS: " + " | ".join(str(x)[:140] for x in page_facts[:8]))
        if value_lines:
            parts.append("VISIBLE VALUE LINES: " + " | ".join(str(x)[:140] for x in value_lines[:10]))
        if likely_answers:
            parts.append("LIKELY ANSWERS ON PAGE: " + " | ".join(str(x)[:140] for x in likely_answers[:6]))
        if relevant_lines:
            parts.append("RELEVANT VISIBLE TEXT: " + " | ".join(str(x)[:160] for x in relevant_lines[:12]))
        if capability_gap:
            parts.append(
                "CAPABILITY GAP: "
                + json.dumps(
                    {
                        "task_operations": capability_gap.get("task_operations", []),
                        "available_operations": capability_gap.get("available_operations", []),
                        "missing_task_operations": capability_gap.get("missing_task_operations", []),
                        "preferred_transition": capability_gap.get("preferred_transition", ""),
                        "read_only_for_task": bool(capability_gap.get("read_only_for_task")),
                    },
                    ensure_ascii=False,
                )
            )
        if headings:
            parts.append("HEADINGS: " + " | ".join(str(x)[:160] for x in headings[:12]))
        if visible_text:
            parts.append("VISIBLE TEXT EXCERPT: " + visible_text[:1800])
        if forms:
            form_lines: List[str] = []
            for idx, form in enumerate(forms[:6], start=1):
                if not isinstance(form, dict):
                    continue
                controls = form.get("controls") if isinstance(form.get("controls"), list) else []
                control_bits: List[str] = []
                for control in controls[:6]:
                    if not isinstance(control, dict):
                        continue
                    blob = _candidate_text(
                        control.get("label"),
                        control.get("placeholder"),
                        control.get("aria_label"),
                        control.get("name"),
                        control.get("text"),
                    )
                    prefix = _candidate_text(control.get("tag"), control.get("type"))
                    if bool(control.get("required")):
                        prefix = ("required " + prefix).strip()
                    value_now = _candidate_text(control.get("value"))
                    if control.get("options"):
                        blob = (blob + " options=" + ",".join(str(x)[:30] for x in (control.get("options") or [])[:4])).strip()
                    if value_now:
                        blob = (blob + f" value={value_now[:40]}").strip()
                    combined = _norm_ws(" ".join(x for x in [prefix, blob] if x))
                    if combined:
                        control_bits.append(combined[:160])
                label = _candidate_text(form.get("name"), form.get("text"), form.get("id"), f"form_{idx}")
                commit_controls = form.get("commit_controls") if isinstance(form.get("commit_controls"), list) else []
                suffix = ""
                if commit_controls:
                    suffix = " ; commits=" + " / ".join(_norm_ws(str(x)[:60]) for x in commit_controls[:3])
                form_lines.append(f"{label[:120]} -> " + " ; ".join(control_bits[:6]) + suffix)
            if form_lines:
                parts.append("FORMS:\n" + "\n".join(f"- {line}" for line in form_lines[:6]))
        if control_groups:
            group_lines: List[str] = []
            for group in control_groups[:6]:
                if not isinstance(group, dict):
                    continue
                controls = group.get("controls") if isinstance(group.get("controls"), list) else []
                group_lines.append(
                    f"{str(group.get('label') or '')[:120]} ({int(group.get('control_count') or 0)} controls) -> "
                    + " ; ".join(str(x)[:120] for x in controls[:6])
                )
            if group_lines:
                parts.append("CONTROL GROUPS:\n" + "\n".join(f"- {line}" for line in group_lines[:6]))
        if active_region:
            region_lines: List[str] = []
            for item in (active_region.get("items") if isinstance(active_region.get("items"), list) else [])[:8]:
                if not isinstance(item, dict):
                    continue
                bits = [
                    str(item.get("id") or "")[:80],
                    _candidate_text(item.get("role"), item.get("region_kind"), item.get("text"), item.get("field_hint"), item.get("href"))[:140],
                ]
                region_lines.append(" | ".join([x for x in bits if x]))
            if region_lines:
                parts.append(
                    "FOCUSED REGION:\n"
                    + f"- {str(active_region.get('label') or '')[:120]} kind={str(active_region.get('region_kind') or '')[:40]}\n"
                    + "\n".join(f"- {line}" for line in region_lines[:8])
                )
        if active_group:
            label = str(active_group.get("label") or "")[:120]
            active_items = active_group.get("items") if isinstance(active_group.get("items"), list) else []
            active_lines: List[str] = []
            for item in active_items[:8]:
                if not isinstance(item, dict):
                    continue
                bits = [
                    str(item.get("id") or "")[:80],
                    _candidate_text(item.get("role"), item.get("text"), item.get("field_hint"), item.get("href"))[:140],
                ]
                active_lines.append(" | ".join([x for x in bits if x]))
            if active_lines:
                parts.append(
                    "ACTIVE CONTROL GROUP:\n"
                    + f"- {label} ({len(active_items)} visible related elements)\n"
                    + "\n".join(f"- {line}" for line in active_lines[:8])
                )
        if cards:
            card_lines: List[str] = []
            for idx, card in enumerate(cards[:6], start=1):
                if not isinstance(card, dict):
                    continue
                facts = card.get("facts") if isinstance(card.get("facts"), list) else []
                actions = card.get("actions") if isinstance(card.get("actions"), list) else []
                action_bits = []
                for action in actions[:3]:
                    if not isinstance(action, dict):
                        continue
                    action_bits.append(_candidate_text(action.get("text"), action.get("href"), action.get("id"))[:90])
                card_lines.append(
                    f"card[{idx}] {str(card.get('label') or '')[:120]} -> "
                    + " | ".join(str(x)[:80] for x in facts[:3])
                    + (" ; actions=" + " / ".join(action_bits) if action_bits else "")
                )
            if card_lines:
                parts.append("ITEM GROUPS:\n" + "\n".join(f"- {line}" for line in card_lines[:6]))
        if candidates:
            sample = []
            for cand in candidates[:24]:
                line = f"{cand.id}: {cand.role}/{cand.type}"
                if cand.text:
                    line += f" text={cand.text[:120]}"
                if cand.field_hint:
                    line += f" field_hint={cand.field_hint[:80]}"
                if cand.href:
                    line += f" href={cand.href[:140]}"
                sample.append(line)
            parts.append("INTERACTIVES: " + " | ".join(sample))
        if visible_text:
            parts.append("VISIBLE_TEXT: " + visible_text[:5000])
        return "\n".join(parts)[:12000]

    def _progress_brief(self, *, state: AgentState) -> str:
        parts: List[str] = []
        last_effect = _candidate_text(state.progress.last_effect)
        if last_effect:
            parts.append(f"last_effect={last_effect}")
        expected_effect = _candidate_text(state.progress.pending_expected_effect)
        if expected_effect:
            parts.append(f"pending_expected_effect={expected_effect}")
        parts.append(f"no_progress_score={int(state.progress.no_progress_score or 0)}")
        parts.append(f"consecutive_no_effect_steps={int(state.progress.consecutive_no_effect_steps or 0)}")
        if state.focus_region.region_label:
            parts.append(
                "active_region="
                + _candidate_text(state.focus_region.region_label, state.focus_region.region_kind, state.focus_region.region_id)[:160]
            )
        recent_effects = []
        for effect in state.progress.recent_effects[-4:]:
            if not isinstance(effect, ProgressEffect):
                continue
            recent_effects.append(
                _norm_ws(
                    f"{effect.step_index}:{effect.action_type or 'action'}:{effect.label or 'unknown'}"
                    + (f":expected={effect.expected_effect}" if effect.expected_effect else "")
                    + (":met" if effect.expected_effect and effect.expected_effect_met else ":miss" if effect.expected_effect else "")
                    + (f":{effect.region_id}" if effect.region_id else "")
                )[:120]
            )
        if recent_effects:
            parts.append("recent=" + " | ".join(recent_effects))
        if state.progress.blocked_regions:
            parts.append("blocked_regions=" + ",".join(state.progress.blocked_regions[-6:]))
        if state.progress.satisfied_constraints:
            parts.append("satisfied_constraints=" + ",".join(state.progress.satisfied_constraints[-8:]))
        return " ; ".join([p for p in parts if p])[:800]

    def _candidate_sig(self, cand: Candidate) -> str:
        selector = cand.selector if isinstance(cand.selector, dict) else {}
        selector_bits = [
            str(selector.get("type") or ""),
            str(selector.get("attribute") or ""),
            str(selector.get("value") or ""),
        ]
        return "|".join(selector_bits + [cand.text[:80], cand.role[:24]])

    def _page_summary_text(self, *, text_ir: Dict[str, Any]) -> str:
        text_ir = text_ir if isinstance(text_ir, dict) else {}
        title = _candidate_text(text_ir.get("title"))
        headings = text_ir.get("headings") if isinstance(text_ir.get("headings"), list) else []
        visible_text = str(text_ir.get("visible_text") or "")
        parts: List[str] = []
        if title:
            parts.append(title[:180])
        if headings:
            parts.append(" | ".join(str(x)[:100] for x in headings[:5]))
        if visible_text:
            parts.append(visible_text[:260])
        return _norm_ws(" ; ".join(part for part in parts if part))[:800]

    def _compute_state_ir_deltas(
        self,
        *,
        state: AgentState,
        url: str,
        text_ir: Dict[str, Any],
        page_ir_text: str,
        candidates: List[Candidate],
    ) -> tuple[str, str]:
        prev_url = str(state.last_url or "")
        prev_summary = str(state.memory.prev_page_summary or "")
        prev_ir = str(state.memory.prev_page_ir_text or "")
        prev_sig_set = set(state.memory.prev_candidate_sigs or [])

        cur_summary = self._page_summary_text(text_ir=text_ir)
        cur_ir = _norm_ws(page_ir_text)[:2000]
        cur_sig_set = {self._candidate_sig(cand) for cand in candidates[:60]}

        added = len(cur_sig_set - prev_sig_set) if prev_sig_set else len(cur_sig_set)
        removed = len(prev_sig_set - cur_sig_set) if prev_sig_set else 0
        unchanged = len(cur_sig_set & prev_sig_set) if prev_sig_set else 0

        summary_changed = "unknown"
        if prev_summary and cur_summary:
            summary_changed = str(prev_summary[:240] != cur_summary[:240]).lower()
        ir_changed = "unknown"
        if prev_ir and cur_ir:
            ir_changed = str(prev_ir[:240] != cur_ir[:240]).lower()

        state_delta = ", ".join(
            [
                f"url_changed={str(bool(prev_url and prev_url != str(url or ''))).lower()}" if prev_url else "url_changed=unknown",
                f"summary_changed={summary_changed}",
                f"candidate_added={added}",
                f"candidate_removed={removed}",
                f"candidate_unchanged={unchanged}",
            ]
        )
        ir_delta = f"ir_changed={ir_changed}"

        state.memory.prev_page_summary = cur_summary
        state.memory.prev_page_ir_text = cur_ir
        state.memory.prev_candidate_sigs = list(cur_sig_set)[:80]
        return state_delta[:240], ir_delta[:120]

    def _select_candidates_for_policy(
        self,
        *,
        candidates: List[Candidate],
        current_url: str,
        state: AgentState,
        max_total: int = 24,
    ) -> List[Candidate]:
        if not candidates:
            return []

        current_path = str(urlsplit(str(current_url or "")).path or "/").rstrip("/") or "/"
        focus_region_id = str(state.focus_region.region_id or "").strip()
        focus_region_context = _norm_ws(state.focus_region.region_context)
        focus_candidate_ids = set(state.focus_region.candidate_ids)
        has_focus = bool(focus_region_id or focus_region_context or focus_candidate_ids)
        focused_local: List[Candidate] = []
        focused_escape: List[Candidate] = []
        focused_extended: List[Candidate] = []
        controls: List[Candidate] = []
        commit_controls: List[Candidate] = []
        contextual: List[Candidate] = []
        global_nav: List[Candidate] = []
        others: List[Candidate] = []

        def same_focus_region(cand: Candidate) -> bool:
            if focus_region_id and cand.region_id and cand.region_id == focus_region_id:
                return True
            if focus_region_id and focus_region_id in set(cand.region_ancestor_ids or []):
                return True
            if focus_region_context and _norm_ws(cand.context) == focus_region_context:
                return True
            if focus_candidate_ids and cand.id in focus_candidate_ids:
                return True
            return False

        for cand in candidates:
            if same_focus_region(cand):
                if self._is_escape_candidate(
                    cand=cand,
                    focus_region_id=focus_region_id,
                    focus_region_context=focus_region_context,
                ):
                    focused_escape.append(cand)
                else:
                    focused_local.append(cand)
                continue
            if has_focus and focus_region_id and focus_region_id in set(cand.region_ancestor_ids or []):
                focused_extended.append(cand)
                continue
            if cand.role == "link" and cand.href:
                try:
                    cand_path = str(urlsplit(str(cand.href or "")).path or "/").rstrip("/") or "/"
                    if cand_path == current_path:
                        continue
                except Exception:
                    pass
            lowered = " ".join(
                [
                    str(cand.text or ""),
                    str(cand.field_hint or ""),
                    str(cand.group_label or ""),
                    str(cand.context or ""),
                ]
            ).lower()
            if cand.role in {"input", "select"} or cand.type in {"textarea"}:
                controls.append(cand)
                continue
            if bool(re.search(r"\b(save|submit|apply|continue|confirm|done|finish|search)\b", lowered)):
                commit_controls.append(cand)
                continue
            if cand.role == "button":
                global_nav.append(cand)
                continue
            if cand.role in {"link", "button"} and len(_norm_ws(cand.context)) >= 40:
                contextual.append(cand)
                continue
            if cand.role == "link":
                global_nav.append(cand)
                continue
            others.append(cand)

        picked: List[Candidate] = []
        seen: set[str] = set()

        def add_many(arr: List[Candidate]) -> None:
            nonlocal picked
            for cand in arr:
                sig = self._candidate_sig(cand)
                if sig in seen:
                    continue
                seen.add(sig)
                picked.append(cand)
                if len(picked) >= max_total:
                    return

        if has_focus:
            add_many(focused_local[:10])
            if len(picked) < max_total:
                add_many(focused_escape[:4])
            if len(picked) < max_total:
                add_many(focused_extended[:4])
            if len(picked) < max_total:
                add_many(commit_controls[:3])
            if len(picked) < max_total:
                add_many(global_nav[:3])
        else:
            add_many(controls[:8])
            if len(picked) < max_total:
                add_many(commit_controls[:6])
            if len(picked) < max_total:
                add_many(contextual[:4])
            if len(picked) < max_total:
                add_many(global_nav[:4])
        if len(picked) < max_total:
            add_many(others[: max_total - len(picked)])
        return picked[:max_total]

    def _is_escape_candidate(self, *, cand: Candidate, focus_region_id: str, focus_region_context: str) -> bool:
        if cand.disabled:
            return False
        if str(cand.field_kind or "").strip().lower() == "pager":
            return False
        blob = " ".join([cand.text, cand.field_hint, cand.group_label, cand.context]).lower()
        if focus_region_id and cand.region_id and cand.region_id == focus_region_id:
            return bool(re.search(r"\b(save|submit|apply|continue|confirm|close|done|cancel|back)\b", blob))
        if focus_region_context and _norm_ws(cand.context) == focus_region_context:
            return bool(re.search(r"\b(save|submit|apply|continue|confirm|close|done|cancel|back)\b", blob))
        return False

    def _partition_candidates(
        self,
        *,
        candidates: List[Candidate],
        state: AgentState,
        max_local: int = 18,
        max_escape: int = 8,
        max_global: int = 18,
    ) -> Dict[str, Any]:
        focus_region_id = str(state.focus_region.region_id or "").strip()
        focus_region_context = _norm_ws(state.focus_region.region_context)
        focus_candidate_ids = set(state.focus_region.candidate_ids)
        local: List[Candidate] = []
        escape: List[Candidate] = []
        global_pool: List[Candidate] = []
        for cand in candidates:
            same_region = False
            if focus_region_id and cand.region_id and cand.region_id == focus_region_id:
                same_region = True
            elif focus_region_id and focus_region_id in set(cand.region_ancestor_ids or []):
                same_region = True
            elif focus_region_context and _norm_ws(cand.context) == focus_region_context:
                same_region = True
            elif focus_candidate_ids and cand.id in focus_candidate_ids:
                same_region = True
            if same_region:
                if self._is_escape_candidate(cand=cand, focus_region_id=focus_region_id, focus_region_context=focus_region_context):
                    escape.append(cand)
                else:
                    local.append(cand)
                continue
            global_pool.append(cand)
        if not focus_region_id and not focus_region_context and not focus_candidate_ids:
            local = candidates[:max_local]
            escape = []
            global_pool = candidates[max_local:]
        return {
            "local": local[:max_local],
            "escape": escape[:max_escape],
            "global": global_pool[:max_global],
            "suppressed_global_count": max(0, len(global_pool) - max_global),
        }

    def _candidate_lines(self, candidates: List[Candidate], *, limit: int = 12) -> str:
        lines: List[str] = []
        for cand in candidates[:limit]:
            bits = [f"[{cand.id}]", f"<{cand.role}>"]
            label = _candidate_text(cand.text, cand.field_hint, cand.href, cand.group_label)
            if label:
                bits.append(label[:140])
            if cand.field_kind:
                bits.append(f"kind={cand.field_kind[:40]}")
            if cand.region_kind:
                bits.append(f"region={cand.region_kind[:32]}")
            if cand.ui_state:
                bits.append(f"state={cand.ui_state[:16]}")
            lines.append(" ".join(bits))
        return "\n".join(lines)

    def _browser_state_snapshot(
        self,
        *,
        url: str,
        text_ir: Dict[str, Any],
        page_observations: Dict[str, Any],
        screenshot_available: bool,
    ) -> str:
        text_ir = text_ir if isinstance(text_ir, dict) else {}
        page_observations = page_observations if isinstance(page_observations, dict) else {}
        page_stats = page_observations.get("page_stats") if isinstance(page_observations.get("page_stats"), dict) else {}
        title = _candidate_text(text_ir.get("title"))
        headings = text_ir.get("headings") if isinstance(text_ir.get("headings"), list) else []
        likely_answers = page_observations.get("likely_answers") if isinstance(page_observations.get("likely_answers"), list) else []
        relevant_lines = page_observations.get("relevant_lines") if isinstance(page_observations.get("relevant_lines"), list) else []
        parts: List[str] = [
            f"Current URL: {_candidate_text(url)}",
            (
                "Page stats: "
                f'{int(page_stats.get("links") or 0)} links, '
                f'{int(page_stats.get("controls") or 0)} controls, '
                f'{int(page_stats.get("forms") or 0)} forms, '
                f'{int(page_stats.get("control_groups") or 0)} control groups, '
                f'{int(page_stats.get("cards") or 0)} cards, '
                f'{int(page_stats.get("visible_text_chars") or 0)} visible chars'
            ),
            f"Screenshot available: {bool(screenshot_available)}",
        ]
        if title:
            parts.append(f"Title: {title[:220]}")
        if headings:
            parts.append("Headings: " + " | ".join(str(x)[:100] for x in headings[:6]))
        if likely_answers:
            parts.append("Current page answers: " + " | ".join(str(x)[:120] for x in likely_answers[:6]))
        elif relevant_lines:
            parts.append("Relevant visible text: " + " | ".join(str(x)[:120] for x in relevant_lines[:8]))
        return "\n".join(parts)[:2400]

    def _browser_state_text(self, candidates: List[Candidate], *, limit: int = 60) -> str:
        class _Node:
            __slots__ = ("name", "children", "items")

            def __init__(self, name: str) -> None:
                self.name = name
                self.children: Dict[str, "_Node"] = {}
                self.items: List[Candidate] = []

        root = _Node("ROOT")
        chosen = candidates[: max(1, int(limit))]
        for cand in chosen:
            chain: List[str] = []
            if cand.group_label:
                chain.append(cand.group_label[:80])
            elif cand.context:
                chain.append(_norm_ws(cand.context)[:80])
            else:
                chain.append("PAGE")
            node = root
            for part in chain[:3]:
                if part not in node.children:
                    node.children[part] = _Node(part)
                node = node.children[part]
            node.items.append(cand)

        def render(node: _Node, indent: str = "") -> List[str]:
            lines: List[str] = []
            for child_name, child in list(node.children.items())[:24]:
                lines.append(f"{indent}{child_name}")
                for cand in child.items[:12]:
                    label = _candidate_text(cand.text, cand.field_hint, cand.href, cand.id)
                    bits = [f"[{cand.id}] <{cand.role}/>"]
                    if label:
                        bits.append(label[:140])
                    if cand.field_kind:
                        bits.append(f"kind={cand.field_kind[:40]}")
                    if cand.region_kind:
                        bits.append(f"region={cand.region_kind[:32]}")
                    if cand.ui_state:
                        bits.append(f"state={cand.ui_state[:20]}")
                    lines.append(f"{indent}  " + " ".join(bit for bit in bits if bit))
                lines.extend(render(child, indent + "  "))
            return lines

        rendered = render(root)
        if not rendered:
            return ""
        return "\n".join(rendered)[:12000]

    def _history_brief(self, history: List[Dict[str, Any]], *, limit: int = 10) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for item in history[-max(1, int(limit)) :]:
            if not isinstance(item, dict):
                continue
            action_raw = item.get("action")
            action = action_raw if isinstance(action_raw, dict) else {}
            action_type = _candidate_text(
                action.get("type"),
                action.get("name"),
                action_raw if isinstance(action_raw, str) else "",
            )
            url = _candidate_text(item.get("url"))
            error = _candidate_text(item.get("error"))
            nested = action.get("raw") if isinstance(action.get("raw"), dict) else {}
            text = _candidate_text(
                item.get("text"),
                action.get("text"),
                action.get("value"),
                nested.get("text"),
                nested.get("value"),
            )
            out.append(
                {
                    "step": int(item.get("step") or 0),
                    "url": url[:420],
                    "action_type": action_type[:80],
                    "done": bool(item.get("done")),
                    "exec_ok": bool(item.get("exec_ok", True)),
                    "error": error[:240] if error else "",
                    "text": text[:240] if text else "",
                }
            )
        return out

    def _typed_values_from_history(self, history: List[Dict[str, Any]]) -> List[str]:
        out: List[str] = []
        for item in history:
            if not isinstance(item, dict):
                continue
            action_raw = item.get("action")
            action: Dict[str, Any] = action_raw if isinstance(action_raw, dict) else {}
            action_type = str(action.get("type") or "")
            if not action_type and isinstance(action_raw, str):
                action_type = str(action_raw)
            if action_type in {"TypeAction", "FillAction"}:
                nested = action.get("raw") if isinstance(action.get("raw"), dict) else {}
                val = _candidate_text(
                    item.get("text"),
                    action.get("text"),
                    action.get("value"),
                    nested.get("text"),
                    nested.get("value"),
                )
                if val:
                    out.append(val)
        return out


    def _history_summary(self, history: List[Dict[str, Any]]) -> str:
        if not isinstance(history, list) or not history:
            return ""
        total = len(history)
        failures = 0
        by_action: Dict[str, int] = {}
        recent_urls: List[str] = []
        for item in history[-120:]:
            if not isinstance(item, dict):
                continue
            action_raw = item.get("action")
            action = action_raw if isinstance(action_raw, dict) else {}
            action_type = _candidate_text(
                action.get("type"),
                action.get("name"),
                action_raw if isinstance(action_raw, str) else "",
                "unknown",
            )
            by_action[action_type] = int(by_action.get(action_type) or 0) + 1
            if not bool(item.get("exec_ok", True)) or bool(_candidate_text(item.get("error"))):
                failures += 1
            url = _candidate_text(item.get("url"))
            if url:
                recent_urls.append(url[:220])
        top_actions = sorted(by_action.items(), key=lambda kv: kv[1], reverse=True)[:6]
        action_line = ", ".join([f"{k}:{v}" for k, v in top_actions]) if top_actions else "none"
        urls_line = ", ".join(_dedupe_keep_order(recent_urls[-12:], 6))
        parts = [
            f"history_total={total}",
            f"history_failures={failures}",
            f"top_actions={action_line}",
        ]
        if urls_line:
            parts.append(f"recent_urls={urls_line}")
        return " ; ".join(parts)[:MAX_HISTORY_SUMMARY_CHARS]

    def _previous_step_verdict(self, history: List[Dict[str, Any]], flags: Dict[str, Any]) -> Dict[str, str]:
        if not isinstance(history, list) or not history:
            return {"status": "n/a", "summary": "No previous step to evaluate."}
        last = history[-1] if isinstance(history[-1], dict) else {}
        if not isinstance(last, dict):
            return {"status": "uncertain", "summary": "Previous step unavailable."}
        err = _candidate_text(last.get("error"))
        if err:
            return {"status": "failure", "summary": f"Previous step error: {err[:180]}"}
        if not bool(last.get("exec_ok", True)):
            return {"status": "failure", "summary": "Previous step execution reported failure."}
        if bool(flags.get("url_changed")) or bool(flags.get("dom_changed")):
            return {"status": "success", "summary": "Previous step changed page state (url/dom diff detected)."}
        if bool(flags.get("no_visual_progress")):
            return {"status": "uncertain", "summary": "No visual progress detected after previous step."}
        return {"status": "uncertain", "summary": "Previous step outcome unclear from current snapshot."}

    def _loop_nudges(self, *, state: AgentState, flags: Dict[str, Any]) -> List[str]:
        n: List[str] = []
        repeat_n = int(state.counters.repeat_action_count or 0)
        stall_n = int(state.counters.stall_count or 0)
        level = str(flags.get("loop_level") or "none")
        if repeat_n >= 2:
            n.append(f"Repeated similar action detected ({repeat_n} consecutive).")
        elif repeat_n == 1:
            n.append("Potential action repetition detected.")
        if stall_n >= 2:
            n.append(f"Stall signal active ({stall_n} no-progress steps).")
        if level == "high":
            n.append("Loop level is HIGH: avoid repeating same element/action.")
        elif level == "low":
            n.append("Loop level is LOW: verify progress before repeating.")
        if bool(flags.get("no_visual_progress")):
            n.append("No visual progress: prefer alternative path, back, or a new target.")
        return n[:6]

    def _tagged_policy_input(
        self,
        *,
        prompt: str,
        task_constraints: Dict[str, str],
        step_index: int,
        mode: str,
        url: str,
        flags: Dict[str, Any],
        state: AgentState,
        history_recent: List[Dict[str, Any]],
        history_summary: str,
        verdict: Dict[str, str],
        loop_nudges: List[str],
        text_ir: Dict[str, Any],
        candidates: List[Candidate],
        state_delta: str = "",
        ir_delta: str = "",
        screenshot_available: bool = False,
    ) -> str:
        text_ir = self._augment_text_ir(text_ir=text_ir, candidates=candidates)
        parts: List[str] = []
        page_observations = self._page_observations(
            prompt=prompt,
            flags=flags,
            state=state,
            text_ir=text_ir,
            candidates=candidates,
            history_recent=history_recent,
        )
        active_group = self.active_group_summary(state=state, candidates=candidates)
        active_region = self.active_region_summary(state=state, candidates=candidates)
        text_ir = dict(text_ir)
        text_ir["active_group"] = active_group
        text_ir["active_region"] = active_region
        text_ir["capability_gap"] = page_observations.get("capability_gap") if isinstance(page_observations, dict) else {}
        page_ir_text = self._page_ir_text(prompt=prompt, text_ir=text_ir, candidates=candidates)
        browser_state_text = self._browser_state_text(candidates, limit=24)
        candidate_partitions = self._partition_candidates(candidates=candidates, state=state)
        progress_brief = self._progress_brief(state=state)
        typed_recent = [
            str(item.get("text") or "")
            for item in history_recent
            if isinstance(item, dict)
            and str(item.get("action_type") or "").lower() in {"typeaction", "fillaction"}
            and str(item.get("text") or "").strip()
        ][:10]
        parts.append(f"TASK: {_candidate_text(prompt)}")
        if task_constraints:
            parts.append("TASK CONSTRAINTS:\n" + json.dumps(task_constraints, ensure_ascii=False))
        parts.append(
            "STEP INFO:\n"
            + f"step_index={int(step_index)}\n"
            + f"mode={str(mode)}\n"
            + f"url={_candidate_text(url)}"
        )
        if progress_brief:
            parts.append("PROGRESS LEDGER:\n" + progress_brief)
        if active_region:
            parts.append("FOCUSED REGION SUMMARY (JSON):\n" + json.dumps(active_region, ensure_ascii=False))
        if candidate_partitions.get("local"):
            parts.append("LOCAL CANDIDATES:\n" + self._candidate_lines(candidate_partitions.get("local") or [], limit=16))
        if candidate_partitions.get("escape"):
            parts.append("ESCAPE / COMMIT CANDIDATES:\n" + self._candidate_lines(candidate_partitions.get("escape") or [], limit=8))
        if state_delta:
            parts.append("STATE DELTA:\n" + state_delta)
        if ir_delta:
            parts.append("PAGE IR DELTA:\n" + ir_delta)
        parts.append("PAGE IR (PRIMARY STRUCTURED STATE):\n" + page_ir_text)
        parts.append("PAGE OBSERVATIONS (GENERIC JSON):\n" + json.dumps(page_observations, ensure_ascii=False))
        parts.append(
            "PAGE GROUPS (JSON):\n"
            + json.dumps(
                {
                    "forms": (text_ir.get("forms") if isinstance(text_ir.get("forms"), list) else [])[:6],
                    "control_groups": (text_ir.get("control_groups") if isinstance(text_ir.get("control_groups"), list) else [])[:8],
                    "cards": (text_ir.get("cards") if isinstance(text_ir.get("cards"), list) else [])[:8],
                    "active_region": text_ir.get("active_region") if isinstance(text_ir.get("active_region"), dict) else {},
                    "active_group": text_ir.get("active_group") if isinstance(text_ir.get("active_group"), dict) else {},
                },
                ensure_ascii=False,
            )
        )
        parts.append(f"SCREENSHOT AVAILABLE: {bool(screenshot_available)}")
        parts.append(
            "PREVIOUS STEP VERDICT:\n"
            + f"status={_candidate_text(verdict.get('status'))}\n"
            + f"summary={_candidate_text(verdict.get('summary'))}\n"
        )
        if history_summary:
            parts.append(f"HISTORY SUMMARY:\n{history_summary}")
        parts.append("HISTORY RECENT (JSON):\n" + json.dumps(history_recent[-HISTORY_RECENT_LIMIT:], ensure_ascii=False))
        parts.append(
            "PLAN AND MEMORY (JSON):\n"
            + json.dumps(
                {
                    "plan": {
                        "active_id": state.plan.active_id,
                        "subgoals": [sg.model_dump(mode="json") for sg in state.plan.subgoals],
                    },
                    "frontier": state.frontier.model_dump(mode="json"),
                    "memory": {
                        "facts": state.memory.facts[-20:],
                        "checkpoints": state.memory.checkpoints[-20:],
                        "visual_notes": state.memory.visual_notes[-8:],
                        "visual_element_hints": state.memory.visual_element_hints[-12:],
                        "strategy_summary": state.memory.strategy_summary,
                        "focus_region": state.focus_region.model_dump(mode="json"),
                        "progress": state.progress.model_dump(mode="json"),
                        "typed_values_recent": typed_recent,
                        "typed_candidate_ids": state.form_progress.typed_candidate_ids[-20:],
                        "typed_selector_sigs": state.form_progress.typed_selector_sigs[-20:],
                        "active_group_label": state.form_progress.active_group_label,
                        "active_group_candidate_ids": state.form_progress.active_group_candidate_ids[-20:],
                    },
                    "counters": state.counters.model_dump(mode="json"),
                    "blocklist_size": len(state.blocklist.element_ids),
                    "session_query": state.session_query,
                },
                ensure_ascii=False,
            )
        )
        if loop_nudges:
            parts.append("LOOP NUDGES:\n" + "\n".join(f"- {x}" for x in loop_nudges))
        parts.append("BROWSER_STATE (interactive elements):\n" + browser_state_text)
        if candidate_partitions.get("global"):
            global_summary = {
                "suppressed_global_count": int(candidate_partitions.get("suppressed_global_count") or 0),
                "global_candidates": [cand.as_obs() for cand in (candidate_partitions.get("global") or [])[:8]],
            }
            parts.append("GLOBAL CANDIDATE SUMMARY (JSON):\n" + json.dumps(global_summary, ensure_ascii=False))
        parts.append("ACTION SHORTLIST (JSON):\n" + json.dumps([cand.as_obs() for cand in candidates[:24]], ensure_ascii=False))
        return "\n\n".join(parts)

    def _augment_text_ir(self, *, text_ir: Dict[str, Any], candidates: List[Candidate]) -> Dict[str, Any]:
        base = dict(text_ir or {}) if isinstance(text_ir, dict) else {}
        groups = self._candidate_groups(candidates)
        existing_groups = base.get("control_groups") if isinstance(base.get("control_groups"), list) else []
        merged_groups: List[Dict[str, Any]] = []
        seen_group_ids: set[str] = set()
        for group in list(existing_groups) + [g for g in groups if str(g.get("kind") or "") == "controls"]:
            if not isinstance(group, dict):
                continue
            group_id = str(group.get("group_id") or "").strip()
            dedupe_key = group_id or _candidate_text(group.get("label"), group.get("context"))
            if not dedupe_key or dedupe_key in seen_group_ids:
                continue
            seen_group_ids.add(dedupe_key)
            merged_groups.append(group)
        base["control_groups"] = merged_groups[:8]
        base["cards"] = self._card_summaries(groups)
        return base

    def active_group_summary(self, *, state: AgentState, candidates: List[Candidate]) -> Dict[str, Any]:
        group_id = str(state.form_progress.active_group_id or "").strip()
        context = _norm_ws(state.form_progress.active_group_context)
        candidate_ids = set(state.form_progress.active_group_candidate_ids)
        if not group_id and not context and not candidate_ids:
            return {}
        items: List[Dict[str, Any]] = []
        for cand in candidates:
            if group_id and cand.group_id and cand.group_id != group_id and cand.id not in candidate_ids:
                continue
            cand_context = _norm_ws(cand.context)
            if context and cand_context and cand_context != context:
                continue
            if not group_id and not context and candidate_ids and cand.id not in candidate_ids:
                continue
            items.append(
                {
                    "id": cand.id,
                    "role": cand.role,
                    "text": cand.text[:120],
                    "field_hint": cand.field_hint[:80],
                    "field_kind": cand.field_kind[:40],
                    "href": cand.href[:140],
                }
            )
        if not items:
            return {}
        return {
            "group_id": group_id,
            "label": str(state.form_progress.active_group_label or "")[:160],
            "context": context[:280],
            "items": items[:10],
        }

    def active_region_summary(self, *, state: AgentState, candidates: List[Candidate]) -> Dict[str, Any]:
        region_id = str(state.focus_region.region_id or "").strip()
        region_context = _norm_ws(state.focus_region.region_context)
        candidate_ids = set(state.focus_region.candidate_ids)
        if not region_id and not region_context and not candidate_ids:
            active_group = self.active_group_summary(state=state, candidates=candidates)
            if not active_group:
                return {}
            return {
                "region_id": str(active_group.get("group_id") or "")[:120],
                "region_kind": "group",
                "label": str(active_group.get("label") or "")[:160],
                "context": str(active_group.get("context") or "")[:280],
                "items": active_group.get("items") if isinstance(active_group.get("items"), list) else [],
            }
        items: List[Dict[str, Any]] = []
        for cand in candidates:
            same_region = False
            if region_id and cand.region_id and cand.region_id == region_id:
                same_region = True
            elif region_context and _norm_ws(cand.context) == region_context:
                same_region = True
            elif candidate_ids and cand.id in candidate_ids:
                same_region = True
            if not same_region:
                continue
            items.append(
                {
                    "id": cand.id,
                    "role": cand.role,
                    "text": cand.text[:120],
                    "field_hint": cand.field_hint[:80],
                    "field_kind": cand.field_kind[:40],
                    "region_kind": cand.region_kind[:40],
                    "href": cand.href[:140],
                }
            )
        if not items:
            return {}
        return {
            "region_id": region_id[:120],
            "region_kind": str(state.focus_region.region_kind or "group")[:40],
            "label": str(state.focus_region.region_label or "")[:160],
            "context": region_context[:280],
            "items": items[:12],
            "recent_region_ids": state.focus_region.recent_region_ids[-8:],
        }

    def build_policy_obs(
        self,
        *,
        task_id: str,
        prompt: str,
        step_index: int,
        url: str,
        mode: str,
        flags: Dict[str, Any],
        state: AgentState,
        text_ir: Dict[str, Any],
        candidates: List[Candidate],
        history: List[Dict[str, Any]],
        screenshot_available: bool = False,
    ) -> Dict[str, Any]:
        text_ir = self._augment_text_ir(text_ir=text_ir, candidates=candidates)
        active = {}
        if state.plan.active_id:
            for sg in state.plan.subgoals:
                if sg.id == state.plan.active_id:
                    active = {"id": sg.id, "text": sg.text, "status": sg.status}
                    break
        history_recent = self._history_brief(history, limit=HISTORY_RECENT_LIMIT)
        history_summary = self._history_summary(history)
        verdict = self._previous_step_verdict(history, flags)
        loop_nudges = self._loop_nudges(state=state, flags=flags)
        typed_values_recent = _dedupe_keep_order(self._typed_values_from_history(history), 10)
        if history_summary:
            state.memory.history_summary = history_summary
        parsed_url = urlsplit(str(url or ""))
        task_constraints = _task_constraints(prompt)
        page_observations = self._page_observations(
            prompt=prompt,
            flags=flags,
            state=state,
            text_ir=text_ir,
            candidates=candidates,
            history_recent=history_recent,
        )
        active_group = self.active_group_summary(state=state, candidates=candidates)
        active_region = self.active_region_summary(state=state, candidates=candidates)
        text_ir = dict(text_ir)
        text_ir["active_region"] = active_region
        text_ir["active_group"] = active_group
        text_ir["capability_gap"] = page_observations.get("capability_gap") if isinstance(page_observations, dict) else {}
        state.memory.strategy_summary = _candidate_text(
            (page_observations.get("capability_gap") if isinstance(page_observations, dict) else {}).get("strategy_summary")
        )[:320]
        policy_candidates = self._select_candidates_for_policy(candidates=candidates, current_url=url, state=state, max_total=60)
        candidate_partitions = self._partition_candidates(candidates=policy_candidates, state=state)
        page_ir_text = self._page_ir_text(prompt=prompt, text_ir=text_ir, candidates=policy_candidates)
        browser_state_text = self._browser_state_text(policy_candidates, limit=24)
        browser_state_snapshot = self._browser_state_snapshot(
            url=url,
            text_ir=text_ir,
            page_observations=page_observations,
            screenshot_available=screenshot_available,
        )
        state_delta, ir_delta = self._compute_state_ir_deltas(
            state=state,
            url=url,
            text_ir=text_ir,
            page_ir_text=page_ir_text,
            candidates=policy_candidates,
        )
        plan_items = [
            {"id": sg.id, "text": sg.text, "status": sg.status}
            for sg in state.plan.subgoals
        ]
        tagged_input = self._tagged_policy_input(
            prompt=prompt,
            task_constraints=task_constraints,
            step_index=step_index,
            mode=mode,
            url=url,
            flags=flags,
            state=state,
            history_recent=history_recent,
            history_summary=history_summary,
            verdict=verdict,
            loop_nudges=loop_nudges,
            text_ir=text_ir,
            candidates=policy_candidates,
            state_delta=state_delta,
            ir_delta=ir_delta,
            screenshot_available=screenshot_available,
        )
        return {
            "task_id": str(task_id or ""),
            "prompt": str(prompt or "")[:1200],
            "task_constraints": task_constraints,
            "step_index": int(step_index),
            "url": str(url or "")[:800],
            "url_parts": {
                "scheme": str(parsed_url.scheme or ""),
                "host": str(parsed_url.netloc or ""),
                "path": str(parsed_url.path or ""),
                "query": str(parsed_url.query or "")[:220],
            },
            "mode": mode,
            "screenshot_available": bool(screenshot_available),
            "flags": flags,
            "page_observations": page_observations,
            "previous_step_verdict": verdict,
            "loop_nudges": loop_nudges,
            "active_subgoal": active,
            "plan": {
                "active_id": state.plan.active_id,
                "subgoals": plan_items,
            },
            "frontier": {
                "pending_urls": state.frontier.pending_urls[:20],
                "pending_elements": state.frontier.pending_elements[:20],
            },
            "session_query": state.session_query,
            "counters": state.counters.model_dump(),
                "memory": {
                    "facts": state.memory.facts[:20],
                    "checkpoints": state.memory.checkpoints[-20:],
                    "visual_notes": state.memory.visual_notes[-8:],
                    "visual_element_hints": state.memory.visual_element_hints[-12:],
                    "obs_candidate_hints": state.memory.obs_candidate_hints[-12:],
                    "strategy_summary": state.memory.strategy_summary,
                    "typed_values_recent": typed_values_recent,
                    "typed_candidate_ids": state.form_progress.typed_candidate_ids[-20:],
                    "typed_selector_sigs": state.form_progress.typed_selector_sigs[-20:],
                "submit_attempt_sigs": state.form_progress.submit_attempt_sigs[-20:],
                "active_group_label": state.form_progress.active_group_label,
                "active_group_candidate_ids": state.form_progress.active_group_candidate_ids[-20:],
                "focus_region": state.focus_region.model_dump(mode="json"),
                "progress": state.progress.model_dump(mode="json"),
            },
            "visited_recent_urls": state.visited.urls[-20:],
            "history_recent": history_recent,
            "history_summary": history_summary,
            "state_delta": state_delta,
            "ir_delta": ir_delta,
            "page_ir_text": page_ir_text,
            "browser_state_snapshot": browser_state_snapshot,
            "browser_state_text": browser_state_text,
            "text_ir": {
                "title": str(text_ir.get("title") or "")[:260],
                "visible_text": str(text_ir.get("visible_text") or "")[:12000],
                "headings": text_ir.get("headings") if isinstance(text_ir.get("headings"), list) else [],
                "forms": text_ir.get("forms") if isinstance(text_ir.get("forms"), list) else [],
                "control_groups": text_ir.get("control_groups") if isinstance(text_ir.get("control_groups"), list) else [],
                "cards": text_ir.get("cards") if isinstance(text_ir.get("cards"), list) else [],
                "active_region": text_ir.get("active_region") if isinstance(text_ir.get("active_region"), dict) else {},
                "active_group": text_ir.get("active_group") if isinstance(text_ir.get("active_group"), dict) else {},
                "capability_gap": text_ir.get("capability_gap") if isinstance(text_ir.get("capability_gap"), dict) else {},
                "llm_extract": text_ir.get("llm_extract") if isinstance(text_ir.get("llm_extract"), dict) else {},
                "html_excerpt": str(text_ir.get("html_excerpt") or "")[:12000],
            },
            "candidates": [cand.as_obs() for cand in policy_candidates[:24]],
            "candidate_partitions": {
                "local": [cand.as_obs() for cand in (candidate_partitions.get("local") or [])[:10]],
                "escape": [cand.as_obs() for cand in (candidate_partitions.get("escape") or [])[:6]],
                "global": [cand.as_obs() for cand in (candidate_partitions.get("global") or [])[:8]],
                "suppressed_global_count": int(candidate_partitions.get("suppressed_global_count") or 0),
            },
            "policy_input_text": tagged_input,
        }


class Router:
    def next_mode(
        self,
        *,
        step_index: int,
        state: AgentState,
        flags: Dict[str, Any],
        prompt: str = "",
    ) -> tuple[str, str]:
        if state.mode == "DONE":
            return "DONE", "already_done"
        if bool(flags.get("captcha_suspected")):
            return "PLAN", "captcha_detected_model_replan"
        if str(flags.get("loop_level") or "none") == "high":
            return "PLAN", "loop_high_model_replan"
        if int(state.counters.stall_count or 0) >= 4 or int(state.counters.repeat_action_count or 0) >= 4:
            return "PLAN", "stalled_model_replan"
        if (
            (bool(flags.get("cookie_banner")) or (bool(flags.get("modal_dialog")) and not bool(flags.get("interactive_modal_form"))))
            and (int(state.counters.repeat_action_count or 0) >= 1 or int(state.counters.stall_count or 0) >= 2)
        ):
            return "PLAN", "popup_stalled_replan"
        if bool(flags.get("cookie_banner")) or (bool(flags.get("modal_dialog")) and not bool(flags.get("interactive_modal_form"))):
            return "POPUP", "popup_detected"
        if int(step_index) == 0:
            return "BOOTSTRAP", "initial_step"
        if state.mode == "STUCK":
            return "PLAN", "recover_from_stuck_mode"
        if state.mode == "REPORT":
            return "REPORT", "report_resume"
        if state.mode in {"BOOTSTRAP", "PLAN"}:
            return "NAV", "progress_after_bootstrap"
        if state.mode in {"NAV", "POPUP", "EXTRACT", "SYNTH"}:
            if bool(flags.get("error_page")):
                return "PLAN", "error_page_replan"
            if (
                (bool(flags.get("product_cards")) or bool(flags.get("results_list")) or bool(flags.get("pricing_table")))
                and not bool(flags.get("error_page"))
            ):
                return "NAV", "extractable_content_visible_in_nav"
            if bool(flags.get("search_box")) and bool(flags.get("dom_changed")):
                return "NAV", "interactive_content_changed_in_nav"
            return "NAV", "continue_navigation"
        return "PLAN", "default_plan"


class Skills:
    def __init__(self, vision_call: Callable[..., Dict[str, Any]] | None = None) -> None:
        self.vision_call = vision_call
        self._vision_cache: Dict[str, Dict[str, Any]] = {}

    def solve_popups(self, *, candidates: List[Candidate]) -> Dict[str, Any]:
        keys = {
            "accept",
            "agree",
            "reject",
            "close",
            "dismiss",
            "continue",
            "ok",
            "cancel",
            "not now",
            "skip",
            "later",
            "no thanks",
        }
        group_field_kinds: Dict[str, set[str]] = {}
        for cand in candidates:
            group_key = str(cand.group_id or _norm_ws(cand.context).lower() or cand.id)
            kinds = group_field_kinds.setdefault(group_key, set())
            if cand.field_kind:
                kinds.add(str(cand.field_kind or "").strip().lower())
        scored: List[tuple[float, Candidate]] = []
        for cand in candidates:
            blob = " ".join([cand.text, cand.context, cand.group_label, cand.field_hint]).lower()
            group_key = str(cand.group_id or _norm_ws(cand.context).lower() or cand.id)
            group_kinds = group_field_kinds.get(group_key) or set()
            if cand.role not in {"button", "link"}:
                continue
            if cand.disabled:
                continue
            if not any(k in blob for k in keys):
                continue
            if cand.field_kind in {"auth_entry", "account_create"}:
                continue
            if {"username", "email", "password", "confirm_password"}.intersection(group_kinds):
                continue
            score = 0.0
            if cand.role == "button":
                score += 2.0
            if any(k in blob for k in {"dialog", "modal", "popup", "overlay", "cookie", "consent", "backdrop"}):
                score += 3.0
            if any(k in (cand.text or "").lower() for k in {"close", "dismiss", "cancel", "ok", "continue"}):
                score += 1.6
            if len(_norm_ws(cand.text)) <= 18:
                score += 0.6
            scored.append((score, cand))
        scored.sort(key=lambda item: item[0], reverse=True)
        hits = [cand.id for _, cand in scored]
        return {
            "pending_elements": hits[:8],
            "primary_element_id": hits[0] if hits else "",
            "checkpoint": f"popup_candidates={len(hits)}",
        }

    def replan(self, *, prompt: str) -> Dict[str, Any]:
        subgoals = _split_prompt_subgoals(prompt)
        return {"subgoals": subgoals}

    def extract_links(self, *, kind: str, candidates: List[Candidate]) -> Dict[str, Any]:
        k = str(kind or "all_links").strip().lower()
        urls: List[str] = []
        for cand in candidates:
            if cand.role != "link" or not cand.href:
                continue
            blob = " ".join([cand.text, cand.context, cand.href]).lower()
            if k == "product_cards" and not any(w in blob for w in {"product", "buy", "pricing", "plan"}):
                continue
            if k == "nav_links" and not any(w in blob for w in {"docs", "about", "pricing", "features", "product", "contact"}):
                continue
            urls.append(cand.href)
        return {"urls": _dedupe_keep_order(urls, MAX_PENDING_URLS)}

    def extract_facts(self, *, schema: str, text_ir: Dict[str, Any], url: str) -> Dict[str, Any]:
        schema_name = str(schema or "generic").strip().lower()
        text = str(text_ir.get("visible_text") or "")
        headings = text_ir.get("headings") if isinstance(text_ir.get("headings"), list) else []
        page_facts = text_ir.get("page_facts") if isinstance(text_ir.get("page_facts"), list) else []
        facts: List[str] = []
        if page_facts:
            facts.extend([str(f)[:180] for f in page_facts[:8] if str(f).strip()])
        if headings:
            facts.extend([f"Heading: {str(h)[:120]}" for h in headings[:4]])
        lines = [ln.strip() for ln in re.split(r"[.\n]", text) if ln.strip()]
        for ln in lines:
            l = ln.lower()
            if schema_name == "pricing":
                if "$" in ln or "price" in l or "plan" in l:
                    facts.append(ln[:220])
            elif schema_name == "product_page":
                if any(k in l for k in {"feature", "product", "integration", "benefit", "use case"}):
                    facts.append(ln[:220])
            else:
                if len(ln) >= 24 and any(ch.isalpha() for ch in ln):
                    facts.append(ln[:220])
            if len(facts) >= 8:
                break
        if not facts and text:
            facts.append(text[:220])
        return {"facts": _dedupe_keep_order([f"{f} (source: {url})" for f in facts], 10)}

    def search_text(self, *, query: str, text_ir: Dict[str, Any]) -> Dict[str, Any]:
        q = _norm_ws(query).lower()
        text = str(text_ir.get("visible_text") or "")
        if not q:
            return {"found": False, "matches": []}
        q_tokens = _tokenize(q)
        lines = [ln.strip() for ln in re.split(r"[\n.]", text) if ln.strip()]
        matches: List[str] = []
        for ln in lines:
            l = ln.lower()
            if q in l:
                matches.append(ln[:220])
                continue
            overlap = len(q_tokens.intersection(_tokenize(l)))
            if q_tokens and overlap >= max(1, min(3, len(q_tokens))):
                matches.append(ln[:220])
            if len(matches) >= 8:
                break
        return {"found": bool(matches), "matches": _dedupe_keep_order(matches, 8)}

    def vision_qa(
        self,
        *,
        task_id: str,
        prompt: str,
        question: str,
        screenshot: Any,
        candidates: List[Candidate],
        text_ir: Dict[str, Any],
        url: str,
    ) -> Dict[str, Any]:
        if self.vision_call is None:
            return {"ok": False, "error": "vision_disabled"}
        image_url = _normalize_screenshot_data_url(screenshot)
        if not image_url:
            return {"ok": False, "error": "missing_screenshot"}
        q = _candidate_text(question, "Describe the screenshot and identify the best next visible targets.")
        signature = _vision_signature(screenshot=screenshot, question=q, url=url)
        if signature and signature in self._vision_cache:
            cached = dict(self._vision_cache[signature])
            cached["cached"] = True
            return cached
        candidate_lines: List[Dict[str, Any]] = []
        for cand in candidates[:28]:
            candidate_lines.append(
                {
                "id": cand.id,
                "role": cand.role,
                "text": cand.text[:120],
                "field_hint": cand.field_hint[:80],
                "href": cand.href[:140],
                "context": cand.context[:180],
                }
            )
        payload = {
            "task": _candidate_text(prompt)[:600],
            "url": str(url or "")[:400],
            "question": q[:500],
            "headings": (text_ir.get("headings") if isinstance(text_ir.get("headings"), list) else [])[:8],
            "control_groups": (text_ir.get("control_groups") if isinstance(text_ir.get("control_groups"), list) else [])[:6],
            "cards": (text_ir.get("cards") if isinstance(text_ir.get("cards"), list) else [])[:6],
            "candidates": candidate_lines,
        }
        system = (
            "You analyze a webpage screenshot for a browser agent.\n"
            "Return exactly one JSON object with keys:\n"
            "- answer: short answer to the question\n"
            "- element_ids: list of candidate ids from the provided candidate list that best match the screenshot and question\n"
            "- signals: short visual observations about layout, panels, cards, dialogs, or likely primary controls\n"
            "- confidence: one of low, medium, high\n"
            "Only return candidate ids that appear in the provided candidate list."
        )
        user_content = [
            {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
            {"type": "image_url", "image_url": {"url": image_url, "detail": _env_str("FSM_VISION_DETAIL", "low") or "low"}},
        ]
        model = _env_str("FSM_VISION_MODEL", _env_str("OPENAI_VISION_MODEL", "gpt-4o-mini"))
        max_tokens = int(_env_str("FSM_VISION_MAX_TOKENS", "350") or "350")
        raw = self.vision_call(
            task_id=task_id or "vision",
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        content = str((((raw or {}).get("choices") or [{}])[0].get("message", {}) or {}).get("content") or "")
        try:
            parsed = json.loads(content)
        except Exception as exc:
            return {"ok": False, "error": f"vision_parse_failed:{str(exc)[:120]}", "raw": content[:500]}
        ids = parsed.get("element_ids") if isinstance(parsed.get("element_ids"), list) else []
        out = {
            "ok": True,
            "answer": _candidate_text(parsed.get("answer"))[:260],
            "element_ids": _dedupe_keep_order([str(x)[:120] for x in ids if str(x).strip()], MAX_VISUAL_HINTS),
            "signals": _dedupe_keep_order([str(x)[:180] for x in (parsed.get("signals") or []) if str(x).strip()], 8),
            "confidence": _candidate_text(parsed.get("confidence"), "low")[:20],
            "model": str((raw or {}).get("model") or model),
            "usage": (raw.get("usage") if isinstance(raw, dict) and isinstance(raw.get("usage"), dict) else {}),
            "signature": signature,
            "cached": False,
        }
        if signature:
            self._vision_cache[signature] = dict(out)
        return out

    def find_elements(
        self,
        *,
        candidates: List[Candidate],
        role: str = "",
        text: str = "",
        limit: int = 8,
    ) -> Dict[str, Any]:
        role_l = str(role or "").strip().lower()
        text_l = str(text or "").strip().lower()
        out: List[str] = []
        for cand in candidates:
            if role_l and cand.role != role_l:
                continue
            blob = " ".join([cand.text, cand.context, cand.href]).lower()
            if text_l and text_l not in blob:
                continue
            out.append(cand.id)
            if len(out) >= max(1, min(int(limit or 8), 20)):
                break
        return {"element_ids": _dedupe_keep_order(out, 20)}

    def select_next_target(self, *, state: AgentState) -> Dict[str, Any]:
        seen = set(state.visited.urls)
        for url in state.frontier.pending_urls:
            if url not in seen:
                return {"url": url}
        return {"url": ""}

    def escalate(self, *, reason: str) -> Dict[str, Any]:
        return {"checkpoint": f"escalate:{_norm_ws(reason)[:180]}"}

    def set_mode(self, *, mode: str) -> Dict[str, Any]:
        m = str(mode or "").strip().upper()
        if m not in FSM_MODES:
            m = "PLAN"
        return {"mode": m}

    def mark_progress(self, *, state: AgentState, subgoal_id: str, status: str) -> Dict[str, Any]:
        sg_id = str(subgoal_id or "")
        sg_status = str(status or "").strip().lower()
        valid = {"pending", "active", "done", "blocked"}
        if sg_status not in valid:
            sg_status = "pending"
        for sg in state.plan.subgoals:
            if sg.id == sg_id:
                sg.status = sg_status  # type: ignore[assignment]
                if sg_status == "active":
                    state.plan.active_id = sg.id
                return {"updated": True}
        return {"updated": False}


class MetaToolExecutor:
    def __init__(self, skills: Skills) -> None:
        self.skills = skills

    def execute(
        self,
        *,
        task_id: str,
        tool_name: str,
        args: Dict[str, Any],
        state: AgentState,
        prompt: str,
        text_ir: Dict[str, Any],
        candidates: List[Candidate],
        url: str,
        screenshot: Any = None,
    ) -> Dict[str, Any]:
        name = str(tool_name or "").strip().upper()
        if name not in META_TOOLS:
            return {"ok": False, "error": f"unknown_meta_tool:{name}"}

        if name == "META.SOLVE_POPUPS":
            result = self.skills.solve_popups(candidates=candidates)
            state.frontier.pending_elements = _dedupe_keep_order(
                state.frontier.pending_elements + list(result.get("pending_elements") or []),
                MAX_PENDING_ELEMENTS,
            )
            state.memory.checkpoints = _dedupe_keep_order(
                state.memory.checkpoints + [str(result.get("checkpoint") or "solve_popups")],
                MAX_CHECKPOINTS,
            )
            return {"ok": True}

        if name == "META.REPLAN":
            replanned = self.skills.replan(prompt=prompt)
            state.plan.subgoals = list(replanned.get("subgoals") or [])
            state.plan.active_id = ""
            if state.plan.subgoals:
                state.plan.subgoals[0].status = "active"
                state.plan.active_id = state.plan.subgoals[0].id
            return {"ok": True}

        if name == "META.EXTRACT_LINKS":
            kind = str(args.get("kind") or "all_links")
            result = self.skills.extract_links(kind=kind, candidates=candidates)
            state.frontier.pending_urls = _dedupe_keep_order(
                state.frontier.pending_urls + list(result.get("urls") or []),
                MAX_PENDING_URLS,
            )
            state.memory.checkpoints = _dedupe_keep_order(
                state.memory.checkpoints + [f"extract_links:{kind}"],
                MAX_CHECKPOINTS,
            )
            return {"ok": True}

        if name == "META.EXTRACT_FACTS":
            schema = str(args.get("schema") or "generic")
            result = self.skills.extract_facts(schema=schema, text_ir=text_ir, url=url)
            state.memory.facts = _dedupe_keep_order(state.memory.facts + list(result.get("facts") or []), MAX_FACTS)
            state.memory.checkpoints = _dedupe_keep_order(
                state.memory.checkpoints + [f"extract_facts:{schema}"],
                MAX_CHECKPOINTS,
            )
            return {"ok": True}

        if name == "META.SEARCH_TEXT":
            query = str(args.get("query") or "")
            result = self.skills.search_text(query=query, text_ir=text_ir)
            matches = result.get("matches") if isinstance(result.get("matches"), list) else []
            if matches:
                state.memory.facts = _dedupe_keep_order(
                    state.memory.facts + [f"Search match: {str(m)[:220]} (source: {url})" for m in matches[:4]],
                    MAX_FACTS,
                )
            state.memory.checkpoints = _dedupe_keep_order(
                state.memory.checkpoints + [f"search_text:{query[:64]}:{len(matches)}"],
                MAX_CHECKPOINTS,
            )
            return {"ok": True, "found": bool(result.get("found"))}

        if name == "META.FIND_ELEMENTS":
            role = str(args.get("role") or "")
            text = str(args.get("text") or "")
            limit = int(args.get("limit") or 8)
            result = self.skills.find_elements(candidates=candidates, role=role, text=text, limit=limit)
            ids = result.get("element_ids") if isinstance(result.get("element_ids"), list) else []
            state.frontier.pending_elements = _dedupe_keep_order(
                state.frontier.pending_elements + [str(x) for x in ids],
                MAX_PENDING_ELEMENTS,
            )
            state.memory.checkpoints = _dedupe_keep_order(
                state.memory.checkpoints + [f"find_elements:{role or 'any'}:{text[:48]}:{len(ids)}"],
                MAX_CHECKPOINTS,
            )
            return {"ok": True, "count": len(ids)}

        if name == "META.VISION_QA":
            question = str(args.get("question") or "").strip()
            if not question:
                question = (
                    "Based on the screenshot and task, which visible candidate ids are the best next targets? "
                    "Prefer current-page controls for narrowing or submitting, and identify the most relevant visible target."
                )
            result = self.skills.vision_qa(
                task_id=task_id,
                prompt=prompt,
                question=question,
                screenshot=screenshot,
                candidates=candidates,
                text_ir=text_ir,
                url=url,
            )
            if not bool(result.get("ok")):
                state.memory.checkpoints = _dedupe_keep_order(
                    state.memory.checkpoints + [f"vision_qa_error:{str(result.get('error') or 'unknown')[:80]}"],
                    MAX_CHECKPOINTS,
                )
                return result
            answer = str(result.get("answer") or "").strip()
            signals = result.get("signals") if isinstance(result.get("signals"), list) else []
            element_ids = result.get("element_ids") if isinstance(result.get("element_ids"), list) else []
            note_parts = [answer] if answer else []
            note_parts.extend(str(x)[:120] for x in signals[:3] if str(x).strip())
            visual_note = " | ".join(note_parts)[:260]
            if visual_note:
                state.memory.visual_notes = _dedupe_keep_order(state.memory.visual_notes + [visual_note], MAX_VISUAL_NOTES)
            if element_ids:
                state.memory.visual_element_hints = _dedupe_keep_order(
                    state.memory.visual_element_hints + [str(x)[:120] for x in element_ids],
                    MAX_VISUAL_HINTS,
                )
                state.frontier.pending_elements = _dedupe_keep_order(
                    state.frontier.pending_elements + [str(x)[:120] for x in element_ids],
                    MAX_PENDING_ELEMENTS,
                )
            if str(result.get("signature") or "").strip():
                state.memory.last_vision_signature = str(result.get("signature") or "")[:64]
            state.memory.checkpoints = _dedupe_keep_order(
                state.memory.checkpoints + [f"vision_qa:{len(element_ids)}:{str(result.get('confidence') or 'low')[:12]}"],
                MAX_CHECKPOINTS,
            )
            return result

        if name == "META.SELECT_NEXT_TARGET":
            result = self.skills.select_next_target(state=state)
            picked = str(result.get("url") or "").strip()
            if picked:
                state.frontier.pending_urls = _dedupe_keep_order([picked] + state.frontier.pending_urls, MAX_PENDING_URLS)
            state.memory.checkpoints = _dedupe_keep_order(
                state.memory.checkpoints + ["select_next_target"],
                MAX_CHECKPOINTS,
            )
            return {"ok": True, "url": picked}

        if name == "META.ESCALATE":
            reason = str(args.get("reason") or "stuck")
            result = self.skills.escalate(reason=reason)
            state.escalated_once = True
            state.memory.checkpoints = _dedupe_keep_order(
                state.memory.checkpoints + [str(result.get("checkpoint") or "escalate")],
                MAX_CHECKPOINTS,
            )
            return {"ok": True}

        if name == "META.SET_MODE":
            mode_out = self.skills.set_mode(mode=str(args.get("mode") or "PLAN"))
            state.mode = str(mode_out.get("mode") or state.mode)
            state.memory.checkpoints = _dedupe_keep_order(
                state.memory.checkpoints + [f"set_mode:{state.mode}"],
                MAX_CHECKPOINTS,
            )
            return {"ok": True, "mode": state.mode}

        if name == "META.MARK_PROGRESS":
            subgoal_id = str(args.get("id") or "")
            status = str(args.get("status") or "done")
            result = self.skills.mark_progress(state=state, subgoal_id=subgoal_id, status=status)
            state.memory.checkpoints = _dedupe_keep_order(
                state.memory.checkpoints + [f"mark_progress:{subgoal_id}:{status}"],
                MAX_CHECKPOINTS,
            )
            return {"ok": bool(result.get("updated"))}

        return {"ok": False, "error": f"unhandled_meta_tool:{name}"}


class Policy:
    def __init__(self, llm_call: Callable[..., Dict[str, Any]]) -> None:
        self.llm_call = llm_call
        self.debug_dir = str(os.getenv("FSM_POLICY_DEBUG_DIR", "") or "").strip()

    def _debug_log(self, task_id: str, payload: Dict[str, Any]) -> None:
        if not self.debug_dir:
            return
        try:
            base = Path(self.debug_dir).expanduser().resolve()
            safe_task = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(task_id or "task"))[:120] or "task"
            _append_jsonl(base / f"{safe_task}.jsonl", payload)
        except Exception:
            return

    def decide(
        self,
        *,
        task_id: str,
        prompt: str,
        mode: str,
        policy_obs: Dict[str, Any],
        allowed_tools: set[str],
        model_name: str,
        plan_model_name: str,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        if mode == "POPUP":
            return {"type": "meta", "name": "META.SOLVE_POPUPS", "arguments": {}}, {"source": "deterministic"}
        if mode == "REPORT":
            facts = policy_obs.get("memory", {}).get("facts") if isinstance(policy_obs.get("memory"), dict) else []
            fact = _candidate_text((facts or [""])[0] if isinstance(facts, list) and facts else "")
            content = fact or "Task appears complete."
            return {"type": "final", "done": True, "content": content}, {"source": "deterministic"}

        meta_enabled = any(str(tool or "").startswith("META.") for tool in allowed_tools)
        direct_mode = mode == "DIRECT"
        if direct_mode:
            system = (
                "You are a web operator.\n"
                "You have the user task, the current page snapshot, the interactive elements, and the allowed browser tools.\n"
                "Choose exactly ONE next step.\n"
                "Return ONE JSON object only. No markdown. No prose.\n"
                "Valid outputs:\n"
                '1) {"type":"browser","tool_call":{...}}\n'
                '2) {"type":"final","done":true,"content":"..."}\n'
                "Rules:\n"
                "- First decide whether the current page already satisfies the task. If yes, finish immediately with final or browser.end.\n"
                "- content must be the actual user-facing answer, result, or extracted value.\n"
                "- Do not keep exploring when CURRENT PAGE ANSWERS, PAGE FACTS, RELEVANT VISIBLE TEXT, or visible text already satisfy the task.\n"
                "- Never return more than one browser action.\n"
                "- Use only element_id values that appear in ACTION SHORTLIST.\n"
                "- For click/type/select, prefer arguments.element_id from the shortlist.\n"
                "- For browser.select, include a non-empty arguments.value.\n"
                "- browser.end is the standard way to finish once the page already satisfies the task.\n"
                "- If the task includes filters or explicit constraints, use visible controls first before opening result items.\n"
                "- After typing into a field, if the page now shows a better submit, suggestion, or next control, use that instead of restarting.\n"
                "- If a visible local form has multiple relevant empty fields, fill the remaining relevant fields before using submit.\n"
                "- Prefer the current visible form or control group over navigation links when that form already covers the task constraints.\n"
                "- Do not type into the same field again if FORM PROGRESS already shows that field/value as filled unless the value is clearly wrong.\n"
                "- Do not navigate to the current page URL again. If the needed form or answer is already visible, act on the current page.\n"
                "- Preserve placeholders such as <username>, <password>, <signup_email> exactly when typing.\n"
            )
        else:
            system = (
            "You are a web automation policy.\n"
            "Given the task and the current page state, choose ONE next step.\n"
            "Return ONE JSON object only. No markdown. No prose. No chain-of-thought.\n"
            "You must choose exactly one of:\n"
            "1) browser tool_call\n"
            + ("2) meta_tool\n" if meta_enabled else "")
            + f"{'3' if meta_enabled else '2'}) final (done=true + content)\n\n"
            "Rules:\n"
            "- Prefer a concrete browser action when there is a reasonable actionable target.\n"
            + ("- Use a meta_tool only when inspection/disambiguation materially improves the next browser action.\n" if meta_enabled else "")
            + "- Never return more than one browser action.\n"
            "- If the current page already contains the answer, return final immediately.\n"
            "- For question-answering and data-extraction tasks, DONE is the correct action once the answer is visible on the current page.\n"
            "- Do NOT keep exploring once CURRENT PAGE FACTS or CURRENT PAGE ANSWERS already answer the task.\n"
            "- Use final/done or browser.end with a concrete content string when the task is satisfied.\n"
            "- content must be the actual answer for the user, not a status message.\n"
            "- Avoid repeating low-value actions when the page did not materially change.\n"
            "- For click/type/select, prefer arguments.element_id from candidates.\n"
            "- For browser.select, provide arguments.value with the option text/value to choose.\n"
            "- Preserve placeholders such as <username>, <password>, <signup_email> exactly when typing.\n"
            "- For informational tasks, use CURRENT PAGE FACTS and CURRENT PAGE ANSWERS before navigating more.\n"
            )
        if direct_mode:
            user_parts = [
                "Choose the next single step.",
                f"TASK: {str(policy_obs.get('prompt') or '')[:1600]}",
                "TASK CONSTRAINTS:",
                json.dumps(
                    (
                        policy_obs.get("task_constraints")
                        if isinstance(policy_obs.get("task_constraints"), dict)
                        else {}
                    ),
                    ensure_ascii=False,
                ),
                f"STEP: {int(policy_obs.get('step_index') or 0)}",
                f"URL: {str(policy_obs.get('url') or '')[:1000]}",
                "",
                "DONE / CONTENT CONTRACT:",
                "- If the answer or completed result is already visible on the current page, return final now.",
                "- final.content or browser.end.arguments.content must be the concrete answer for the user.",
                "- Do not return generic status text like 'task complete' or 'done'.",
                "- Before taking another action, check CURRENT PAGE SNAPSHOT, CURRENT PAGE ANSWERS, and CURRENT PAGE.",
                "- If a select already shows the target value, do not select the same value again.",
                "- If the current page shows grouped controls or filters, use their current values before choosing another action.",
                "- If the task includes explicit constraints such as search terms, genre, year, date, or sort values, satisfy those constraints with visible controls before opening result items.",
                "- If a visible local form already contains relevant fields for the task, complete the remaining relevant fields before clicking submit.",
                "- Prefer the current visible form or control group over navigation links when the needed fields are already on the page.",
                "",
                "CURRENT PAGE SNAPSHOT:",
                str(policy_obs.get("browser_state_snapshot") or "")[:3000],
                "",
                "CURRENT PAGE ANSWERS:",
                json.dumps(
                    {
                        "likely_answers": (
                            policy_obs.get("page_observations", {}).get("likely_answers")
                            if isinstance(policy_obs.get("page_observations"), dict)
                            and isinstance(policy_obs.get("page_observations", {}).get("likely_answers"), list)
                            else []
                        )[:6],
                        "informational_task": (
                            policy_obs.get("page_observations", {}).get("informational_task")
                            if isinstance(policy_obs.get("page_observations"), dict)
                            else False
                        ),
                        "relevant_lines": (
                            policy_obs.get("page_observations", {}).get("relevant_lines")
                            if isinstance(policy_obs.get("page_observations"), dict)
                            and isinstance(policy_obs.get("page_observations", {}).get("relevant_lines"), list)
                            else []
                        )[:10],
                        "page_stats": (
                            policy_obs.get("page_observations", {}).get("page_stats")
                            if isinstance(policy_obs.get("page_observations"), dict)
                            and isinstance(policy_obs.get("page_observations", {}).get("page_stats"), dict)
                            else {}
                        ),
                    },
                    ensure_ascii=False,
                ),
                "",
                "FORM PROGRESS (JSON):",
                json.dumps(
                    {
                        "typed_values_recent": (
                            policy_obs.get("memory", {}).get("typed_values_recent")
                            if isinstance(policy_obs.get("memory"), dict)
                            and isinstance(policy_obs.get("memory", {}).get("typed_values_recent"), list)
                            else []
                        )[:8],
                        "typed_candidate_ids": (
                            policy_obs.get("memory", {}).get("typed_candidate_ids")
                            if isinstance(policy_obs.get("memory"), dict)
                            and isinstance(policy_obs.get("memory", {}).get("typed_candidate_ids"), list)
                            else []
                        )[:12],
                        "active_group_label": (
                            policy_obs.get("memory", {}).get("active_group_label")
                            if isinstance(policy_obs.get("memory"), dict)
                            else ""
                        ),
                    },
                    ensure_ascii=False,
                ),
                "",
                "CONTROL GROUPS (JSON):",
                json.dumps(
                    (
                        policy_obs.get("text_ir", {}).get("control_groups")
                        if isinstance(policy_obs.get("text_ir"), dict)
                        and isinstance(policy_obs.get("text_ir", {}).get("control_groups"), list)
                        else []
                    )[:8],
                    ensure_ascii=False,
                ),
                "",
                "VISIBLE FORMS (JSON):",
                json.dumps(
                    (
                        policy_obs.get("text_ir", {}).get("forms")
                        if isinstance(policy_obs.get("text_ir"), dict)
                        and isinstance(policy_obs.get("text_ir", {}).get("forms"), list)
                        else []
                    )[:6],
                    ensure_ascii=False,
                ),
                "",
                "CURRENT PAGE:",
                str(policy_obs.get("page_ir_text") or "")[:14000],
                "",
                "INTERACTIVE ELEMENTS (tree-style):",
                str(policy_obs.get("browser_state_text") or "")[:9000],
                "",
                "ACTION SHORTLIST (JSON):",
                json.dumps(
                    (policy_obs.get("candidates") if isinstance(policy_obs.get("candidates"), list) else [])[:14],
                    ensure_ascii=False,
                ),
                "",
                "ALLOWED BROWSER TOOLS: " + ", ".join(sorted([t for t in list(allowed_tools) if str(t).startswith("browser.")]) if allowed_tools else []),
                "",
                "Output schema examples:",
                '{"type":"browser","tool_call":{"name":"browser.click","arguments":{"element_id":"el_123"}}}',
                '{"type":"browser","tool_call":{"name":"browser.end","arguments":{"content":"The total value is 2844."}}}',
                '{"type":"final","done":true,"content":"The total value is 2844."}',
            ]
        else:
            user_parts = [
            "You have a task and must choose the next single browser step.",
            f"TASK: {str(policy_obs.get('prompt') or '')[:1600]}",
            f"STEP: {int(policy_obs.get('step_index') or 0)}",
            f"MODE: {mode}",
            f"URL: {str(policy_obs.get('url') or '')[:1000]}",
            f"SCREENSHOT_AVAILABLE: {bool(policy_obs.get('screenshot_available'))}",
            "",
            "DONE / CONTENT CONTRACT:",
            "- If the task is already answered by the current page, return final now.",
            "- final.content must contain the user-facing answer.",
            "- Do not output a browser action when the answer is already visible.",
            "- Prefer quoting the visible metric / value directly in content.",
            "",
            "PAGE IR (PRIMARY STRUCTURED STATE):",
            str(policy_obs.get("page_ir_text") or "")[:14000],
            "",
            "CURRENT PAGE ANSWERS:",
            json.dumps(
                {
                    "likely_answers": (
                        policy_obs.get("page_observations", {}).get("likely_answers")
                        if isinstance(policy_obs.get("page_observations"), dict)
                        and isinstance(policy_obs.get("page_observations", {}).get("likely_answers"), list)
                        else []
                    )[:6],
                    "page_fact_count": (
                        policy_obs.get("page_observations", {}).get("page_fact_count")
                        if isinstance(policy_obs.get("page_observations"), dict)
                        else 0
                    ),
                    "informational_task": (
                        policy_obs.get("page_observations", {}).get("informational_task")
                        if isinstance(policy_obs.get("page_observations"), dict)
                        else False
                    ),
                    "relevant_lines": (
                        policy_obs.get("page_observations", {}).get("relevant_lines")
                        if isinstance(policy_obs.get("page_observations"), dict)
                        and isinstance(policy_obs.get("page_observations", {}).get("relevant_lines"), list)
                        else []
                    )[:10],
                },
                ensure_ascii=False,
            ),
            "",
            "STATE DELTA:",
            str(policy_obs.get("state_delta") or "")[:600],
            "",
            "PAGE IR DELTA:",
            str(policy_obs.get("ir_delta") or "")[:600],
            "",
            "BROWSER_STATE (interactive elements):",
            str(policy_obs.get("browser_state_text") or "")[:14000],
            "",
            "PAGE OBSERVATIONS (GENERIC JSON):",
            json.dumps(policy_obs.get("page_observations") if isinstance(policy_obs.get("page_observations"), dict) else {}, ensure_ascii=False),
            "",
            "PAGE GROUPS (FORMS / CONTROL GROUPS / ITEM GROUPS JSON):",
            json.dumps(
                {
                    "forms": (
                        policy_obs.get("text_ir", {}).get("forms")
                        if isinstance(policy_obs.get("text_ir"), dict) and isinstance(policy_obs.get("text_ir", {}).get("forms"), list)
                        else []
                    )[:6],
                    "control_groups": (
                        policy_obs.get("text_ir", {}).get("control_groups")
                        if isinstance(policy_obs.get("text_ir"), dict)
                        and isinstance(policy_obs.get("text_ir", {}).get("control_groups"), list)
                        else []
                    )[:8],
                    "cards": (
                        policy_obs.get("text_ir", {}).get("cards")
                        if isinstance(policy_obs.get("text_ir"), dict) and isinstance(policy_obs.get("text_ir", {}).get("cards"), list)
                        else []
                    )[:8],
                },
                ensure_ascii=False,
            ),
            "",
            "PREVIOUS STEP VERDICT:",
            json.dumps(policy_obs.get("previous_step_verdict") if isinstance(policy_obs.get("previous_step_verdict"), dict) else {}, ensure_ascii=False),
            ]
            history_summary = str(policy_obs.get("history_summary") or "").strip()
            if history_summary:
                user_parts.extend(["", "HISTORY SUMMARY:", history_summary[:2000]])
            history_recent = policy_obs.get("history_recent") if isinstance(policy_obs.get("history_recent"), list) else []
            if history_recent:
                user_parts.extend(["", "HISTORY RECENT (JSON):", json.dumps(history_recent[:10], ensure_ascii=False)])
            loop_nudges = policy_obs.get("loop_nudges") if isinstance(policy_obs.get("loop_nudges"), list) else []
            if loop_nudges:
                user_parts.extend(["", "LOOP NUDGES:"] + [f"- {str(item)[:220]}" for item in loop_nudges[:8]])
            memory = policy_obs.get("memory") if isinstance(policy_obs.get("memory"), dict) else {}
            strategy_summary = str(memory.get("strategy_summary") or "").strip()
            capability_gap = (
                policy_obs.get("page_observations", {}).get("capability_gap")
                if isinstance(policy_obs.get("page_observations"), dict)
                and isinstance(policy_obs.get("page_observations", {}).get("capability_gap"), dict)
                else {}
            )
            if strategy_summary or capability_gap:
                user_parts.extend(
                    [
                        "",
                        "STRATEGY / CAPABILITY GAP:",
                        json.dumps(
                            {
                                "strategy_summary": strategy_summary,
                                "capability_gap": capability_gap,
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
            plan = policy_obs.get("plan") if isinstance(policy_obs.get("plan"), dict) else {}
            counters = policy_obs.get("counters") if isinstance(policy_obs.get("counters"), dict) else {}
            visual_notes = memory.get("visual_notes") if isinstance(memory.get("visual_notes"), list) else []
            visual_hints = memory.get("visual_element_hints") if isinstance(memory.get("visual_element_hints"), list) else []
            if visual_notes or visual_hints:
                user_parts.extend(
                    [
                        "",
                        "VISUAL OBSERVATIONS:",
                        json.dumps(
                            {"notes": visual_notes[:8], "element_hints": visual_hints[:12]},
                            ensure_ascii=False,
                        ),
                    ]
                )
            user_parts.extend(
                [
                "",
                "PLAN / MEMORY (JSON):",
                json.dumps(
                    {
                        "task_constraints": policy_obs.get("task_constraints") if isinstance(policy_obs.get("task_constraints"), dict) else {},
                        "active_subgoal": policy_obs.get("active_subgoal") if isinstance(policy_obs.get("active_subgoal"), dict) else {},
                        "plan": plan,
                        "memory": memory,
                        "counters": counters,
                        "frontier": policy_obs.get("frontier") if isinstance(policy_obs.get("frontier"), dict) else {},
                    },
                    ensure_ascii=False,
                ),
                "ACTION SHORTLIST (JSON):",
                json.dumps(
                    (policy_obs.get("candidates") if isinstance(policy_obs.get("candidates"), list) else [])[:24],
                    ensure_ascii=False,
                ),
                "",
                "ALLOWED BROWSER TOOLS: " + ", ".join(sorted([t for t in list(allowed_tools) if str(t).startswith("browser.")]) if allowed_tools else []),
                "",
                "Output schema examples:",
                '{"type":"browser","tool_call":{"name":"browser.click","arguments":{"element_id":"el_123"}}}',
                '{"type":"browser","tool_call":{"name":"browser.select","arguments":{"element_id":"el_456","value":"Comedy"}}}',
                '{"type":"browser","tool_call":{"name":"browser.end","arguments":{"content":"The total value is 2844."}}}',
                '{"type":"final","done":true,"content":"The total value is 2844."}',
                "",
                "Instructions:",
                "- Output JSON only.",
                "- Return ONE step only.",
                "- Do not burn steps on the same no-op pattern if nothing changed.",
                "- For browser.select, include a non-empty arguments.value.",
                "- If the task is about narrowing results, use current-page controls before opening result items.",
                "- Preserve placeholders exactly when typing.",
                "- For informational tasks, prefer answering from what is already visible on the current page before opening more pages.",
                "- If CURRENT PAGE ANSWERS are sufficient, finish now.",
                "- Do not return content like 'task completed'; return the actual answer.",
                ]
            )
            if meta_enabled:
                user_parts.insert(-5, '{"type":"meta","meta_tool":{"name":"META.FIND_ELEMENTS","arguments":{"role":"input","text":"search","limit":6}}}')
                user_parts.insert(-5, '{"type":"meta","meta_tool":{"name":"META.VISION_QA","arguments":{"question":"Which visible control best applies the current filters?"}}}')
        user_text = "\n".join([part for part in user_parts if part is not None])[:45000]
        model = plan_model_name if (mode == "PLAN" or mode == "STUCK") else model_name
        self._debug_log(
            str(task_id or "task"),
            {
                "ts": _utc_now(),
                "phase": "llm_request",
                "mode": mode,
                "model": model,
                "system": system,
                "user": user_text,
                "allowed_tools_count": len(allowed_tools),
            },
        )
        try:
            raw = self.llm_call(
                task_id=str(task_id or "local"),
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_text},
                ],
                temperature=0.1,
                max_tokens=1600,
            )
            content = str(
                (((raw or {}).get("choices") or [{}])[0].get("message", {}) or {}).get("content") or ""
            )
            usage = self._normalize_usage(raw)
            try:
                obj = self._parse_json(content)
                normalized = self._normalize_decision(obj, allowed_tools)
            except Exception as parse_or_schema_err:
                repaired = self._attempt_repair(
                    task_id=str(task_id or "local"),
                    model=model,
                    mode=mode,
                    raw_content=content,
                    allowed_tools=allowed_tools,
                )
                if repaired is None:
                    raise parse_or_schema_err
                normalized, repair_usage, repair_model, repair_content = repaired
                self._debug_log(
                    str(task_id or "task"),
                    {
                        "ts": _utc_now(),
                        "phase": "llm_repair_response",
                        "mode": mode,
                        "model": str(repair_model or model),
                        "raw_content": repair_content,
                        "normalized": normalized,
                        "usage": repair_usage,
                        "original_error": str(parse_or_schema_err),
                    },
                )
                return normalized, {
                    "source": "llm_repair",
                    "usage": repair_usage,
                    "model": str(repair_model or model),
                }
            self._debug_log(
                str(task_id or "task"),
                {
                    "ts": _utc_now(),
                    "phase": "llm_response",
                    "mode": mode,
                    "model": str((raw or {}).get("model") or model),
                    "raw_content": content,
                    "parsed": obj,
                    "normalized": normalized,
                    "usage": usage,
                },
            )
            return normalized, {
                "source": "llm",
                "usage": usage,
                "model": str((raw or {}).get("model") or model),
            }
        except Exception as e:
            self._debug_log(
                str(task_id or "task"),
                {
                    "ts": _utc_now(),
                    "phase": "llm_error",
                    "mode": mode,
                    "model": model,
                    "error": str(e),
                },
            )
            fallback = self._fallback(prompt=prompt, mode=mode, policy_obs=policy_obs, allowed_tools=allowed_tools)
            self._debug_log(
                str(task_id or "task"),
                {
                    "ts": _utc_now(),
                    "phase": "fallback_decision",
                    "mode": mode,
                    "decision": fallback,
                },
            )
            return fallback, {"source": "fallback"}

    def _normalize_usage(self, raw: Dict[str, Any] | None) -> Dict[str, int]:
        usage = (raw or {}).get("usage") if isinstance((raw or {}).get("usage"), dict) else {}
        return {
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
        }

    def _extract_first_json_object(self, raw: str) -> str | None:
        if not raw:
            return None
        in_str = False
        esc = False
        depth = 0
        start = -1
        for i, ch in enumerate(raw):
            if in_str:
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = True
                    continue
                if ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
                continue
            if ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start >= 0:
                        return raw[start : i + 1]
        return None

    def _parse_json(self, content: str) -> Dict[str, Any]:
        raw = str(content or "").strip()
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        first_obj = self._extract_first_json_object(raw)
        if first_obj:
            try:
                obj = json.loads(first_obj)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
            # Weak fallback for single-quoted dict-like outputs.
            try:
                lit = ast.literal_eval(first_obj)
                if isinstance(lit, dict):
                    return json.loads(json.dumps(lit, ensure_ascii=False))
            except Exception:
                pass
        raise ValueError("invalid_json_policy_output")

    def _normalize_decision(self, obj: Dict[str, Any], allowed_tools: set[str]) -> Dict[str, Any]:
        t = str(obj.get("type") or "").strip().lower()
        if not t:
            if isinstance(obj.get("tool_call"), dict):
                t = "browser"
            elif isinstance(obj.get("meta_tool"), dict):
                t = "meta"
            elif str(obj.get("name") or obj.get("tool") or "").strip():
                maybe_name = _canonical_allowed_tool_name(str(obj.get("name") or obj.get("tool") or ""))
                if maybe_name.startswith("browser."):
                    t = "browser"
                elif maybe_name.startswith("meta.") or maybe_name.startswith("META."):
                    t = "meta"
            elif bool(obj.get("done")) or isinstance(obj.get("content"), str):
                t = "final"
        if t == "final":
            return {
                "type": "final",
                "done": True,
                "content": _candidate_text(obj.get("content"), "Task complete."),
                "reasoning": _candidate_text(obj.get("reasoning")),
            }
        if t == "meta":
            mt = obj.get("meta_tool") if isinstance(obj.get("meta_tool"), dict) else {}
            name = str(mt.get("name") or obj.get("name") or "").strip().upper()
            if name in META_TOOLS and (not allowed_tools or name in allowed_tools):
                return {
                    "type": "meta",
                    "name": name,
                    "arguments": (
                        mt.get("arguments")
                        if isinstance(mt.get("arguments"), dict)
                        else (obj.get("arguments") if isinstance(obj.get("arguments"), dict) else {})
                    ),
                    "reasoning": _candidate_text(obj.get("reasoning")),
                }
        if t == "browser":
            tc = obj.get("tool_call") if isinstance(obj.get("tool_call"), dict) else {}
            if not tc:
                tc = {
                    "name": str(obj.get("name") or obj.get("tool") or "").strip(),
                    "arguments": obj.get("arguments") if isinstance(obj.get("arguments"), dict) else {},
                }
            raw_name = str(tc.get("name") or "").strip()
            name = _canonical_allowed_tool_name(raw_name)
            if name == "browser.end":
                args = tc.get("arguments") if isinstance(tc.get("arguments"), dict) else {}
                return {
                    "type": "final",
                    "done": True,
                    "content": _candidate_text(args.get("content"), obj.get("content"), "Task complete."),
                    "reasoning": _candidate_text(obj.get("reasoning")),
                }
            if name.startswith("browser.") and (not allowed_tools or name in allowed_tools):
                return {
                    "type": "browser",
                    "tool_call": {
                        "name": name,
                        "arguments": tc.get("arguments") if isinstance(tc.get("arguments"), dict) else {},
                    },
                    "reasoning": _candidate_text(obj.get("reasoning")),
                }
        raise ValueError("invalid_policy_decision")

    def _attempt_repair(
        self,
        *,
        task_id: str,
        model: str,
        mode: str,
        raw_content: str,
        allowed_tools: set[str],
    ) -> tuple[Dict[str, Any], Dict[str, int], str, str] | None:
        repair_system = (
            "You fix malformed agent policy outputs.\n"
            "Return exactly ONE valid JSON object and nothing else.\n"
            "Output must match one of:\n"
            "1) {\"type\":\"browser\",\"tool_call\":{\"name\":\"browser.<tool>\",\"arguments\":{}}}\n"
            "2) {\"type\":\"meta\",\"meta_tool\":{\"name\":\"META.<TOOL>\",\"arguments\":{}}}\n"
            "3) {\"type\":\"final\",\"done\":true,\"content\":\"...\"}"
        )
        repair_user = {
            "mode": mode,
            "allowed_browser_tools": sorted(list(allowed_tools)) if allowed_tools else [],
            "raw_response": str(raw_content or "")[:12000],
        }
        try:
            raw = self.llm_call(
                task_id=str(task_id or "local"),
                model=str(model),
                messages=[
                    {"role": "system", "content": repair_system},
                    {"role": "user", "content": json.dumps(repair_user, ensure_ascii=False)},
                ],
                temperature=0.0,
                max_tokens=500,
            )
            content = str((((raw or {}).get("choices") or [{}])[0].get("message", {}) or {}).get("content") or "")
            obj = self._parse_json(content)
            normalized = self._normalize_decision(obj, allowed_tools)
            usage = self._normalize_usage(raw)
            model_name = str((raw or {}).get("model") or model or "")
            return normalized, usage, model_name, content
        except Exception:
            return None

    def _fallback(
        self,
        *,
        prompt: str,
        mode: str,
        policy_obs: Dict[str, Any],
        allowed_tools: set[str],
    ) -> Dict[str, Any]:
        def allow(name: str) -> bool:
            return (not allowed_tools) or (name in allowed_tools)

        candidates = policy_obs.get("candidates") if isinstance(policy_obs.get("candidates"), list) else []
        partitions = policy_obs.get("candidate_partitions") if isinstance(policy_obs.get("candidate_partitions"), dict) else {}
        local_candidates = partitions.get("local") if isinstance(partitions.get("local"), list) else []
        escape_candidates = partitions.get("escape") if isinstance(partitions.get("escape"), list) else []
        global_candidates = partitions.get("global") if isinstance(partitions.get("global"), list) else []
        memory = policy_obs.get("memory") if isinstance(policy_obs.get("memory"), dict) else {}
        visual_hints = {
            str(x)
            for x in (
                memory.get("visual_element_hints")
                if isinstance(memory.get("visual_element_hints"), list)
                else []
            )
            if str(x).strip()
        }
        typed_candidate_ids = {
            str(x)
            for x in (
                memory.get("typed_candidate_ids")
                if isinstance(memory.get("typed_candidate_ids"), list)
                else []
            )
            if str(x).strip()
        }
        first = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
        flags = policy_obs.get("flags") if isinstance(policy_obs.get("flags"), dict) else {}
        counters = policy_obs.get("counters") if isinstance(policy_obs.get("counters"), dict) else {}
        loop_level = str(flags.get("loop_level") or "none")
        stall_count = int(counters.get("stall_count") or 0)
        repeat_count = int(counters.get("repeat_action_count") or 0)
        route_like_stuck = mode in {"STUCK", "PLAN"} or loop_level == "high" or stall_count >= 4 or repeat_count >= 4

        def candidate_id(item: Dict[str, Any]) -> str:
            return str(item.get("id") or item.get("element_id") or item.get("_element_id") or "").strip()

        def browser_action_for_candidate(item: Dict[str, Any]) -> Dict[str, Any] | None:
            role = str(item.get("role") or "").strip().lower()
            selector = item.get("selector") if isinstance(item.get("selector"), dict) else None
            if not isinstance(selector, dict):
                return None
            element_id = candidate_id(item)
            if role in {"input", "textarea"} and element_id and element_id in typed_candidate_ids:
                return None
            if role in {"button", "link"} and allow("browser.click"):
                return {
                    "type": "browser",
                    "tool_call": {
                        "name": "browser.click",
                        "arguments": {
                            "selector": selector,
                            "element_id": element_id,
                        },
                    },
                }
            if role in {"input", "textarea"} and allow("browser.type"):
                return {
                    "type": "browser",
                    "tool_call": {
                        "name": "browser.type",
                        "arguments": {
                            "selector": selector,
                            "element_id": element_id,
                            "text": "<text>",
                        },
                    },
                }
            if role == "select" and allow("browser.select"):
                return {
                    "type": "browser",
                    "tool_call": {
                        "name": "browser.select",
                        "arguments": {
                            "selector": selector,
                            "element_id": element_id,
                            "value": "Option",
                        },
                    },
                }
            if role == "input" and allow("browser.click"):
                return {
                    "type": "browser",
                    "tool_call": {
                        "name": "browser.click",
                        "arguments": {
                            "selector": selector,
                            "element_id": element_id,
                        },
                    },
                }
            return None

        if route_like_stuck and escape_candidates:
            for cand in escape_candidates:
                if isinstance(cand, dict):
                    act = browser_action_for_candidate(cand)
                    if act is not None:
                        return act
        if route_like_stuck:
            if allow("browser.back"):
                return {"type": "browser", "tool_call": {"name": "browser.back", "arguments": {}}}
        ordered = []
        ordered.extend([cand for cand in local_candidates if isinstance(cand, dict)])
        ordered.extend([cand for cand in escape_candidates if isinstance(cand, dict)])
        ordered.extend([cand for cand in candidates if isinstance(cand, dict)])
        ordered.extend([cand for cand in global_candidates if isinstance(cand, dict)])
        for cand in ordered:
            if not isinstance(cand, dict):
                continue
            if candidate_id(cand) in visual_hints:
                action = browser_action_for_candidate(cand)
                if action is not None:
                    return action
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            action = browser_action_for_candidate(cand)
            if action is not None:
                return action
        if first and allow("browser.click"):
            sel = first.get("selector") if isinstance(first.get("selector"), dict) else None
            if sel:
                return {"type": "browser", "tool_call": {"name": "browser.click", "arguments": {"selector": sel}}}
        if allow("browser.wait"):
            return {"type": "browser", "tool_call": {"name": "browser.wait", "arguments": {"time_seconds": 1.0}}}
        if allow("browser.scroll"):
            return {"type": "browser", "tool_call": {"name": "browser.scroll", "arguments": {"direction": "down", "amount": 600}}}
        return {"type": "final", "done": True, "content": "No safe browser action available from allowed_tools."}


class FSMOperator:
    def __init__(
        self,
        llm_call: Callable[..., Dict[str, Any]],
        vision_call: Callable[..., Dict[str, Any]] | None = None,
    ) -> None:
        self.flags = FlagDetector()
        self.extractor = CandidateExtractor()
        self.ranker = CandidateRanker()
        self.obs_builder = ObsBuilder()
        self._router: Router | None = None
        self._vision_call = vision_call
        self._skills: Skills | None = None
        self._meta: MetaToolExecutor | None = None
        self.policy = Policy(llm_call=llm_call)
        self.obs_extract_call = llm_call
        self.debug_dir = str(os.getenv("FSM_POLICY_DEBUG_DIR", "") or "").strip()
        self.trace_json = _env_bool("FSM_TRACE_JSON", False)
        self.trace_dir = str(
            os.getenv("FSM_TRACE_DIR", self.debug_dir or str((Path(__file__).resolve().parent / "data" / "fsm_traces").resolve()))
            or ""
        ).strip()

    def _allow_control_meta_tools(self) -> bool:
        return _env_bool("FSM_ALLOW_CONTROL_META_TOOLS", False)

    @property
    def router(self) -> Router:
        if self._router is None:
            self._router = Router()
        return self._router

    @property
    def skills(self) -> Skills:
        skills, _ = self._ensure_legacy_support()
        return skills

    @property
    def meta(self) -> MetaToolExecutor:
        _, meta = self._ensure_legacy_support()
        return meta

    def _ensure_legacy_support(self) -> tuple[Skills, MetaToolExecutor]:
        if self._skills is None:
            self._skills = Skills(vision_call=self._vision_call)
        if self._meta is None:
            self._meta = MetaToolExecutor(self._skills)
        return self._skills, self._meta

    def _obs_extract_mode(self) -> str:
        mode = _env_str("FSM_OBS_EXTRACT_MODE", "auto").lower()
        if mode not in {"off", "auto", "always"}:
            mode = "auto"
        return mode

    def _completion_only_result(
        self,
        *,
        prompt: str,
        url: str,
        step_index: int,
        state: AgentState,
        text_ir: Dict[str, Any],
        flags: Dict[str, Any],
    ) -> tuple[bool, str, str]:
        if bool(flags.get("captcha_suspected")):
            return False, "", "Current page is blocked by a challenge."
        if _looks_like_informational_task(prompt):
            best_page_evidence = _best_page_evidence(prompt, text_ir)
            if best_page_evidence and _page_context_ready_for_informational_answer(prompt, url, text_ir):
                ok, _ = self._pre_done_verification(
                    step_index=step_index,
                    state=state,
                    prompt=prompt,
                    text_ir=text_ir,
                    content=best_page_evidence,
                    flags=flags,
                )
                if ok:
                    return True, best_page_evidence, "Current page contains a concrete answer."
            return False, "", "Current page does not yet show a concrete answer."
        if state.mode == "REPORT" and state.memory.facts:
            content = _candidate_text(state.memory.facts[0])
            if content:
                return True, content, "Current evidence is sufficient."
        return False, "", "Current page is not yet sufficient to conclude completion."

    def _obs_extract_signature(self, *, dom_hash: str, url: str) -> str:
        raw = json.dumps({"dom_hash": str(dom_hash or "")[:64], "url": str(url or "")[:240]}, ensure_ascii=True, sort_keys=True)
        return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:16]

    def _should_obs_extract(
        self,
        *,
        state: AgentState,
        flags: Dict[str, Any],
        text_ir: Dict[str, Any],
        url: str,
    ) -> bool:
        mode = self._obs_extract_mode()
        if mode == "off":
            return False
        dom_hash = str(flags.get("dom_hash") or "")[:64]
        signature = self._obs_extract_signature(dom_hash=dom_hash, url=url)
        cached_sig = self._obs_extract_signature(dom_hash=state.memory.obs_extract_dom_hash, url=url)
        if mode != "always" and signature and cached_sig and signature == cached_sig and state.memory.obs_extract_payload:
            return False
        if mode == "always":
            return True
        if bool(flags.get("url_changed")) or bool(flags.get("dom_changed")):
            return True
        if int(state.progress.no_progress_score or 0) >= 5:
            return True
        forms = text_ir.get("forms") if isinstance(text_ir.get("forms"), list) else []
        cards = text_ir.get("cards") if isinstance(text_ir.get("cards"), list) else []
        return not forms and not cards

    def _obs_extract_messages(
        self,
        *,
        prompt: str,
        url: str,
        text_ir: Dict[str, Any],
        candidates: List[Candidate],
    ) -> List[Dict[str, Any]]:
        candidate_lines: List[str] = []
        for cand in candidates[:40]:
            label = _candidate_text(cand.text, cand.field_hint, cand.href, cand.group_label, cand.region_label)
            candidate_lines.append(
                f"[{cand.id}] role={cand.role} kind={cand.field_kind or cand.region_kind or cand.type} label={label[:140]} context={cand.context[:180]}"
            )
        user_payload = {
            "task": str(prompt or "")[:1200],
            "url": str(url or "")[:400],
            "title": str(text_ir.get("title") or "")[:240],
            "headings": list(text_ir.get("headings") or [])[:10],
            "visible_text": str(text_ir.get("visible_text") or "")[:5000],
            "html_excerpt": str(text_ir.get("html_excerpt") or "")[:6000],
            "candidates": candidate_lines[:40],
        }
        system = (
            "You extract structured browser observations from HTML for a separate policy model. "
            "Return strict JSON only with keys: "
            "page_kind, summary, regions, forms, facts, primary_candidate_ids. "
            "regions must be a list of objects with kind, label, candidate_ids. "
            "forms must be a list of objects with label, fields, commit_ids. "
            "facts must be short strings for visible metrics, totals, or key-value facts already present on the page. "
            "primary_candidate_ids should identify the most likely commit/apply/save/search/open targets visible right now. "
            "Do not choose the next action. Do not narrate. Be concise and structural."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

    def _safe_json_load(self, raw: Any) -> Dict[str, Any]:
        text = str(raw or "").strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except Exception:
            match = re.search(r"\{.*\}", text, flags=re.S)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    return {}
        return {}

    def _normalize_obs_extract_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        normalized = {
            "page_kind": str(payload.get("page_kind") or "")[:80],
            "summary": str(payload.get("summary") or "")[:600],
            "regions": [],
            "forms": [],
            "facts": [],
            "primary_candidate_ids": [],
        }
        for item in list(payload.get("regions") or [])[:8]:
            if not isinstance(item, dict):
                continue
            normalized["regions"].append(
                {
                    "kind": str(item.get("kind") or "")[:40],
                    "label": str(item.get("label") or "")[:160],
                    "candidate_ids": [str(x)[:120] for x in list(item.get("candidate_ids") or [])[:8] if str(x).strip()],
                }
            )
        for item in list(payload.get("forms") or [])[:8]:
            if not isinstance(item, dict):
                continue
            normalized["forms"].append(
                {
                    "label": str(item.get("label") or "")[:160],
                    "fields": [str(x)[:80] for x in list(item.get("fields") or [])[:8] if str(x).strip()],
                    "commit_ids": [str(x)[:120] for x in list(item.get("commit_ids") or [])[:6] if str(x).strip()],
                }
            )
        normalized["facts"] = [str(x)[:180] for x in list(payload.get("facts") or [])[:8] if str(x).strip()]
        normalized["primary_candidate_ids"] = [str(x)[:120] for x in list(payload.get("primary_candidate_ids") or [])[:12] if str(x).strip()]
        if not (
            normalized["summary"]
            or normalized["regions"]
            or normalized["forms"]
            or normalized["facts"]
            or normalized["primary_candidate_ids"]
        ):
            return {}
        return normalized

    def _maybe_extract_observation(
        self,
        *,
        task_id: str,
        prompt: str,
        url: str,
        flags: Dict[str, Any],
        state: AgentState,
        text_ir: Dict[str, Any],
        candidates: List[Candidate],
    ) -> Dict[str, Any]:
        if not self._should_obs_extract(state=state, flags=flags, text_ir=text_ir, url=url):
            return state.memory.obs_extract_payload if isinstance(state.memory.obs_extract_payload, dict) else {}
        model_name = _env_str("OPENAI_OBS_MODEL", "gpt-4o-mini") or "gpt-4o-mini"
        messages = self._obs_extract_messages(prompt=prompt, url=url, text_ir=text_ir, candidates=candidates)
        try:
            raw = self.obs_extract_call(
                task_id=task_id,
                messages=messages,
                model=model_name,
                temperature=0.0,
                max_tokens=int(os.getenv("FSM_OBS_MAX_TOKENS", "450")),
            )
        except Exception:
            return {}
        content = ""
        try:
            content = str(((raw.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
        except Exception:
            content = ""
        parsed = self._normalize_obs_extract_payload(self._safe_json_load(content))
        if not parsed:
            return {}
        state.memory.obs_extract_dom_hash = str(flags.get("dom_hash") or "")[:64]
        clean_payload = dict(parsed)
        state.memory.obs_extract_payload = clean_payload
        primary_ids = [str(x)[:120] for x in list(parsed.get("primary_candidate_ids") or [])[:12] if str(x).strip()]
        state.memory.obs_candidate_hints = _dedupe_keep_order(primary_ids, MAX_VISUAL_HINTS)
        clean_payload["__model"] = str(raw.get("model") or model_name)
        clean_payload["__usage"] = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
        return clean_payload

    def _debug_log(self, task_id: str, payload: Dict[str, Any]) -> None:
        if not self.debug_dir and not self.trace_json:
            return
        try:
            base = Path(self.trace_dir if self.trace_json else self.debug_dir).expanduser().resolve()
            safe_task = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(task_id or "task"))[:120] or "task"
            event = {"ts": _utc_now(), **payload}
            _append_jsonl(base / f"{safe_task}.fsm.jsonl", event)
        except Exception:
            return

    def _state_delta(self, before: AgentState, after: AgentState) -> Dict[str, Any]:
        return {
            "mode": {"from": before.mode, "to": after.mode},
            "stall_count": {
                "from": int(before.counters.stall_count or 0),
                "to": int(after.counters.stall_count or 0),
            },
            "repeat_action_count": {
                "from": int(before.counters.repeat_action_count or 0),
                "to": int(after.counters.repeat_action_count or 0),
            },
            "facts_count": {"from": len(before.memory.facts), "to": len(after.memory.facts)},
            "frontier_urls_count": {
                "from": len(before.frontier.pending_urls),
                "to": len(after.frontier.pending_urls),
            },
            "frontier_elements_count": {
                "from": len(before.frontier.pending_elements),
                "to": len(after.frontier.pending_elements),
            },
            "blocklist_count": {
                "from": len(before.blocklist.element_ids),
                "to": len(after.blocklist.element_ids),
            },
        }

    def _active_region_debug(self, *, state: AgentState) -> Dict[str, Any]:
        return {
            "region_id": str(state.focus_region.region_id or "")[:120],
            "region_kind": str(state.focus_region.region_kind or "")[:40],
            "region_label": str(state.focus_region.region_label or "")[:160],
            "candidate_count": len(state.focus_region.candidate_ids),
        }

    def _last_effect_label(
        self,
        *,
        history: List[Dict[str, Any]],
        flags: Dict[str, Any],
        done: bool = False,
    ) -> str:
        if done:
            return "COMPLETED"
        last = history[-1] if history else {}
        if isinstance(last, dict):
            if not bool(last.get("exec_ok", True)) or bool(_candidate_text(last.get("error"))):
                return "BLOCKED"
        if bool(flags.get("url_changed")):
            return "ADVANCED"
        if bool(flags.get("dom_changed")):
            return "LOCAL_PROGRESS"
        if bool(flags.get("no_visual_progress")):
            return "NO_VISIBLE_CHANGE"
        return "UNKNOWN"

    def _predict_expected_effect(
        self,
        *,
        action: Dict[str, Any],
        candidate: Candidate | None,
    ) -> str:
        action_type = str(action.get("type") or "").strip()
        if action_type in {"TypeAction", "FillAction"}:
            return "field_filled"
        if action_type == "SelectDropDownOptionAction":
            return "selection_changed"
        if action_type == "NavigateAction":
            return "navigation"
        if action_type == "ClickAction":
            blob = " ".join(
                [
                    str((candidate.text if candidate is not None else "") or ""),
                    str((candidate.field_hint if candidate is not None else "") or ""),
                    str((candidate.group_label if candidate is not None else "") or ""),
                    str((candidate.context if candidate is not None else "") or ""),
                    str((candidate.field_kind if candidate is not None else "") or ""),
                ]
            ).lower()
            if candidate is not None and (candidate.role == "link" or candidate.href):
                return "navigation"
            if re.search(r"\b(save|submit|apply|continue|confirm|search|done|finish)\b", blob):
                return "submit_effect"
            if re.search(r"\b(close|cancel|back)\b", blob):
                return "region_change"
            return "ui_change"
        return ""

    def _expected_effect_met(
        self,
        *,
        expected_effect: str,
        flags: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> bool:
        if not expected_effect:
            return False
        last = history[-1] if history and isinstance(history[-1], dict) else {}
        if isinstance(last, dict) and ((not bool(last.get("exec_ok", True))) or bool(_candidate_text(last.get("error")))):
            return False
        if expected_effect == "navigation":
            return bool(flags.get("url_changed"))
        if expected_effect in {"submit_effect", "region_change", "ui_change"}:
            return bool(flags.get("url_changed")) or bool(flags.get("dom_changed")) or not bool(flags.get("no_visual_progress"))
        if expected_effect == "field_filled":
            return not bool(flags.get("no_visual_progress"))
        if expected_effect == "selection_changed":
            return bool(flags.get("dom_changed")) or not bool(flags.get("no_visual_progress"))
        return False

    def _no_progress_score(self, *, state: AgentState, flags: Dict[str, Any], history: List[Dict[str, Any]]) -> int:
        score = 0
        score += max(0, int(state.counters.stall_count or 0))
        score += max(0, int(state.counters.repeat_action_count or 0))
        if bool(flags.get("no_visual_progress")):
            score += 2
        if str(flags.get("loop_level") or "none") == "high":
            score += 3
        recent_effects = state.progress.recent_effects[-3:]
        if recent_effects:
            if all(str(effect.label or "") in {"NO_VISIBLE_CHANGE", "BLOCKED"} for effect in recent_effects):
                score += 3
            if len({str(effect.target_id or "") for effect in recent_effects if str(effect.target_id or "")}) == 1:
                score += 2
            if all(bool(effect.expected_effect) and not bool(effect.expected_effect_met) for effect in recent_effects):
                score += 2
        recent_failures = 0
        for item in history[-3:]:
            if isinstance(item, dict) and ((not bool(item.get("exec_ok", True))) or bool(_candidate_text(item.get("error")))):
                recent_failures += 1
        score += recent_failures * 2
        return min(score, 12)

    def _record_progress_effect(
        self,
        *,
        step_index: int,
        history: List[Dict[str, Any]],
        state: AgentState,
        flags: Dict[str, Any],
    ) -> None:
        if not history:
            state.progress.last_effect = ""
            state.progress.no_progress_score = self._no_progress_score(state=state, flags=flags, history=history)
            return
        last = history[-1]
        if not isinstance(last, dict):
            return
        action = last.get("action") if isinstance(last.get("action"), dict) else {}
        label = self._last_effect_label(history=history, flags=flags, done=False)
        target_id = str(action.get("_element_id") or action.get("element_id") or "")[:120]
        region_id = str(state.focus_region.region_id or state.form_progress.active_group_id or "")[:120]
        error = str(last.get("error") or "")[:220]
        expected_effect = str(state.progress.pending_expected_effect or "")[:40]
        expected_effect_met = self._expected_effect_met(
            expected_effect=expected_effect,
            flags=flags,
            history=history,
        ) if expected_effect else False
        prev_effect = state.progress.recent_effects[-1] if state.progress.recent_effects else None
        prev_region_id = str(prev_effect.region_id or "") if isinstance(prev_effect, ProgressEffect) else ""
        prev_target_id = str(prev_effect.target_id or "") if isinstance(prev_effect, ProgressEffect) else ""
        effect = ProgressEffect(
            step_index=max(0, int(step_index) - 1),
            label=label,
            action_type=str(action.get("type") or "")[:80],
            target_id=target_id,
            region_id=region_id,
            expected_effect=expected_effect,
            expected_effect_met=expected_effect_met,
            region_changed=bool(prev_region_id and region_id and prev_region_id != region_id),
            repeated_target=bool(prev_target_id and target_id and prev_target_id == target_id),
            url_changed=bool(flags.get("url_changed")),
            dom_changed=bool(flags.get("dom_changed")),
            exec_ok=bool(last.get("exec_ok", last.get("success", True))),
            error=error,
        )
        state.progress.recent_effects = (state.progress.recent_effects + [effect])[-16:]
        state.progress.pending_expected_effect = ""
        state.progress.pending_expected_target_id = ""
        state.progress.pending_expected_region_id = ""
        state.progress.pending_expected_action_type = ""
        state.progress.last_effect = label
        state.progress.no_progress_score = self._no_progress_score(state=state, flags=flags, history=history)
        if label in {"NO_VISIBLE_CHANGE", "BLOCKED"} or (expected_effect and not expected_effect_met):
            state.progress.consecutive_no_effect_steps = int(state.progress.consecutive_no_effect_steps or 0) + 1
        else:
            state.progress.consecutive_no_effect_steps = 0
        if region_id:
            state.progress.region_attempts[region_id] = int(state.progress.region_attempts.get(region_id) or 0) + 1
            if label in {"NO_VISIBLE_CHANGE", "BLOCKED"} and int(state.progress.region_attempts.get(region_id) or 0) >= 2:
                state.progress.blocked_regions = _dedupe_keep_order(
                    state.progress.blocked_regions + [region_id],
                    MAX_PENDING_ELEMENTS,
                )
        pattern = ""
        action_type = str(action.get("type") or "").strip()
        if label in {"ADVANCED", "LOCAL_PROGRESS"} and action_type in {"TypeAction", "SelectDropDownOptionAction", "ClickAction"}:
            pattern = f"{action_type.lower()}_progress"
            state.progress.successful_patterns = _dedupe_keep_order(
                state.progress.successful_patterns + [pattern],
                32,
            )
        elif (label in {"NO_VISIBLE_CHANGE", "BLOCKED"} or (expected_effect and not expected_effect_met)) and action_type:
            pattern = f"{action_type.lower()}_no_effect"
            if "intercepts pointer events" in error.lower():
                pattern = "popup_intercept"
            elif expected_effect and not expected_effect_met:
                pattern = f"{action_type.lower()}_expected_{expected_effect}_miss"
            state.progress.failed_patterns = _dedupe_keep_order(
                state.progress.failed_patterns + [pattern],
                32,
            )

    def _store_expected_effect(
        self,
        *,
        action: Dict[str, Any],
        ranked_candidates: List[Candidate],
        state: AgentState,
    ) -> None:
        candidate = self._candidate_for_action(action=action, ranked_candidates=ranked_candidates)
        expected_effect = self._predict_expected_effect(action=action, candidate=candidate)
        state.progress.pending_expected_effect = expected_effect[:40]
        state.progress.pending_expected_action_type = str(action.get("type") or "")[:80]
        state.progress.pending_expected_target_id = str(action.get("_element_id") or action.get("element_id") or "")[:120]
        state.progress.pending_expected_region_id = str(
            (candidate.region_id if candidate is not None else "") or state.focus_region.region_id or ""
        )[:120]

    def _apply_stagnation_policy(self, *, state: AgentState, flags: Dict[str, Any]) -> None:
        no_progress_score = int(state.progress.no_progress_score or 0)
        region_id = str(state.focus_region.region_id or "").strip()
        if region_id and (
            no_progress_score >= 6
            or int(state.progress.consecutive_no_effect_steps or 0) >= 2
            or region_id in set(state.progress.blocked_regions)
        ):
            state.focus_region.recent_region_ids = _dedupe_keep_order(
                state.focus_region.recent_region_ids + [region_id],
                MAX_PENDING_ELEMENTS,
            )
            state.focus_region.region_id = ""
            state.focus_region.region_kind = ""
            state.focus_region.region_label = ""
            state.focus_region.region_context = ""
            state.focus_region.candidate_ids = []
            state.memory.checkpoints = _dedupe_keep_order(
                state.memory.checkpoints + [f"release_region:{region_id}"],
                MAX_CHECKPOINTS,
            )
        if no_progress_score >= 7 and state.mode in {"NAV", "EXTRACT"}:
            state.mode = "PLAN"

    def _maybe_promote_to_plan_from_capability_gap(
        self,
        *,
        state: AgentState,
        policy_obs: Dict[str, Any],
        route_reason: str,
    ) -> str:
        page_obs = policy_obs.get("page_observations") if isinstance(policy_obs.get("page_observations"), dict) else {}
        capability_gap = page_obs.get("capability_gap") if isinstance(page_obs.get("capability_gap"), dict) else {}
        if state.mode not in {"BOOTSTRAP", "NAV", "EXTRACT"}:
            return route_reason
        if not bool(capability_gap.get("read_only_for_task")):
            return route_reason
        if not str(capability_gap.get("preferred_transition") or "").strip():
            return route_reason
        state.mode = "PLAN"
        return "capability_gap_model_replan"

    def _vision_mode(self) -> str:
        mode = _env_str("FSM_VISION_MODE", "auto").lower()
        if mode not in {"off", "auto", "always"}:
            mode = "auto"
        return mode

    def _default_vision_question(self, *, prompt: str, state: AgentState, text_ir: Dict[str, Any]) -> str:
        control_groups = text_ir.get("control_groups") if isinstance(text_ir.get("control_groups"), list) else []
        cards = text_ir.get("cards") if isinstance(text_ir.get("cards"), list) else []
        last_action_type = str(state.last_action_sig or "").split("|", 1)[0].strip().lower()
        active_group_label = str(state.form_progress.active_group_label or "").strip()
        if active_group_label and last_action_type == "selectdropdownoptionaction":
            return (
                "Focus on the currently visible control group "
                f"'{active_group_label[:120]}'. "
                "After the current selection, which visible candidate ids are the best follow-up controls? "
                "Prefer apply/search/submit or the next related control, and avoid repeating the same select."
            )
        if control_groups and cards:
            return (
                "Given the screenshot and task, which visible candidate ids are the best next targets? "
                "Prefer current-page controls or apply/submit actions over unrelated cards unless the target item is clearly identified."
            )
        if control_groups:
            return (
                "Given the screenshot and task, which visible controls or buttons best advance the task on the current page?"
            )
        if cards:
            return (
                "Given the screenshot and task, which visible candidate ids correspond to the most relevant item or item action?"
            )
        if state.mode in {"PLAN", "STUCK"}:
            return "Describe the visible UI and identify the best visible next target from the candidate list."
        return "Which visible candidate ids best match the next useful action for this task?"

    def _should_auto_vision(
        self,
        *,
        screenshot: Any,
        state: AgentState,
        flags: Dict[str, Any],
        text_ir: Dict[str, Any],
        question: str,
        url: str,
    ) -> bool:
        skills, _ = self._ensure_legacy_support()
        if skills.vision_call is None:
            return False
        mode = self._vision_mode()
        if mode == "off":
            return False
        signature = _vision_signature(screenshot=screenshot, question=question, url=url)
        if not signature:
            return False
        if signature == str(state.memory.last_vision_signature or ""):
            return False
        if mode == "always":
            return True
        if state.mode in {"PLAN", "STUCK"}:
            return True
        last_action_type = str(state.last_action_sig or "").split("|", 1)[0].strip().lower()
        if (
            last_action_type == "selectdropdownoptionaction"
            and bool(state.form_progress.active_group_candidate_ids)
            and (bool(flags.get("no_visual_progress")) or int(state.counters.repeat_action_count or 0) >= 1)
        ):
            return True
        if str(flags.get("loop_level") or "none") == "high":
            return True
        if int(state.counters.stall_count or 0) >= 2 or int(state.counters.repeat_action_count or 0) >= 1:
            return True
        return False

    def _pre_done_verification(
        self,
        *,
        step_index: int,
        state: AgentState,
        prompt: str,
        text_ir: Dict[str, Any],
        content: str,
        flags: Dict[str, Any],
    ) -> tuple[bool, str]:
        text = _candidate_text(content)
        if not text:
            return False, "empty_final_content"
        informational_task = _looks_like_informational_task(prompt)
        best_page_evidence = _best_page_evidence(prompt, text_ir) if informational_task else ""
        local_page_evidence_ok = bool(best_page_evidence) and _content_supported_by_page_evidence(prompt, text, text_ir)
        if informational_task and _looks_like_vague_informational_answer(text):
            return False, "vague_informational_answer"
        lowered = text.lower()
        generic = {
            "task complete",
            "task completed",
            "done",
            "completed",
            "no safe browser action available",
        }
        if state.mode == "REPORT" and len(state.memory.facts) >= 1 and lowered in (generic | {"task completed."}):
            return True, "report_resume_default_content"
        if lowered in generic:
            return False, "generic_final_content"
        if informational_task and state.mode != "REPORT" and not local_page_evidence_ok:
            return False, "informational_answer_not_supported_by_local_evidence"
        if int(step_index) <= 0 and len(state.memory.facts) < 1 and not local_page_evidence_ok:
            return False, "too_early_without_facts"
        min_facts = 1 if state.mode == "REPORT" else 2
        if len(state.memory.facts) < min_facts and not local_page_evidence_ok:
            return False, "insufficient_evidence_for_final"
        if bool(flags.get("captcha_suspected")):
            return False, "captcha_unresolved"
        if informational_task and best_page_evidence and not local_page_evidence_ok and len(state.memory.facts) < min_facts:
            return False, "content_not_supported_by_page_evidence"
        return True, "ok"

    def _run_direct_loop(
        self,
        *,
        task_id: str,
        prompt: str,
        url: str,
        step_index: int,
        state: AgentState,
        text_ir: Dict[str, Any],
        candidates: List[Candidate],
        ranked: List[Candidate],
        policy_obs: Dict[str, Any],
        browser_allowed: set[str],
        model_name: str,
        usage_payload: Dict[str, Any],
        policy_model_used: str,
    ) -> tuple[Dict[str, Any] | None, bool, str, str, str]:
        chosen_action: Dict[str, Any] | None = None
        done = False
        content = ""
        policy_reasoning = ""
        direct_browser_allowed = {tool for tool in browser_allowed if tool != "browser.back"}
        decision, usage = self.policy.decide(
            task_id=task_id or "task",
            prompt=prompt,
            mode="DIRECT",
            policy_obs=policy_obs,
            allowed_tools=direct_browser_allowed,
            model_name=model_name,
            plan_model_name=model_name,
        )
        if usage:
            usage_payload["usage"] = _merge_usage_dicts(
                usage_payload.get("usage") if isinstance(usage_payload.get("usage"), dict) else {},
                usage.get("usage") if isinstance(usage.get("usage"), dict) else {},
            )
            if usage.get("model"):
                policy_model_used = str(usage.get("model"))
        policy_reasoning = _candidate_text(decision.get("reasoning"), policy_reasoning)
        dtype = str(decision.get("type") or "").strip().lower()
        if dtype == "final":
            final_content = _candidate_text(decision.get("content"), "")
            lowered = final_content.lower()
            generic = {
                "",
                "task complete",
                "task completed",
                "done",
                "completed",
                "no safe browser action available",
            }
            is_generic_final = lowered in generic or lowered.startswith("no safe browser action available")
            if final_content and not is_generic_final and not _looks_like_vague_informational_answer(final_content):
                done = True
                content = final_content
                state.mode = "DONE"
        elif dtype == "browser":
            ranked_ids = {cand.id for cand in ranked if cand.id}
            action_candidates = list(ranked)
            action_candidates.extend(cand for cand in candidates if cand.id not in ranked_ids)
            chosen_action = self._browser_action_from_tool_call(
                tool_call=decision.get("tool_call") if isinstance(decision.get("tool_call"), dict) else {},
                ranked_candidates=action_candidates,
                state=state,
                prompt=prompt,
                allowed=direct_browser_allowed,
                current_url=url,
            )
        return chosen_action, done, content, policy_reasoning, policy_model_used

    def _run_legacy_loop(
        self,
        *,
        task_id: str,
        prompt: str,
        url: str,
        step_index: int,
        history: List[Dict[str, Any]],
        screenshot: Any,
        state: AgentState,
        flags: Dict[str, Any],
        text_ir: Dict[str, Any],
        candidates: List[Candidate],
        ranked: List[Candidate],
        policy_obs: Dict[str, Any],
        allowed: set[str],
        model_name: str,
        plan_model_name: str,
        usage_payload: Dict[str, Any],
        policy_model_used: str,
        route_reason: str,
    ) -> tuple[Dict[str, Any] | None, bool, str, str, str, List[str]]:
        chosen_action: Dict[str, Any] | None = None
        done = False
        content = ""
        policy_reasoning = ""
        meta_exec_trace: List[str] = []
        step_vision_signatures: set[str] = set()

        pre_action, pre_done, pre_content, pre_note = self._deterministic_pre_action(
            prompt=prompt,
            url=url,
            step_index=step_index,
            history=history,
            state=state,
            flags=flags,
            ranked_candidates=ranked,
            allowed=allowed,
        )
        if pre_note:
            meta_exec_trace.append(f"PRE:{pre_note}")
            self._debug_log(
                task_id or "task",
                {
                    "phase": "deterministic_pre_action",
                    "note": pre_note,
                    "action": pre_action,
                    "pre_done": bool(pre_done),
                },
            )
        if pre_done:
            if pre_note == "wait_only_complete":
                ok, reason = True, "ok"
            else:
                ok, reason = self._pre_done_verification(
                    step_index=step_index,
                    state=state,
                    prompt=prompt,
                    text_ir=text_ir,
                    content=pre_content,
                    flags=flags,
                )
            if ok:
                done = True
                content = pre_content
                state.mode = "DONE"
            else:
                meta_exec_trace.append(f"BLOCK_DONE:{reason}")
        elif pre_action is not None:
            chosen_action = pre_action

        if not done and chosen_action is None:
            auto_page_answer = ""
            if (
                int(step_index) >= 1
                and _looks_like_informational_task(prompt)
                and not bool(flags.get("captcha_suspected"))
                and _runtime_page_evidence_ready(prompt, url, text_ir, step_index=step_index)
            ):
                auto_page_answer = _best_page_evidence(prompt, text_ir)
            if auto_page_answer:
                ok, reason = self._pre_done_verification(
                    step_index=step_index,
                    state=state,
                    prompt=prompt,
                    text_ir=text_ir,
                    content=auto_page_answer,
                    flags=flags,
                )
                if ok:
                    done = True
                    content = auto_page_answer
                    state.mode = "DONE"
                    meta_exec_trace.append("PRE:page_evidence_final")
                else:
                    meta_exec_trace.append(f"BLOCK_DONE:{reason}")

        if not done and chosen_action is None:
            auto_vision_question = self._default_vision_question(prompt=prompt, state=state, text_ir=text_ir)
            if self._should_auto_vision(
                screenshot=screenshot,
                state=state,
                flags=flags,
                text_ir=text_ir,
                question=auto_vision_question,
                url=url,
            ):
                _, meta = self._ensure_legacy_support()
                vision_result = meta.execute(
                    task_id=task_id,
                    tool_name="META.VISION_QA",
                    args={"question": auto_vision_question},
                    state=state,
                    prompt=prompt,
                    text_ir=text_ir,
                    candidates=ranked,
                    url=url,
                    screenshot=screenshot,
                )
                meta_exec_trace.append("AUTO:META.VISION_QA")
                self._debug_log(
                    task_id,
                    {
                        "phase": "auto_vision",
                        "question": auto_vision_question,
                        "result": vision_result,
                    },
                )
                if isinstance(vision_result.get("usage"), dict):
                    usage_payload["usage"] = _merge_usage_dicts(
                        usage_payload.get("usage") if isinstance(usage_payload.get("usage"), dict) else {},
                        vision_result.get("usage") if isinstance(vision_result.get("usage"), dict) else {},
                    )
                    helper_model = str(vision_result.get("model") or "").strip()
                    if helper_model:
                        usage_payload["helper_models"] = _dedupe_keep_order(
                            list(usage_payload.get("helper_models") or []) + [helper_model],
                            8,
                        )
                ranked = self.ranker.rank(
                    task=prompt,
                    mode=state.mode,
                    flags=flags,
                    candidates=candidates,
                    state=state,
                    current_url=url,
                    top_k=120,
                )
                policy_obs = self.obs_builder.build_policy_obs(
                    task_id=task_id,
                    prompt=prompt,
                    step_index=step_index,
                    url=url,
                    mode=state.mode,
                    flags=flags,
                    state=state,
                    text_ir=text_ir,
                    candidates=ranked,
                    history=history,
                    screenshot_available=bool(_normalize_screenshot_data_url(screenshot)),
                )
                route_reason = self._maybe_promote_to_plan_from_capability_gap(
                    state=state,
                    policy_obs=policy_obs,
                    route_reason=route_reason,
                )

        if not done and chosen_action is None:
            for _ in range(MAX_INTERNAL_META_STEPS + 1):
                decision, usage = self.policy.decide(
                    task_id=task_id or "task",
                    prompt=prompt,
                    mode=state.mode,
                    policy_obs=policy_obs,
                    allowed_tools=allowed,
                    model_name=model_name,
                    plan_model_name=plan_model_name,
                )
                if usage:
                    usage_payload["usage"] = _merge_usage_dicts(
                        usage_payload.get("usage") if isinstance(usage_payload.get("usage"), dict) else {},
                        usage.get("usage") if isinstance(usage.get("usage"), dict) else {},
                    )
                    if usage.get("model"):
                        policy_model_used = str(usage.get("model"))
                policy_reasoning = _candidate_text(decision.get("reasoning"), policy_reasoning)
                dtype = str(decision.get("type") or "").strip().lower()
                if dtype == "final":
                    final_content = _candidate_text(decision.get("content"), "Task completed.")
                    can_early_finish = bool(state.memory.facts) and state.mode in {"REPORT", "DONE"}
                    ok, done_reason = self._pre_done_verification(
                        step_index=step_index,
                        state=state,
                        prompt=prompt,
                        text_ir=text_ir,
                        content=final_content,
                        flags=flags,
                    )
                    if int(step_index) == 0 and not history and not can_early_finish:
                        meta_exec_trace.append("BLOCK_EARLY_FINAL")
                    elif not ok:
                        meta_exec_trace.append(f"BLOCK_DONE:{done_reason}")
                        best_page_evidence = _best_page_evidence(prompt, text_ir)
                        if (
                            _looks_like_informational_task(prompt)
                            and best_page_evidence
                            and _runtime_page_evidence_ready(prompt, url, text_ir, step_index=step_index)
                        ):
                            ok2, reason2 = self._pre_done_verification(
                                step_index=step_index,
                                state=state,
                                prompt=prompt,
                                text_ir=text_ir,
                                content=best_page_evidence,
                                flags=flags,
                            )
                            if ok2:
                                done = True
                                content = best_page_evidence
                                state.mode = "DONE"
                                meta_exec_trace.append("RESCUE_DONE:page_evidence")
                                break
                            meta_exec_trace.append(f"BLOCK_DONE:{reason2}")
                        if best_page_evidence:
                            decision = {"type": "meta", "name": "META.EXTRACT_FACTS", "arguments": {"schema": "generic"}}
                            dtype = "meta"
                        elif state.mode in {"NAV", "PLAN"} and not _looks_like_informational_task(prompt):
                            decision = {"type": "meta", "name": "META.EXTRACT_LINKS", "arguments": {"kind": "all_links"}}
                            dtype = "meta"
                        elif len(state.memory.facts) < 2:
                            decision = {"type": "meta", "name": "META.EXTRACT_FACTS", "arguments": {"schema": "generic"}}
                            dtype = "meta"
                        else:
                            decision = {"type": "meta", "name": "META.SEARCH_TEXT", "arguments": {"query": prompt[:80]}}
                            dtype = "meta"
                    else:
                        done = True
                        content = final_content
                        state.mode = "DONE"
                        break
                if dtype == "browser":
                    chosen_action = self._browser_action_from_tool_call(
                        tool_call=decision.get("tool_call") if isinstance(decision.get("tool_call"), dict) else {},
                        ranked_candidates=ranked,
                        state=state,
                        prompt=prompt,
                        allowed=allowed,
                        current_url=url,
                    )
                    if chosen_action is not None:
                        break
                if dtype == "meta":
                    if state.counters.meta_steps_used >= MAX_INTERNAL_META_STEPS:
                        break
                    meta_name = str(decision.get("name") or "")
                    meta_args = decision.get("arguments") if isinstance(decision.get("arguments"), dict) else {}
                    if meta_name == "META.VISION_QA":
                        vision_question = str(
                            meta_args.get("question") or self._default_vision_question(prompt=prompt, state=state, text_ir=text_ir)
                        )
                        if not self._should_auto_vision(
                            screenshot=screenshot,
                            state=state,
                            flags=flags,
                            text_ir=text_ir,
                            question=vision_question,
                            url=url,
                        ):
                            meta_exec_trace.append("BLOCK:META.VISION_QA")
                            break
                        vision_sig = _vision_signature(
                            screenshot=screenshot,
                            question=vision_question,
                            url=url,
                        )
                        if vision_sig and vision_sig in step_vision_signatures:
                            meta_exec_trace.append("SKIP:META.VISION_QA_DUP")
                            break
                    _, meta = self._ensure_legacy_support()
                    result = meta.execute(
                        task_id=task_id,
                        tool_name=meta_name,
                        args=meta_args,
                        state=state,
                        prompt=prompt,
                        text_ir=text_ir,
                        candidates=ranked,
                        url=url,
                        screenshot=screenshot,
                    )
                    state.counters.meta_steps_used += 1
                    meta_exec_trace.append(meta_name)
                    if meta_name == "META.VISION_QA":
                        vision_sig = str(result.get("signature") or "")
                        if vision_sig:
                            step_vision_signatures.add(vision_sig)
                    self._debug_log(
                        task_id,
                        {
                            "phase": "meta_tool",
                            "name": meta_name,
                            "args": meta_args,
                            "result": result,
                            "meta_steps_used": int(state.counters.meta_steps_used or 0),
                        },
                    )
                    if isinstance(result.get("usage"), dict):
                        usage_payload["usage"] = _merge_usage_dicts(
                            usage_payload.get("usage") if isinstance(usage_payload.get("usage"), dict) else {},
                            result.get("usage") if isinstance(result.get("usage"), dict) else {},
                        )
                        helper_model = str(result.get("model") or "").strip()
                        if helper_model:
                            usage_payload["helper_models"] = _dedupe_keep_order(
                                list(usage_payload.get("helper_models") or []) + [helper_model],
                                8,
                            )
                    if meta_name == "META.SET_MODE":
                        route_reason = f"meta_set_mode:{state.mode}"
                    ranked = self.ranker.rank(
                        task=prompt,
                        mode=state.mode,
                        flags=flags,
                        candidates=candidates,
                        state=state,
                        current_url=url,
                        top_k=120,
                    )
                    policy_obs = self.obs_builder.build_policy_obs(
                        task_id=task_id,
                        prompt=prompt,
                        step_index=step_index,
                        url=url,
                        mode=state.mode,
                        flags=flags,
                        state=state,
                        text_ir=text_ir,
                        candidates=ranked,
                        history=history,
                        screenshot_available=bool(_normalize_screenshot_data_url(screenshot)),
                    )
                    route_reason = self._maybe_promote_to_plan_from_capability_gap(
                        state=state,
                        policy_obs=policy_obs,
                        route_reason=route_reason,
                    )
                    if not bool(result.get("ok")):
                        break
                    continue
                break

        if not done and chosen_action is None:
            chosen_action = self._fallback_from_ranked_candidates(
                prompt=prompt,
                ranked_candidates=ranked,
                state=state,
                allowed=allowed,
                current_url=url,
            )
            if chosen_action is not None:
                meta_exec_trace.append("FALLBACK:top_ranked_candidate")

        if not done and chosen_action is None:
            chosen_action, done, content, stuck_note = self._stuck_recovery(
                prompt=prompt,
                url=url,
                step_index=step_index,
                state=state,
                allowed=allowed,
            )
            if stuck_note:
                meta_exec_trace.append(f"EMERGENCY_RECOVERY:{stuck_note}")

        if not done and chosen_action is None:
            fallback = self.policy._fallback(prompt=prompt, mode=state.mode, policy_obs=policy_obs, allowed_tools=allowed)
            if str(fallback.get("type") or "") == "final":
                done = True
                content = _candidate_text(fallback.get("content"), "Task complete.")
                state.mode = "DONE"
            else:
                chosen_action = self._browser_action_from_tool_call(
                    tool_call=fallback.get("tool_call") if isinstance(fallback.get("tool_call"), dict) else {},
                    ranked_candidates=ranked,
                    state=state,
                    prompt=prompt,
                    allowed=allowed,
                    current_url=url,
                )

        return chosen_action, done, content, policy_reasoning, policy_model_used, meta_exec_trace

    def _finalize_chosen_action(
        self,
        *,
        done: bool,
        direct_loop: bool,
        chosen_action: Dict[str, Any] | None,
        prompt: str,
        history: List[Dict[str, Any]],
        ranked: List[Candidate],
        state: AgentState,
        step_index: int,
    ) -> tuple[List[Dict[str, Any]], str, Dict[str, Any] | None]:
        actions: List[Dict[str, Any]] = []
        browser_tool_name = ""
        if done or chosen_action is None:
            state.last_action_sig = ""
            state.last_action_element_id = ""
            return actions, browser_tool_name, chosen_action

        if direct_loop:
            # Keep the direct path close to: obs -> LLM -> browser action.
            # Do not rewrite the model's action beyond structural normalization that already
            # happened in _browser_action_from_tool_call.
            pass
        else:
            chosen_action = self._guard_delete_task_against_unrelated_form_edits(
                action=chosen_action,
                prompt=prompt,
                ranked_candidates=ranked,
                state=state,
            )
            chosen_action = self._guard_missing_group_inputs(
                action=chosen_action,
                prompt=prompt,
                history=history,
                ranked_candidates=ranked,
                state=state,
            )
            chosen_action = self._guard_submit_without_inputs(
                action=chosen_action,
                prompt=prompt,
                history=history,
                ranked_candidates=ranked,
                state=state,
            )
            chosen_action = self._guard_redundant_type_action(
                action=chosen_action,
                prompt=prompt,
                history=history,
                ranked_candidates=ranked,
                state=state,
            )
            chosen_action = self._guard_redundant_select_action(
                action=chosen_action,
                history=history,
                ranked_candidates=ranked,
                state=state,
            )
            chosen_action = self._promote_click_input_to_type(
                action=chosen_action,
                prompt=prompt,
                ranked_candidates=ranked,
                state=state,
            )
        actions = [chosen_action]
        browser_tool_name = str(self._tool_name_for_action(chosen_action))
        if not direct_loop:
            self._remember_form_progress_from_action(prompt=prompt, action=chosen_action, ranked_candidates=ranked, state=state)
            self._store_expected_effect(action=chosen_action, ranked_candidates=ranked, state=state)
        sig = self._action_signature(chosen_action)
        state.counters.repeat_action_count = state.counters.repeat_action_count + 1 if sig == state.last_action_sig else 0
        state.last_action_sig = sig
        state.last_action_element_id = str(chosen_action.get("_element_id") or "")[:120]
        if state.last_action_element_id and int(state.counters.repeat_action_count or 0) >= 2:
            state.blocklist.element_ids = _dedupe_keep_order(
                state.blocklist.element_ids + [state.last_action_element_id],
                MAX_PENDING_ELEMENTS,
            )
            state.blocklist.until_step = max(state.blocklist.until_step, int(step_index) + 2)
        return actions, browser_tool_name, chosen_action

    def _build_run_output(
        self,
        *,
        task_id: str,
        route_reason: str,
        mode_in: str,
        mode_out: str,
        direct_loop: bool,
        prompt: str,
        text_ir: Dict[str, Any],
        policy_obs: Dict[str, Any],
        state_before: AgentState,
        state: AgentState,
        meta_exec_trace: List[str],
        chosen_action: Dict[str, Any] | None,
        actions: List[Dict[str, Any]],
        done: bool,
        content: str,
        browser_tool_name: str,
        include_reasoning: bool,
        policy_reasoning: str,
        ranked: List[Candidate],
        flags: Dict[str, Any],
        history: List[Dict[str, Any]],
        usage_payload: Dict[str, Any],
        policy_model_used: str,
        model_name: str,
    ) -> Dict[str, Any]:
        final_content = _candidate_text(content)
        if done and not final_content:
            final_content = "Task completed."

        reasoning = None
        if include_reasoning:
            if direct_loop and policy_reasoning:
                reasoning = policy_reasoning[:600]
            else:
                best_fact = _best_page_evidence(prompt, text_ir)
                goal = _candidate_text(prompt) or "Complete the task."
                if done and final_content:
                    current_page = final_content
                    decision_text = f"Return final answer now: {final_content}"
                else:
                    current_page = best_fact or _candidate_text((text_ir.get("likely_answers") or [""])[0] if isinstance(text_ir.get("likely_answers"), list) else "") or "Current page answer not clear yet."
                if not done and chosen_action is not None:
                    decision_text = f"Take one browser action: {browser_tool_name or str(chosen_action.get('type') or 'action')}"
                elif not done:
                    decision_text = "No safe browser action selected."
                reasoning = (
                    f"Goal: {goal}. "
                    f"Current page: {current_page}. "
                    f"Decision: {decision_text}."
                )[:600]

        selected_candidate = None
        if chosen_action is not None:
            selected_candidate = self._candidate_for_action(action=chosen_action, ranked_candidates=ranked)
        last_effect = self._last_effect_label(history=history, flags=flags, done=done)
        no_progress_score = self._no_progress_score(state=state, flags=flags, history=history)
        out: Dict[str, Any] = {
            "protocol_version": "1.0",
            "actions": actions[:1],
            "done": bool(done),
            "content": final_content if done else None,
            "state_out": state.to_state_out(),
        }
        if isinstance(reasoning, str) and reasoning:
            out["reasoning"] = reasoning
        usage = usage_payload.get("usage") if isinstance(usage_payload.get("usage"), dict) else None
        if usage:
            out["usage"] = usage
            out["total_tokens"] = int(usage.get("total_tokens") or 0)
        out["model"] = str(policy_model_used or model_name)
        helper_models = [str(m).strip() for m in list(usage_payload.get("helper_models") or []) if str(m).strip()]
        if helper_models:
            out["helper_models"] = helper_models
        self._debug_log(
            task_id,
            {
                "phase": "run_end",
                "route_reason": route_reason,
                "mode_in": mode_in,
                "mode_out": mode_out,
                "meta_trace": meta_exec_trace,
                "browser_tool": browser_tool_name,
                "done": bool(done),
                "content": _candidate_text(content)[:260] if done else "",
                "state_delta": self._state_delta(state_before, state),
                "active_region": self._active_region_debug(state=state),
                "last_effect": last_effect,
                "no_progress_score": int(no_progress_score),
                "candidate_count_local": int((policy_obs.get("page_observations") if isinstance(policy_obs.get("page_observations"), dict) else {}).get("local_candidate_count") or 0),
                "candidate_count_global": int((policy_obs.get("page_observations") if isinstance(policy_obs.get("page_observations"), dict) else {}).get("global_candidate_count") or 0),
                "selected_candidate_id": str((selected_candidate.id if selected_candidate is not None else "") or "")[:120],
                "selected_region_id": str((selected_candidate.region_id if selected_candidate is not None else "") or "")[:120],
                "usage": usage_payload.get("usage") if isinstance(usage_payload.get("usage"), dict) else {},
            },
        )
        return out

    def _build_completion_only_output(
        self,
        *,
        task_id: str,
        prompt: str,
        url: str,
        step_index: int,
        state: AgentState,
        text_ir: Dict[str, Any],
        flags: Dict[str, Any],
        include_reasoning: bool,
        usage_payload: Dict[str, Any],
        policy_model_used: str,
    ) -> Dict[str, Any]:
        done, content, policy_reasoning = self._completion_only_result(
            prompt=prompt,
            url=url,
            step_index=step_index,
            state=state,
            text_ir=text_ir,
            flags=flags,
        )
        if done:
            state.mode = "DONE"
        out = {
            "protocol_version": "1.0",
            "done": bool(done),
            "content": content if done else None,
            "reasoning": policy_reasoning[:200] if include_reasoning and policy_reasoning else None,
            "actions": [],
            "state_out": state.to_state_out(),
            "usage": usage_payload.get("usage") if isinstance(usage_payload.get("usage"), dict) else None,
            "model": policy_model_used,
            "helper_models": usage_payload.get("helper_models") if isinstance(usage_payload.get("helper_models"), list) else [],
        }
        self._debug_log(
            task_id,
            {
                "phase": "completion_only_result",
                "done": bool(done),
                "content": _candidate_text(content),
                "reasoning": policy_reasoning,
            },
        )
        return out

    def _prepare_run_context(
        self,
        *,
        task_id: str,
        prompt: str,
        url: str,
        html: str,
        screenshot: Any,
        step_index: int,
        history: List[Dict[str, Any]],
        state: AgentState,
        allowed: set[str],
        model_override: str,
    ) -> Dict[str, Any]:
        direct_loop = _env_bool("FSM_DIRECT_LOOP", True)
        flags = self.flags.detect(snapshot_html=html, url=url, history=history, state=state)
        state.counters.stall_count = int(flags.get("stall_count_suggested") or 0)
        self._record_progress_effect(
            step_index=step_index,
            history=history,
            state=state,
            flags=flags,
        )
        if bool(flags.get("url_changed")) or bool(flags.get("dom_changed")):
            state.memory.visual_notes = []
            state.memory.visual_element_hints = []
            state.memory.last_vision_signature = ""
        if bool(flags.get("url_changed")):
            state.form_progress.active_group_id = ""
            state.form_progress.active_group_label = ""
            state.form_progress.active_group_context = ""
            state.form_progress.active_group_candidate_ids = []
            state.focus_region.region_id = ""
            state.focus_region.region_kind = ""
            state.focus_region.region_label = ""
            state.focus_region.region_context = ""
            state.focus_region.candidate_ids = []
        state.visited.urls = _dedupe_keep_order(state.visited.urls + ([url] if url else []), MAX_VISITED_URLS)
        if url:
            state.visited.page_hashes[url[:MAX_STR]] = str(flags.get("dom_hash") or "")[:64]

        if direct_loop:
            routed_mode = "DIRECT"
            route_reason = "direct_loop"
            state.mode = routed_mode
        else:
            routed_mode, route_reason = self.router.next_mode(step_index=step_index, state=state, flags=flags, prompt=prompt)
            state.mode = routed_mode
            mode_before_stagnation = state.mode
            self._apply_stagnation_policy(state=state, flags=flags)
            if state.mode == "PLAN" and mode_before_stagnation != "PLAN":
                route_reason = "stagnation_replan"
        self._debug_log(
            task_id,
            {
                "phase": "flags_router",
                "flags": flags,
                "route_reason": route_reason,
                "mode_routed": routed_mode,
                "last_effect": state.progress.last_effect,
                "no_progress_score": int(state.progress.no_progress_score or 0),
            },
        )

        text_ir = self.obs_builder.build_text_ir(html)
        candidates = self.extractor.extract(snapshot_html=html, url=url)
        obs_extract = self._maybe_extract_observation(
            task_id=task_id,
            prompt=prompt,
            url=url,
            flags=flags,
            state=state,
            text_ir=text_ir,
            candidates=candidates,
        )
        obs_extract_usage = obs_extract.get("__usage") if isinstance(obs_extract.get("__usage"), dict) else {}
        obs_extract_model = str(obs_extract.get("__model") or "").strip() if isinstance(obs_extract, dict) else ""
        usage_payload: Dict[str, Any] = {"helper_models": []}
        if obs_extract:
            text_ir = dict(text_ir)
            text_ir["llm_extract"] = {k: v for k, v in obs_extract.items() if not str(k).startswith("__")}
        ranked = self.ranker.rank(
            task=prompt,
            mode=state.mode,
            flags=flags,
            candidates=candidates,
            state=state,
            current_url=url,
            top_k=120,
        )
        policy_obs = self.obs_builder.build_policy_obs(
            task_id=task_id,
            prompt=prompt,
            step_index=step_index,
            url=url,
            mode=state.mode,
            flags=flags,
            state=state,
            text_ir=text_ir,
            candidates=ranked,
            history=history,
            screenshot_available=bool(_normalize_screenshot_data_url(screenshot)),
        )
        if not direct_loop:
            route_reason = self._maybe_promote_to_plan_from_capability_gap(
                state=state,
                policy_obs=policy_obs,
                route_reason=route_reason,
            )
            if state.mode == "PLAN" and str(policy_obs.get("mode") or "") != "PLAN":
                policy_obs = self.obs_builder.build_policy_obs(
                    task_id=task_id,
                    prompt=prompt,
                    step_index=step_index,
                    url=url,
                    mode=state.mode,
                    flags=flags,
                    state=state,
                    text_ir=text_ir,
                    candidates=ranked,
                    history=history,
                    screenshot_available=bool(_normalize_screenshot_data_url(screenshot)),
                )
        if obs_extract_usage:
            usage_payload["usage"] = _merge_usage_dicts(
                usage_payload.get("usage") if isinstance(usage_payload.get("usage"), dict) else {},
                obs_extract_usage,
            )
        if obs_extract_model:
            usage_payload["helper_models"] = _dedupe_keep_order(
                list(usage_payload.get("helper_models") or []) + [obs_extract_model],
                8,
            )
        default_model_name = _env_str("OPENAI_MODEL", "gpt-5.2") or "gpt-5.2"
        model_name = str(model_override or default_model_name)
        plan_model_name = str(model_override or _env_str("OPENAI_PLAN_MODEL", default_model_name) or default_model_name)
        policy_model_used = plan_model_name if state.mode in {"PLAN", "STUCK"} else model_name
        browser_allowed = {tool for tool in allowed if str(tool).startswith("browser.")}
        if not browser_allowed:
            browser_allowed = set(SUPPORTED_BROWSER_TOOL_NAMES)
        return {
            "flags": flags,
            "route_reason": route_reason,
            "text_ir": text_ir,
            "candidates": candidates,
            "ranked": ranked,
            "policy_obs": policy_obs,
            "usage_payload": usage_payload,
            "model_name": model_name,
            "plan_model_name": plan_model_name,
            "policy_model_used": policy_model_used,
            "direct_loop": direct_loop,
            "browser_allowed": browser_allowed,
        }

    def run(self, *, payload: Dict[str, Any], model_override: str = "") -> Dict[str, Any]:
        prompt = str(payload.get("prompt") or payload.get("task_prompt") or "")
        url = str(payload.get("url") or "")
        html = str(payload.get("snapshot_html") or "")
        screenshot = payload.get("screenshot")
        task_id = str(payload.get("task_id") or "")
        step_index = int(payload.get("step_index") or 0)
        completion_only = bool(payload.get("completion_only"))
        include_reasoning = bool(payload.get("include_reasoning"))
        history = payload.get("history") if isinstance(payload.get("history"), list) else []
        state = AgentState.from_state_in(payload.get("state_in"), prompt=prompt)
        state_before = state.model_copy(deep=True)
        allowed = self._normalize_allowed(payload.get("allowed_tools"))
        mode_in = state.mode
        if url and not state.session_query:
            state.session_query = _query_map(url)
        self._debug_log(
            task_id,
            {
                "phase": "run_start",
                "task_id": task_id,
                "step_index": int(step_index),
                "mode_in": mode_in,
                "url": _candidate_text(url),
                "active_region": self._active_region_debug(state=state),
                "allowed_tools": sorted(list(allowed)),
                "history_len": len(history),
            },
        )
        prepared = self._prepare_run_context(
            task_id=task_id,
            prompt=prompt,
            url=url,
            html=html,
            screenshot=screenshot,
            step_index=step_index,
            history=history,
            state=state,
            allowed=allowed,
            model_override=model_override,
        )
        flags = prepared["flags"]
        route_reason = prepared["route_reason"]
        text_ir = prepared["text_ir"]
        candidates = prepared["candidates"]
        ranked = prepared["ranked"]
        policy_obs = prepared["policy_obs"]
        usage_payload = prepared["usage_payload"]
        model_name = prepared["model_name"]
        plan_model_name = prepared["plan_model_name"]
        policy_model_used = prepared["policy_model_used"]
        direct_loop = prepared["direct_loop"]
        browser_allowed = prepared["browser_allowed"]
        meta_exec_trace: List[str] = []
        policy_reasoning = ""
        chosen_action: Dict[str, Any] | None = None
        done = False
        content = ""
        if completion_only:
            return self._build_completion_only_output(
                task_id=task_id,
                prompt=prompt,
                url=url,
                step_index=step_index,
                state=state,
                text_ir=text_ir,
                flags=flags,
                include_reasoning=include_reasoning,
                usage_payload=usage_payload,
                policy_model_used=policy_model_used,
            )
        if direct_loop:
            chosen_action, done, content, policy_reasoning, policy_model_used = self._run_direct_loop(
                task_id=task_id,
                prompt=prompt,
                url=url,
                step_index=step_index,
                state=state,
                text_ir=text_ir,
                candidates=candidates,
                ranked=ranked,
                policy_obs=policy_obs,
                browser_allowed=browser_allowed,
                model_name=model_name,
                usage_payload=usage_payload,
                policy_model_used=policy_model_used,
            )
        else:
            chosen_action, done, content, policy_reasoning, policy_model_used, meta_exec_trace = self._run_legacy_loop(
                task_id=task_id,
                prompt=prompt,
                url=url,
                step_index=step_index,
                history=history,
                screenshot=screenshot,
                state=state,
                flags=flags,
                text_ir=text_ir,
                candidates=candidates,
                ranked=ranked,
                policy_obs=policy_obs,
                allowed=allowed,
                model_name=model_name,
                plan_model_name=plan_model_name,
                usage_payload=usage_payload,
                policy_model_used=policy_model_used,
                route_reason=route_reason,
            )

        actions, browser_tool_name, chosen_action = self._finalize_chosen_action(
            done=done,
            direct_loop=direct_loop,
            chosen_action=chosen_action,
            prompt=prompt,
            history=history,
            ranked=ranked,
            state=state,
            step_index=step_index,
        )

        state.last_url = str(url or "")[:MAX_STR]
        state.last_dom_hash = str(flags.get("dom_hash") or "")[:64]
        state.blocklist.until_step = max(state.blocklist.until_step, int(step_index) + 1 if state.blocklist.element_ids else state.blocklist.until_step)
        if done:
            state.mode = "DONE"
        mode_out = state.mode
        return self._build_run_output(
            task_id=task_id,
            route_reason=route_reason,
            mode_in=mode_in,
            mode_out=mode_out,
            direct_loop=direct_loop,
            prompt=prompt,
            text_ir=text_ir,
            policy_obs=policy_obs,
            state_before=state_before,
            state=state,
            meta_exec_trace=meta_exec_trace,
            chosen_action=chosen_action,
            actions=actions,
            done=done,
            content=content,
            browser_tool_name=browser_tool_name,
            include_reasoning=include_reasoning,
            policy_reasoning=policy_reasoning,
            ranked=ranked,
            flags=flags,
            history=history,
            usage_payload=usage_payload,
            policy_model_used=policy_model_used,
            model_name=model_name,
        )

    def _deterministic_pre_action(
        self,
        *,
        prompt: str,
        url: str,
        step_index: int,
        history: List[Dict[str, Any]],
        state: AgentState,
        flags: Dict[str, Any],
        ranked_candidates: List[Candidate],
        allowed: set[str],
    ) -> tuple[Dict[str, Any] | None, bool, str, str]:
        def allow(name: str) -> bool:
            return (not allowed) or (name in allowed)

        last_action_type = str(state.last_action_sig or "").split("|", 1)[0].strip().lower()
        recent_errors = [
            str(item.get("error") or "").lower()
            for item in history[-4:]
            if isinstance(item, dict) and str(item.get("error") or "").strip()
        ]
        if bool(flags.get("cookie_banner")) or (bool(flags.get("modal_dialog")) and not bool(flags.get("interactive_modal_form"))):
            if allow("browser.send_keys") and any("intercepts pointer events" in err for err in recent_errors):
                return {"type": "SendKeysIWAAction", "keys": "Escape"}, False, "", "popup_escape"
            skills, _ = self._ensure_legacy_support()
            popup_result = skills.solve_popups(candidates=ranked_candidates)
            primary_id = str(popup_result.get("primary_element_id") or "").strip()
            if primary_id and allow("browser.click"):
                for cand in ranked_candidates:
                    if cand.id != primary_id:
                        continue
                    selector = _sanitize_selector(cand.selector)
                    if isinstance(selector, dict):
                        return {"type": "ClickAction", "selector": selector, "_element_id": cand.id}, False, "", "popup_click"
            if allow("browser.send_keys"):
                return {"type": "SendKeysIWAAction", "keys": "Escape"}, False, "", "popup_escape"

        recent_wait_only = True
        wait_steps = 0
        for item in history[-4:]:
            if not isinstance(item, dict):
                continue
            action = item.get("action") if isinstance(item.get("action"), dict) else {}
            action_type = str(action.get("type") or item.get("action") or "").strip().lower()
            if not action_type:
                continue
            if action_type == "waitaction":
                wait_steps += 1
                continue
            recent_wait_only = False
            break
        if (
            last_action_type == "waitaction"
            and int(step_index) >= 1
            and recent_wait_only
            and wait_steps >= 1
            and not _task_constraints(prompt)
        ):
            return None, True, "Task completed.", "wait_only_complete"

        # Give async side effects one short cycle to land before declaring stuck.
        if (
            bool(flags.get("no_visual_progress"))
            and last_action_type in {"clickaction", "typeaction", "selectdropdownoptionaction"}
            and allow("browser.wait")
            and int(state.counters.stall_count or 0) <= 2
        ):
            checkpoint = "post_action_async_wait"
            recent = state.memory.checkpoints[-1] if state.memory.checkpoints else ""
            if recent != checkpoint:
                state.memory.checkpoints = _dedupe_keep_order(
                    state.memory.checkpoints + [checkpoint],
                    MAX_CHECKPOINTS,
                )
                return {"type": "WaitAction", "time_seconds": 1.0}, False, "", "post_action_wait"

        return None, False, "", ""

    def _typed_values_from_history(self, history: List[Dict[str, Any]]) -> List[str]:
        out: List[str] = []
        for item in history:
            if not isinstance(item, dict):
                continue
            action_raw = item.get("action")
            action: Dict[str, Any] = action_raw if isinstance(action_raw, dict) else {}
            action_type = str(action.get("type") or "")
            if not action_type and isinstance(action_raw, str):
                action_type = str(action_raw)
            if action_type in {"TypeAction", "FillAction"}:
                nested = action.get("raw") if isinstance(action.get("raw"), dict) else {}
                val = _candidate_text(
                    item.get("text"),
                    action.get("text"),
                    action.get("value"),
                    nested.get("text"),
                    nested.get("value"),
                )
                if val:
                    out.append(val)
        return out

    def _normalized_field_value(self, value: Any) -> str:
        text = _norm_ws(str(value or ""))
        if not text:
            return ""
        if re.fullmatch(r"<[^>]+>", text):
            return text[:220]
        cleaned = text.strip().strip(" \t\r\n'\"`")
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned[:220]

    def _field_value_is_usable(self, *, value: Any, candidate: Candidate | None = None) -> bool:
        normalized = self._normalized_field_value(value)
        if not normalized:
            return False
        lowered = normalized.lower()
        if lowered in {"none", "null", "nil", "undefined", "n/a", "na"}:
            return False
        if re.fullmatch(r"[-_./,:;!?|]+", normalized):
            return False
        if re.fullmatch(r"<[^>]+>", normalized):
            return True
        if not re.search(r"[A-Za-z0-9]", normalized):
            return False
        field_kind = str((candidate.field_kind if candidate is not None else "") or "").strip().lower()
        input_type = str((candidate.input_type if candidate is not None else "") or "").strip().lower()
        if field_kind == "email" or input_type == "email":
            return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", normalized))
        if field_kind in {"password", "confirm_password"} or input_type == "password":
            return len(normalized) >= 6
        if field_kind in {"username", "name"}:
            return len(normalized) >= 2
        return len(normalized) >= 1

    def _remembered_value_for_candidate(self, *, candidate: Candidate, state: AgentState) -> str:
        by_id = str(state.form_progress.typed_values_by_candidate.get(candidate.id) or "")
        if self._field_value_is_usable(value=by_id, candidate=candidate):
            return self._normalized_field_value(by_id)
        sig = self._selector_signature(candidate.selector)
        by_sig = str(state.form_progress.typed_values_by_selector.get(sig) or "") if sig else ""
        if self._field_value_is_usable(value=by_sig, candidate=candidate):
            return self._normalized_field_value(by_sig)
        return ""

    def _candidate_has_usable_typed_value(
        self,
        *,
        candidate: Candidate,
        history: List[Dict[str, Any]],
        state: AgentState,
    ) -> bool:
        remembered = self._remembered_value_for_candidate(candidate=candidate, state=state)
        if remembered:
            return True
        target_sig = self._selector_signature(candidate.selector)
        target_id = str(candidate.id or "")
        had_typed_marker = target_id in set(state.form_progress.typed_candidate_ids)
        if target_sig and target_sig in set(state.form_progress.typed_selector_sigs):
            had_typed_marker = True
        has_explicit_value = target_id in state.form_progress.typed_values_by_candidate
        if target_sig and target_sig in state.form_progress.typed_values_by_selector:
            has_explicit_value = True
        for item in history:
            if not isinstance(item, dict):
                continue
            action_raw = item.get("action")
            action = action_raw if isinstance(action_raw, dict) else {}
            action_type = str(action.get("type") or "")
            if not action_type and isinstance(action_raw, str):
                action_type = action_raw
            if action_type not in {"TypeAction", "FillAction"}:
                continue
            nested = action.get("raw") if isinstance(action.get("raw"), dict) else {}
            item_candidate_id = _candidate_text(
                item.get("candidate_id"),
                item.get("element_id"),
                action.get("_element_id"),
                action.get("element_id"),
                nested.get("_element_id"),
                nested.get("element_id"),
            )
            value = _candidate_text(
                item.get("text"),
                action.get("text"),
                action.get("value"),
                nested.get("text"),
                nested.get("value"),
            )
            if target_id and item_candidate_id == target_id:
                return self._field_value_is_usable(value=value, candidate=candidate)
            sel = action.get("selector")
            if not isinstance(sel, dict):
                sel = nested.get("selector") if isinstance(nested.get("selector"), dict) else None
            if target_sig and self._selector_signature(sel if isinstance(sel, dict) else None) == target_sig:
                return self._field_value_is_usable(value=value, candidate=candidate)
        if has_explicit_value:
            return False
        return had_typed_marker

    def _coerce_type_text(
        self,
        *,
        text: str,
        prompt: str,
        candidate: Candidate | None,
        state: AgentState,
    ) -> str:
        normalized = self._normalized_field_value(text)
        if candidate is not None:
            if not self._candidate_accepts_typed_text(candidate=candidate):
                return ""
            inferred = self._infer_input_text(prompt=prompt, candidate=candidate)
            inferred = self._normalized_field_value(inferred)
            field_kind = str(candidate.field_kind or "").strip().lower()
            if field_kind in {"password", "confirm_password"}:
                next_password = self._next_password_value(prompt=prompt, candidate=candidate, state=state)
                if self._field_value_is_usable(value=next_password, candidate=candidate):
                    return next_password
            if field_kind in {"username", "email", "password", "confirm_password"} and self._field_value_is_usable(value=inferred, candidate=candidate):
                return inferred
        remembered = self._remembered_value_for_candidate(candidate=candidate, state=state) if candidate is not None else ""
        if candidate is not None and self._field_value_is_usable(value=normalized, candidate=candidate):
            if remembered and normalized != remembered and len(normalized) < len(remembered):
                return remembered
            return normalized
        if remembered:
            return remembered
        if candidate is not None and self._field_value_is_usable(value=inferred, candidate=candidate):
            return inferred
        return normalized

    def _candidate_accepts_typed_text(self, *, candidate: Candidate | None) -> bool:
        if candidate is None:
            return False
        if candidate.disabled:
            return False
        role = str(candidate.role or "").strip().lower()
        input_type = str(candidate.input_type or "").strip().lower()
        if role not in {"input", "textarea"}:
            return False
        if input_type in {"checkbox", "radio", "submit", "button", "reset", "image", "hidden", "file"}:
            return False
        return True

    def _next_password_value(self, *, prompt: str, candidate: Candidate | None, state: AgentState) -> str:
        _, passwords = self._extract_credentials(prompt)
        if not passwords:
            return ""
        used_values: List[str] = []
        if candidate is not None:
            remembered = self._remembered_value_for_candidate(candidate=candidate, state=state)
            if remembered:
                used_values.append(self._normalized_field_value(remembered))
        for value in list(state.form_progress.typed_values_by_candidate.values()) + list(state.form_progress.typed_values_by_selector.values()):
            normalized = self._normalized_field_value(value)
            if normalized:
                used_values.append(normalized)
        used_set = {value for value in used_values if value}
        for password in passwords:
            normalized = self._normalized_field_value(password)
            if normalized and normalized not in used_set:
                return normalized
        return self._normalized_field_value(passwords[0])

    def _extract_credentials(self, prompt: str) -> tuple[List[str], List[str]]:
        text = str(prompt or "")
        identifiers = _dedupe_keep_order(
            re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text),
            4,
        )
        for hit in re.finditer(
            r"\b(?:username|user|email|login)\b\s*(?:equals|=|is|:)?\s*(?:'([^']+)'|\"([^\"]+)\"|(<[^>]+>)|([^\s,;]+))",
            text,
            flags=re.I,
        ):
            value = next((g for g in hit.groups() if g), "")
            value = _norm_ws(value).strip(" \t\r\n'\"`.,;:!?")
            if value and value.lower() not in {"and", "then", "attempt", "retry", "with", "to"}:
                identifiers.append(value[:120])
        for token in re.findall(r"<(?:username|user|email|signup_username|signup_email)>", text, flags=re.I):
            identifiers.append(token[:120])
        passwords: List[str] = []
        for hit in re.finditer(
            r"\b(?:password|pass|pwd)\b\s*(?:equals|=|is|:)?\s*(?:'([^']+)'|\"([^\"]+)\"|(<[^>]+>)|([^\s,;]+))",
            text,
            flags=re.I,
        ):
            value = next((g for g in hit.groups() if g), "")
            value = _norm_ws(value).strip(" \t\r\n'\"`.,;:!?")
            if value and value.lower() not in {"and", "then", "attempt", "retry", "with"}:
                passwords.append(value[:80])
        for token in re.findall(r"<(?:password|pass|pwd|signup_password)>", text, flags=re.I):
            passwords.append(token[:80])
        return _dedupe_keep_order(identifiers, 6), _dedupe_keep_order(passwords, 6)

    def _prompt_seed(self, prompt: str) -> str:
        return hashlib.sha1(str(prompt or "").encode("utf-8", errors="ignore")).hexdigest()[:8] or "autoppia"

    def _email_domain_constraint(self, prompt: str) -> str:
        text = str(prompt or "")
        match = re.search(
            r"\b(?:ends with|ending with|domain(?: is| equals)?|suffix(?: is| equals)?)\b[^@]{0,40}@([A-Za-z0-9.-]+\.[A-Za-z]{2,})",
            text,
            flags=re.I,
        )
        if match:
            return str(match.group(1) or "").strip().lower()
        email_match = re.search(r"@([A-Za-z0-9.-]+\.[A-Za-z]{2,})", text)
        if email_match:
            return str(email_match.group(1) or "").strip().lower()
        return "example.com"

    def _quoted_values(self, prompt: str) -> List[str]:
        out: List[str] = []
        for pattern in (r"'([^']*)'", r'"([^"]*)"'):
            for hit in re.finditer(pattern, str(prompt or "")):
                value = _norm_ws(hit.group(1) or "")
                if not value:
                    continue
                lowered = value.lower()
                if any(token in lowered for token in (" equals", " password", " username", " email")):
                    continue
                out.append(value[:120])
        return _dedupe_keep_order(out, 8)

    def _prompt_field_needs(self, prompt: str) -> set[str]:
        return CandidateRanker()._prompt_field_needs(prompt)

    def _uses_signup_placeholders(self, prompt: str) -> bool:
        text = str(prompt or "").lower()
        return any(term in text for term in ("register", "sign up", "signup", "create account", "create an account"))

    def _benchmark_placeholder_for_field(self, *, field_kind: str, prompt: str) -> str:
        kind = str(field_kind or "").strip().lower()
        if self._uses_signup_placeholders(prompt):
            if kind == "username":
                return "<signup_username>"
            if kind == "email":
                return "<signup_email>"
            if kind in {"password", "confirm_password"}:
                return "<signup_password>"
        if kind == "username":
            return "<username>"
        if kind in {"password", "confirm_password"}:
            return "<password>"
        return ""

    def _generated_value_for_field(self, *, field_kind: str, prompt: str) -> str:
        placeholder = self._benchmark_placeholder_for_field(field_kind=field_kind, prompt=prompt)
        if placeholder:
            return placeholder
        seed = self._prompt_seed(prompt)
        if field_kind == "email":
            return f"autoppia.{seed}@{self._email_domain_constraint(prompt)}"
        if field_kind == "password":
            return f"Auto{seed}Pass1!"
        if field_kind == "confirm_password":
            return self._generated_value_for_field(field_kind="password", prompt=prompt)
        if field_kind == "username":
            return f"autoppia_{seed}"
        if field_kind == "name":
            return f"Autoppia {seed[:4]}"
        return f"autoppia-{seed}"

    def _normalize_allowed(self, allowed_tools: Any) -> set[str]:
        out: set[str] = set()
        if not isinstance(allowed_tools, list):
            out.update(SUPPORTED_BROWSER_TOOL_NAMES)
            out.update(OBS_META_TOOLS)
            if self._allow_control_meta_tools():
                out.update(CONTROL_META_TOOLS)
            return out
        for item in allowed_tools:
            if not isinstance(item, dict):
                continue
            raw_name = str(item.get("name") or "").strip()
            if not raw_name and isinstance(item.get("function"), dict):
                raw_name = str((item.get("function") or {}).get("name") or "").strip()
            canonical = _canonical_allowed_tool_name(raw_name)
            if canonical:
                if canonical.startswith("browser.") and canonical not in SUPPORTED_BROWSER_TOOL_NAMES:
                    continue
                if canonical.startswith("META."):
                    if canonical not in META_TOOLS:
                        continue
                    if (not self._allow_control_meta_tools()) and canonical in CONTROL_META_TOOLS:
                        continue
                out.add(canonical)
        return out

    def _tool_name_for_action(self, action: Dict[str, Any]) -> str:
        t = str(action.get("type") or "")
        t_l = t.lower()
        mapping = {
            "navigateaction": "browser.navigate",
            "clickaction": "browser.click",
            "typeaction": "browser.type",
            "scrollaction": "browser.scroll",
            "waitaction": "browser.wait",
            "selectdropdownoptionaction": "browser.select",
            "sendkeysiwaaction": "browser.send_keys",
            "holdkeyaction": "browser.hold_key",
        }
        return mapping.get(t_l, "")

    def _candidate_for_action(self, *, action: Dict[str, Any], ranked_candidates: List[Candidate]) -> Candidate | None:
        element_id = str(action.get("_element_id") or "")
        if element_id:
            for cand in ranked_candidates:
                if cand.id == element_id:
                    return cand
        selector = _sanitize_selector(action.get("selector") if isinstance(action.get("selector"), dict) else None)
        if not isinstance(selector, dict):
            return None
        try:
            selector_key = json.dumps(selector, ensure_ascii=True, sort_keys=True)
        except Exception:
            return None
        for cand in ranked_candidates:
            try:
                cand_key = json.dumps(cand.selector, ensure_ascii=True, sort_keys=True)
            except Exception:
                continue
            if cand_key == selector_key:
                return cand
        return None

    def _same_group_candidates(self, *, target: Candidate, ranked_candidates: List[Candidate]) -> List[Candidate]:
        if target.group_id:
            return [cand for cand in ranked_candidates if cand.group_id == target.group_id and cand.id != target.id]
        target_context = _norm_ws(target.context)
        if not target_context:
            return []
        return [cand for cand in ranked_candidates if _norm_ws(cand.context) == target_context and cand.id != target.id]

    def _remember_active_group(self, *, target: Candidate, ranked_candidates: List[Candidate], state: AgentState) -> None:
        context = _norm_ws(target.context)
        if not context and not target.group_id:
            return
        related_ids = [target.id]
        related_ids.extend([cand.id for cand in self._same_group_candidates(target=target, ranked_candidates=ranked_candidates)])
        state.form_progress.active_group_id = str(target.group_id or "")[:120]
        state.form_progress.active_group_context = context[:320]
        state.form_progress.active_group_label = _candidate_text(target.group_label, self.obs_builder._group_label_for_candidate(target))[:160]
        state.form_progress.active_group_candidate_ids = _dedupe_keep_order(related_ids, MAX_PENDING_ELEMENTS)
        focus_region_id = str(target.region_id or target.group_id or "")[:120]
        focus_region_context = context[:320]
        state.focus_region.region_id = focus_region_id
        state.focus_region.region_kind = str(target.region_kind or "group")[:40]
        state.focus_region.region_label = _candidate_text(
            target.region_label,
            target.group_label,
            self.obs_builder._group_label_for_candidate(target),
        )[:160]
        state.focus_region.region_context = focus_region_context
        state.focus_region.candidate_ids = _dedupe_keep_order(related_ids, MAX_PENDING_ELEMENTS)
        if focus_region_id:
            state.focus_region.recent_region_ids = _dedupe_keep_order(
                state.focus_region.recent_region_ids + [focus_region_id],
                MAX_PENDING_ELEMENTS,
            )

    def _candidate_constraint_keys(self, *, candidate: Candidate, task_constraints: Dict[str, str]) -> set[str]:
        return self.ranker._candidate_constraint_keys(cand=candidate, task_constraints=task_constraints)

    def _record_constraint_progress(
        self,
        *,
        prompt: str,
        action: Dict[str, Any],
        target: Candidate | None,
        state: AgentState,
    ) -> None:
        task_constraints = _task_constraints(prompt)
        if not task_constraints or target is None:
            return
        candidate_keys = self._candidate_constraint_keys(candidate=target, task_constraints=task_constraints)
        action_text = _candidate_text(action.get("text"), target.text, target.field_hint)
        for key in candidate_keys:
            state.progress.attempted_constraints[key] = int(state.progress.attempted_constraints.get(key) or 0) + 1
            expected = str(task_constraints.get(key) or "")
            if _constraint_value_matches(expected, action_text) or _constraint_value_matches(expected, target.text):
                state.progress.satisfied_constraints = _dedupe_keep_order(
                    state.progress.satisfied_constraints + [key],
                    32,
                )

    def _infer_input_text(self, *, prompt: str, candidate: Candidate) -> str:
        prompt_text = str(prompt or "")
        if not self._candidate_accepts_typed_text(candidate=candidate):
            return ""
        blob = " ".join([candidate.text, candidate.context, candidate.href, candidate.field_hint, candidate.field_kind]).lower()
        quoted_values = self._quoted_values(prompt_text)
        emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", prompt_text)
        identifiers, passwords = self._extract_credentials(prompt_text)
        explicit_constraints = _task_constraints(prompt_text)
        field_kind = str(candidate.field_kind or "")

        if field_kind in {"password", "confirm_password"} or any(k in blob for k in ("password", "pass", "pwd")):
            if passwords:
                return passwords[0]
            return self._generated_value_for_field(field_kind=field_kind or "password", prompt=prompt_text)
        if field_kind == "email" or any(k in blob for k in ("email", "mail")):
            if emails:
                return emails[0]
            if explicit_constraints.get("email"):
                return explicit_constraints["email"]
            return self._generated_value_for_field(field_kind="email", prompt=prompt_text)
        if field_kind == "username" or any(k in blob for k in ("username", "user", "login")):
            username_like = [value for value in identifiers if "@" not in str(value)]
            if username_like:
                return username_like[0]
            if explicit_constraints.get("username"):
                return explicit_constraints["username"]
            if explicit_constraints.get("user"):
                return explicit_constraints["user"]
            return self._generated_value_for_field(field_kind="username", prompt=prompt_text)
        if field_kind == "name" or "name" in blob:
            m_not = re.search(r"\bname\b[^.]{0,100}\bnot\b[^'\"<]*['\"]([^'\"]+)['\"]", prompt_text, flags=re.I)
            if m_not:
                banned = _norm_ws(m_not.group(1))
                if banned.lower() != "autouser":
                    return "AutoUser"
                return "ExampleUser"
            m_eq = re.search(r"\bname\b[^.]{0,100}\b(?:equals|is|=)\s*['\"]([^'\"]+)['\"]", prompt_text, flags=re.I)
            if m_eq:
                val = _norm_ws(m_eq.group(1))
                if val:
                    return val
            return self._generated_value_for_field(field_kind="name", prompt=prompt_text)
        if field_kind == "search" or any(k in blob for k in ("search", "query")):
            if quoted_values:
                return quoted_values[0][:80]
            m_movie = re.search(r"\bmovie\b[^'\"<]*['\"]([^'\"]+)['\"]", prompt_text, flags=re.I)
            if m_movie:
                return _norm_ws(m_movie.group(1))[:80]
            return "search"
        if any(k in blob for k in ("rating", "score")):
            m_num = re.search(r"\b(?:rating|score)\b[^0-9]{0,20}([0-9]+(?:\.[0-9]+)?)", prompt_text, flags=re.I)
            if m_num:
                return m_num.group(1)
            return "4.0"
        if field_kind == "year" or any(k in blob for k in ("year", "date")):
            m_year = re.search(r"\b(19[0-9]{2}|20[0-9]{2}|21[0-9]{2})\b", prompt_text)
            if m_year:
                return m_year.group(1)
            return "2020"
        if quoted_values:
            return quoted_values[0][:120]
        if self._prompt_field_needs(prompt_text):
            return self._generated_value_for_field(field_kind=field_kind or "text", prompt=prompt_text)
        return "autoppia"

    def _selector_signature(self, selector: Dict[str, Any] | None) -> str:
        selector = _sanitize_selector(selector)
        if not isinstance(selector, dict):
            return ""
        try:
            return json.dumps(selector, ensure_ascii=True, sort_keys=True)
        except Exception:
            return ""

    def _typed_selector_signatures_from_history(self, history: List[Dict[str, Any]]) -> set[str]:
        out: set[str] = set()
        for item in history:
            if not isinstance(item, dict):
                continue
            action_raw = item.get("action")
            action = action_raw if isinstance(action_raw, dict) else {}
            action_type = str(action.get("type") or action.get("name") or "")
            if not action_type and isinstance(action_raw, str):
                action_type = action_raw
            if action_type not in {"TypeAction", "FillAction"}:
                continue
            sel = action.get("selector")
            if not isinstance(sel, dict):
                nested = action.get("raw") if isinstance(action.get("raw"), dict) else {}
                sel = nested.get("selector")
            sig = self._selector_signature(sel if isinstance(sel, dict) else None)
            if sig:
                out.add(sig)
        return out

    def _selected_selector_signatures_from_history(self, history: List[Dict[str, Any]]) -> set[str]:
        out: set[str] = set()
        for item in history:
            if not isinstance(item, dict):
                continue
            action_raw = item.get("action")
            action = action_raw if isinstance(action_raw, dict) else {}
            action_type = str(action.get("type") or action.get("name") or "")
            if not action_type and isinstance(action_raw, str):
                action_type = action_raw
            if action_type not in {"SelectAction", "SelectDropDownOptionAction"}:
                continue
            sel = action.get("selector")
            if not isinstance(sel, dict):
                nested = action.get("raw") if isinstance(action.get("raw"), dict) else {}
                sel = nested.get("selector")
            sig = self._selector_signature(sel if isinstance(sel, dict) else None)
            if sig:
                out.add(sig)
        return out

    def _typed_candidate_ids_from_history(self, history: List[Dict[str, Any]]) -> set[str]:
        out: set[str] = set()
        for item in history:
            if not isinstance(item, dict):
                continue
            action_raw = item.get("action")
            action = action_raw if isinstance(action_raw, dict) else {}
            action_type = str(action.get("type") or action.get("name") or "")
            if not action_type and isinstance(action_raw, str):
                action_type = action_raw
            if action_type not in {"TypeAction", "FillAction"}:
                continue
            nested = action.get("raw") if isinstance(action.get("raw"), dict) else {}
            candidate_id = _candidate_text(
                item.get("candidate_id"),
                item.get("element_id"),
                action.get("_element_id"),
                action.get("element_id"),
                nested.get("_element_id"),
                nested.get("element_id"),
            )
            if candidate_id:
                out.add(candidate_id)
        return out

    def _typed_selector_signatures(self, history: List[Dict[str, Any]], state: AgentState) -> set[str]:
        out = set(self._typed_selector_signatures_from_history(history))
        out.update(state.form_progress.typed_selector_sigs)
        return out

    def _typed_candidate_ids(self, history: List[Dict[str, Any]], state: AgentState) -> set[str]:
        out = set(self._typed_candidate_ids_from_history(history))
        out.update(state.form_progress.typed_candidate_ids)
        return out

    def _is_submit_like(self, cand: Candidate) -> bool:
        if self.ranker._is_section_switch_candidate(cand):
            return False
        if cand.field_kind == "submit":
            return True
        if cand.role in {"button", "input"} and cand.field_kind == "account_create":
            return True
        if cand.role == "select":
            return False
        if cand.role == "input" and str(cand.input_type or "").strip().lower() in {"submit", "button", "image", "reset"}:
            return True
        if cand.role not in {"button", "input"}:
            return False
        blob = " ".join([cand.text, cand.field_hint]).lower()
        return any(
            k in blob
            for k in ("submit", "save", "apply", "find", "search", "filter", "go", "continue", "send", "sign up", "signup", "register", "login", "sign in", "create account")
        )

    def _is_search_or_filter_input(self, cand: Candidate) -> bool:
        return str(cand.field_kind or "") in {"search", "sort"}

    def _redirect_delete_only_form_action(
        self,
        *,
        prompt: str,
        ranked_candidates: List[Candidate],
        target: Candidate | None,
        state: AgentState,
    ) -> Dict[str, Any] | None:
        task_ops = self.ranker._task_operation_hints(prompt)
        if task_ops.intersection({"create", "update"}) or "delete" not in task_ops:
            return None
        has_password_input_visible = any(
            cand.role == "input" and cand.field_kind in {"password", "confirm_password"}
            for cand in ranked_candidates
        )
        if has_password_input_visible:
            return None
        if target is not None and target.role not in {"input", "select"}:
            return None
        for preferred_state in ("inactive", ""):
            for cand in ranked_candidates:
                if cand.id in state.blocklist.element_ids:
                    continue
                if not self.ranker._is_section_switch_candidate(cand):
                    continue
                if preferred_state and cand.ui_state != preferred_state:
                    continue
                if cand.ui_state == "active":
                    continue
                return {"type": "ClickAction", "selector": cand.selector, "_element_id": cand.id}
        for cand in ranked_candidates:
            if cand.id in state.blocklist.element_ids:
                continue
            if cand.role in {"button", "link"}:
                return {"type": "ClickAction", "selector": cand.selector, "_element_id": cand.id}
        return None

    def _guard_delete_task_against_unrelated_form_edits(
        self,
        *,
        action: Dict[str, Any] | None,
        prompt: str,
        ranked_candidates: List[Candidate],
        state: AgentState,
    ) -> Dict[str, Any] | None:
        if not isinstance(action, dict):
            return action
        task_ops = self.ranker._task_operation_hints(prompt)
        if task_ops.intersection({"create", "update"}) or "delete" not in task_ops:
            return action
        if str(action.get("type") or "") not in {"TypeAction", "SelectDropDownOptionAction"}:
            return action
        has_password_input_visible = any(
            cand.role == "input" and cand.field_kind in {"password", "confirm_password"}
            for cand in ranked_candidates
        )
        if has_password_input_visible:
            return action
        mutation_candidates = [
            cand
            for cand in ranked_candidates
            if self.ranker._candidate_action_tags(cand).intersection({"delete"})
            and cand.id not in state.blocklist.element_ids
            and cand.role in {"button", "link"}
        ]
        if mutation_candidates:
            best = mutation_candidates[0]
            return {"type": "ClickAction", "selector": best.selector, "_element_id": best.id}
        if any(cand.role in {"button", "link"} for cand in ranked_candidates):
            target = self._candidate_for_action(action=action, ranked_candidates=ranked_candidates)
            if target is not None and target.role in {"input", "select"}:
                redirect = self._redirect_delete_only_form_action(
                    prompt=prompt,
                    ranked_candidates=ranked_candidates,
                    target=target,
                    state=state,
                )
                if redirect is not None:
                    return redirect
        return action

    def _guard_submit_without_inputs(
        self,
        *,
        action: Dict[str, Any] | None,
        prompt: str,
        history: List[Dict[str, Any]],
        ranked_candidates: List[Candidate],
        state: AgentState,
    ) -> Dict[str, Any] | None:
        if not isinstance(action, dict) or str(action.get("type") or "") != "ClickAction":
            return action
        target = self._candidate_for_action(action=action, ranked_candidates=ranked_candidates)
        if target is None or not self._is_submit_like(target):
            return action
        typed_sigs = self._typed_selector_signatures(history, state)
        typed_candidate_ids = self._typed_candidate_ids(history, state)
        for cand in ranked_candidates:
            if cand.role != "input":
                continue
            if cand.id in state.blocklist.element_ids:
                continue
            if self._is_search_or_filter_input(cand):
                continue
            sig = self._selector_signature(cand.selector)
            if (
                cand.id in typed_candidate_ids or (sig and sig in typed_sigs)
            ) and self._candidate_has_usable_typed_value(candidate=cand, history=history, state=state):
                continue
            text = self._infer_input_text(prompt=prompt, candidate=cand)
            if not text:
                continue
            return {
                "type": "TypeAction",
                "selector": cand.selector,
                "text": text[:220],
                "_element_id": cand.id,
            }
        sel_sig = self._selector_signature(target.selector)
        if sel_sig:
            state.form_progress.submit_attempt_sigs = _dedupe_keep_order(
                state.form_progress.submit_attempt_sigs + [sel_sig],
                MAX_PENDING_ELEMENTS * 2,
            )
        return action

    def _guard_missing_group_inputs(
        self,
        *,
        action: Dict[str, Any] | None,
        prompt: str,
        history: List[Dict[str, Any]],
        ranked_candidates: List[Candidate],
        state: AgentState,
    ) -> Dict[str, Any] | None:
        if not isinstance(action, dict):
            return action
        action_type = str(action.get("type") or "")
        if action_type not in {"ClickAction", "SelectDropDownOptionAction"}:
            return action
        target = self._candidate_for_action(action=action, ranked_candidates=ranked_candidates)
        if target is not None and self.ranker._is_section_switch_candidate(target):
            return action
        typed_sigs = self._typed_selector_signatures(history, state)
        typed_candidate_ids = self._typed_candidate_ids(history, state)
        prompt_needs = self._prompt_field_needs(prompt)
        group_id = str(state.form_progress.active_group_id or "").strip()
        group_context = _norm_ws(state.form_progress.active_group_context)
        group_candidate_ids = set(state.form_progress.active_group_candidate_ids)

        missing_inputs: List[Candidate] = []
        for cand in ranked_candidates:
            if cand.role != "input":
                continue
            if cand.id in state.blocklist.element_ids:
                continue
            if self._is_search_or_filter_input(cand):
                continue
            cand_sig = self._selector_signature(cand.selector)
            if (
                cand.id in typed_candidate_ids or (cand_sig and cand_sig in typed_sigs)
            ) and self._candidate_has_usable_typed_value(candidate=cand, history=history, state=state):
                continue
            if group_id or group_context or group_candidate_ids:
                same_group = False
                if group_id and cand.group_id == group_id:
                    same_group = True
                elif group_context and _norm_ws(cand.context) == group_context:
                    same_group = True
                elif cand.id in group_candidate_ids:
                    same_group = True
                if not same_group:
                    continue
            text = self._infer_input_text(prompt=prompt, candidate=cand)
            if not text:
                continue
            missing_inputs.append(cand)
        if not missing_inputs:
            target_group_candidates: List[Candidate] = []
            for cand in ranked_candidates:
                if group_id and cand.group_id == group_id:
                    target_group_candidates.append(cand)
                elif group_context and _norm_ws(cand.context) == group_context:
                    target_group_candidates.append(cand)
                elif group_candidate_ids and cand.id in group_candidate_ids:
                    target_group_candidates.append(cand)
            typed_kinds = {
                cand.field_kind
                for cand in target_group_candidates
                if cand.field_kind
                and (
                    cand.id in typed_candidate_ids
                    or (self._selector_signature(cand.selector) and self._selector_signature(cand.selector) in typed_sigs)
                )
            }
            required_kinds = {kind for kind in prompt_needs if any(c.field_kind == kind for c in target_group_candidates)}
            if "password" in required_kinds and any(c.field_kind == "confirm_password" for c in target_group_candidates):
                required_kinds.add("confirm_password")
            if (
                action_type == "SelectDropDownOptionAction"
                and target_group_candidates
                and required_kinds
                and required_kinds.issubset(typed_kinds)
                and target is not None
                and target.field_kind not in required_kinds
            ):
                for cand in target_group_candidates:
                    if cand.id in state.blocklist.element_ids:
                        continue
                    if cand.role == "button" and self._is_submit_like(cand):
                        return {
                            "type": "ClickAction",
                            "selector": cand.selector,
                            "_element_id": cand.id,
                        }
            return action
        if not prompt_needs and not any(c.field_kind in {"username", "email", "password", "confirm_password"} for c in missing_inputs):
            return action
        if target is not None and target.role == "input" and target.id not in typed_candidate_ids:
            return action

        def score(cand: Candidate) -> tuple[int, int]:
            score_value = 0
            if cand.field_kind in prompt_needs:
                score_value += 5
            if cand.field_kind in {"username", "email", "password", "confirm_password"}:
                score_value += 3
            if str(cand.input_type or "").strip().lower() in {"email", "password"}:
                score_value += 1
            return (-score_value, ranked_candidates.index(cand))

        best = sorted(missing_inputs, key=score)[0]
        text = self._infer_input_text(prompt=prompt, candidate=best)
        if not text:
            return action
        return {
            "type": "TypeAction",
            "selector": best.selector,
            "text": text[:220],
            "_element_id": best.id,
        }

    def _guard_redundant_type_action(
        self,
        *,
        action: Dict[str, Any] | None,
        prompt: str,
        history: List[Dict[str, Any]],
        ranked_candidates: List[Candidate],
        state: AgentState,
    ) -> Dict[str, Any] | None:
        if not isinstance(action, dict) or str(action.get("type") or "") != "TypeAction":
            return action
        target = self._candidate_for_action(action=action, ranked_candidates=ranked_candidates)
        if target is None:
            return action
        typed_sigs = self._typed_selector_signatures(history, state)
        typed_candidate_ids = self._typed_candidate_ids(history, state)
        target_sig = self._selector_signature(target.selector)
        already_typed_target = (
            (target.id in typed_candidate_ids) or (bool(target_sig) and target_sig in typed_sigs)
        ) and self._candidate_has_usable_typed_value(candidate=target, history=history, state=state)
        if not already_typed_target:
            return action
        same_group = self._same_group_candidates(target=target, ranked_candidates=ranked_candidates)
        for cand in same_group:
            if cand.id in state.blocklist.element_ids:
                continue
            if cand.role not in {"button", "select", "input"}:
                continue
            if cand.role == "button" and self._is_submit_like(cand):
                submit_sig = self._selector_signature(cand.selector)
                if submit_sig and submit_sig in set(state.form_progress.submit_attempt_sigs):
                    continue
                return {
                    "type": "ClickAction",
                    "selector": cand.selector,
                    "_element_id": cand.id,
                }
        for cand in ranked_candidates:
            if cand.role != "input":
                continue
            if cand.id in state.blocklist.element_ids:
                continue
            if self._is_search_or_filter_input(cand):
                continue
            cand_sig = self._selector_signature(cand.selector)
            if (
                cand.id in typed_candidate_ids or (cand_sig and cand_sig in typed_sigs)
            ) and self._candidate_has_usable_typed_value(candidate=cand, history=history, state=state):
                continue
            text = self._infer_input_text(prompt=prompt, candidate=cand)
            if not text:
                continue
            return {
                "type": "TypeAction",
                "selector": cand.selector,
                "text": text[:220],
                "_element_id": cand.id,
            }
        for cand in ranked_candidates:
            if cand.id in state.blocklist.element_ids:
                continue
            if not self._is_submit_like(cand):
                continue
            submit_sig = self._selector_signature(cand.selector)
            if submit_sig and submit_sig in set(state.form_progress.submit_attempt_sigs):
                continue
            return {
                "type": "ClickAction",
                "selector": cand.selector,
                "_element_id": cand.id,
            }
        return action

    def _guard_redundant_select_action(
        self,
        *,
        action: Dict[str, Any] | None,
        history: List[Dict[str, Any]],
        ranked_candidates: List[Candidate],
        state: AgentState,
    ) -> Dict[str, Any] | None:
        if not isinstance(action, dict) or str(action.get("type") or "") != "SelectDropDownOptionAction":
            return action
        target = self._candidate_for_action(action=action, ranked_candidates=ranked_candidates)
        if target is None:
            return action
        selected_sigs = self._selected_selector_signatures_from_history(history)
        target_sig = self._selector_signature(target.selector)
        if not target_sig or target_sig not in selected_sigs:
            return action
        same_group = self._same_group_candidates(target=target, ranked_candidates=ranked_candidates)
        for cand in same_group:
            if cand.id in state.blocklist.element_ids:
                continue
            if cand.role == "button" and self._is_submit_like(cand):
                submit_sig = self._selector_signature(cand.selector)
                if submit_sig and submit_sig in set(state.form_progress.submit_attempt_sigs):
                    continue
                return {
                    "type": "ClickAction",
                    "selector": cand.selector,
                    "_element_id": cand.id,
                }
        for cand in same_group:
            if cand.id in state.blocklist.element_ids:
                continue
            if cand.role != "button":
                continue
            return {
                "type": "ClickAction",
                "selector": cand.selector,
                "_element_id": cand.id,
            }
        for cand in ranked_candidates:
            if cand.id in state.blocklist.element_ids:
                continue
            if not self._is_submit_like(cand):
                continue
            submit_sig = self._selector_signature(cand.selector)
            if submit_sig and submit_sig in set(state.form_progress.submit_attempt_sigs):
                continue
            return {
                "type": "ClickAction",
                "selector": cand.selector,
                "_element_id": cand.id,
            }
        return action

    def _promote_click_input_to_type(
        self,
        *,
        action: Dict[str, Any] | None,
        prompt: str,
        ranked_candidates: List[Candidate],
        state: AgentState,
    ) -> Dict[str, Any] | None:
        if not isinstance(action, dict):
            return action
        if str(action.get("type") or "") != "ClickAction":
            return action
        cand = self._candidate_for_action(action=action, ranked_candidates=ranked_candidates)
        if cand is None or cand.role != "input":
            return action
        repeat_n = int(state.counters.repeat_action_count or 0)
        if repeat_n < 1 and int(state.counters.stall_count or 0) < 2:
            return action
        text = self._infer_input_text(prompt=prompt, candidate=cand)
        if not text:
            return action
        return {
            "type": "TypeAction",
            "selector": cand.selector,
            "text": text[:220],
            "_element_id": cand.id,
        }

    def _browser_action_from_tool_call(
        self,
        *,
        tool_call: Dict[str, Any],
        ranked_candidates: List[Candidate],
        state: AgentState,
        prompt: str,
        allowed: set[str],
        current_url: str = "",
    ) -> Dict[str, Any] | None:
        name = str(tool_call.get("name") or "").strip().lower()
        if not name or not name.startswith("browser."):
            return None
        if allowed and name not in allowed:
            return None
        args = tool_call.get("arguments") if isinstance(tool_call.get("arguments"), dict) else {}
        action_type = _action_type_for_tool(name)
        if not action_type:
            return None

        if name == "browser.back":
            return {"type": "NavigateAction", "go_back": True, "go_forward": False}

        if name == "browser.click":
            selector = _sanitize_selector(args.get("selector") if isinstance(args.get("selector"), dict) else None)
            element_id = str(args.get("element_id") or args.get("_element_id") or "")
            selector, element_id = self._resolve_selector_and_element_id(
                selector=selector,
                element_id=element_id,
                ranked_candidates=ranked_candidates,
                state=state,
                prefer_roles={"button", "link", "input"},
                allow_unmatched_selector=True,
            )
            if selector is None and ranked_candidates:
                for cand in ranked_candidates:
                    if cand.id not in state.blocklist.element_ids:
                        selector = cand.selector
                        element_id = cand.id
                        break
            selector = _sanitize_selector(selector if isinstance(selector, dict) else None)
            if not isinstance(selector, dict):
                return None
            selected_candidate = self._candidate_for_action(
                action={"type": "ClickAction", "selector": selector, "_element_id": element_id},
                ranked_candidates=ranked_candidates,
            )
            if (
                selected_candidate is not None
                and selected_candidate.href
                and ((not allowed) or ("browser.navigate" in allowed))
            ):
                normalized_href = self._normalize_url_for_session(
                    target_url=selected_candidate.href,
                    current_url=current_url,
                    state=state,
                )
                same_visible_link = False
                if normalized_href:
                    cand_href = str(selected_candidate.href or "")
                    if normalized_href == cand_href:
                        same_visible_link = True
                    else:
                        norm_parts = urlsplit(normalized_href)
                        cand_parts = urlsplit(cand_href)
                        if (
                            norm_parts.scheme
                            and cand_parts.scheme
                            and norm_parts.scheme == cand_parts.scheme
                            and norm_parts.netloc == cand_parts.netloc
                            and ((norm_parts.path or "/") == (cand_parts.path or "/"))
                        ):
                            same_visible_link = True
                if normalized_href and not same_visible_link:
                    return {"type": "NavigateAction", "url": normalized_href, "go_back": False, "go_forward": False}
            if ((not allowed) or ("browser.navigate" in allowed)) and str(selector.get("type") or "") == "attributeValueSelector":
                attr = str(selector.get("attribute") or "").strip().lower()
                raw_val = str(selector.get("value") or "").strip()
                if attr == "href" and raw_val:
                    normalized_sel_href = self._normalize_url_for_session(
                        target_url=raw_val,
                        current_url=current_url,
                        state=state,
                    )
                    raw_safe_href = _safe_url(raw_val, base=current_url)
                    same_visible_selector_link = False
                    if normalized_sel_href and raw_safe_href:
                        if normalized_sel_href == raw_safe_href:
                            same_visible_selector_link = True
                        else:
                            norm_parts = urlsplit(normalized_sel_href)
                            raw_parts = urlsplit(raw_safe_href)
                            if (
                                norm_parts.scheme
                                and raw_parts.scheme
                                and norm_parts.scheme == raw_parts.scheme
                                and norm_parts.netloc == raw_parts.netloc
                                and ((norm_parts.path or "/") == (raw_parts.path or "/"))
                            ):
                                same_visible_selector_link = True
                    if normalized_sel_href and not same_visible_selector_link:
                        return {"type": "NavigateAction", "url": normalized_sel_href, "go_back": False, "go_forward": False}
            out = {"type": "ClickAction", "selector": selector}
            if element_id:
                out["_element_id"] = element_id
            return out

        if name == "browser.navigate":
            nav_url = self._normalize_url_for_session(
                target_url=str(args.get("url") or ""),
                current_url=current_url,
                state=state,
            )
            if not nav_url:
                return None
            current_safe = _safe_url(current_url, base=current_url)
            if current_safe and nav_url == current_safe:
                return {"type": "WaitAction", "time_seconds": 0.5}
            target_parts = urlsplit(nav_url)
            current_parts = urlsplit(str(current_safe or current_url or ""))
            if (
                ((not allowed) or ("browser.click" in allowed))
                and target_parts.scheme
                and target_parts.netloc
                and current_parts.scheme
                and current_parts.netloc
                and target_parts.scheme == current_parts.scheme
                and target_parts.netloc == current_parts.netloc
            ):
                for cand in ranked_candidates:
                    if cand.id in state.blocklist.element_ids:
                        continue
                    if cand.role != "link" or not cand.href:
                        continue
                    cand_url = self._normalize_url_for_session(
                        target_url=str(cand.href or ""),
                        current_url=current_url,
                        state=state,
                    )
                    same_visible_link = False
                    if cand_url:
                        if cand_url == nav_url:
                            same_visible_link = True
                        else:
                            cand_parts = urlsplit(cand_url)
                            if (
                                cand_parts.scheme == target_parts.scheme
                                and cand_parts.netloc == target_parts.netloc
                                and ((cand_parts.path or "/") == (target_parts.path or "/"))
                            ):
                                same_visible_link = True
                    if same_visible_link and isinstance(cand.selector, dict):
                        out = {"type": "ClickAction", "selector": cand.selector}
                        if cand.id:
                            out["_element_id"] = cand.id
                        return out
            return {"type": "NavigateAction", "url": nav_url, "go_back": False, "go_forward": False}

        if name == "browser.type":
            selector = _sanitize_selector(args.get("selector") if isinstance(args.get("selector"), dict) else None)
            text = _candidate_text(args.get("text"), args.get("value"))
            element_id = str(args.get("element_id") or args.get("_element_id") or "")
            selector, element_id = self._resolve_selector_and_element_id(
                selector=selector,
                element_id=element_id,
                ranked_candidates=ranked_candidates,
                state=state,
                prefer_roles={"input", "textarea"},
                allow_unmatched_selector=False,
            )
            if not isinstance(selector, dict):
                for cand in ranked_candidates:
                    if cand.role == "input" and cand.id not in state.blocklist.element_ids:
                        selector = cand.selector
                        element_id = cand.id
                        break
            selector = _sanitize_selector(selector if isinstance(selector, dict) else None)
            if not isinstance(selector, dict):
                return None
            resolved = self._candidate_for_action(
                action={"type": "TypeAction", "selector": selector, "_element_id": element_id},
                ranked_candidates=ranked_candidates,
            )
            if resolved is not None and resolved.role != "input":
                for cand in ranked_candidates:
                    if cand.role == "input" and cand.id not in state.blocklist.element_ids:
                        selector = cand.selector
                        element_id = cand.id
                        resolved = cand
                        break
            selector = _sanitize_selector(selector if isinstance(selector, dict) else None)
            if not isinstance(selector, dict):
                return None
            resolved_candidate = resolved
            if resolved_candidate is None:
                resolved_candidate = self._candidate_for_action(
                    action={"type": "TypeAction", "selector": selector, "_element_id": element_id},
                    ranked_candidates=ranked_candidates,
                )
            if resolved_candidate is not None and not self._candidate_accepts_typed_text(candidate=resolved_candidate):
                input_type = str(resolved_candidate.input_type or "").strip().lower()
                if input_type in {"checkbox", "radio"} and ((not allowed) or ("browser.click" in allowed)):
                    out = {"type": "ClickAction", "selector": selector}
                    if element_id:
                        out["_element_id"] = element_id
                    return out
                fallback_candidate = next(
                    (
                        cand
                        for cand in ranked_candidates
                        if cand.id not in state.blocklist.element_ids and self._candidate_accepts_typed_text(candidate=cand)
                    ),
                    None,
                )
                if fallback_candidate is None:
                    return None
                selector = _sanitize_selector(fallback_candidate.selector)
                if not isinstance(selector, dict):
                    return None
                element_id = fallback_candidate.id
                resolved_candidate = fallback_candidate
            redirected = self._redirect_delete_only_form_action(
                prompt=prompt,
                ranked_candidates=ranked_candidates,
                target=resolved_candidate,
                state=state,
            )
            if redirected is not None:
                return redirected
            text = self._coerce_type_text(
                text=text,
                prompt=prompt,
                candidate=resolved_candidate,
                state=state,
            )
            if not text:
                text = " "
            out = {"type": "TypeAction", "selector": selector, "text": text[:220]}
            if element_id:
                out["_element_id"] = element_id
            return out

        if name == "browser.scroll":
            direction = str(args.get("direction") or "down").strip().lower()
            raw_amount = args.get("amount")
            amount = 650
            if isinstance(raw_amount, (int, float)):
                amount = int(raw_amount)
            else:
                raw_text = str(raw_amount or "").strip().lower()
                if raw_text:
                    if raw_text in {"page", "screen", "viewport"}:
                        amount = 650
                    else:
                        try:
                            amount = int(float(raw_text))
                        except (TypeError, ValueError):
                            amount = 650
            return {
                "type": "ScrollAction",
                "direction": "up" if direction == "up" else "down",
                "up": bool(direction == "up"),
                "down": bool(direction != "up"),
                "amount": max(120, min(amount, 1400)),
            }

        if name == "browser.wait":
            seconds = float(args.get("time_seconds") or 1.0)
            return {"type": "WaitAction", "time_seconds": max(0.2, min(seconds, 8.0))}

        if name == "browser.select":
            selector = _sanitize_selector(args.get("selector") if isinstance(args.get("selector"), dict) else None)
            text = _candidate_text(args.get("text"), args.get("value"), "Option")
            element_id = str(args.get("element_id") or args.get("_element_id") or "")
            selector, element_id = self._resolve_selector_and_element_id(
                selector=selector,
                element_id=element_id,
                ranked_candidates=ranked_candidates,
                state=state,
                prefer_roles={"select"},
                allow_unmatched_selector=False,
            )
            if not isinstance(selector, dict):
                for cand in ranked_candidates:
                    if cand.role == "select" and cand.id not in state.blocklist.element_ids:
                        selector = cand.selector
                        element_id = cand.id
                        break
            selector = _sanitize_selector(selector if isinstance(selector, dict) else None)
            if not isinstance(selector, dict):
                return None
            resolved = self._candidate_for_action(
                action={"type": "SelectDropDownOptionAction", "selector": selector, "_element_id": element_id},
                ranked_candidates=ranked_candidates,
            )
            if resolved is None or resolved.role != "select":
                selector = None
                element_id = ""
                for cand in ranked_candidates:
                    if cand.role == "select" and cand.id not in state.blocklist.element_ids:
                        selector = cand.selector
                        element_id = cand.id
                        break
            selector = _sanitize_selector(selector if isinstance(selector, dict) else None)
            if not isinstance(selector, dict):
                return None
            redirected = self._redirect_delete_only_form_action(
                prompt=prompt,
                ranked_candidates=ranked_candidates,
                target=resolved,
                state=state,
            )
            if redirected is not None:
                return redirected
            out = {"type": "SelectDropDownOptionAction", "selector": selector, "text": text}
            if element_id:
                out["_element_id"] = element_id
            return out

        if name == "browser.send_keys":
            keys = _candidate_text(args.get("keys"), "Enter")
            return {"type": "SendKeysIWAAction", "keys": keys}

        if name == "browser.hold_key":
            key = _candidate_text(args.get("key"), "Control")
            return {"type": "HoldKeyAction", "key": key}
        return None

    def _resolve_selector_and_element_id(
        self,
        *,
        selector: Dict[str, Any] | None,
        element_id: str,
        ranked_candidates: List[Candidate],
        state: AgentState,
        prefer_roles: set[str],
        allow_unmatched_selector: bool = True,
    ) -> tuple[Dict[str, Any] | None, str]:
        selector = _sanitize_selector(selector)
        if element_id:
            for cand in ranked_candidates:
                if cand.id == element_id and ((not prefer_roles) or (cand.role in prefer_roles)):
                    return cand.selector, cand.id

        if not isinstance(selector, dict):
            return selector, element_id

        sel_type = str(selector.get("type") or "").strip().lower()
        if sel_type == "text":
            needle = _candidate_text(selector.get("value")).lower()
            if needle:
                ordered = sorted(
                    ranked_candidates,
                    key=lambda c: (0 if c.role in prefer_roles else 1),
                )
                for cand in ordered:
                    if cand.id in state.blocklist.element_ids:
                        continue
                    blob = " ".join([cand.text, cand.context, cand.href]).lower()
                    if needle in blob:
                        return cand.selector, cand.id
            return (None, "") if not allow_unmatched_selector else (selector, element_id)

        match_id = self._candidate_id_for_selector(
            selector=selector,
            ranked_candidates=ranked_candidates,
            prefer_roles=prefer_roles,
        )
        if match_id:
            for cand in ranked_candidates:
                if cand.id == match_id:
                    return cand.selector, cand.id
        if not allow_unmatched_selector:
            return None, ""
        return selector, element_id

    def _candidate_id_for_selector(
        self,
        *,
        selector: Dict[str, Any],
        ranked_candidates: List[Candidate],
        prefer_roles: set[str] | None = None,
    ) -> str:
        selector = _sanitize_selector(selector)
        if not isinstance(selector, dict):
            return ""
        try:
            target = json.dumps(selector, ensure_ascii=True, sort_keys=True)
        except Exception:
            return ""
        for cand in ranked_candidates:
            if prefer_roles and cand.role not in prefer_roles:
                continue
            try:
                cand_sel = json.dumps(_sanitize_selector(cand.selector), ensure_ascii=True, sort_keys=True)
            except Exception:
                continue
            if cand_sel == target:
                return cand.id
        return ""

    def _normalize_url_for_session(
        self,
        *,
        target_url: str,
        current_url: str,
        state: AgentState,
    ) -> str:
        nav_url = _safe_url(target_url, base=current_url)
        if not nav_url:
            return ""
        if not state.session_query:
            return nav_url
        try:
            target = urlsplit(nav_url)
            current = urlsplit(str(current_url or ""))
            if current.scheme and current.netloc and target.netloc and target.netloc != current.netloc:
                return nav_url
            target_query = _query_map(nav_url)
            needs_pin = False
            for key, value in state.session_query.items():
                if not key:
                    continue
                if target_query.get(key) != value:
                    target_query[key] = value
                    needs_pin = True
            if needs_pin:
                return _with_query(nav_url, target_query)
            return nav_url
        except Exception:
            return nav_url

    def _stuck_recovery(
        self,
        *,
        prompt: str,
        url: str,
        step_index: int,
        state: AgentState,
        allowed: set[str],
    ) -> tuple[Dict[str, Any] | None, bool, str, str]:
        def allow(tool: str) -> bool:
            return (not allowed) or (tool in allowed)

        if state.last_action_element_id:
            state.blocklist.element_ids = _dedupe_keep_order(
                state.blocklist.element_ids + [state.last_action_element_id],
                MAX_PENDING_ELEMENTS,
            )
            state.blocklist.until_step = max(state.blocklist.until_step, int(step_index) + 2)

        last_action_type = str(state.last_action_sig or "").split("|", 1)[0].strip().lower()
        if (
            last_action_type in {"clickaction", "typeaction", "selectdropdownoptionaction"}
            and allow("browser.wait")
            and int(state.counters.stall_count or 0) <= 3
        ):
            return (
                {"type": "WaitAction", "time_seconds": 1.2},
                False,
                "",
                "wait_for_async_side_effect",
            )

        current_url = str(url or "").strip().lower()
        if allow("browser.back") and current_url not in {"", "about:blank"}:
            return (
                {"type": "NavigateAction", "go_back": True, "go_forward": False},
                False,
                "",
                "back_navigation",
            )
        if allow("browser.scroll"):
            return (
                {"type": "ScrollAction", "direction": "down", "up": False, "down": True, "amount": 700},
                False,
                "",
                "last_resort_scroll",
            )
        return (None, False, "", "")

    def _fallback_from_ranked_candidates(
        self,
        *,
        prompt: str,
        ranked_candidates: List[Candidate],
        state: AgentState,
        allowed: set[str],
        current_url: str,
    ) -> Dict[str, Any] | None:
        def allow(tool: str) -> bool:
            return (not allowed) or (tool in allowed)

        focus_region_id = str(state.focus_region.region_id or "").strip()
        focus_region_context = _norm_ws(state.focus_region.region_context)
        focus_candidate_ids = set(state.focus_region.candidate_ids)
        local_ranked: List[Candidate] = []
        escape_ranked: List[Candidate] = []
        global_ranked: List[Candidate] = []
        for cand in ranked_candidates:
            same_region = False
            if focus_region_id and cand.region_id and cand.region_id == focus_region_id:
                same_region = True
            elif focus_region_id and focus_region_id in set(cand.region_ancestor_ids or []):
                same_region = True
            elif focus_region_context and _norm_ws(cand.context) == focus_region_context:
                same_region = True
            elif focus_candidate_ids and cand.id in focus_candidate_ids:
                same_region = True
            if same_region:
                blob = " ".join([cand.text, cand.field_hint, cand.group_label, cand.context]).lower()
                if str(cand.field_kind or "").strip().lower() != "pager" and re.search(r"\b(save|submit|apply|continue|confirm|close|done|cancel|back)\b", blob):
                    escape_ranked.append(cand)
                else:
                    local_ranked.append(cand)
            else:
                global_ranked.append(cand)
        visual_hints = set(state.memory.visual_element_hints or [])
        if visual_hints:
            for cand in local_ranked + escape_ranked + global_ranked:
                if cand.id not in visual_hints or cand.id in state.blocklist.element_ids:
                    continue
                if cand.role in {"link", "button"} and allow("browser.click"):
                    return {"type": "ClickAction", "selector": cand.selector, "_element_id": cand.id}
                if cand.role == "input" and allow("browser.type"):
                    text = self._infer_input_text(prompt=prompt, candidate=cand)
                    if text:
                        return {"type": "TypeAction", "selector": cand.selector, "text": text[:220], "_element_id": cand.id}
                if cand.role == "select" and allow("browser.select"):
                    text = self._infer_input_text(prompt=prompt, candidate=cand)
                    if text:
                        return {"type": "SelectDropDownOptionAction", "selector": cand.selector, "text": text, "_element_id": cand.id}

        ordered_candidates = local_ranked[:10] + escape_ranked[:8] + global_ranked[:10]
        for cand in ordered_candidates:
            if cand.id in state.blocklist.element_ids:
                continue
            if cand.role in {"link", "button"} and allow("browser.click"):
                return {"type": "ClickAction", "selector": cand.selector, "_element_id": cand.id}
            if cand.role == "input" and allow("browser.type"):
                text = self._infer_input_text(prompt=prompt, candidate=cand)
                if text:
                    return {"type": "TypeAction", "selector": cand.selector, "text": text[:220], "_element_id": cand.id}
            if cand.role == "select" and allow("browser.select"):
                text = self._infer_input_text(prompt=prompt, candidate=cand)
                if text:
                    return {"type": "SelectDropDownOptionAction", "selector": cand.selector, "text": text, "_element_id": cand.id}
        return None

    def _action_signature(self, action: Dict[str, Any]) -> str:
        t = str(action.get("type") or "").strip()
        selector = ""
        if isinstance(action.get("selector"), dict):
            try:
                selector = json.dumps(action.get("selector"), ensure_ascii=True, sort_keys=True)[:220]
            except Exception:
                selector = str(action.get("selector"))
        url = str(action.get("url") or "")[:220]
        txt = str(action.get("text") or "")[:100]
        return f"{t}|{selector}|{url}|{txt}"

    def _remember_form_progress_from_action(
        self,
        *,
        prompt: str,
        action: Dict[str, Any],
        ranked_candidates: List[Candidate],
        state: AgentState,
    ) -> None:
        if not isinstance(action, dict):
            return
        action_type = str(action.get("type") or "").strip()
        target = self._candidate_for_action(action=action, ranked_candidates=ranked_candidates)
        if target is not None and target.context and action_type in {"TypeAction", "ClickAction", "SelectDropDownOptionAction"}:
            self._remember_active_group(target=target, ranked_candidates=ranked_candidates, state=state)
            self._record_constraint_progress(prompt=prompt, action=action, target=target, state=state)
        element_id = str(action.get("_element_id") or action.get("element_id") or "").strip()
        if action_type == "TypeAction":
            cand_id = element_id
            sel = action.get("selector") if isinstance(action.get("selector"), dict) else None
            sel_sig = self._selector_signature(sel if isinstance(sel, dict) else None)
            typed_value = self._normalized_field_value(action.get("text"))
            if cand_id:
                state.form_progress.typed_candidate_ids = _dedupe_keep_order(
                    state.form_progress.typed_candidate_ids + [cand_id],
                    MAX_PENDING_ELEMENTS * 2,
                )
                if typed_value:
                    state.form_progress.typed_values_by_candidate[cand_id] = typed_value
            if sel_sig:
                state.form_progress.typed_selector_sigs = _dedupe_keep_order(
                    state.form_progress.typed_selector_sigs + [sel_sig],
                    MAX_PENDING_ELEMENTS * 2,
                )
                if typed_value:
                    state.form_progress.typed_values_by_selector[sel_sig] = typed_value
            return
        if action_type == "ClickAction":
            cand_id = element_id
            if cand_id and cand_id in state.form_progress.typed_candidate_ids:
                state.form_progress.typed_candidate_ids = [
                    x for x in state.form_progress.typed_candidate_ids if x != cand_id
                ]
            return
        if action_type == "SelectDropDownOptionAction":
            return
