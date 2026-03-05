from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Callable, Dict, List, Literal
from urllib.parse import urljoin, urlsplit, urlunsplit

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

META_TOOLS = {
    "META.SOLVE_POPUPS",
    "META.REPLAN",
    "META.EXTRACT_LINKS",
    "META.EXTRACT_FACTS",
    "META.SELECT_NEXT_TARGET",
    "META.ESCALATE",
    "META.SET_MODE",
    "META.MARK_PROGRESS",
}

MAX_INTERNAL_META_STEPS = 3
MAX_FACTS = 32
MAX_CHECKPOINTS = 20
MAX_PENDING_URLS = 30
MAX_PENDING_ELEMENTS = 50
MAX_VISITED_URLS = 80
MAX_PAGE_HASHES = 80
MAX_STR = 400


def _norm_ws(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _dom_digest(html: str) -> str:
    try:
        data = str(html or "").encode("utf-8", errors="ignore")
        return hashlib.sha256(data).hexdigest()[:16]
    except Exception:
        return ""


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]{2,}", str(text or "").lower())}


def _safe_url(raw: str, base: str = "") -> str:
    txt = str(raw or "").strip()
    if not txt:
        return ""
    if txt.startswith("http://") or txt.startswith("https://"):
        return txt
    if txt.startswith("/"):
        return urljoin(base or "", txt)
    if "." in txt and " " not in txt and "/" not in txt:
        return f"https://{txt}"
    return txt


def _first_host_in_text(text: str) -> str:
    hits = re.findall(r"\b[a-z0-9.-]+\.[a-z]{2,}\b", str(text or "").lower())
    return hits[0] if hits else ""


def _candidate_text(*parts: Any) -> str:
    for item in parts:
        if isinstance(item, str):
            cleaned = _norm_ws(item)
            if cleaned:
                return cleaned[:MAX_STR]
    return ""


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
    counters: AgentCounters = Field(default_factory=AgentCounters)
    blocklist: AgentBlocklist = Field(default_factory=AgentBlocklist)
    last_url: str = ""
    last_dom_hash: str = ""
    last_action_sig: str = ""
    last_action_element_id: str = ""
    escalated_once: bool = False

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
        self.blocklist.element_ids = _dedupe_keep_order(self.blocklist.element_ids, MAX_PENDING_ELEMENTS)
        self.last_url = str(self.last_url or "")[:MAX_STR]
        self.last_dom_hash = str(self.last_dom_hash or "")[:64]
        self.last_action_sig = str(self.last_action_sig or "")[:MAX_STR]
        self.last_action_element_id = str(self.last_action_element_id or "")[:120]
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
        captcha_suspected = any(token in lower for token in ("captcha", "recaptcha", "hcaptcha", "cloudflare challenge"))
        login_form = ("type=\"password\"" in lower) or ("signin" in text) or ("log in" in text) or ("sign in" in text)
        search_box = ("type=\"search\"" in lower) or ("placeholder=\"search" in lower) or ("search" in text[:1600])
        product_cards = text.count("add to cart") >= 1 or text.count("price") >= 3 or "product" in text
        results_list = text.count("result") >= 2 or ("<li" in lower and lower.count("<li") >= 6)
        pricing_table = ("pricing" in text and "$" in text) or ("<table" in lower and "price" in text)
        error_page = any(token in text for token in ("error 404", "error 500", "not found", "access denied", "site can’t be reached", "temporarily unavailable"))
        url_changed = bool(state.last_url and str(url or "") != state.last_url)
        dom_changed = bool(state.last_dom_hash and digest != state.last_dom_hash)
        repeat_hint = int(state.counters.repeat_action_count or 0)
        stall_suggested = int(state.counters.stall_count or 0)
        if not url_changed and not dom_changed:
            stall_suggested += 1
        loop_level = "none"
        if repeat_hint >= 1 or stall_suggested >= 1:
            loop_level = "low"
        if repeat_hint >= 2 or stall_suggested >= 2:
            loop_level = "high"
        return {
            "cookie_banner": bool(cookie_banner),
            "modal_dialog": bool(modal_dialog),
            "captcha_suspected": bool(captcha_suspected),
            "login_form": bool(login_form),
            "search_box": bool(search_box),
            "product_cards": bool(product_cards),
            "results_list": bool(results_list),
            "pricing_table": bool(pricing_table),
            "error_page": bool(error_page),
            "url_changed": bool(url_changed),
            "dom_changed": bool(dom_changed),
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
    bbox: Dict[str, float] | None = None

    def as_obs(self) -> Dict[str, Any]:
        out = {
            "id": self.id,
            "role": self.role,
            "type": self.type,
            "text": self.text[:140],
            "href": self.href[:240] if self.href else "",
            "context": self.context[:220] if self.context else "",
            "selector": self.selector,
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
        out: List[Candidate] = []
        for node in soup.select("a,button,input,select,textarea,[role='button'],[role='link']"):
            try:
                attrs = node.attrs if isinstance(getattr(node, "attrs", None), dict) else {}
                tag = str(getattr(node, "name", "") or "").lower()
                role = str(attrs.get("role") or "").strip().lower()
                role_name = self._role_name(tag, role)
                if not role_name:
                    continue
                text = _norm_ws(node.get_text(" ", strip=True))
                if not text:
                    text = _candidate_text(attrs.get("aria-label"), attrs.get("placeholder"), attrs.get("value"))
                href = _safe_url(str(attrs.get("href") or ""), base=str(url or ""))
                dom_path = self._dom_path(node)
                selector = self._selector_for(tag=tag, attrs=attrs, text=text, href=href, dom_path=dom_path)
                if not isinstance(selector, dict):
                    continue
                stable_id = self._stable_id(attrs=attrs, selector=selector, text=text, href=href, dom_path=dom_path)
                context = self._context(node)
                out.append(
                    Candidate(
                        id=stable_id,
                        role=role_name,
                        type=tag or role_name,
                        text=text[:160],
                        href=href[:320],
                        context=context[:260],
                        selector=selector,
                        dom_path=dom_path[:260],
                        bbox=None,
                    )
                )
            except Exception:
                continue
        dedup: Dict[str, Candidate] = {}
        for cand in out:
            dedup[cand.id] = cand
        return list(dedup.values())[:220]

    def _role_name(self, tag: str, role: str) -> str:
        if tag == "a" or role == "link":
            return "link"
        if tag == "button" or role == "button":
            return "button"
        if tag in {"input", "textarea"}:
            return "input"
        if tag == "select":
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

    def _selector_for(self, *, tag: str, attrs: Dict[str, Any], text: str, href: str, dom_path: str) -> Dict[str, Any]:
        node_id = _norm_ws(attrs.get("id"))
        if node_id:
            return {"type": "attributeValueSelector", "attribute": "id", "value": node_id, "case_sensitive": False}
        node_name = _norm_ws(attrs.get("name"))
        if node_name and tag in {"input", "textarea", "select"}:
            return {"type": "attributeValueSelector", "attribute": "name", "value": node_name, "case_sensitive": False}
        if href and tag == "a":
            return {"type": "attributeValueSelector", "attribute": "href", "value": href, "case_sensitive": False}
        if text and tag in {"button", "a"}:
            clean = text.replace('"', "'")[:120]
            xpath = f"//{tag}[contains(normalize-space(.), \"{clean}\")]"
            return {"type": "xpathSelector", "value": xpath, "case_sensitive": False}
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


class CandidateRanker:
    def rank(
        self,
        *,
        task: str,
        mode: str,
        flags: Dict[str, Any],
        candidates: List[Candidate],
        state: AgentState,
        top_k: int = 30,
    ) -> List[Candidate]:
        task_tokens = _tokenize(task)
        blocked = set(state.blocklist.element_ids if state.blocklist.until_step > 0 else [])
        scored: List[tuple[float, Candidate]] = []
        for cand in candidates:
            if cand.id in blocked:
                continue
            blob = " ".join([cand.text, cand.href, cand.context]).lower()
            score = 0.0
            if cand.role == "button":
                score += 3.2
            elif cand.role == "link":
                score += 2.4
            elif cand.role == "input":
                score += 2.0
            elif cand.role == "select":
                score += 1.8
            if cand.href:
                score += 0.5
            overlap = len(task_tokens.intersection(_tokenize(blob)))
            score += min(6.0, overlap * 1.1)
            if mode == "POPUP":
                if any(k in blob for k in ("accept", "reject", "agree", "close", "dismiss", "continue")):
                    score += 6.0
            if mode == "EXTRACT" and cand.role == "link":
                score += 0.8
            if bool(flags.get("search_box")) and cand.role == "input" and "search" in blob:
                score += 3.0
            scored.append((score, cand))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [cand for _, cand in scored[: max(1, int(top_k or 30))]]


class ObsBuilder:
    def build_text_ir(self, snapshot_html: str) -> Dict[str, Any]:
        html = str(snapshot_html or "")
        if not html:
            return {"visible_text": "", "headings": []}
        if BeautifulSoup is None:
            cleaned = _norm_ws(re.sub(r"<[^>]+>", " ", html))
            return {"visible_text": cleaned[:2600], "headings": []}
        try:
            soup = BeautifulSoup(html, "lxml")
            for node in soup(["script", "style", "noscript"]):
                try:
                    node.decompose()
                except Exception:
                    pass
            headings: List[str] = []
            for h in soup.find_all(["h1", "h2", "h3"], limit=12):
                txt = _norm_ws(h.get_text(" ", strip=True))
                if txt:
                    headings.append(txt[:140])
            text = _norm_ws(soup.get_text(" ", strip=True))
            return {
                "visible_text": text[:2600],
                "headings": _dedupe_keep_order(headings, 10),
            }
        except Exception:
            cleaned = _norm_ws(re.sub(r"<[^>]+>", " ", html))
            return {"visible_text": cleaned[:2600], "headings": []}

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
    ) -> Dict[str, Any]:
        active = {}
        if state.plan.active_id:
            for sg in state.plan.subgoals:
                if sg.id == state.plan.active_id:
                    active = {"id": sg.id, "text": sg.text, "status": sg.status}
                    break
        return {
            "task_id": str(task_id or ""),
            "prompt": str(prompt or "")[:MAX_STR],
            "step_index": int(step_index),
            "url": str(url or "")[:MAX_STR],
            "mode": mode,
            "flags": flags,
            "active_subgoal": active,
            "frontier": {
                "pending_urls": state.frontier.pending_urls[:8],
                "pending_elements": state.frontier.pending_elements[:8],
            },
            "counters": state.counters.model_dump(),
            "memory": {
                "facts": state.memory.facts[:6],
                "checkpoints": state.memory.checkpoints[-6:],
            },
            "text_ir": {
                "visible_text": str(text_ir.get("visible_text") or "")[:2200],
                "headings": text_ir.get("headings") if isinstance(text_ir.get("headings"), list) else [],
            },
            "candidates": [cand.as_obs() for cand in candidates[:30]],
        }


class Router:
    def next_mode(self, *, step_index: int, state: AgentState, flags: Dict[str, Any]) -> tuple[str, str]:
        if state.mode == "DONE":
            return "DONE", "already_done"
        if bool(flags.get("captcha_suspected")):
            return "STUCK", "captcha_detected"
        if bool(flags.get("cookie_banner")) or bool(flags.get("modal_dialog")):
            return "POPUP", "popup_detected"
        if str(flags.get("loop_level") or "none") == "high":
            return "STUCK", "loop_high"
        if int(state.counters.stall_count or 0) >= 2 or int(state.counters.repeat_action_count or 0) >= 2:
            return "STUCK", "stalled_or_repeating"
        if int(step_index) == 0:
            return "BOOTSTRAP", "initial_step"
        if state.mode in {"BOOTSTRAP", "PLAN"}:
            return "NAV", "progress_after_bootstrap"
        if state.mode in {"NAV", "POPUP"}:
            if bool(flags.get("product_cards")) or bool(flags.get("results_list")) or bool(flags.get("pricing_table")):
                return "EXTRACT", "extractable_content_visible"
            return "NAV", "continue_navigation"
        if state.mode == "EXTRACT":
            if len(state.memory.facts) >= 2:
                return "SYNTH", "facts_collected"
            return "EXTRACT", "collecting_facts"
        if state.mode == "SYNTH":
            return "REPORT", "synthesis_ready"
        if state.mode == "REPORT":
            return "DONE", "report_emitted"
        return "PLAN", "default_plan"


class Skills:
    def solve_popups(self, *, candidates: List[Candidate]) -> Dict[str, Any]:
        keys = {"accept", "agree", "reject", "close", "dismiss", "continue", "ok"}
        hits = []
        for cand in candidates:
            blob = " ".join([cand.text, cand.context]).lower()
            if cand.role in {"button", "link"} and any(k in blob for k in keys):
                hits.append(cand.id)
        return {"pending_elements": hits[:8], "checkpoint": f"popup_candidates={len(hits)}"}

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
        facts: List[str] = []
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
        tool_name: str,
        args: Dict[str, Any],
        state: AgentState,
        prompt: str,
        text_ir: Dict[str, Any],
        candidates: List[Candidate],
        url: str,
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
        if mode == "EXTRACT":
            return {"type": "meta", "name": "META.EXTRACT_FACTS", "arguments": {"schema": "generic"}}, {"source": "deterministic"}
        if mode == "SYNTH":
            return {"type": "meta", "name": "META.EXTRACT_FACTS", "arguments": {"schema": "generic"}}, {"source": "deterministic"}
        if mode == "REPORT":
            facts = policy_obs.get("memory", {}).get("facts") if isinstance(policy_obs.get("memory"), dict) else []
            fact = _candidate_text((facts or [""])[0] if isinstance(facts, list) and facts else "")
            content = fact or "Task appears complete."
            return {"type": "final", "done": True, "content": content}, {"source": "deterministic"}

        system = (
            "You are an operator policy. Return strict JSON object only. "
            "Pick exactly one: browser tool_call, meta_tool, or final. "
            "Never emit more than one browser action per step."
        )
        user = {
            "mode": mode,
            "allowed_browser_tools": sorted(list(allowed_tools)) if allowed_tools else [],
            "allowed_meta_tools": sorted(list(META_TOOLS)),
            "policy_obs": policy_obs,
            "output_schema": {
                "browser": {
                    "type": "browser",
                    "tool_call": {"name": "browser.click", "arguments": {}},
                    "reasoning": "short string",
                },
                "meta": {
                    "type": "meta",
                    "meta_tool": {"name": "META.REPLAN", "arguments": {}},
                    "reasoning": "short string",
                },
                "final": {
                    "type": "final",
                    "done": True,
                    "content": "final answer",
                    "reasoning": "short string",
                },
            },
        }
        model = plan_model_name if (mode == "PLAN" or mode == "STUCK") else model_name
        try:
            raw = self.llm_call(
                task_id=str(task_id or "local"),
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user, ensure_ascii=True)},
                ],
                temperature=0.1,
                max_tokens=280,
            )
            content = str(
                (((raw or {}).get("choices") or [{}])[0].get("message", {}) or {}).get("content") or ""
            )
            obj = self._parse_json(content)
            usage = (raw or {}).get("usage") if isinstance((raw or {}).get("usage"), dict) else {}
            return self._normalize_decision(obj, allowed_tools), {
                "source": "llm",
                "usage": {
                    "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                    "completion_tokens": int(usage.get("completion_tokens") or 0),
                    "total_tokens": int(usage.get("total_tokens") or 0),
                },
                "model": str((raw or {}).get("model") or model),
            }
        except Exception:
            return self._fallback(prompt=prompt, mode=mode, policy_obs=policy_obs, allowed_tools=allowed_tools), {"source": "fallback"}

    def _parse_json(self, content: str) -> Dict[str, Any]:
        raw = str(content or "").strip()
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(raw[start : end + 1])
            if isinstance(obj, dict):
                return obj
        raise ValueError("invalid_json_policy_output")

    def _normalize_decision(self, obj: Dict[str, Any], allowed_tools: set[str]) -> Dict[str, Any]:
        t = str(obj.get("type") or "").strip().lower()
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
            if name in META_TOOLS:
                return {
                    "type": "meta",
                    "name": name,
                    "arguments": mt.get("arguments") if isinstance(mt.get("arguments"), dict) else {},
                    "reasoning": _candidate_text(obj.get("reasoning")),
                }
        if t == "browser":
            tc = obj.get("tool_call") if isinstance(obj.get("tool_call"), dict) else {}
            name = str(tc.get("name") or "").strip().lower()
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
        first = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
        if mode == "STUCK":
            if allow("browser.scroll"):
                return {"type": "browser", "tool_call": {"name": "browser.scroll", "arguments": {"direction": "down", "amount": 650}}}
            if allow("browser.back"):
                return {"type": "browser", "tool_call": {"name": "browser.back", "arguments": {}}}
            if allow("browser.navigate"):
                host = _first_host_in_text(prompt)
                if host:
                    return {"type": "browser", "tool_call": {"name": "browser.navigate", "arguments": {"url": f"https://{host}"}}}
        if first and allow("browser.click"):
            sel = first.get("selector") if isinstance(first.get("selector"), dict) else None
            if sel:
                return {"type": "browser", "tool_call": {"name": "browser.click", "arguments": {"selector": sel}}}
        if allow("browser.scroll"):
            return {"type": "browser", "tool_call": {"name": "browser.scroll", "arguments": {"direction": "down", "amount": 600}}}
        return {"type": "final", "done": True, "content": "No safe browser action available from allowed_tools."}


class FSMOperator:
    def __init__(self, llm_call: Callable[..., Dict[str, Any]]) -> None:
        self.flags = FlagDetector()
        self.extractor = CandidateExtractor()
        self.ranker = CandidateRanker()
        self.obs_builder = ObsBuilder()
        self.router = Router()
        self.skills = Skills()
        self.meta = MetaToolExecutor(self.skills)
        self.policy = Policy(llm_call=llm_call)

    def run(self, *, payload: Dict[str, Any], model_override: str = "") -> Dict[str, Any]:
        prompt = str(payload.get("prompt") or payload.get("task_prompt") or "")
        url = str(payload.get("url") or "")
        html = str(payload.get("snapshot_html") or "")
        task_id = str(payload.get("task_id") or "")
        step_index = int(payload.get("step_index") or 0)
        include_reasoning = bool(payload.get("include_reasoning"))
        history = payload.get("history") if isinstance(payload.get("history"), list) else []
        state = AgentState.from_state_in(payload.get("state_in"), prompt=prompt)
        allowed = self._normalize_allowed(payload.get("allowed_tools"))
        mode_in = state.mode

        flags = self.flags.detect(snapshot_html=html, url=url, history=history, state=state)
        state.counters.stall_count = int(flags.get("stall_count_suggested") or 0)
        state.visited.urls = _dedupe_keep_order(state.visited.urls + ([url] if url else []), MAX_VISITED_URLS)
        if url:
            state.visited.page_hashes[url[:MAX_STR]] = str(flags.get("dom_hash") or "")[:64]

        routed_mode, route_reason = self.router.next_mode(step_index=step_index, state=state, flags=flags)
        state.mode = routed_mode

        text_ir = self.obs_builder.build_text_ir(html)
        candidates = self.extractor.extract(snapshot_html=html, url=url)
        ranked = self.ranker.rank(task=prompt, mode=state.mode, flags=flags, candidates=candidates, state=state, top_k=30)

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
        )

        meta_exec_trace: List[str] = []
        policy_reasoning = ""
        usage_payload: Dict[str, Any] = {}
        model_name = str(model_override or "gpt-4o-mini")
        plan_model_name = str(model_override or "gpt-4o")
        chosen_action: Dict[str, Any] | None = None
        done = False
        content = ""
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
        if pre_done:
            done = True
            content = pre_content
            state.mode = "DONE"
        elif pre_action is not None:
            chosen_action = pre_action

        if not done and chosen_action is None and state.mode == "STUCK":
            chosen_action, done, content, stuck_note = self._stuck_recovery(
                prompt=prompt,
                url=url,
                step_index=step_index,
                state=state,
                allowed=allowed,
            )
            if stuck_note:
                meta_exec_trace.append(f"STUCK_RECOVERY:{stuck_note}")
        elif not done and chosen_action is None:
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
                    usage_payload = usage
                policy_reasoning = _candidate_text(decision.get("reasoning"), policy_reasoning)
                dtype = str(decision.get("type") or "").strip().lower()
                if dtype == "final":
                    done = True
                    content = _candidate_text(decision.get("content"), "Task completed.")
                    state.mode = "DONE"
                    break
                if dtype == "browser":
                    chosen_action = self._browser_action_from_tool_call(
                        tool_call=decision.get("tool_call") if isinstance(decision.get("tool_call"), dict) else {},
                        ranked_candidates=ranked,
                        state=state,
                        allowed=allowed,
                    )
                    if chosen_action is not None:
                        break
                if dtype == "meta":
                    if state.counters.meta_steps_used >= MAX_INTERNAL_META_STEPS:
                        break
                    meta_name = str(decision.get("name") or "")
                    meta_args = decision.get("arguments") if isinstance(decision.get("arguments"), dict) else {}
                    result = self.meta.execute(
                        tool_name=meta_name,
                        args=meta_args,
                        state=state,
                        prompt=prompt,
                        text_ir=text_ir,
                        candidates=ranked,
                        url=url,
                    )
                    state.counters.meta_steps_used += 1
                    meta_exec_trace.append(meta_name)
                    if meta_name == "META.SET_MODE":
                        routed_mode, route_reason = state.mode, f"meta_set_mode:{state.mode}"
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
                    )
                    if not bool(result.get("ok")):
                        break
                    continue
                break

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
                    allowed=allowed,
                )

        actions: List[Dict[str, Any]] = []
        browser_tool_name = ""
        if not done and chosen_action is not None:
            actions = [chosen_action]
            browser_tool_name = str(self._tool_name_for_action(chosen_action))
            sig = self._action_signature(chosen_action)
            state.counters.repeat_action_count = state.counters.repeat_action_count + 1 if sig == state.last_action_sig else 0
            state.last_action_sig = sig
            state.last_action_element_id = str(chosen_action.get("_element_id") or "")[:120]
        else:
            state.last_action_sig = ""
            state.last_action_element_id = ""

        state.last_url = str(url or "")[:MAX_STR]
        state.last_dom_hash = str(flags.get("dom_hash") or "")[:64]
        state.blocklist.until_step = max(state.blocklist.until_step, int(step_index) + 1 if state.blocklist.element_ids else state.blocklist.until_step)
        if state.mode == "REPORT" and done:
            state.mode = "DONE"
        mode_out = state.mode

        reasoning = None
        if include_reasoning:
            flag_brief = ",".join(
                [
                    name
                    for name in ("cookie_banner", "modal_dialog", "captcha_suspected", "error_page", "url_changed", "dom_changed")
                    if bool(flags.get(name))
                ]
            ) or "none"
            reasoning = (
                f"mode_in={mode_in}; mode_out={mode_out}; router={route_reason}; "
                f"flags={flag_brief}; meta_steps={meta_exec_trace or ['none']}; "
                f"browser_tool={browser_tool_name or 'none'}; policy={policy_reasoning or 'n/a'}"
            )[:600]

        out: Dict[str, Any] = {
            "protocol_version": "1.0",
            "actions": actions[:1],  # Hard guarantee: max 1 browser action in stateful mode.
            "done": bool(done),
            "content": _candidate_text(content) if done else None,
            "state_out": state.to_state_out(),
        }
        if isinstance(reasoning, str) and reasoning:
            out["reasoning"] = reasoning
        usage = usage_payload.get("usage") if isinstance(usage_payload.get("usage"), dict) else None
        if usage:
            out["usage"] = usage
            out["total_tokens"] = int(usage.get("total_tokens") or 0)
        if usage_payload.get("model"):
            out["model"] = str(usage_payload.get("model"))
        return out

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
        prompt_l = str(prompt or "").lower()
        current_host = str(urlsplit(str(url or "")).hostname or "").lower()
        current_path = str(urlsplit(str(url or "")).path or "").lower()
        host_hint = _first_host_in_text(prompt_l)
        visited_joined = " | ".join(state.visited.urls).lower()
        has_studio_seen = "studio" in visited_joined
        has_login_seen = ("login" in visited_joined) or ("sign-in" in visited_joined) or ("signin" in visited_joined)
        auth_url_hint = ("login" in current_path) or ("sign-in" in current_path) or ("signin" in current_path)

        def allow(name: str) -> bool:
            return (not allowed) or (name in allowed)

        # Short deterministic completion path for "wait and finish" tasks.
        if ("wait" in prompt_l or "espera" in prompt_l) and int(step_index) >= 1:
            return None, True, "Wait completed. Task done.", "wait_then_done"
        if ("wait" in prompt_l or "espera" in prompt_l) and int(step_index) == 0 and allow("browser.wait"):
            return {"type": "WaitAction", "time_seconds": 1.0}, False, "", "wait_first_step"

        # Navigate to task host when prompt contains an explicit domain.
        if host_hint and allow("browser.navigate"):
            if not current_host or (current_host != host_hint and not current_host.endswith(f".{host_hint}")):
                return (
                    {"type": "NavigateAction", "url": f"https://{host_hint}/", "go_back": False, "go_forward": False},
                    False,
                    "",
                    "navigate_host_hint",
                )

        # Guided click for common navigation intents.
        if (
            "studio" in prompt_l
            and host_hint
            and allow("browser.navigate")
            and not has_studio_seen
            and not auth_url_hint
        ):
            if "studio" not in current_path and (current_host == host_hint or current_host.endswith(f".{host_hint}")):
                return (
                    {
                        "type": "NavigateAction",
                        "url": f"https://app.{host_hint}/studio",
                        "go_back": False,
                        "go_forward": False,
                    },
                    False,
                    "",
                    "navigate_studio_hint",
                )
        if (
            any(k in prompt_l for k in ("login", "sign in", "sign-in"))
            and host_hint
            and allow("browser.navigate")
            and (has_studio_seen or "studio" not in prompt_l)
            and not has_login_seen
        ):
            if ("login" not in current_path and "sign-in" not in current_path and "signin" not in current_path) and (
                current_host == host_hint or current_host.endswith(f".{host_hint}")
            ):
                return (
                    {
                        "type": "NavigateAction",
                        "url": f"https://app.{host_hint}/auth/sign-in?redirectURL=%2F",
                        "go_back": False,
                        "go_forward": False,
                    },
                    False,
                    "",
                    "navigate_login_hint",
                )

        if allow("browser.click"):
            for kw, note in (
                ("studio", "open_studio"),
                ("login", "open_login"),
                ("sign in", "open_signin"),
                ("sign-in", "open_signin"),
            ):
                if kw not in prompt_l:
                    continue
                cand = self._pick_candidate_by_keywords(ranked_candidates, include=[kw], roles={"link", "button"})
                if cand is not None:
                    return {"type": "ClickAction", "selector": cand.selector, "_element_id": cand.id}, False, "", note

        # Deterministic credential progression on auth pages.
        if bool(flags.get("login_form")) or auth_url_hint:
            typed_values = self._typed_values_from_history(history)
            typed_lower = {v.lower() for v in typed_values}
            emails, passwords = self._extract_credentials(prompt)

            email_next = next((e for e in emails if e.lower() not in typed_lower), "")
            pwd_next = next((p for p in passwords if p.lower() not in typed_lower), "")

            if email_next and allow("browser.type"):
                cand = self._pick_candidate_by_keywords(
                    ranked_candidates,
                    include=["email", "user", "username", "mail"],
                    roles={"input"},
                )
                if cand is not None:
                    return {"type": "TypeAction", "selector": cand.selector, "text": email_next, "_element_id": cand.id}, False, "", "type_email"
                return {
                    "type": "TypeAction",
                    "selector": {"type": "attributeValueSelector", "attribute": "id", "value": "email", "case_sensitive": False},
                    "text": email_next,
                }, False, "", "type_email_fallback"

            if pwd_next and allow("browser.type"):
                cand = self._pick_candidate_by_keywords(
                    ranked_candidates,
                    include=["password", "pass"],
                    roles={"input"},
                )
                if cand is not None:
                    return {"type": "TypeAction", "selector": cand.selector, "text": pwd_next, "_element_id": cand.id}, False, "", "type_password"
                return {
                    "type": "TypeAction",
                    "selector": {"type": "attributeValueSelector", "attribute": "id", "value": "password", "case_sensitive": False},
                    "text": pwd_next,
                }, False, "", "type_password_fallback"

            if allow("browser.click"):
                cand = self._pick_candidate_by_keywords(
                    ranked_candidates,
                    include=["login", "sign in", "continue", "submit"],
                    roles={"button", "link"},
                )
                if cand is not None:
                    return {"type": "ClickAction", "selector": cand.selector, "_element_id": cand.id}, False, "", "submit_login"

        return None, False, "", ""

    def _pick_candidate_by_keywords(
        self,
        candidates: List[Candidate],
        *,
        include: List[str],
        roles: set[str],
    ) -> Candidate | None:
        terms = [str(t).lower() for t in include if str(t).strip()]
        if not terms:
            return None
        for cand in candidates:
            if roles and cand.role not in roles:
                continue
            blob = " ".join(
                [
                    cand.text,
                    cand.context,
                    cand.href,
                    json.dumps(cand.selector, ensure_ascii=True, sort_keys=True),
                ]
            ).lower()
            if any(term in blob for term in terms):
                return cand
        return None

    def _typed_values_from_history(self, history: List[Dict[str, Any]]) -> List[str]:
        out: List[str] = []
        for item in history:
            if not isinstance(item, dict):
                continue
            action = item.get("action") if isinstance(item.get("action"), dict) else {}
            action_type = str(action.get("type") or "")
            if action_type in {"TypeAction", "FillAction"}:
                val = _candidate_text(action.get("text"), action.get("value"))
                if val:
                    out.append(val)
        return out

    def _extract_credentials(self, prompt: str) -> tuple[List[str], List[str]]:
        text = str(prompt or "")
        emails = _dedupe_keep_order(
            re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text),
            4,
        )
        passwords: List[str] = []
        for hit in re.finditer(r"\b(?:password|pass)\b\s+([^\s,;]+)", text, flags=re.I):
            value = _norm_ws(hit.group(1)).strip(".,;:!?")
            if value and value.lower() not in {"and", "then", "attempt", "retry", "with"}:
                passwords.append(value[:80])
        if not passwords and re.search(r"\bpassword\b", text, flags=re.I):
            passwords.append("password")
        return emails, _dedupe_keep_order(passwords, 5)

    def _normalize_allowed(self, allowed_tools: Any) -> set[str]:
        out: set[str] = set()
        if not isinstance(allowed_tools, list):
            return out
        for item in allowed_tools:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip().lower()
            if name:
                out.add(name)
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

    def _browser_action_from_tool_call(
        self,
        *,
        tool_call: Dict[str, Any],
        ranked_candidates: List[Candidate],
        state: AgentState,
        allowed: set[str],
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
            selector = args.get("selector") if isinstance(args.get("selector"), dict) else None
            element_id = str(args.get("element_id") or "")
            if element_id:
                for cand in ranked_candidates:
                    if cand.id == element_id:
                        selector = cand.selector
                        break
            if selector is None and ranked_candidates:
                for cand in ranked_candidates:
                    if cand.id not in state.blocklist.element_ids:
                        selector = cand.selector
                        element_id = cand.id
                        break
            if not isinstance(selector, dict):
                return None
            out = {"type": "ClickAction", "selector": selector}
            if element_id:
                out["_element_id"] = element_id
            return out

        if name == "browser.navigate":
            nav_url = _safe_url(str(args.get("url") or ""))
            if not nav_url:
                return None
            return {"type": "NavigateAction", "url": nav_url, "go_back": False, "go_forward": False}

        if name == "browser.type":
            selector = args.get("selector") if isinstance(args.get("selector"), dict) else None
            text = _candidate_text(args.get("text"), args.get("value"))
            if not text:
                text = " "
            if not isinstance(selector, dict):
                for cand in ranked_candidates:
                    if cand.role == "input" and cand.id not in state.blocklist.element_ids:
                        selector = cand.selector
                        break
            if not isinstance(selector, dict):
                return None
            return {"type": "TypeAction", "selector": selector, "text": text[:220]}

        if name == "browser.scroll":
            direction = str(args.get("direction") or "down").strip().lower()
            amount = int(args.get("amount") or 650)
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
            selector = args.get("selector") if isinstance(args.get("selector"), dict) else None
            text = _candidate_text(args.get("text"), args.get("value"), "Option")
            if not isinstance(selector, dict):
                for cand in ranked_candidates:
                    if cand.role == "select" and cand.id not in state.blocklist.element_ids:
                        selector = cand.selector
                        break
            if not isinstance(selector, dict):
                return None
            return {"type": "SelectDropDownOptionAction", "selector": selector, "text": text}

        if name == "browser.send_keys":
            keys = _candidate_text(args.get("keys"), "Enter")
            return {"type": "SendKeysIWAAction", "keys": keys}

        if name == "browser.hold_key":
            key = _candidate_text(args.get("key"), "Control")
            return {"type": "HoldKeyAction", "key": key}
        return None

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

        if allow("browser.scroll"):
            return (
                {"type": "ScrollAction", "direction": "down", "up": False, "down": True, "amount": 700},
                False,
                "",
                "block_element_scroll",
            )
        if allow("browser.back"):
            return (
                {"type": "NavigateAction", "go_back": True, "go_forward": False},
                False,
                "",
                "back_navigation",
            )
        if allow("browser.navigate"):
            host_hint = _first_host_in_text(prompt)
            if host_hint:
                return (
                    {"type": "NavigateAction", "url": f"https://{host_hint}/", "go_back": False, "go_forward": False},
                    False,
                    "",
                    "navigate_task_host",
                )
            try:
                parsed = urlsplit(str(url or ""))
                if parsed.scheme and parsed.netloc:
                    root = urlunsplit((parsed.scheme, parsed.netloc, "/", "", ""))
                    return (
                        {"type": "NavigateAction", "url": root, "go_back": False, "go_forward": False},
                        False,
                        "",
                        "navigate_host_root",
                    )
            except Exception:
                pass
        if not state.escalated_once:
            state.escalated_once = True
            state.memory.checkpoints = _dedupe_keep_order(
                state.memory.checkpoints + ["escalate:stuck_recovery"],
                MAX_CHECKPOINTS,
            )
            if allow("browser.wait"):
                return (
                    {"type": "WaitAction", "time_seconds": 1.2},
                    False,
                    "",
                    "escalate_wait",
                )
        best_effort = (
            f"Could not complete the task safely due to repeated no-progress state. "
            f"Current url: {str(url or '')[:220]}"
        )
        state.mode = "DONE"
        return (None, True, best_effort, "final_best_effort")

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
