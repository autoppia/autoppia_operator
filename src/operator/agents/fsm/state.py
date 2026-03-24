from __future__ import annotations

import contextlib
import re
from typing import Any, Literal

from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from .utils import (
    FSM_MODES,
    MAX_CHECKPOINTS,
    MAX_FACTS,
    MAX_HISTORY_SUMMARY_CHARS,
    MAX_PAGE_HASHES,
    MAX_PENDING_ELEMENTS,
    MAX_PENDING_URLS,
    MAX_STR,
    MAX_VISITED_URLS,
    MAX_VISUAL_HINTS,
    MAX_VISUAL_NOTES,
    _dedupe_keep_order,
    _dom_digest,
    _norm_ws,
    _normalize_working_state,
)


class Subgoal(BaseModel):
    id: str
    text: str
    status: Literal["pending", "active", "done", "blocked"] = "pending"


class AgentPlan(BaseModel):
    subgoals: list[Subgoal] = Field(default_factory=list)
    active_id: str = ""


class AgentFrontier(BaseModel):
    pending_urls: list[str] = Field(default_factory=list)
    pending_elements: list[str] = Field(default_factory=list)


class AgentVisited(BaseModel):
    urls: list[str] = Field(default_factory=list)
    page_hashes: dict[str, str] = Field(default_factory=dict)


class AgentMemory(BaseModel):
    facts: list[str] = Field(default_factory=list)
    checkpoints: list[str] = Field(default_factory=list)
    visual_notes: list[str] = Field(default_factory=list)
    visual_element_hints: list[str] = Field(default_factory=list)
    last_vision_signature: str = ""
    history_summary: str = ""
    strategy_summary: str = ""
    prev_page_summary: str = ""
    prev_page_ir_text: str = ""
    prev_candidate_sigs: list[str] = Field(default_factory=list)
    obs_extract_dom_hash: str = ""
    obs_extract_payload: dict[str, Any] = Field(default_factory=dict)
    obs_candidate_hints: list[str] = Field(default_factory=list)
    reasoning_trace: dict[str, str] = Field(default_factory=dict)
    working_state: dict[str, Any] = Field(default_factory=dict)


class AgentFormProgress(BaseModel):
    typed_selector_sigs: list[str] = Field(default_factory=list)
    typed_candidate_ids: list[str] = Field(default_factory=list)
    typed_values_by_selector: dict[str, str] = Field(default_factory=dict)
    typed_values_by_candidate: dict[str, str] = Field(default_factory=dict)
    submit_attempt_sigs: list[str] = Field(default_factory=list)
    active_group_id: str = ""
    active_group_label: str = ""
    active_group_context: str = ""
    active_group_candidate_ids: list[str] = Field(default_factory=list)


class AgentFocusRegion(BaseModel):
    region_id: str = ""
    region_kind: str = ""
    region_label: str = ""
    region_context: str = ""
    candidate_ids: list[str] = Field(default_factory=list)
    recent_region_ids: list[str] = Field(default_factory=list)


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
    recent_effects: list[ProgressEffect] = Field(default_factory=list)
    region_attempts: dict[str, int] = Field(default_factory=dict)
    blocked_regions: list[str] = Field(default_factory=list)
    successful_patterns: list[str] = Field(default_factory=list)
    failed_patterns: list[str] = Field(default_factory=list)
    satisfied_constraints: list[str] = Field(default_factory=list)
    attempted_constraints: dict[str, int] = Field(default_factory=dict)
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
    element_ids: list[str] = Field(default_factory=list)
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
    session_query: dict[str, str] = Field(default_factory=dict)
    score_feedback: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_state_in(cls, state_in: Any, prompt: str) -> AgentState:
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

    def _sanitize(self) -> AgentState:
        self.mode = self.mode if self.mode in FSM_MODES else "BOOTSTRAP"
        self.frontier.pending_urls = _dedupe_keep_order(self.frontier.pending_urls, MAX_PENDING_URLS)
        self.frontier.pending_elements = _dedupe_keep_order(self.frontier.pending_elements, MAX_PENDING_ELEMENTS)
        self.visited.urls = _dedupe_keep_order(self.visited.urls, MAX_VISITED_URLS)
        # Keep deterministic dict size by insertion order.
        if len(self.visited.page_hashes) > MAX_PAGE_HASHES:
            trimmed: dict[str, str] = {}
            for key in list(self.visited.page_hashes.keys())[-MAX_PAGE_HASHES:]:
                trimmed[str(key)[:MAX_STR]] = str(self.visited.page_hashes.get(key) or "")[:64]
            self.visited.page_hashes = trimmed
        else:
            self.visited.page_hashes = {str(k)[:MAX_STR]: str(v)[:64] for k, v in self.visited.page_hashes.items()}
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
        self.memory.reasoning_trace = {str(k)[:40]: _norm_ws(v)[:280] for k, v in self.memory.reasoning_trace.items() if str(k).strip() and _norm_ws(v)}
        self.memory.working_state = _normalize_working_state(self.memory.working_state)
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
        self.form_progress.typed_values_by_selector = {str(k)[:220]: str(v)[:220] for k, v in self.form_progress.typed_values_by_selector.items() if str(k).strip() and str(v).strip()}
        self.form_progress.typed_values_by_candidate = {str(k)[:120]: str(v)[:220] for k, v in self.form_progress.typed_values_by_candidate.items() if str(k).strip() and str(v).strip()}
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
        self.progress.region_attempts = {str(k)[:120]: max(0, int(v or 0)) for k, v in self.progress.region_attempts.items() if str(k).strip()}
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
        self.progress.attempted_constraints = {str(k)[:80]: max(0, int(v or 0)) for k, v in self.progress.attempted_constraints.items() if str(k).strip()}
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
            trimmed_q: dict[str, str] = {}
            for key in list(self.session_query.keys())[:16]:
                trimmed_q[str(key)[:80]] = str(self.session_query.get(key) or "")[:120]
            self.session_query = trimmed_q
        else:
            self.session_query = {str(k)[:80]: str(v)[:120] for k, v in self.session_query.items() if str(k).strip()}
        raw_score_feedback = self.score_feedback if isinstance(self.score_feedback, dict) else {}
        self.score_feedback = (
            {
                "enabled": bool(raw_score_feedback.get("enabled", False)),
                "score": max(0.0, min(1.0, float(raw_score_feedback.get("score", 0.0) or 0.0))),
                "success": bool(raw_score_feedback.get("success", False)),
                "tests_passed": max(0, int(raw_score_feedback.get("tests_passed", 0) or 0)),
                "total_tests": max(0, int(raw_score_feedback.get("total_tests", 0) or 0)),
            }
            if raw_score_feedback
            else {}
        )
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

    def to_state_out(self) -> dict[str, Any]:
        self._sanitize()
        return self.model_dump(mode="json", exclude_none=True)


def _split_prompt_subgoals(prompt: str) -> list[Subgoal]:
    raw = _norm_ws(prompt)
    if not raw:
        return []
    # Do not split on dots to avoid breaking hostnames like autoppia.com.
    parts = [p.strip() for p in re.split(r"\bthen\b|;|,|\band\b", raw, flags=re.I) if p.strip()]
    out: list[Subgoal] = []
    for idx, part in enumerate(parts[:8]):
        out.append(Subgoal(id=f"sg_{idx + 1}", text=part[:MAX_STR], status="pending"))
    return out or [Subgoal(id="sg_1", text=raw[:MAX_STR], status="pending")]


class FlagDetector:
    def detect(
        self,
        *,
        snapshot_html: str,
        url: str,
        history: list[dict[str, Any]],
        state: AgentState,
    ) -> dict[str, Any]:
        html = str(snapshot_html or "")
        lower = html.lower()
        text = self._visible_text(html).lower()
        digest = _dom_digest(html)
        cookie_banner = any(token in lower for token in ("cookie", "cookies", "gdpr", "consent"))
        modal_dialog = any(
            token in lower
            for token in (
                'role="dialog"',
                "aria-modal",
                "<dialog",
                "modal",
                "popup",
                "pop-up",
            )
        )
        interactive_modal_form = self._interactive_modal_form(html)
        captcha_suspected = any(token in lower for token in ("captcha", "recaptcha", "hcaptcha", "cloudflare challenge"))
        login_form = ('type="password"' in lower) or ("signin" in text) or ("log in" in text) or ("sign in" in text)
        search_box = ('type="search"' in lower) or ('placeholder="search' in lower) or bool(re.search(r"(?:id|name|aria-label)=['\"][^'\"]*search", lower))
        product_cards = text.count("add to cart") >= 1 or text.count("watch trailer") >= 2 or text.count("add to watchlist") >= 1 or lower.count("movie-card") >= 2 or lower.count("product-card") >= 2
        results_list = ("search results" in text) or ("results for" in text) or ("result-item" in lower) or ("results-list" in lower) or (search_box and text.count("result") >= 2)
        pricing_table = ("pricing" in text and "$" in text) or ("<table" in lower and "price" in text)
        hard_error_tokens = (
            "error 404",
            "404 page",
            "error 500",
            "internal server error",
            "access denied",
            "site can't be reached",
            "site can't be reached",
            "temporarily unavailable",
            "service unavailable",
            "bad gateway",
        )
        error_page = any(token in text for token in hard_error_tokens)
        if not error_page and "not found" in text:
            error_page = any(
                token in text
                for token in (
                    "page not found",
                    "resource not found",
                    "this page could not be found",
                    "404",
                )
            )
        url_changed = bool(state.last_url and str(url or "") != state.last_url)
        dom_changed = bool(state.last_dom_hash and digest != state.last_dom_hash)
        repeat_hint = int(state.counters.repeat_action_count or 0)
        stall_suggested = int(state.counters.stall_count or 0)
        last_action_type = str(state.last_action_sig or "").split("|", 1)[0].strip().lower()
        no_visual_progress = (not url_changed) and (not dom_changed)
        # A click/type/select can legitimately trigger async backend events without visible DOM changes.
        if no_visual_progress and last_action_type not in {
            "clickaction",
            "typeaction",
            "selectdropdownoptionaction",
        }:
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
                with contextlib.suppress(Exception):
                    tag.decompose()
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
                        if any(
                            token in blob
                            for token in (
                                "email",
                                "password",
                                "username",
                                "user name",
                                "sign in",
                                "log in",
                            )
                        ):
                            return True
                except Exception:
                    continue
        except Exception:
            return False
        return False


__all__ = [name for name in globals() if not name.startswith("__")]
