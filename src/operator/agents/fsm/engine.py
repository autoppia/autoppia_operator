from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from src.operator.agents.fsm import (
    AgentState,
    Candidate,
    CandidateExtractor,
    CandidateRanker,
    FlagDetector,
    ObsBuilder,
)

from .meta_tools import MetaToolExecutor, Router, Skills
from .policy import Policy
from .state import ProgressEffect
from .utils import (
    _REPO_ROOT,
    CONTROL_META_TOOLS,
    MAX_CHECKPOINTS,
    MAX_INTERNAL_META_STEPS,
    MAX_PENDING_ELEMENTS,
    MAX_STR,
    MAX_VISITED_URLS,
    MAX_VISUAL_HINTS,
    _action_type_for_tool,
    _append_jsonl,
    _best_page_evidence,
    _candidate_text,
    _canonical_allowed_tool_name,
    _constraint_value_matches,
    _content_supported_by_page_evidence,
    _dedupe_keep_order,
    _empty_call_breakdown,
    _env_bool,
    _env_int,
    _env_str,
    _looks_like_informational_task,
    _looks_like_vague_informational_answer,
    _merge_usage_dicts,
    _meta_tools,
    _norm_ws,
    _normalize_reasoning_trace,
    _normalize_use_case_info,
    _normalize_working_state,
    _obs_meta_tools,
    _page_context_ready_for_informational_answer,
    _query_map,
    _runtime_page_evidence_ready,
    _safe_url,
    _sanitize_selector,
    _screenshot_available,
    _supported_browser_tool_names,
    _task_constraints,
    _usage_breakdown_template,
    _use_vision,
    _utc_now,
    _vision_signature,
    _with_query,
)


class FSMOperator:
    def __init__(
        self,
        llm_call: Callable[..., dict[str, Any]],
        vision_call: Callable[..., dict[str, Any]] | None = None,
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
            os.getenv(
                "FSM_TRACE_DIR",
                self.debug_dir or str((_REPO_ROOT / "data" / "fsm_traces").resolve()),
            )
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
        skills, _ = self._ensure_support_tools()
        return skills

    @property
    def meta(self) -> MetaToolExecutor:
        _, meta = self._ensure_support_tools()
        return meta

    def _ensure_support_tools(self) -> tuple[Skills, MetaToolExecutor]:
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
        text_ir: dict[str, Any],
        flags: dict[str, Any],
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
                    return (
                        True,
                        best_page_evidence,
                        "Current page contains a concrete answer.",
                    )
            return False, "", "Current page does not yet show a concrete answer."
        if state.mode == "REPORT" and state.memory.facts:
            content = _candidate_text(state.memory.facts[0])
            if content:
                return True, content, "Current evidence is sufficient."
        return False, "", "Current page is not yet sufficient to conclude completion."

    def _obs_extract_signature(self, *, dom_hash: str, url: str) -> str:
        raw = json.dumps(
            {"dom_hash": str(dom_hash or "")[:64], "url": str(url or "")[:240]},
            ensure_ascii=True,
            sort_keys=True,
        )
        return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:16]

    def _should_obs_extract(
        self,
        *,
        state: AgentState,
        flags: dict[str, Any],
        text_ir: dict[str, Any],
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
        text_ir: dict[str, Any],
        candidates: list[Candidate],
    ) -> list[dict[str, Any]]:
        candidate_lines: list[str] = []
        for cand in candidates[:40]:
            label = _candidate_text(
                cand.text,
                cand.field_hint,
                cand.href,
                cand.group_label,
                cand.region_label,
            )
            candidate_lines.append(f"[{cand.id}] role={cand.role} kind={cand.field_kind or cand.region_kind or cand.type} label={label[:140]} context={cand.context[:180]}")
        user_payload = {
            "task": str(prompt or "")[:1200],
            "url": str(url or "")[:400],
            "title": str(text_ir.get("title") or "")[:240],
            "headings": list(text_ir.get("headings") or [])[:10],
            "visible_text": str(text_ir.get("visible_text") or "")[:5000],
            "html_excerpt": str(text_ir.get("html_excerpt") or "")[:6000],
            "candidates": candidate_lines[:40],
        }
        system = "You extract structured browser observations from HTML for a separate policy model. Return strict JSON only with keys: page_kind, summary, regions, forms, facts, primary_candidate_ids. regions must be a list of objects with kind, label, candidate_ids. forms must be a list of objects with label, fields, commit_ids. facts must be short strings for visible metrics, totals, or key-value facts already present on the page. primary_candidate_ids should identify the most likely commit/apply/save/search/open targets visible right now. Do not choose the next action. Do not narrate. Be concise and structural."
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

    def _safe_json_load(self, raw: Any) -> dict[str, Any]:
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

    def _normalize_obs_extract_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
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
        if not (normalized["summary"] or normalized["regions"] or normalized["forms"] or normalized["facts"] or normalized["primary_candidate_ids"]):
            return {}
        return normalized

    def _maybe_extract_observation(
        self,
        *,
        task_id: str,
        prompt: str,
        url: str,
        flags: dict[str, Any],
        state: AgentState,
        text_ir: dict[str, Any],
        candidates: list[Candidate],
    ) -> dict[str, Any]:
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

    def _debug_log(self, task_id: str, payload: dict[str, Any]) -> None:
        if not self.debug_dir and not self.trace_json:
            return
        try:
            base = Path(self.trace_dir if self.trace_json else self.debug_dir).expanduser().resolve()
            safe_task = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(task_id or "task"))[:120] or "task"
            event = {"ts": _utc_now(), **payload}
            _append_jsonl(base / f"{safe_task}.fsm.jsonl", event)
        except Exception:
            return

    def _state_delta(self, before: AgentState, after: AgentState) -> dict[str, Any]:
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
            "facts_count": {
                "from": len(before.memory.facts),
                "to": len(after.memory.facts),
            },
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

    def _active_region_debug(self, *, state: AgentState) -> dict[str, Any]:
        return {
            "region_id": str(state.focus_region.region_id or "")[:120],
            "region_kind": str(state.focus_region.region_kind or "")[:40],
            "region_label": str(state.focus_region.region_label or "")[:160],
            "candidate_count": len(state.focus_region.candidate_ids),
        }

    def _last_effect_label(
        self,
        *,
        history: list[dict[str, Any]],
        flags: dict[str, Any],
        done: bool = False,
    ) -> str:
        if done:
            return "COMPLETED"
        last = history[-1] if history else {}
        if isinstance(last, dict) and (not bool(last.get("exec_ok", True)) or bool(_candidate_text(last.get("error")))):
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
        action: dict[str, Any],
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
        flags: dict[str, Any],
        history: list[dict[str, Any]],
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

    def _no_progress_score(self, *, state: AgentState, flags: dict[str, Any], history: list[dict[str, Any]]) -> int:
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
        history: list[dict[str, Any]],
        state: AgentState,
        flags: dict[str, Any],
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
        expected_effect_met = (
            self._expected_effect_met(
                expected_effect=expected_effect,
                flags=flags,
                history=history,
            )
            if expected_effect
            else False
        )
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
        state.progress.recent_effects = ([*state.progress.recent_effects, effect])[-16:]
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
                    [*state.progress.blocked_regions, region_id],
                    MAX_PENDING_ELEMENTS,
                )
        pattern = ""
        action_type = str(action.get("type") or "").strip()
        if label in {"ADVANCED", "LOCAL_PROGRESS"} and action_type in {
            "TypeAction",
            "SelectDropDownOptionAction",
            "ClickAction",
        }:
            pattern = f"{action_type.lower()}_progress"
            state.progress.successful_patterns = _dedupe_keep_order(
                [*state.progress.successful_patterns, pattern],
                32,
            )
        elif (label in {"NO_VISIBLE_CHANGE", "BLOCKED"} or (expected_effect and not expected_effect_met)) and action_type:
            pattern = f"{action_type.lower()}_no_effect"
            if "intercepts pointer events" in error.lower():
                pattern = "popup_intercept"
            elif expected_effect and not expected_effect_met:
                pattern = f"{action_type.lower()}_expected_{expected_effect}_miss"
            state.progress.failed_patterns = _dedupe_keep_order(
                [*state.progress.failed_patterns, pattern],
                32,
            )

    def _store_expected_effect(
        self,
        *,
        action: dict[str, Any],
        ranked_candidates: list[Candidate],
        state: AgentState,
    ) -> None:
        candidate = self._candidate_for_action(action=action, ranked_candidates=ranked_candidates)
        expected_effect = self._predict_expected_effect(action=action, candidate=candidate)
        state.progress.pending_expected_effect = expected_effect[:40]
        state.progress.pending_expected_action_type = str(action.get("type") or "")[:80]
        state.progress.pending_expected_target_id = str(action.get("_element_id") or action.get("element_id") or "")[:120]
        state.progress.pending_expected_region_id = str((candidate.region_id if candidate is not None else "") or state.focus_region.region_id or "")[:120]

    def _apply_stagnation_policy(self, *, state: AgentState, flags: dict[str, Any]) -> None:
        no_progress_score = int(state.progress.no_progress_score or 0)
        region_id = str(state.focus_region.region_id or "").strip()
        if region_id and (no_progress_score >= 6 or int(state.progress.consecutive_no_effect_steps or 0) >= 2 or region_id in set(state.progress.blocked_regions)):
            state.focus_region.recent_region_ids = _dedupe_keep_order(
                [*state.focus_region.recent_region_ids, region_id],
                MAX_PENDING_ELEMENTS,
            )
            state.focus_region.region_id = ""
            state.focus_region.region_kind = ""
            state.focus_region.region_label = ""
            state.focus_region.region_context = ""
            state.focus_region.candidate_ids = []
            state.memory.checkpoints = _dedupe_keep_order(
                [*state.memory.checkpoints, f"release_region:{region_id}"],
                MAX_CHECKPOINTS,
            )
        if no_progress_score >= 7 and state.mode in {"NAV", "EXTRACT"}:
            state.mode = "PLAN"

    def _maybe_promote_to_plan_from_capability_gap(
        self,
        *,
        state: AgentState,
        policy_obs: dict[str, Any],
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

    def _default_vision_question(self, *, prompt: str, state: AgentState, text_ir: dict[str, Any]) -> str:
        control_groups = text_ir.get("control_groups") if isinstance(text_ir.get("control_groups"), list) else []
        cards = text_ir.get("cards") if isinstance(text_ir.get("cards"), list) else []
        last_action_type = str(state.last_action_sig or "").split("|", 1)[0].strip().lower()
        active_group_label = str(state.form_progress.active_group_label or "").strip()
        if active_group_label and last_action_type == "selectdropdownoptionaction":
            return f"Focus on the currently visible control group '{active_group_label[:120]}'. After the current selection, which visible candidate ids are the best follow-up controls? Prefer apply/search/submit or the next related control, and avoid repeating the same select."
        if control_groups and cards:
            return "Given the screenshot and task, which visible candidate ids are the best next targets? Prefer current-page controls or apply/submit actions over unrelated cards unless the target item is clearly identified."
        if control_groups:
            return "Given the screenshot and task, which visible controls or buttons best advance the task on the current page?"
        if cards:
            return "Given the screenshot and task, which visible candidate ids correspond to the most relevant item or item action?"
        if state.mode in {"PLAN", "STUCK"}:
            return "Describe the visible UI and identify the best visible next target from the candidate list."
        return "Which visible candidate ids best match the next useful action for this task?"

    def _should_auto_vision(
        self,
        *,
        screenshot: Any,
        state: AgentState,
        flags: dict[str, Any],
        text_ir: dict[str, Any],
        question: str,
        url: str,
    ) -> bool:
        skills, _ = self._ensure_support_tools()
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
        return bool(int(state.counters.stall_count or 0) >= 2 or int(state.counters.repeat_action_count or 0) >= 1)

    def _pre_done_verification(
        self,
        *,
        step_index: int,
        state: AgentState,
        prompt: str,
        text_ir: dict[str, Any],
        content: str,
        flags: dict[str, Any],
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
        text_ir: dict[str, Any],
        candidates: list[Candidate],
        ranked: list[Candidate],
        policy_obs: dict[str, Any],
        browser_allowed: set[str],
        model_name: str,
        usage_payload: dict[str, Any],
        policy_model_used: str,
    ) -> tuple[list[dict[str, Any]], bool, str, str, str]:
        chosen_actions: list[dict[str, Any]] = []
        done = False
        content = ""
        policy_reasoning = ""
        direct_browser_allowed = {tool for tool in browser_allowed if tool != "browser.go_back"}
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
                usage if isinstance(usage, dict) else {},
            )
            usage_breakdown = usage_payload.get("usage_breakdown") if isinstance(usage_payload.get("usage_breakdown"), dict) else _usage_breakdown_template()
            usage_breakdown["policy"] = _merge_usage_dicts(
                usage_breakdown.get("policy") if isinstance(usage_breakdown.get("policy"), dict) else {},
                usage if isinstance(usage, dict) else {},
            )
            usage_payload["usage_breakdown"] = usage_breakdown
            call_breakdown = usage_payload.get("call_breakdown") if isinstance(usage_payload.get("call_breakdown"), dict) else _empty_call_breakdown()
            call_breakdown["policy_llm_calls"] = int(call_breakdown.get("policy_llm_calls") or 0) + 1
            usage_payload["call_breakdown"] = call_breakdown
        self._remember_reasoning_trace(state=state, decision=decision)
        self._remember_working_state(state=state, decision=decision)
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
            raw_calls = decision.get("tool_calls") if isinstance(decision.get("tool_calls"), list) else []
            if not raw_calls and isinstance(decision.get("tool_call"), dict):
                raw_calls = [decision.get("tool_call")]
            max_actions_per_step = max(1, min(_env_int("FSM_MAX_ACTIONS_PER_STEP", 3), 5))
            for raw_call in raw_calls[:max_actions_per_step]:
                if not isinstance(raw_call, dict):
                    continue
                action = self._browser_action_from_tool_call(
                    tool_call=raw_call,
                    ranked_candidates=action_candidates,
                    state=state,
                    prompt=prompt,
                    allowed=direct_browser_allowed,
                    current_url=url,
                )
                if action is not None:
                    chosen_actions.append(action)
        return chosen_actions, done, content, policy_reasoning, policy_model_used

    def _run_meta_loop(
        self,
        *,
        task_id: str,
        prompt: str,
        web_project_id: str,
        use_case: dict[str, str],
        url: str,
        html: str,
        step_index: int,
        history: list[dict[str, Any]],
        screenshot: Any,
        state: AgentState,
        flags: dict[str, Any],
        text_ir: dict[str, Any],
        candidates: list[Candidate],
        ranked: list[Candidate],
        policy_obs: dict[str, Any],
        allowed: set[str],
        model_name: str,
        plan_model_name: str,
        usage_payload: dict[str, Any],
        policy_model_used: str,
        route_reason: str,
    ) -> tuple[list[dict[str, Any]], bool, str, str, str, list[str]]:
        chosen_actions: list[dict[str, Any]] = []
        done = False
        content = ""
        policy_reasoning = ""
        meta_exec_trace: list[str] = []
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
            chosen_actions = [pre_action]

        if not done and not chosen_actions:
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

        if not done and not chosen_actions:
            auto_vision_question = self._default_vision_question(prompt=prompt, state=state, text_ir=text_ir)
            if self._should_auto_vision(
                screenshot=screenshot,
                state=state,
                flags=flags,
                text_ir=text_ir,
                question=auto_vision_question,
                url=url,
            ):
                _, meta = self._ensure_support_tools()
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
                    usage_breakdown = usage_payload.get("usage_breakdown") if isinstance(usage_payload.get("usage_breakdown"), dict) else _usage_breakdown_template()
                    usage_breakdown["vision"] = _merge_usage_dicts(
                        usage_breakdown.get("vision") if isinstance(usage_breakdown.get("vision"), dict) else {},
                        vision_result.get("usage") if isinstance(vision_result.get("usage"), dict) else {},
                    )
                    usage_payload["usage_breakdown"] = usage_breakdown
                    call_breakdown = usage_payload.get("call_breakdown") if isinstance(usage_payload.get("call_breakdown"), dict) else _empty_call_breakdown()
                    call_breakdown["vision_llm_calls"] = int(call_breakdown.get("vision_llm_calls") or 0) + 1
                    usage_payload["call_breakdown"] = call_breakdown
                    helper_model = str(vision_result.get("model") or "").strip()
                    if helper_model:
                        usage_payload["helper_models"] = _dedupe_keep_order(
                            [*list(usage_payload.get("helper_models") or []), helper_model],
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
                    web_project_id=web_project_id,
                    use_case=use_case,
                    snapshot_html=html,
                    step_index=step_index,
                    url=url,
                    mode=state.mode,
                    flags=flags,
                    state=state,
                    text_ir=text_ir,
                    candidates=ranked,
                    history=history,
                    screenshot_available=_screenshot_available(screenshot),
                )
                route_reason = self._maybe_promote_to_plan_from_capability_gap(
                    state=state,
                    policy_obs=policy_obs,
                    route_reason=route_reason,
                )

        if not done and not chosen_actions:
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
                        usage if isinstance(usage, dict) else {},
                    )
                    usage_breakdown = usage_payload.get("usage_breakdown") if isinstance(usage_payload.get("usage_breakdown"), dict) else _usage_breakdown_template()
                    usage_breakdown["policy"] = _merge_usage_dicts(
                        usage_breakdown.get("policy") if isinstance(usage_breakdown.get("policy"), dict) else {},
                        usage if isinstance(usage, dict) else {},
                    )
                    usage_payload["usage_breakdown"] = usage_breakdown
                    call_breakdown = usage_payload.get("call_breakdown") if isinstance(usage_payload.get("call_breakdown"), dict) else _empty_call_breakdown()
                    call_breakdown["policy_llm_calls"] = int(call_breakdown.get("policy_llm_calls") or 0) + 1
                    usage_payload["call_breakdown"] = call_breakdown
                self._remember_reasoning_trace(state=state, decision=decision)
                self._remember_working_state(state=state, decision=decision)
                policy_reasoning = _candidate_text(decision.get("reasoning"), policy_reasoning)
                dtype = str(decision.get("type") or "").strip().lower()
                if dtype == "final":
                    final_content = _candidate_text(decision.get("content"), "Task completed.")
                    can_early_finish = bool(state.memory.facts) and state.mode in {
                        "REPORT",
                        "DONE",
                    }
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
                        if _looks_like_informational_task(prompt) and best_page_evidence and _runtime_page_evidence_ready(prompt, url, text_ir, step_index=step_index):
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
                            decision = {
                                "type": "meta",
                                "name": "META.EXTRACT_FACTS",
                                "arguments": {"schema": "generic"},
                            }
                            dtype = "meta"
                        elif state.mode in {
                            "NAV",
                            "PLAN",
                        } and not _looks_like_informational_task(prompt):
                            decision = {
                                "type": "meta",
                                "name": "META.EXTRACT_LINKS",
                                "arguments": {"kind": "all_links"},
                            }
                            dtype = "meta"
                        elif len(state.memory.facts) < 2:
                            decision = {
                                "type": "meta",
                                "name": "META.EXTRACT_FACTS",
                                "arguments": {"schema": "generic"},
                            }
                            dtype = "meta"
                        else:
                            decision = {
                                "type": "meta",
                                "name": "META.SEARCH_TEXT",
                                "arguments": {"query": prompt[:80]},
                            }
                            dtype = "meta"
                    else:
                        done = True
                        content = final_content
                        state.mode = "DONE"
                        break
                if dtype == "browser":
                    raw_calls = decision.get("tool_calls") if isinstance(decision.get("tool_calls"), list) else []
                    if not raw_calls and isinstance(decision.get("tool_call"), dict):
                        raw_calls = [decision.get("tool_call")]
                    max_actions_per_step = max(1, min(_env_int("FSM_MAX_ACTIONS_PER_STEP", 3), 5))
                    for raw_call in raw_calls[:max_actions_per_step]:
                        if not isinstance(raw_call, dict):
                            continue
                        action = self._browser_action_from_tool_call(
                            tool_call=raw_call,
                            ranked_candidates=ranked,
                            state=state,
                            prompt=prompt,
                            allowed=allowed,
                            current_url=url,
                        )
                        if action is not None:
                            chosen_actions.append(action)
                    if chosen_actions:
                        break
                if dtype == "meta":
                    if state.counters.meta_steps_used >= MAX_INTERNAL_META_STEPS:
                        break
                    meta_name = str(decision.get("name") or "")
                    meta_args = decision.get("arguments") if isinstance(decision.get("arguments"), dict) else {}
                    if meta_name == "META.VISION_QA":
                        vision_question = str(meta_args.get("question") or self._default_vision_question(prompt=prompt, state=state, text_ir=text_ir))
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
                    _, meta = self._ensure_support_tools()
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
                        if meta_name == "META.VISION_QA":
                            usage_breakdown = usage_payload.get("usage_breakdown") if isinstance(usage_payload.get("usage_breakdown"), dict) else _usage_breakdown_template()
                            usage_breakdown["vision"] = _merge_usage_dicts(
                                usage_breakdown.get("vision") if isinstance(usage_breakdown.get("vision"), dict) else {},
                                result.get("usage") if isinstance(result.get("usage"), dict) else {},
                            )
                            usage_payload["usage_breakdown"] = usage_breakdown
                            call_breakdown = usage_payload.get("call_breakdown") if isinstance(usage_payload.get("call_breakdown"), dict) else _empty_call_breakdown()
                            call_breakdown["vision_llm_calls"] = int(call_breakdown.get("vision_llm_calls") or 0) + 1
                            usage_payload["call_breakdown"] = call_breakdown
                        helper_model = str(result.get("model") or "").strip()
                        if helper_model:
                            usage_payload["helper_models"] = _dedupe_keep_order(
                                [*list(usage_payload.get("helper_models") or []), helper_model],
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
                        web_project_id=web_project_id,
                        use_case=use_case,
                        snapshot_html=html,
                        step_index=step_index,
                        url=url,
                        mode=state.mode,
                        flags=flags,
                        state=state,
                        text_ir=text_ir,
                        candidates=ranked,
                        history=history,
                        screenshot_available=_screenshot_available(screenshot),
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

        if not done and not chosen_actions:
            fallback_action = self._fallback_from_ranked_candidates(
                prompt=prompt,
                ranked_candidates=ranked,
                state=state,
                allowed=allowed,
                current_url=url,
            )
            if fallback_action is not None:
                chosen_actions = [fallback_action]
                meta_exec_trace.append("FALLBACK:top_ranked_candidate")

        if not done and not chosen_actions:
            stuck_action, done, content, stuck_note = self._stuck_recovery(
                prompt=prompt,
                url=url,
                step_index=step_index,
                state=state,
                allowed=allowed,
            )
            if stuck_action is not None:
                chosen_actions = [stuck_action]
            if stuck_note:
                meta_exec_trace.append(f"EMERGENCY_RECOVERY:{stuck_note}")

        if not done and not chosen_actions:
            fallback = self.policy._fallback(
                prompt=prompt,
                mode=state.mode,
                policy_obs=policy_obs,
                allowed_tools=allowed,
            )
            if str(fallback.get("type") or "") == "final":
                done = True
                content = _candidate_text(fallback.get("content"), "Task complete.")
                state.mode = "DONE"
            else:
                fallback_action = self._browser_action_from_tool_call(
                    tool_call=fallback.get("tool_call") if isinstance(fallback.get("tool_call"), dict) else {},
                    ranked_candidates=ranked,
                    state=state,
                    prompt=prompt,
                    allowed=allowed,
                    current_url=url,
                )
                if fallback_action is not None:
                    chosen_actions = [fallback_action]

        return (
            chosen_actions,
            done,
            content,
            policy_reasoning,
            policy_model_used,
            meta_exec_trace,
        )

    def _finalize_chosen_action(
        self,
        *,
        done: bool,
        direct_loop: bool,
        chosen_actions: list[dict[str, Any]] | None,
        prompt: str,
        history: list[dict[str, Any]],
        ranked: list[Candidate],
        state: AgentState,
        step_index: int,
    ) -> tuple[list[dict[str, Any]], str, dict[str, Any] | None]:
        actions: list[dict[str, Any]] = []
        browser_tool_name = ""
        chosen_actions = list(chosen_actions or [])
        if done or not chosen_actions:
            state.last_action_sig = ""
            state.last_action_element_id = ""
            return actions, browser_tool_name, None

        simulated_history = list(history)
        max_actions_per_step = max(1, min(_env_int("FSM_MAX_ACTIONS_PER_STEP", 3), 5))
        for chosen_action in chosen_actions[:max_actions_per_step]:
            if direct_loop:
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
                    history=simulated_history,
                    ranked_candidates=ranked,
                    state=state,
                )
                chosen_action = self._guard_submit_without_inputs(
                    action=chosen_action,
                    prompt=prompt,
                    history=simulated_history,
                    ranked_candidates=ranked,
                    state=state,
                )
                chosen_action = self._guard_redundant_type_action(
                    action=chosen_action,
                    prompt=prompt,
                    history=simulated_history,
                    ranked_candidates=ranked,
                    state=state,
                )
                chosen_action = self._guard_redundant_select_action(
                    action=chosen_action,
                    history=simulated_history,
                    ranked_candidates=ranked,
                    state=state,
                )
                chosen_action = self._promote_click_input_to_type(
                    action=chosen_action,
                    prompt=prompt,
                    ranked_candidates=ranked,
                    state=state,
                )
            actions.append(chosen_action)
            browser_tool_name = str(self._tool_name_for_action(chosen_action))
            if not direct_loop:
                self._remember_form_progress_from_action(
                    prompt=prompt,
                    action=chosen_action,
                    ranked_candidates=ranked,
                    state=state,
                )
                self._store_expected_effect(action=chosen_action, ranked_candidates=ranked, state=state)
            simulated_history.append({"step": int(step_index), "action": chosen_action, "exec_ok": True})
        chosen_action = actions[-1] if actions else None
        if chosen_action is None:
            return [], "", None
        sig = self._action_signature(chosen_action)
        state.counters.repeat_action_count = state.counters.repeat_action_count + 1 if sig == state.last_action_sig else 0
        state.last_action_sig = sig
        state.last_action_element_id = str(chosen_action.get("_element_id") or "")[:120]
        if state.last_action_element_id and int(state.counters.repeat_action_count or 0) >= 2:
            state.blocklist.element_ids = _dedupe_keep_order(
                [*state.blocklist.element_ids, state.last_action_element_id],
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
        text_ir: dict[str, Any],
        policy_obs: dict[str, Any],
        state_before: AgentState,
        state: AgentState,
        meta_exec_trace: list[str],
        chosen_action: dict[str, Any] | None,
        actions: list[dict[str, Any]],
        done: bool,
        content: str,
        browser_tool_name: str,
        include_reasoning: bool,
        policy_reasoning: str,
        ranked: list[Candidate],
        flags: dict[str, Any],
        history: list[dict[str, Any]],
        usage_payload: dict[str, Any],
        policy_model_used: str,
        model_name: str,
    ) -> dict[str, Any]:
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
                    current_page = (
                        best_fact or _candidate_text((text_ir.get("likely_answers") or [""])[0] if isinstance(text_ir.get("likely_answers"), list) else "") or "Current page answer not clear yet."
                    )
                if not done and chosen_action is not None:
                    decision_text = f"Take browser action sequence ending with: {browser_tool_name or str(chosen_action.get('type') or 'action')}"
                elif not done:
                    decision_text = "No safe browser action selected."
                reasoning = (f"Goal: {goal}. Current page: {current_page}. Decision: {decision_text}.")[:600]

        selected_candidate = None
        if chosen_action is not None:
            selected_candidate = self._candidate_for_action(action=chosen_action, ranked_candidates=ranked)
        last_effect = self._last_effect_label(history=history, flags=flags, done=done)
        no_progress_score = self._no_progress_score(state=state, flags=flags, history=history)
        out: dict[str, Any] = {
            "protocol_version": "1.0",
            "actions": actions,
            "done": bool(done),
            "content": final_content if done else None,
            "state_out": state.to_state_out(),
        }
        if isinstance(reasoning, str) and reasoning:
            out["reasoning"] = reasoning
        if include_reasoning and isinstance(state.memory.reasoning_trace, dict) and state.memory.reasoning_trace:
            out["reasoning_trace"] = dict(state.memory.reasoning_trace)
        if include_reasoning and isinstance(state.memory.working_state, dict) and state.memory.working_state:
            out["working_state"] = dict(state.memory.working_state)
        usage = usage_payload.get("usage") if isinstance(usage_payload.get("usage"), dict) else None
        if usage:
            out["usage"] = usage
            out["total_tokens"] = int(usage.get("total_tokens") or 0)
        call_breakdown = usage_payload.get("call_breakdown") if isinstance(usage_payload.get("call_breakdown"), dict) else {}
        if call_breakdown:
            out["call_breakdown"] = {
                "policy_llm_calls": int(call_breakdown.get("policy_llm_calls") or 0),
                "obs_extract_llm_calls": int(call_breakdown.get("obs_extract_llm_calls") or 0),
                "vision_llm_calls": int(call_breakdown.get("vision_llm_calls") or 0),
            }
        usage_breakdown = usage_payload.get("usage_breakdown") if isinstance(usage_payload.get("usage_breakdown"), dict) else {}
        if usage_breakdown:
            out["usage_breakdown"] = {
                "policy": _merge_usage_dicts(
                    {},
                    usage_breakdown.get("policy") if isinstance(usage_breakdown.get("policy"), dict) else {},
                ),
                "obs_extract": _merge_usage_dicts(
                    {},
                    usage_breakdown.get("obs_extract") if isinstance(usage_breakdown.get("obs_extract"), dict) else {},
                ),
                "vision": _merge_usage_dicts(
                    {},
                    usage_breakdown.get("vision") if isinstance(usage_breakdown.get("vision"), dict) else {},
                ),
            }
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

    def _remember_reasoning_trace(self, *, state: AgentState, decision: dict[str, Any]) -> None:
        trace = _normalize_reasoning_trace(decision.get("reasoning_trace"))
        if not trace:
            return
        state.memory.reasoning_trace = trace
        next_expected = _candidate_text(trace.get("next_expected_proof"))
        if next_expected:
            state.progress.pending_expected_effect = next_expected[:40]

    def _remember_working_state(self, *, state: AgentState, decision: dict[str, Any]) -> None:
        ws = _normalize_working_state(decision.get("working_state"))
        if ws:
            state.memory.working_state = ws

    def _build_completion_only_output(
        self,
        *,
        task_id: str,
        prompt: str,
        url: str,
        step_index: int,
        state: AgentState,
        text_ir: dict[str, Any],
        flags: dict[str, Any],
        include_reasoning: bool,
        usage_payload: dict[str, Any],
        policy_model_used: str,
    ) -> dict[str, Any]:
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
            "call_breakdown": usage_payload.get("call_breakdown") if isinstance(usage_payload.get("call_breakdown"), dict) else {},
            "usage_breakdown": usage_payload.get("usage_breakdown") if isinstance(usage_payload.get("usage_breakdown"), dict) else {},
        }
        if include_reasoning and isinstance(state.memory.reasoning_trace, dict) and state.memory.reasoning_trace:
            out["reasoning_trace"] = dict(state.memory.reasoning_trace)
        if include_reasoning and isinstance(state.memory.working_state, dict) and state.memory.working_state:
            out["working_state"] = dict(state.memory.working_state)
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
        web_project_id: str,
        use_case: dict[str, str],
        url: str,
        html: str,
        screenshot: Any,
        step_index: int,
        history: list[dict[str, Any]],
        state: AgentState,
        allowed: set[str],
        model_override: str,
    ) -> dict[str, Any]:
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
        usage_payload: dict[str, Any] = {
            "helper_models": [],
            "call_breakdown": _empty_call_breakdown(),
            "usage_breakdown": _usage_breakdown_template(),
        }
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
            web_project_id=web_project_id,
            use_case=use_case,
            snapshot_html=html,
            step_index=step_index,
            url=url,
            mode=state.mode,
            flags=flags,
            state=state,
            text_ir=text_ir,
            candidates=ranked,
            history=history,
            screenshot_available=_screenshot_available(screenshot),
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
                    web_project_id=web_project_id,
                    use_case=use_case,
                    snapshot_html=html,
                    step_index=step_index,
                    url=url,
                    mode=state.mode,
                    flags=flags,
                    state=state,
                    text_ir=text_ir,
                    candidates=ranked,
                    history=history,
                    screenshot_available=_screenshot_available(screenshot),
                )
        if obs_extract_usage:
            usage_payload["usage"] = _merge_usage_dicts(
                usage_payload.get("usage") if isinstance(usage_payload.get("usage"), dict) else {},
                obs_extract_usage,
            )
            usage_breakdown = usage_payload.get("usage_breakdown") if isinstance(usage_payload.get("usage_breakdown"), dict) else _usage_breakdown_template()
            usage_breakdown["obs_extract"] = _merge_usage_dicts(
                usage_breakdown.get("obs_extract") if isinstance(usage_breakdown.get("obs_extract"), dict) else {},
                obs_extract_usage,
            )
            usage_payload["usage_breakdown"] = usage_breakdown
        if obs_extract_model:
            usage_payload["helper_models"] = _dedupe_keep_order(
                [*list(usage_payload.get("helper_models") or []), obs_extract_model],
                8,
            )
        if obs_extract_usage or obs_extract_model:
            call_breakdown = usage_payload.get("call_breakdown") if isinstance(usage_payload.get("call_breakdown"), dict) else _empty_call_breakdown()
            call_breakdown["obs_extract_llm_calls"] = int(call_breakdown.get("obs_extract_llm_calls") or 0) + 1
            usage_payload["call_breakdown"] = call_breakdown
        default_model_name = _env_str("OPENAI_MODEL", "gpt-5.2") or "gpt-5.2"
        model_name = str(model_override or default_model_name)
        plan_model_name = str(model_override or _env_str("OPENAI_PLAN_MODEL", default_model_name) or default_model_name)
        policy_model_used = plan_model_name if state.mode in {"PLAN", "STUCK"} else model_name
        browser_allowed = {tool for tool in allowed if str(tool).startswith("browser.")}
        if not browser_allowed:
            browser_allowed = set(_supported_browser_tool_names())
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

    def run(self, *, payload: dict[str, Any], model_override: str = "") -> dict[str, Any]:
        prompt = str(payload.get("prompt") or payload.get("task_prompt") or "")
        web_project_id = _candidate_text(payload.get("web_project_id"))
        use_case = _normalize_use_case_info(payload.get("use_case"))
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
            web_project_id=web_project_id,
            use_case=use_case,
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
        meta_exec_trace: list[str] = []
        policy_reasoning = ""
        chosen_actions: list[dict[str, Any]] = []
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
            chosen_actions, done, content, policy_reasoning, policy_model_used = self._run_direct_loop(
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
            (
                chosen_actions,
                done,
                content,
                policy_reasoning,
                policy_model_used,
                meta_exec_trace,
            ) = self._run_meta_loop(
                task_id=task_id,
                prompt=prompt,
                web_project_id=web_project_id,
                use_case=use_case,
                url=url,
                html=html,
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
            chosen_actions=chosen_actions,
            prompt=prompt,
            history=history,
            ranked=ranked,
            state=state,
            step_index=step_index,
        )

        state.last_url = str(url or "")[:MAX_STR]
        state.last_dom_hash = str(flags.get("dom_hash") or "")[:64]
        state.blocklist.until_step = max(
            state.blocklist.until_step,
            int(step_index) + 1 if state.blocklist.element_ids else state.blocklist.until_step,
        )
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
        history: list[dict[str, Any]],
        state: AgentState,
        flags: dict[str, Any],
        ranked_candidates: list[Candidate],
        allowed: set[str],
    ) -> tuple[dict[str, Any] | None, bool, str, str]:
        def allow(name: str) -> bool:
            return (not allowed) or (name in allowed)

        last_action_type = str(state.last_action_sig or "").split("|", 1)[0].strip().lower()
        recent_errors = [str(item.get("error") or "").lower() for item in history[-4:] if isinstance(item, dict) and str(item.get("error") or "").strip()]
        if bool(flags.get("cookie_banner")) or (bool(flags.get("modal_dialog")) and not bool(flags.get("interactive_modal_form"))):
            if allow("browser.send_keys") and any("intercepts pointer events" in err for err in recent_errors):
                return (
                    {"type": "SendKeysIWAAction", "keys": "Escape"},
                    False,
                    "",
                    "popup_escape",
                )
            skills, _ = self._ensure_support_tools()
            popup_result = skills.solve_popups(candidates=ranked_candidates)
            primary_id = str(popup_result.get("primary_element_id") or "").strip()
            if primary_id and allow("browser.click"):
                for cand in ranked_candidates:
                    if cand.id != primary_id:
                        continue
                    selector = _sanitize_selector(cand.selector)
                    if isinstance(selector, dict):
                        return (
                            {
                                "type": "ClickAction",
                                "selector": selector,
                                "_element_id": cand.id,
                            },
                            False,
                            "",
                            "popup_click",
                        )
            if allow("browser.send_keys"):
                return (
                    {"type": "SendKeysIWAAction", "keys": "Escape"},
                    False,
                    "",
                    "popup_escape",
                )

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
        if last_action_type == "waitaction" and int(step_index) >= 1 and recent_wait_only and wait_steps >= 1 and not _task_constraints(prompt):
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
                    [*state.memory.checkpoints, checkpoint],
                    MAX_CHECKPOINTS,
                )
                return (
                    {"type": "WaitAction", "time_seconds": 1.0},
                    False,
                    "",
                    "post_action_wait",
                )

        return None, False, "", ""

    def _typed_values_from_history(self, history: list[dict[str, Any]]) -> list[str]:
        out: list[str] = []
        for item in history:
            if not isinstance(item, dict):
                continue
            action_raw = item.get("action")
            action: dict[str, Any] = action_raw if isinstance(action_raw, dict) else {}
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
        history: list[dict[str, Any]],
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
            if field_kind in {
                "username",
                "email",
                "password",
                "confirm_password",
            } and self._field_value_is_usable(value=inferred, candidate=candidate):
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
        return input_type not in {"checkbox", "radio", "submit", "button", "reset", "image", "hidden", "file"}

    def _next_password_value(self, *, prompt: str, candidate: Candidate | None, state: AgentState) -> str:
        _, passwords = self._extract_credentials(prompt)
        if not passwords:
            return ""
        used_values: list[str] = []
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

    def _extract_credentials(self, prompt: str) -> tuple[list[str], list[str]]:
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
            if value and value.lower() not in {
                "and",
                "then",
                "attempt",
                "retry",
                "with",
                "to",
            }:
                identifiers.append(value[:120])
        for token in re.findall(r"<(?:username|user|email|signup_username|signup_email)>", text, flags=re.I):
            identifiers.append(token[:120])
        passwords: list[str] = []
        for hit in re.finditer(
            r"\b(?:password|pass|pwd)\b\s*(?:equals|=|is|:)?\s*(?:'([^']+)'|\"([^\"]+)\"|(<[^>]+>)|([^\s,;]+))",
            text,
            flags=re.I,
        ):
            value = next((g for g in hit.groups() if g), "")
            value = _norm_ws(value).strip(" \t\r\n'\"`.,;:!?")
            if value and value.lower() not in {
                "and",
                "then",
                "attempt",
                "retry",
                "with",
            }:
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

    def _quoted_values(self, prompt: str) -> list[str]:
        out: list[str] = []
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
        return any(
            term in text
            for term in (
                "register",
                "sign up",
                "signup",
                "create account",
                "create an account",
            )
        )

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
            out.update(_supported_browser_tool_names())
            out.update(_obs_meta_tools())
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
                if canonical.startswith("browser.") and canonical not in _supported_browser_tool_names():
                    continue
                if canonical.startswith("META."):
                    if canonical not in _meta_tools():
                        continue
                    if (not self._allow_control_meta_tools()) and canonical in CONTROL_META_TOOLS:
                        continue
                out.add(canonical)
        return out

    def _tool_name_for_action(self, action: dict[str, Any]) -> str:
        t = str(action.get("type") or "")
        t_l = t.lower()
        mapping = {
            "searchaction": "browser.search",
            "navigateaction": "browser.navigate",
            "gobackaction": "browser.go_back",
            "clickaction": "browser.click",
            "doubleclickaction": "browser.dblclick",
            "rightclickaction": "browser.rightclick",
            "middleclickaction": "browser.middleclick",
            "tripleclickaction": "browser.tripleclick",
            "typeaction": "browser.input",
            "scrollaction": "browser.scroll",
            "waitaction": "browser.wait",
            "selectdropdownoptionaction": "browser.select_dropdown",
            "getdropdownoptionsaction": "browser.dropdown_options",
            "hoveraction": "browser.hover",
            "screenshotaction": "browser.screenshot",
            "sendkeysiwaaction": "browser.send_keys",
            "holdkeyaction": "browser.hold_key",
            "extractaction": "browser.extract",
        }
        return mapping.get(t_l, "")

    def _candidate_for_action(self, *, action: dict[str, Any], ranked_candidates: list[Candidate]) -> Candidate | None:
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

    def _same_group_candidates(self, *, target: Candidate, ranked_candidates: list[Candidate]) -> list[Candidate]:
        if target.group_id:
            return [cand for cand in ranked_candidates if cand.group_id == target.group_id and cand.id != target.id]
        target_context = _norm_ws(target.context)
        if not target_context:
            return []
        return [cand for cand in ranked_candidates if _norm_ws(cand.context) == target_context and cand.id != target.id]

    def _remember_active_group(
        self,
        *,
        target: Candidate,
        ranked_candidates: list[Candidate],
        state: AgentState,
    ) -> None:
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
                [*state.focus_region.recent_region_ids, focus_region_id],
                MAX_PENDING_ELEMENTS,
            )

    def _candidate_constraint_keys(self, *, candidate: Candidate, task_constraints: dict[str, str]) -> set[str]:
        return self.ranker._candidate_constraint_keys(cand=candidate, task_constraints=task_constraints)

    def _record_constraint_progress(
        self,
        *,
        prompt: str,
        action: dict[str, Any],
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
                    [*state.progress.satisfied_constraints, key],
                    32,
                )

    def _infer_input_text(self, *, prompt: str, candidate: Candidate) -> str:
        prompt_text = str(prompt or "")
        if not self._candidate_accepts_typed_text(candidate=candidate):
            return ""
        blob = " ".join(
            [
                candidate.text,
                candidate.context,
                candidate.href,
                candidate.field_hint,
                candidate.field_kind,
            ]
        ).lower()
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
            m_not = re.search(
                r"\bname\b[^.]{0,100}\bnot\b[^'\"<]*['\"]([^'\"]+)['\"]",
                prompt_text,
                flags=re.I,
            )
            if m_not:
                banned = _norm_ws(m_not.group(1))
                if banned.lower() != "autouser":
                    return "AutoUser"
                return "ExampleUser"
            m_eq = re.search(
                r"\bname\b[^.]{0,100}\b(?:equals|is|=)\s*['\"]([^'\"]+)['\"]",
                prompt_text,
                flags=re.I,
            )
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
            m_num = re.search(
                r"\b(?:rating|score)\b[^0-9]{0,20}([0-9]+(?:\.[0-9]+)?)",
                prompt_text,
                flags=re.I,
            )
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

    def _selector_signature(self, selector: dict[str, Any] | None) -> str:
        selector = _sanitize_selector(selector)
        if not isinstance(selector, dict):
            return ""
        try:
            return json.dumps(selector, ensure_ascii=True, sort_keys=True)
        except Exception:
            return ""

    def _typed_selector_signatures_from_history(self, history: list[dict[str, Any]]) -> set[str]:
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

    def _selected_selector_signatures_from_history(self, history: list[dict[str, Any]]) -> set[str]:
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

    def _typed_candidate_ids_from_history(self, history: list[dict[str, Any]]) -> set[str]:
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

    def _typed_selector_signatures(self, history: list[dict[str, Any]], state: AgentState) -> set[str]:
        out = set(self._typed_selector_signatures_from_history(history))
        out.update(state.form_progress.typed_selector_sigs)
        return out

    def _typed_candidate_ids(self, history: list[dict[str, Any]], state: AgentState) -> set[str]:
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
        if cand.role == "input" and str(cand.input_type or "").strip().lower() in {
            "submit",
            "button",
            "image",
            "reset",
        }:
            return True
        if cand.role not in {"button", "input"}:
            return False
        blob = " ".join([cand.text, cand.field_hint]).lower()
        return any(
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

    def _is_search_or_filter_input(self, cand: Candidate) -> bool:
        return str(cand.field_kind or "") in {"search", "sort"}

    def _redirect_delete_only_form_action(
        self,
        *,
        prompt: str,
        ranked_candidates: list[Candidate],
        target: Candidate | None,
        state: AgentState,
    ) -> dict[str, Any] | None:
        task_ops = self.ranker._task_operation_hints(prompt)
        if task_ops.intersection({"create", "update"}) or "delete" not in task_ops:
            return None
        has_password_input_visible = any(cand.role == "input" and cand.field_kind in {"password", "confirm_password"} for cand in ranked_candidates)
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
                return {
                    "type": "ClickAction",
                    "selector": cand.selector,
                    "_element_id": cand.id,
                }
        for cand in ranked_candidates:
            if cand.id in state.blocklist.element_ids:
                continue
            if cand.role in {"button", "link"}:
                return {
                    "type": "ClickAction",
                    "selector": cand.selector,
                    "_element_id": cand.id,
                }
        return None

    def _guard_delete_task_against_unrelated_form_edits(
        self,
        *,
        action: dict[str, Any] | None,
        prompt: str,
        ranked_candidates: list[Candidate],
        state: AgentState,
    ) -> dict[str, Any] | None:
        if not isinstance(action, dict):
            return action
        task_ops = self.ranker._task_operation_hints(prompt)
        if task_ops.intersection({"create", "update"}) or "delete" not in task_ops:
            return action
        if str(action.get("type") or "") not in {
            "TypeAction",
            "SelectDropDownOptionAction",
        }:
            return action
        has_password_input_visible = any(cand.role == "input" and cand.field_kind in {"password", "confirm_password"} for cand in ranked_candidates)
        if has_password_input_visible:
            return action
        mutation_candidates = [
            cand for cand in ranked_candidates if self.ranker._candidate_action_tags(cand).intersection({"delete"}) and cand.id not in state.blocklist.element_ids and cand.role in {"button", "link"}
        ]
        if mutation_candidates:
            best = mutation_candidates[0]
            return {
                "type": "ClickAction",
                "selector": best.selector,
                "_element_id": best.id,
            }
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
        action: dict[str, Any] | None,
        prompt: str,
        history: list[dict[str, Any]],
        ranked_candidates: list[Candidate],
        state: AgentState,
    ) -> dict[str, Any] | None:
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
            if (cand.id in typed_candidate_ids or (sig and sig in typed_sigs)) and self._candidate_has_usable_typed_value(candidate=cand, history=history, state=state):
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
                [*state.form_progress.submit_attempt_sigs, sel_sig],
                MAX_PENDING_ELEMENTS * 2,
            )
        return action

    def _guard_missing_group_inputs(
        self,
        *,
        action: dict[str, Any] | None,
        prompt: str,
        history: list[dict[str, Any]],
        ranked_candidates: list[Candidate],
        state: AgentState,
    ) -> dict[str, Any] | None:
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

        missing_inputs: list[Candidate] = []
        for cand in ranked_candidates:
            if cand.role != "input":
                continue
            if cand.id in state.blocklist.element_ids:
                continue
            if self._is_search_or_filter_input(cand):
                continue
            cand_sig = self._selector_signature(cand.selector)
            if (cand.id in typed_candidate_ids or (cand_sig and cand_sig in typed_sigs)) and self._candidate_has_usable_typed_value(candidate=cand, history=history, state=state):
                continue
            if group_id or group_context or group_candidate_ids:
                same_group = False
                if (group_id and cand.group_id == group_id) or (group_context and _norm_ws(cand.context) == group_context) or cand.id in group_candidate_ids:
                    same_group = True
                if not same_group:
                    continue
            text = self._infer_input_text(prompt=prompt, candidate=cand)
            if not text:
                continue
            missing_inputs.append(cand)
        if not missing_inputs:
            target_group_candidates: list[Candidate] = []
            for cand in ranked_candidates:
                if (group_id and cand.group_id == group_id) or (group_context and _norm_ws(cand.context) == group_context) or (group_candidate_ids and cand.id in group_candidate_ids):
                    target_group_candidates.append(cand)
            typed_kinds = {
                cand.field_kind
                for cand in target_group_candidates
                if cand.field_kind and (cand.id in typed_candidate_ids or (self._selector_signature(cand.selector) and self._selector_signature(cand.selector) in typed_sigs))
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
        action: dict[str, Any] | None,
        prompt: str,
        history: list[dict[str, Any]],
        ranked_candidates: list[Candidate],
        state: AgentState,
    ) -> dict[str, Any] | None:
        if not isinstance(action, dict) or str(action.get("type") or "") != "TypeAction":
            return action
        target = self._candidate_for_action(action=action, ranked_candidates=ranked_candidates)
        if target is None:
            return action
        typed_sigs = self._typed_selector_signatures(history, state)
        typed_candidate_ids = self._typed_candidate_ids(history, state)
        target_sig = self._selector_signature(target.selector)
        already_typed_target = ((target.id in typed_candidate_ids) or (bool(target_sig) and target_sig in typed_sigs)) and self._candidate_has_usable_typed_value(
            candidate=target, history=history, state=state
        )
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
            if (cand.id in typed_candidate_ids or (cand_sig and cand_sig in typed_sigs)) and self._candidate_has_usable_typed_value(candidate=cand, history=history, state=state):
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
        action: dict[str, Any] | None,
        history: list[dict[str, Any]],
        ranked_candidates: list[Candidate],
        state: AgentState,
    ) -> dict[str, Any] | None:
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
        action: dict[str, Any] | None,
        prompt: str,
        ranked_candidates: list[Candidate],
        state: AgentState,
    ) -> dict[str, Any] | None:
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
        tool_call: dict[str, Any],
        ranked_candidates: list[Candidate],
        state: AgentState,
        prompt: str,
        allowed: set[str],
        current_url: str = "",
    ) -> dict[str, Any] | None:
        normalized_allowed = {_canonical_allowed_tool_name(t) for t in allowed}
        normalized_allowed.discard("")
        name = _canonical_allowed_tool_name(str(tool_call.get("name") or "").strip().lower())
        if not name or not name.startswith("browser."):
            return None
        if normalized_allowed and name not in normalized_allowed:
            return None
        args = tool_call.get("arguments") if isinstance(tool_call.get("arguments"), dict) else {}
        action_type = _action_type_for_tool(name)
        if not action_type:
            return None

        def resolve_target(*, prefer_roles: set[str], allow_unmatched_selector: bool) -> tuple[dict[str, Any] | None, str, int | None]:
            selector = _sanitize_selector(args.get("selector") if isinstance(args.get("selector"), dict) else None)
            element_id = str(args.get("element_id") or args.get("_element_id") or "")
            raw_index = args.get("index")
            indexed_candidate = self._candidate_by_index(ranked_candidates, raw_index)
            if indexed_candidate is not None and ((not prefer_roles) or indexed_candidate.role in prefer_roles):
                return indexed_candidate.selector, indexed_candidate.id, int(raw_index)
            selector, element_id = self._resolve_selector_and_element_id(
                selector=selector,
                element_id=element_id,
                ranked_candidates=ranked_candidates,
                state=state,
                prefer_roles=prefer_roles,
                allow_unmatched_selector=allow_unmatched_selector,
            )
            return selector, element_id, None

        if name == "browser.go_back":
            return {"type": "GoBackAction"}

        if name == "browser.search":
            query = _candidate_text(args.get("query"))
            engine = _candidate_text(args.get("engine"), "duckduckgo")
            if not query:
                return None
            return {"type": "SearchAction", "query": query[:300], "engine": engine}

        if name == "browser.click":
            selector, element_id, _ = resolve_target(prefer_roles={"button", "link", "input"}, allow_unmatched_selector=True)
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
                action={
                    "type": "ClickAction",
                    "selector": selector,
                    "_element_id": element_id,
                },
                ranked_candidates=ranked_candidates,
            )
            if selected_candidate is not None and selected_candidate.href and ((not normalized_allowed) or ("browser.navigate" in normalized_allowed)):
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
                    return {
                        "type": "NavigateAction",
                        "url": normalized_href,
                        "go_back": False,
                        "go_forward": False,
                    }
            if ((not normalized_allowed) or ("browser.navigate" in normalized_allowed)) and str(selector.get("type") or "") == "attributeValueSelector":
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
                        return {
                            "type": "NavigateAction",
                            "url": normalized_sel_href,
                            "go_back": False,
                            "go_forward": False,
                        }
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
                ((not normalized_allowed) or ("browser.click" in normalized_allowed))
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
                            if cand_parts.scheme == target_parts.scheme and cand_parts.netloc == target_parts.netloc and ((cand_parts.path or "/") == (target_parts.path or "/")):
                                same_visible_link = True
                    if same_visible_link and isinstance(cand.selector, dict):
                        out = {"type": "ClickAction", "selector": cand.selector}
                        if cand.id:
                            out["_element_id"] = cand.id
                        return out
            return {
                "type": "NavigateAction",
                "url": nav_url,
                "go_back": False,
                "go_forward": False,
            }

        if name == "browser.input":
            text = _candidate_text(args.get("text"), args.get("value"))
            selector, element_id, _ = resolve_target(prefer_roles={"input", "textarea"}, allow_unmatched_selector=False)
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
                action={
                    "type": "TypeAction",
                    "selector": selector,
                    "_element_id": element_id,
                },
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
                    action={
                        "type": "TypeAction",
                        "selector": selector,
                        "_element_id": element_id,
                    },
                    ranked_candidates=ranked_candidates,
                )
            if resolved_candidate is not None and not self._candidate_accepts_typed_text(candidate=resolved_candidate):
                input_type = str(resolved_candidate.input_type or "").strip().lower()
                if input_type in {"checkbox", "radio"} and ((not normalized_allowed) or ("browser.click" in normalized_allowed)):
                    out = {"type": "ClickAction", "selector": selector}
                    if element_id:
                        out["_element_id"] = element_id
                    return out
                fallback_candidate = next(
                    (cand for cand in ranked_candidates if cand.id not in state.blocklist.element_ids and self._candidate_accepts_typed_text(candidate=cand)),
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
            if isinstance(raw_amount, int | float):
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

        if name == "browser.select_dropdown":
            text = _candidate_text(args.get("text"), args.get("value"), "Option")
            selector, element_id, _ = resolve_target(prefer_roles={"select"}, allow_unmatched_selector=False)
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
                action={
                    "type": "SelectDropDownOptionAction",
                    "selector": selector,
                    "_element_id": element_id,
                },
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
            out = {
                "type": "SelectDropDownOptionAction",
                "selector": selector,
                "text": text,
            }
            if element_id:
                out["_element_id"] = element_id
            return out

        if name == "browser.dropdown_options":
            selector, element_id, _ = resolve_target(prefer_roles={"select"}, allow_unmatched_selector=False)
            selector = _sanitize_selector(selector if isinstance(selector, dict) else None)
            if not isinstance(selector, dict):
                return None
            out = {"type": "GetDropDownOptionsAction", "selector": selector}
            if element_id:
                out["_element_id"] = element_id
            return out

        if name == "browser.hover":
            selector, element_id, _ = resolve_target(
                prefer_roles={"button", "link", "input", "select"},
                allow_unmatched_selector=True,
            )
            selector = _sanitize_selector(selector if isinstance(selector, dict) else None)
            if not isinstance(selector, dict):
                return None
            out = {"type": "HoverAction", "selector": selector}
            if element_id:
                out["_element_id"] = element_id
            return out

        if name in {
            "browser.dblclick",
            "browser.rightclick",
            "browser.middleclick",
            "browser.tripleclick",
        }:
            selector, element_id, _ = resolve_target(prefer_roles={"button", "link", "input"}, allow_unmatched_selector=True)
            selector = _sanitize_selector(selector if isinstance(selector, dict) else None)
            if not isinstance(selector, dict):
                return None
            click_type_map = {
                "browser.dblclick": "DoubleClickAction",
                "browser.rightclick": "RightClickAction",
                "browser.middleclick": "MiddleClickAction",
                "browser.tripleclick": "TripleClickAction",
            }
            out = {"type": click_type_map[name], "selector": selector}
            if element_id:
                out["_element_id"] = element_id
            return out

        if name == "browser.screenshot":
            if not _use_vision():
                return None
            file_path = _candidate_text(args.get("file_path"), args.get("file_name"), "")
            full_page = bool(args.get("full_page"))
            return {
                "type": "ScreenshotAction",
                "file_path": file_path,
                "full_page": full_page,
            }

        if name == "browser.send_keys":
            keys = _candidate_text(args.get("keys"), "Enter")
            return {"type": "SendKeysIWAAction", "keys": keys}

        if name == "browser.hold_key":
            key = _candidate_text(args.get("key"), "Control")
            return {"type": "HoldKeyAction", "key": key}
        if name == "browser.extract":
            query = _candidate_text(args.get("query"), "")
            selector, _, _ = resolve_target(prefer_roles=set(), allow_unmatched_selector=True)
            out = {"type": "ExtractAction", "query": query}
            if isinstance(selector, dict):
                out["selector"] = selector
            if "include_html" in args:
                out["include_html"] = bool(args.get("include_html"))
            if "max_chars" in args:
                with contextlib.suppress(Exception):
                    out["max_chars"] = int(args.get("max_chars") or 0)
            return out
        return None

    def _resolve_selector_and_element_id(
        self,
        *,
        selector: dict[str, Any] | None,
        element_id: str,
        ranked_candidates: list[Candidate],
        state: AgentState,
        prefer_roles: set[str],
        allow_unmatched_selector: bool = True,
    ) -> tuple[dict[str, Any] | None, str]:
        selector = _sanitize_selector(selector)
        if element_id:
            for cand in ranked_candidates:
                if cand.id == element_id and ((not prefer_roles) or (cand.role in prefer_roles)):
                    return cand.selector, cand.id

        if not isinstance(selector, dict):
            return selector, element_id

        sel_type = str(selector.get("type") or "").strip().lower()
        if sel_type in {"text", "tagcontainsselector"}:
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

    def _candidate_by_index(self, candidates: list[Candidate], raw_index: Any) -> Candidate | None:
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            return None
        if index < 0 or index >= len(candidates):
            return None
        return candidates[index]

    def _candidate_id_for_selector(
        self,
        *,
        selector: dict[str, Any],
        ranked_candidates: list[Candidate],
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
    ) -> tuple[dict[str, Any] | None, bool, str, str]:
        normalized_allowed = {_canonical_allowed_tool_name(t) for t in allowed}
        normalized_allowed.discard("")

        def allow(tool: str) -> bool:
            return (not normalized_allowed) or (_canonical_allowed_tool_name(tool) in normalized_allowed)

        if state.last_action_element_id:
            state.blocklist.element_ids = _dedupe_keep_order(
                [*state.blocklist.element_ids, state.last_action_element_id],
                MAX_PENDING_ELEMENTS,
            )
            state.blocklist.until_step = max(state.blocklist.until_step, int(step_index) + 2)

        last_action_type = str(state.last_action_sig or "").split("|", 1)[0].strip().lower()
        if last_action_type in {"clickaction", "typeaction", "selectdropdownoptionaction"} and allow("browser.wait") and int(state.counters.stall_count or 0) <= 3:
            return (
                {"type": "WaitAction", "time_seconds": 1.2},
                False,
                "",
                "wait_for_async_side_effect",
            )

        current_url = str(url or "").strip().lower()
        if allow("browser.go_back") and current_url not in {"", "about:blank"}:
            return (
                {"type": "NavigateAction", "go_back": True, "go_forward": False},
                False,
                "",
                "back_navigation",
            )
        if allow("browser.scroll"):
            return (
                {
                    "type": "ScrollAction",
                    "direction": "down",
                    "up": False,
                    "down": True,
                    "amount": 700,
                },
                False,
                "",
                "last_resort_scroll",
            )
        return (None, False, "", "")

    def _fallback_from_ranked_candidates(
        self,
        *,
        prompt: str,
        ranked_candidates: list[Candidate],
        state: AgentState,
        allowed: set[str],
        current_url: str,
    ) -> dict[str, Any] | None:
        normalized_allowed = {_canonical_allowed_tool_name(t) for t in allowed}
        normalized_allowed.discard("")

        def allow(tool: str) -> bool:
            return (not normalized_allowed) or (_canonical_allowed_tool_name(tool) in normalized_allowed)

        focus_region_id = str(state.focus_region.region_id or "").strip()
        focus_region_context = _norm_ws(state.focus_region.region_context)
        focus_candidate_ids = set(state.focus_region.candidate_ids)
        local_ranked: list[Candidate] = []
        escape_ranked: list[Candidate] = []
        global_ranked: list[Candidate] = []
        for cand in ranked_candidates:
            same_region = False
            if (
                (focus_region_id and cand.region_id and cand.region_id == focus_region_id)
                or (focus_region_id and focus_region_id in set(cand.region_ancestor_ids or []))
                or (focus_region_context and _norm_ws(cand.context) == focus_region_context)
                or (focus_candidate_ids and cand.id in focus_candidate_ids)
            ):
                same_region = True
            if same_region:
                blob = " ".join([cand.text, cand.field_hint, cand.group_label, cand.context]).lower()
                if str(cand.field_kind or "").strip().lower() != "pager" and re.search(
                    r"\b(save|submit|apply|continue|confirm|close|done|cancel|back)\b",
                    blob,
                ):
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
                    return {
                        "type": "ClickAction",
                        "selector": cand.selector,
                        "_element_id": cand.id,
                    }
                if cand.role == "input" and allow("browser.input"):
                    text = self._infer_input_text(prompt=prompt, candidate=cand)
                    if text:
                        return {
                            "type": "TypeAction",
                            "selector": cand.selector,
                            "text": text[:220],
                            "_element_id": cand.id,
                        }
                if cand.role == "select" and allow("browser.select_dropdown"):
                    text = self._infer_input_text(prompt=prompt, candidate=cand)
                    if text:
                        return {
                            "type": "SelectDropDownOptionAction",
                            "selector": cand.selector,
                            "text": text,
                            "_element_id": cand.id,
                        }

        ordered_candidates = local_ranked[:10] + escape_ranked[:8] + global_ranked[:10]
        for cand in ordered_candidates:
            if cand.id in state.blocklist.element_ids:
                continue
            if cand.role in {"link", "button"} and allow("browser.click"):
                return {
                    "type": "ClickAction",
                    "selector": cand.selector,
                    "_element_id": cand.id,
                }
            if cand.role == "input" and allow("browser.input"):
                text = self._infer_input_text(prompt=prompt, candidate=cand)
                if text:
                    return {
                        "type": "TypeAction",
                        "selector": cand.selector,
                        "text": text[:220],
                        "_element_id": cand.id,
                    }
            if cand.role == "select" and allow("browser.select_dropdown"):
                text = self._infer_input_text(prompt=prompt, candidate=cand)
                if text:
                    return {
                        "type": "SelectDropDownOptionAction",
                        "selector": cand.selector,
                        "text": text,
                        "_element_id": cand.id,
                    }
        return None

    def _action_signature(self, action: dict[str, Any]) -> str:
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
        action: dict[str, Any],
        ranked_candidates: list[Candidate],
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
                    [*state.form_progress.typed_candidate_ids, cand_id],
                    MAX_PENDING_ELEMENTS * 2,
                )
                if typed_value:
                    state.form_progress.typed_values_by_candidate[cand_id] = typed_value
            if sel_sig:
                state.form_progress.typed_selector_sigs = _dedupe_keep_order(
                    [*state.form_progress.typed_selector_sigs, sel_sig],
                    MAX_PENDING_ELEMENTS * 2,
                )
                if typed_value:
                    state.form_progress.typed_values_by_selector[sel_sig] = typed_value
            return
        if action_type == "ClickAction":
            cand_id = element_id
            if cand_id and cand_id in state.form_progress.typed_candidate_ids:
                state.form_progress.typed_candidate_ids = [x for x in state.form_progress.typed_candidate_ids if x != cand_id]
            return
        if action_type == "SelectDropDownOptionAction":
            return
