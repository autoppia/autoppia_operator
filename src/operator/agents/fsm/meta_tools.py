from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from src.operator.agents.fsm import AgentState, Candidate

from .state import _split_prompt_subgoals
from .utils import (
    FSM_MODES,
    MAX_CHECKPOINTS,
    MAX_FACTS,
    MAX_PENDING_ELEMENTS,
    MAX_PENDING_URLS,
    MAX_VISUAL_HINTS,
    MAX_VISUAL_NOTES,
    _candidate_text,
    _dedupe_keep_order,
    _env_str,
    _meta_tools,
    _norm_ws,
    _normalize_screenshot_data_url,
    _tokenize,
    _vision_signature,
)


class Router:
    def next_mode(
        self,
        *,
        step_index: int,
        state: AgentState,
        flags: dict[str, Any],
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
        if (bool(flags.get("cookie_banner")) or (bool(flags.get("modal_dialog")) and not bool(flags.get("interactive_modal_form")))) and (
            int(state.counters.repeat_action_count or 0) >= 1 or int(state.counters.stall_count or 0) >= 2
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
            if (bool(flags.get("product_cards")) or bool(flags.get("results_list")) or bool(flags.get("pricing_table"))) and not bool(flags.get("error_page")):
                return "NAV", "extractable_content_visible_in_nav"
            if bool(flags.get("search_box")) and bool(flags.get("dom_changed")):
                return "NAV", "interactive_content_changed_in_nav"
            return "NAV", "continue_navigation"
        return "PLAN", "default_plan"


class Skills:
    def __init__(self, vision_call: Callable[..., dict[str, Any]] | None = None) -> None:
        self.vision_call = vision_call
        self._vision_cache: dict[str, dict[str, Any]] = {}

    def solve_popups(self, *, candidates: list[Candidate]) -> dict[str, Any]:
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
        group_field_kinds: dict[str, set[str]] = {}
        for cand in candidates:
            group_key = str(cand.group_id or _norm_ws(cand.context).lower() or cand.id)
            kinds = group_field_kinds.setdefault(group_key, set())
            if cand.field_kind:
                kinds.add(str(cand.field_kind or "").strip().lower())
        scored: list[tuple[float, Candidate]] = []
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
            if any(
                k in blob
                for k in {
                    "dialog",
                    "modal",
                    "popup",
                    "overlay",
                    "cookie",
                    "consent",
                    "backdrop",
                }
            ):
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

    def replan(self, *, prompt: str) -> dict[str, Any]:
        subgoals = _split_prompt_subgoals(prompt)
        return {"subgoals": subgoals}

    def extract_links(self, *, kind: str, candidates: list[Candidate]) -> dict[str, Any]:
        k = str(kind or "all_links").strip().lower()
        urls: list[str] = []
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

    def extract_facts(self, *, schema: str, text_ir: dict[str, Any], url: str) -> dict[str, Any]:
        schema_name = str(schema or "generic").strip().lower()
        text = str(text_ir.get("visible_text") or "")
        headings = text_ir.get("headings") if isinstance(text_ir.get("headings"), list) else []
        page_facts = text_ir.get("page_facts") if isinstance(text_ir.get("page_facts"), list) else []
        facts: list[str] = []
        if page_facts:
            facts.extend([str(f)[:180] for f in page_facts[:8] if str(f).strip()])
        if headings:
            facts.extend([f"Heading: {str(h)[:120]}" for h in headings[:4]])
        lines = [ln.strip() for ln in re.split(r"[.\n]", text) if ln.strip()]
        for ln in lines:
            line_lower = ln.lower()
            if schema_name == "pricing":
                if "$" in ln or "price" in line_lower or "plan" in line_lower:
                    facts.append(ln[:220])
            elif schema_name == "product_page":
                if any(
                    k in line_lower
                    for k in {
                        "feature",
                        "product",
                        "integration",
                        "benefit",
                        "use case",
                    }
                ):
                    facts.append(ln[:220])
            else:
                if len(ln) >= 24 and any(ch.isalpha() for ch in ln):
                    facts.append(ln[:220])
            if len(facts) >= 8:
                break
        if not facts and text:
            facts.append(text[:220])
        return {"facts": _dedupe_keep_order([f"{f} (source: {url})" for f in facts], 10)}

    def search_text(self, *, query: str, text_ir: dict[str, Any]) -> dict[str, Any]:
        q = _norm_ws(query).lower()
        text = str(text_ir.get("visible_text") or "")
        if not q:
            return {"found": False, "matches": []}
        q_tokens = _tokenize(q)
        lines = [ln.strip() for ln in re.split(r"[\n.]", text) if ln.strip()]
        matches: list[str] = []
        for ln in lines:
            line = ln.lower()
            if q in line:
                matches.append(ln[:220])
                continue
            overlap = len(q_tokens.intersection(_tokenize(line)))
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
        candidates: list[Candidate],
        text_ir: dict[str, Any],
        url: str,
    ) -> dict[str, Any]:
        if self.vision_call is None:
            return {"ok": False, "error": "vision_disabled"}
        image_url = _normalize_screenshot_data_url(screenshot)
        if not image_url:
            return {"ok": False, "error": "missing_screenshot"}
        q = _candidate_text(
            question,
            "Describe the screenshot and identify the best next visible targets.",
        )
        signature = _vision_signature(screenshot=screenshot, question=q, url=url)
        if signature and signature in self._vision_cache:
            cached = dict(self._vision_cache[signature])
            cached["cached"] = True
            return cached
        candidate_lines: list[dict[str, Any]] = []
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
        system = "You analyze a webpage screenshot for a browser agent.\nReturn exactly one JSON object with keys:\n- answer: short answer to the question\n- element_ids: list of candidate ids from the provided candidate list that best match the screenshot and question\n- signals: short visual observations about layout, panels, cards, dialogs, or likely primary controls\n- confidence: one of low, medium, high\nOnly return candidate ids that appear in the provided candidate list."
        user_content = [
            {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                    "detail": _env_str("FSM_VISION_DETAIL", "low") or "low",
                },
            },
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
            return {
                "ok": False,
                "error": f"vision_parse_failed:{str(exc)[:120]}",
                "raw": content[:500],
            }
        ids = parsed.get("element_ids") if isinstance(parsed.get("element_ids"), list) else []
        out = {
            "ok": True,
            "answer": _candidate_text(parsed.get("answer"))[:260],
            "element_ids": _dedupe_keep_order([str(x)[:120] for x in ids if str(x).strip()], MAX_VISUAL_HINTS),
            "signals": _dedupe_keep_order(
                [str(x)[:180] for x in (parsed.get("signals") or []) if str(x).strip()],
                8,
            ),
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
        candidates: list[Candidate],
        role: str = "",
        text: str = "",
        limit: int = 8,
    ) -> dict[str, Any]:
        role_l = str(role or "").strip().lower()
        text_l = str(text or "").strip().lower()
        out: list[str] = []
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

    def select_next_target(self, *, state: AgentState) -> dict[str, Any]:
        seen = set(state.visited.urls)
        for url in state.frontier.pending_urls:
            if url not in seen:
                return {"url": url}
        return {"url": ""}

    def escalate(self, *, reason: str) -> dict[str, Any]:
        return {"checkpoint": f"escalate:{_norm_ws(reason)[:180]}"}

    def set_mode(self, *, mode: str) -> dict[str, Any]:
        m = str(mode or "").strip().upper()
        if m not in FSM_MODES:
            m = "PLAN"
        return {"mode": m}

    def mark_progress(self, *, state: AgentState, subgoal_id: str, status: str) -> dict[str, Any]:
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
        args: dict[str, Any],
        state: AgentState,
        prompt: str,
        text_ir: dict[str, Any],
        candidates: list[Candidate],
        url: str,
        screenshot: Any = None,
    ) -> dict[str, Any]:
        name = str(tool_name or "").strip().upper()
        allowed_meta_tools = _meta_tools()
        if self.skills.vision_call is not None:
            allowed_meta_tools = set(allowed_meta_tools)
            allowed_meta_tools.add("META.VISION_QA")
        if name not in allowed_meta_tools:
            return {"ok": False, "error": f"unknown_meta_tool:{name}"}

        if name == "META.SOLVE_POPUPS":
            result = self.skills.solve_popups(candidates=candidates)
            state.frontier.pending_elements = _dedupe_keep_order(
                state.frontier.pending_elements + list(result.get("pending_elements") or []),
                MAX_PENDING_ELEMENTS,
            )
            state.memory.checkpoints = _dedupe_keep_order(
                [*state.memory.checkpoints, str(result.get("checkpoint") or "solve_popups")],
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
                [*state.memory.checkpoints, f"extract_links:{kind}"],
                MAX_CHECKPOINTS,
            )
            return {"ok": True}

        if name == "META.EXTRACT_FACTS":
            schema = str(args.get("schema") or "generic")
            result = self.skills.extract_facts(schema=schema, text_ir=text_ir, url=url)
            state.memory.facts = _dedupe_keep_order(state.memory.facts + list(result.get("facts") or []), MAX_FACTS)
            state.memory.checkpoints = _dedupe_keep_order(
                [*state.memory.checkpoints, f"extract_facts:{schema}"],
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
                [*state.memory.checkpoints, f"search_text:{query[:64]}:{len(matches)}"],
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
                [*state.memory.checkpoints, f"find_elements:{role or 'any'}:{text[:48]}:{len(ids)}"],
                MAX_CHECKPOINTS,
            )
            return {"ok": True, "count": len(ids)}

        if name == "META.VISION_QA":
            question = str(args.get("question") or "").strip()
            if not question:
                question = "Based on the screenshot and task, which visible candidate ids are the best next targets? Prefer current-page controls for narrowing or submitting, and identify the most relevant visible target."
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
                    [*state.memory.checkpoints, f"vision_qa_error:{str(result.get('error') or 'unknown')[:80]}"],
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
                state.memory.visual_notes = _dedupe_keep_order([*state.memory.visual_notes, visual_note], MAX_VISUAL_NOTES)
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
                [*state.memory.checkpoints, f"vision_qa:{len(element_ids)}:{str(result.get('confidence') or 'low')[:12]}"],
                MAX_CHECKPOINTS,
            )
            return result

        if name == "META.SELECT_NEXT_TARGET":
            result = self.skills.select_next_target(state=state)
            picked = str(result.get("url") or "").strip()
            if picked:
                state.frontier.pending_urls = _dedupe_keep_order([picked, *state.frontier.pending_urls], MAX_PENDING_URLS)
            state.memory.checkpoints = _dedupe_keep_order(
                [*state.memory.checkpoints, "select_next_target"],
                MAX_CHECKPOINTS,
            )
            return {"ok": True, "url": picked}

        if name == "META.ESCALATE":
            reason = str(args.get("reason") or "stuck")
            result = self.skills.escalate(reason=reason)
            state.escalated_once = True
            state.memory.checkpoints = _dedupe_keep_order(
                [*state.memory.checkpoints, str(result.get("checkpoint") or "escalate")],
                MAX_CHECKPOINTS,
            )
            return {"ok": True}

        if name == "META.SET_MODE":
            mode_out = self.skills.set_mode(mode=str(args.get("mode") or "PLAN"))
            state.mode = str(mode_out.get("mode") or state.mode)
            state.memory.checkpoints = _dedupe_keep_order(
                [*state.memory.checkpoints, f"set_mode:{state.mode}"],
                MAX_CHECKPOINTS,
            )
            return {"ok": True, "mode": state.mode}

        if name == "META.MARK_PROGRESS":
            subgoal_id = str(args.get("id") or "")
            status = str(args.get("status") or "done")
            result = self.skills.mark_progress(state=state, subgoal_id=subgoal_id, status=status)
            state.memory.checkpoints = _dedupe_keep_order(
                [*state.memory.checkpoints, f"mark_progress:{subgoal_id}:{status}"],
                MAX_CHECKPOINTS,
            )
            return {"ok": bool(result.get("updated"))}

        return {"ok": False, "error": f"unhandled_meta_tool:{name}"}


__all__ = [name for name in globals() if not name.startswith("__")]
