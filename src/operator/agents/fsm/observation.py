from __future__ import annotations

import contextlib
import hashlib
import json
import re
from typing import Any
from urllib.parse import urlsplit

from bs4 import BeautifulSoup

from agent_test_support import _norm_ws

from .candidates import Candidate
from .state import AgentState, ProgressEffect
from .utils import (
    HISTORY_RECENT_LIMIT,
    MAX_HISTORY_SUMMARY_CHARS,
    _anchor_overlap_score,
    _build_site_knowledge,
    _candidate_text,
    _dedupe_keep_order,
    _env_bool,
    _env_int,
    _fact_overlap_score,
    _labelish_text,
    _looks_like_informational_task,
    _normalize_use_case_info,
    _normalize_working_state,
    _supported_browser_tool_names,
    _task_constraints,
    _unavailable_browser_tools,
    _value_line_text,
    _valueish_text,
    _working_state_summary,
)


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
        if re.search(
            r"<\s*(username|email|password|signup_email|signup_password)\s*>",
            prompt,
            flags=re.I,
        ):
            return True
        return bool(re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", prompt))

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
        candidates: list[Candidate],
    ) -> dict[str, Any]:
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
        active_manage_context = bool(local_mutation_controls_visible) and (bool(state.form_progress.active_group_candidate_ids) or bool(state.form_progress.active_group_label))
        strategy_parts: list[str] = []
        if read_only_for_task:
            strategy_parts.append("Current page appears read-only for the requested operation: missing " + ", ".join(missing_ops[:3]) + ".")
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

    def build_text_ir(self, snapshot_html: str) -> dict[str, Any]:
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
                with contextlib.suppress(Exception):
                    node.decompose()
            title = ""
            try:
                title_tag = soup.find("title")
                title = _norm_ws(title_tag.get_text(" ", strip=True) if title_tag else "")
            except Exception:
                title = ""
            headings: list[str] = []
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

    def _visible_lines(self, soup: Any) -> list[str]:
        try:
            raw = str((soup.body or soup).get_text("\n", strip=True))
        except Exception:
            raw = ""
        lines = [_norm_ws(line)[:220] for line in raw.splitlines() if _norm_ws(line)]
        return _dedupe_keep_order(lines, 160)

    def _extract_page_facts(self, *, soup: Any, visible_lines: list[str]) -> list[str]:
        facts: list[str] = []

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
                if "toggle" in lowered or lowered == "search":
                    score -= 3
                if score > best_score:
                    best = candidate
                    best_score = score
            return best

        # Table / row pairs
        try:
            for row in soup.find_all(["tr"], limit=80):
                cells = [_norm_ws(cell.get_text(" ", strip=True)) for cell in row.find_all(["th", "td"], limit=4) if _norm_ws(cell.get_text(" ", strip=True))]
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

    def _extract_visible_value_lines(self, *, visible_lines: list[str]) -> list[str]:
        lines: list[str] = []
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
                if "toggle" in lowered or lowered == "search":
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
        visible_lines: list[str],
        page_facts: list[str],
        value_lines: list[str],
    ) -> list[str]:
        ranked: list[tuple[int, str]] = []
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
        fallback: list[str] = []
        for item in value_lines[:12] + visible_lines[:12]:
            clean = _norm_ws(item)
            if clean and clean not in fallback:
                fallback.append(clean[:180])
        return fallback[:16]

    def _likely_answers(self, *, prompt: str, page_facts: list[str]) -> list[str]:
        scored: list[tuple[int, str]] = []
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

    def _extract_forms(self, soup: Any) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for idx, form in enumerate(soup.find_all("form", limit=12), start=1):
            try:
                attrs = form.attrs if isinstance(getattr(form, "attrs", None), dict) else {}
                form_text = _norm_ws(form.get_text(" ", strip=True))
                controls: list[dict[str, Any]] = []
                commit_controls: list[str] = []
                for control in form.find_all(["input", "textarea", "select", "button"], limit=20):
                    c_attrs = control.attrs if isinstance(getattr(control, "attrs", None), dict) else {}
                    tag = str(getattr(control, "name", "") or "").lower()
                    label = self._field_hint_from_node(control)
                    required = ("required" in c_attrs) or (str(c_attrs.get("aria-required") or "").strip().lower() == "true")
                    options: list[str] = []
                    current_value = _norm_ws(c_attrs.get("value"))
                    if tag == "select":
                        selected_texts: list[str] = []
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
                    if (
                        tag == "button"
                        or str(c_attrs.get("type") or "").strip().lower() in {"submit", "button"}
                        or any(
                            token in role_blob
                            for token in (
                                "submit",
                                "save",
                                "apply",
                                "search",
                                "find",
                                "continue",
                                "register",
                                "sign up",
                                "log in",
                                "sign in",
                            )
                        )
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

    def _control_groups_from_forms(self, forms: list[dict[str, Any]]) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        for idx, form in enumerate(forms[:8], start=1):
            if not isinstance(form, dict):
                continue
            controls = form.get("controls") if isinstance(form.get("controls"), list) else []
            if not controls:
                continue
            summary: list[str] = []
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

    def _extract_control_panels(self, soup: Any) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
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
                summary: list[str] = []
                for control in controls[:8]:
                    c_attrs = control.attrs if isinstance(getattr(control, "attrs", None), dict) else {}
                    tag = str(getattr(control, "name", "") or "").lower()
                    label = self._field_hint_from_node(control)
                    options: list[str] = []
                    current_value = _norm_ws(c_attrs.get("value"))
                    if tag == "select":
                        selected_texts: list[str] = []
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

    def _candidate_groups(self, candidates: list[Candidate]) -> list[dict[str, Any]]:
        groups: dict[str, dict[str, Any]] = {}
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
        ranked: list[dict[str, Any]] = []
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

    def _card_summaries(self, groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cards: list[dict[str, Any]] = []
        for group in groups:
            if not isinstance(group, dict) or str(group.get("kind") or "") != "items":
                continue
            items = group.get("items") if isinstance(group.get("items"), list) else []
            actions: list[dict[str, Any]] = []
            facts: list[str] = []
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
        flags: dict[str, Any],
        state: AgentState,
        text_ir: dict[str, Any],
        candidates: list[Candidate],
        history_recent: list[dict[str, Any]],
    ) -> dict[str, Any]:
        text_ir = text_ir if isinstance(text_ir, dict) else {}
        role_counts: dict[str, int] = {}
        type_counts: dict[str, int] = {}
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
        recent_failures = sum(1 for item in history_recent if isinstance(item, dict) and ((not bool(item.get("exec_ok", True))) or bool(_candidate_text(item.get("error")))))
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

    def _page_ir_text(self, *, prompt: str, text_ir: dict[str, Any], candidates: list[Candidate]) -> str:
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
        parts: list[str] = []
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
            form_lines: list[str] = []
            for idx, form in enumerate(forms[:6], start=1):
                if not isinstance(form, dict):
                    continue
                controls = form.get("controls") if isinstance(form.get("controls"), list) else []
                control_bits: list[str] = []
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
            group_lines: list[str] = []
            for group in control_groups[:6]:
                if not isinstance(group, dict):
                    continue
                controls = group.get("controls") if isinstance(group.get("controls"), list) else []
                group_lines.append(f"{str(group.get('label') or '')[:120]} ({int(group.get('control_count') or 0)} controls) -> " + " ; ".join(str(x)[:120] for x in controls[:6]))
            if group_lines:
                parts.append("CONTROL GROUPS:\n" + "\n".join(f"- {line}" for line in group_lines[:6]))
        if active_region:
            region_lines: list[str] = []
            for item in (active_region.get("items") if isinstance(active_region.get("items"), list) else [])[:8]:
                if not isinstance(item, dict):
                    continue
                bits = [
                    str(item.get("id") or "")[:80],
                    _candidate_text(
                        item.get("role"),
                        item.get("region_kind"),
                        item.get("text"),
                        item.get("field_hint"),
                        item.get("href"),
                    )[:140],
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
            active_lines: list[str] = []
            for item in active_items[:8]:
                if not isinstance(item, dict):
                    continue
                bits = [
                    str(item.get("id") or "")[:80],
                    _candidate_text(
                        item.get("role"),
                        item.get("text"),
                        item.get("field_hint"),
                        item.get("href"),
                    )[:140],
                ]
                active_lines.append(" | ".join([x for x in bits if x]))
            if active_lines:
                parts.append("ACTIVE CONTROL GROUP:\n" + f"- {label} ({len(active_items)} visible related elements)\n" + "\n".join(f"- {line}" for line in active_lines[:8]))
        if cards:
            card_lines: list[str] = []
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
                    f"card[{idx}] {str(card.get('label') or '')[:120]} -> " + " | ".join(str(x)[:80] for x in facts[:3]) + (" ; actions=" + " / ".join(action_bits) if action_bits else "")
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
        parts: list[str] = []
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
                + _candidate_text(
                    state.focus_region.region_label,
                    state.focus_region.region_kind,
                    state.focus_region.region_id,
                )[:160]
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

    def _active_objective_summary(
        self,
        *,
        prompt: str,
        state: AgentState,
        page_observations: dict[str, Any],
        active_subgoal: dict[str, Any],
        active_region: dict[str, Any],
        active_group: dict[str, Any],
    ) -> dict[str, Any]:
        expected_effect = _candidate_text(state.progress.pending_expected_effect)
        expected_action_type = _candidate_text(state.progress.pending_expected_action_type)
        no_progress_score = int(state.progress.no_progress_score or 0)
        consecutive_no_effect = int(state.progress.consecutive_no_effect_steps or 0)
        local_candidate_count = int(page_observations.get("local_candidate_count") or 0) if isinstance(page_observations, dict) else 0
        blocked_regions = set(state.progress.blocked_regions or [])
        region_id = str(active_region.get("region_id") or "") if isinstance(active_region, dict) else ""
        visible_answer_signals = bool(
            isinstance(page_observations, dict)
            and (
                (page_observations.get("likely_answers") if isinstance(page_observations.get("likely_answers"), list) else [])
                or (page_observations.get("relevant_lines") if isinstance(page_observations.get("relevant_lines"), list) else [])
            )
        )
        focus = {
            "mode": str(state.mode or "")[:40],
            "goal": "best_visible_next_step",
            "reason": "",
            "active_subgoal": active_subgoal if isinstance(active_subgoal, dict) else {},
            "focused_region": active_region if isinstance(active_region, dict) else {},
            "active_group": active_group if isinstance(active_group, dict) else {},
            "expected_effect": expected_effect,
            "expected_action_type": expected_action_type,
            "local_candidate_count": local_candidate_count,
            "frontier_url_count": len(state.frontier.pending_urls),
            "frontier_element_count": len(state.frontier.pending_elements),
            "no_progress_score": no_progress_score,
            "consecutive_no_effect_steps": consecutive_no_effect,
            "visible_answer_signals": visible_answer_signals,
        }
        if visible_answer_signals:
            focus["goal"] = "verify_or_finish_from_visible_page"
            focus["reason"] = "The current page already exposes likely answer evidence."
        elif region_id and region_id in blocked_regions:
            focus["goal"] = "change_region_or_strategy"
            focus["reason"] = "The current focused region is blocked or repeatedly unproductive."
        elif no_progress_score >= 4 or consecutive_no_effect >= 2:
            focus["goal"] = "change_target_without_repeating"
            focus["reason"] = "Recent actions did not produce visible progress."
        elif expected_effect:
            focus["goal"] = "confirm_expected_effect"
            focus["reason"] = "A previous action implied an expected next effect that should be verified or advanced."
        elif local_candidate_count > 0 and active_region:
            focus["goal"] = "advance_current_region"
            focus["reason"] = "There is an active region with local candidates still available."
        elif active_subgoal:
            focus["goal"] = "advance_active_subgoal"
            focus["reason"] = "There is an explicit active subgoal to satisfy."
        elif state.frontier.pending_elements or state.frontier.pending_urls:
            focus["goal"] = "use_frontier_before_random_exploration"
            focus["reason"] = "Previously discovered frontier items exist."
        else:
            focus["reason"] = "Choose the best visible next step from the current page."
        focus["task_excerpt"] = _candidate_text(prompt)[:220]
        return focus

    def _working_state_summary(
        self,
        *,
        prompt: str,
        state: AgentState,
        text_ir: dict[str, Any],
        page_observations: dict[str, Any],
        active_region: dict[str, Any],
        active_group: dict[str, Any],
    ) -> dict[str, Any]:
        previous = state.memory.working_state if isinstance(state.memory.working_state, dict) else {}
        page_kind = _candidate_text(
            page_observations.get("page_kind") if isinstance(page_observations, dict) else None,
            text_ir.get("page_kind") if isinstance(text_ir, dict) else None,
            text_ir.get("title") if isinstance(text_ir, dict) else None,
        )
        region_label = _candidate_text(
            active_region.get("label") if isinstance(active_region, dict) else None,
            active_group.get("label") if isinstance(active_group, dict) else None,
            state.focus_region.region_label,
            state.focus_region.region_kind,
        )
        workflow = _candidate_text(
            state.plan.subgoals[0].text if state.plan.subgoals else "",
            state.plan.active_id,
            prompt,
        )
        completed_fields = []
        typed_by_candidate = state.form_progress.typed_values_by_candidate if isinstance(state.form_progress.typed_values_by_candidate, dict) else {}
        typed_by_selector = state.form_progress.typed_values_by_selector if isinstance(state.form_progress.typed_values_by_selector, dict) else {}
        for key in list(typed_by_candidate.keys())[-6:]:
            completed_fields.append(str(key))
        if not completed_fields:
            for key in list(typed_by_selector.keys())[-6:]:
                completed_fields.append(str(key))
        pending_fields: list[str] = []
        field_labels: list[str] = []
        for item in list(active_region.get("items") or [])[:8] if isinstance(active_region, dict) else []:
            if not isinstance(item, dict):
                continue
            item_kind = str(item.get("kind") or "").lower()
            if item_kind not in {"input", "textarea", "select"}:
                continue
            label = _candidate_text(item.get("label"), item.get("text"), item.get("id"))
            if label:
                field_labels.append(label)
        for label in field_labels:
            lower = label.lower()
            if not any(lower in str(done).lower() or str(done).lower() in lower for done in completed_fields):
                pending_fields.append(label)
        completion_missing: list[str] = []
        expected_effect = _candidate_text(state.progress.pending_expected_effect)
        if expected_effect:
            completion_missing.append(expected_effect)
        likely_answers = (page_observations.get("likely_answers") if isinstance(page_observations, dict) else []) or []
        relevant_lines = (page_observations.get("relevant_lines") if isinstance(page_observations, dict) else []) or []
        if not likely_answers and not relevant_lines:
            completion_missing.append("Visible completion evidence")
        completion_state = "in_progress"
        if likely_answers or relevant_lines:
            completion_state = "answer_visible_or_checkable"
        if pending_fields:
            completion_state = "awaiting_local_completion"
        return _normalize_working_state(
            {
                "current_page_kind": page_kind or previous.get("current_page_kind"),
                "active_region": region_label or previous.get("active_region"),
                "active_workflow": workflow or previous.get("active_workflow"),
                "completed_fields": completed_fields or previous.get("completed_fields"),
                "pending_fields": pending_fields or previous.get("pending_fields"),
                "completion_evidence_missing": completion_missing or previous.get("completion_evidence_missing"),
                "next_milestone": expected_effect or previous.get("next_milestone") or "Advance the active workflow with visible controls.",
                "completion_state": completion_state,
            }
        )

    def _local_workflow_closure_summary(
        self,
        *,
        state: AgentState,
        active_region: dict[str, Any],
        candidates: list[Candidate],
    ) -> dict[str, Any]:
        if not isinstance(active_region, dict) or not active_region:
            return {}
        typed_ids = set(state.form_progress.typed_candidate_ids or [])
        indexed = {cand.id: idx for idx, cand in enumerate(candidates[:24]) if isinstance(cand, Candidate) and cand.id}
        commit_controls: list[dict[str, Any]] = []
        for item in list(active_region.get("items") or [])[:12]:
            if not isinstance(item, dict):
                continue
            field_kind = str(item.get("field_kind") or "").strip().lower()
            role = str(item.get("role") or "").strip().lower()
            label = _candidate_text(item.get("text"), item.get("id"))
            blob = f"{field_kind} {role} {label or ''}".lower()
            if field_kind in {"submit", "account_create", "auth_entry"} or re.search(r"\b(save|submit|post|publish|apply|create|send|confirm)\b", blob):
                item_id = str(item.get("id") or "")
                commit_controls.append(
                    {
                        "id": item_id,
                        "index": indexed.get(item_id),
                        "label": label or item_id,
                        "field_kind": field_kind,
                        "role": role,
                    }
                )
        input_ids = {
            str(item.get("id") or "")
            for item in list(active_region.get("items") or [])[:12]
            if isinstance(item, dict) and str(item.get("role") or "").strip().lower() in {"input", "textarea", "select"}
        }
        completed_ids = [item_id for item_id in input_ids if item_id and item_id in typed_ids]
        pending_ids = [item_id for item_id in input_ids if item_id and item_id not in typed_ids]
        ready_to_commit = bool(commit_controls) and (not input_ids or len(completed_ids) >= max(1, len(input_ids)))
        return {
            "active_region_label": _candidate_text(active_region.get("label"), active_region.get("region_id")),
            "commit_controls": commit_controls[:4],
            "completed_input_ids": completed_ids[:8],
            "pending_input_ids": pending_ids[:8],
            "ready_to_commit": ready_to_commit,
        }

    def _avoid_repeating_summary(self, *, state: AgentState, candidates: list[Candidate]) -> dict[str, Any]:
        indexed_candidates = self._indexed_candidate_obs(candidates, limit=24)
        current_by_id = {
            str(item.get("id") or ""): {
                "index": item.get("index"),
                "id": item.get("id"),
                "role": item.get("role"),
                "text": item.get("text"),
                "context": item.get("context"),
            }
            for item in indexed_candidates
            if isinstance(item, dict) and str(item.get("id") or "")
        }
        discouraged_targets: list[dict[str, Any]] = []
        seen_target_ids: set[str] = set()
        for effect in reversed(state.progress.recent_effects[-8:]):
            if not isinstance(effect, ProgressEffect):
                continue
            target_id = str(effect.target_id or "").strip()
            if not target_id or target_id in seen_target_ids:
                continue
            if not (effect.repeated_target or effect.label in {"NO_VISIBLE_CHANGE", "BLOCKED"} or not effect.exec_ok):
                continue
            candidate_info = current_by_id.get(target_id)
            if not candidate_info:
                continue
            reason = "repeated_target" if effect.repeated_target else "no_visible_change" if effect.label == "NO_VISIBLE_CHANGE" else "blocked"
            discouraged_targets.append(
                {
                    **candidate_info,
                    "reason": reason,
                    "action_type": str(effect.action_type or "")[:80],
                }
            )
            seen_target_ids.add(target_id)
        repeated_regions = []
        blocked_region_ids = list(state.progress.blocked_regions[-6:])
        focus_recent = list(state.focus_region.recent_region_ids[-6:])
        for region_id in _dedupe_keep_order(blocked_region_ids + focus_recent, 8):
            if not region_id:
                continue
            repeated_regions.append(region_id)
        return {
            "should_change_approach": bool(
                discouraged_targets
                or repeated_regions
                or int(state.counters.repeat_action_count or 0) >= 1
                or int(state.counters.stall_count or 0) >= 2
                or int(state.progress.no_progress_score or 0) >= 4
            ),
            "repeat_action_count": int(state.counters.repeat_action_count or 0),
            "stall_count": int(state.counters.stall_count or 0),
            "failed_patterns": list(state.progress.failed_patterns[-6:]),
            "blocked_regions": blocked_region_ids,
            "recent_region_ids": focus_recent,
            "discouraged_targets": discouraged_targets[:6],
        }

    def _candidate_sig(self, cand: Candidate) -> str:
        selector = cand.selector if isinstance(cand.selector, dict) else {}
        selector_bits = [
            str(selector.get("type") or ""),
            str(selector.get("attribute") or ""),
            str(selector.get("value") or ""),
        ]
        return "|".join([*selector_bits, cand.text[:80], cand.role[:24]])

    def _page_summary_text(self, *, text_ir: dict[str, Any]) -> str:
        text_ir = text_ir if isinstance(text_ir, dict) else {}
        title = _candidate_text(text_ir.get("title"))
        headings = text_ir.get("headings") if isinstance(text_ir.get("headings"), list) else []
        visible_text = str(text_ir.get("visible_text") or "")
        parts: list[str] = []
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
        text_ir: dict[str, Any],
        page_ir_text: str,
        candidates: list[Candidate],
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
        candidates: list[Candidate],
        current_url: str,
        state: AgentState,
        max_total: int = 24,
    ) -> list[Candidate]:
        if not candidates:
            return []

        current_path = str(urlsplit(str(current_url or "")).path or "/").rstrip("/") or "/"
        focus_region_id = str(state.focus_region.region_id or "").strip()
        focus_region_context = _norm_ws(state.focus_region.region_context)
        focus_candidate_ids = set(state.focus_region.candidate_ids)
        has_focus = bool(focus_region_id or focus_region_context or focus_candidate_ids)
        focused_local: list[Candidate] = []
        focused_escape: list[Candidate] = []
        focused_extended: list[Candidate] = []
        controls: list[Candidate] = []
        commit_controls: list[Candidate] = []
        contextual: list[Candidate] = []
        global_nav: list[Candidate] = []
        others: list[Candidate] = []

        def same_focus_region(cand: Candidate) -> bool:
            if focus_region_id and cand.region_id and cand.region_id == focus_region_id:
                return True
            if focus_region_id and focus_region_id in set(cand.region_ancestor_ids or []):
                return True
            if focus_region_context and _norm_ws(cand.context) == focus_region_context:
                return True
            return bool(focus_candidate_ids and cand.id in focus_candidate_ids)

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
            if bool(
                re.search(
                    r"\b(save|submit|apply|continue|confirm|done|finish|search)\b",
                    lowered,
                )
            ):
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

        picked: list[Candidate] = []
        seen: set[str] = set()

        def add_many(arr: list[Candidate]) -> None:
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
            return bool(
                re.search(
                    r"\b(save|submit|apply|continue|confirm|close|done|cancel|back)\b",
                    blob,
                )
            )
        if focus_region_context and _norm_ws(cand.context) == focus_region_context:
            return bool(
                re.search(
                    r"\b(save|submit|apply|continue|confirm|close|done|cancel|back)\b",
                    blob,
                )
            )
        return False

    def _partition_candidates(
        self,
        *,
        candidates: list[Candidate],
        state: AgentState,
        max_local: int = 18,
        max_escape: int = 8,
        max_global: int = 18,
    ) -> dict[str, Any]:
        focus_region_id = str(state.focus_region.region_id or "").strip()
        focus_region_context = _norm_ws(state.focus_region.region_context)
        focus_candidate_ids = set(state.focus_region.candidate_ids)
        local: list[Candidate] = []
        escape: list[Candidate] = []
        global_pool: list[Candidate] = []
        for cand in candidates:
            same_region = False
            if (
                (focus_region_id and cand.region_id and cand.region_id == focus_region_id)
                or (focus_region_id and focus_region_id in set(cand.region_ancestor_ids or []))
                or (focus_region_context and _norm_ws(cand.context) == focus_region_context)
                or (focus_candidate_ids and cand.id in focus_candidate_ids)
            ):
                same_region = True
            if same_region:
                if self._is_escape_candidate(
                    cand=cand,
                    focus_region_id=focus_region_id,
                    focus_region_context=focus_region_context,
                ):
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

    def _candidate_lines(self, candidates: list[Candidate], *, limit: int = 12) -> str:
        lines: list[str] = []
        for idx, cand in enumerate(candidates[:limit]):
            bits = [f"[index={idx}] [{cand.id}]", f"<{cand.role}>"]
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

    def _indexed_candidate_obs(self, candidates: list[Candidate], *, limit: int) -> list[dict[str, Any]]:
        return [cand.as_obs(index=idx) for idx, cand in enumerate(candidates[:limit])]

    def _browser_state_snapshot(
        self,
        *,
        url: str,
        text_ir: dict[str, Any],
        page_observations: dict[str, Any],
        screenshot_available: bool,
    ) -> str:
        text_ir = text_ir if isinstance(text_ir, dict) else {}
        page_observations = page_observations if isinstance(page_observations, dict) else {}
        page_stats = page_observations.get("page_stats") if isinstance(page_observations.get("page_stats"), dict) else {}
        title = _candidate_text(text_ir.get("title"))
        headings = text_ir.get("headings") if isinstance(text_ir.get("headings"), list) else []
        likely_answers = page_observations.get("likely_answers") if isinstance(page_observations.get("likely_answers"), list) else []
        relevant_lines = page_observations.get("relevant_lines") if isinstance(page_observations.get("relevant_lines"), list) else []
        parts: list[str] = [
            f"Current URL: {_candidate_text(url)}",
            (
                f"Page stats: {int(page_stats.get('links') or 0)} links, {int(page_stats.get('controls') or 0)} controls, {int(page_stats.get('forms') or 0)} forms, {int(page_stats.get('control_groups') or 0)} control groups, {int(page_stats.get('cards') or 0)} cards, {int(page_stats.get('visible_text_chars') or 0)} visible chars"
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

    def _browser_state_text(self, candidates: list[Candidate], *, limit: int = 60) -> str:
        class _Node:
            __slots__ = ("children", "items", "name")

            def __init__(self, name: str) -> None:
                self.name = name
                self.children: dict[str, _Node] = {}
                self.items: list[Candidate] = []

        root = _Node("ROOT")
        chosen = candidates[: max(1, int(limit))]
        for cand in chosen:
            chain: list[str] = []
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

        def render(node: _Node, indent: str = "") -> list[str]:
            lines: list[str] = []
            for child_name, child in list(node.children.items())[:24]:
                lines.append(f"{indent}{child_name}")
                for cand in child.items[:12]:
                    label = _candidate_text(cand.text, cand.field_hint, cand.href, cand.id)
                    candidate_index = chosen.index(cand) if cand in chosen else -1
                    bits = [f"[index={candidate_index}] [{cand.id}] <{cand.role}/>"]
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

    def _history_brief(self, history: list[dict[str, Any]], *, limit: int = 10) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
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

    def _history_summary(self, history: list[dict[str, Any]]) -> str:
        if not isinstance(history, list) or not history:
            return ""
        total = len(history)
        failures = 0
        by_action: dict[str, int] = {}
        recent_urls: list[str] = []
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

    def _recent_failures(self, history: list[dict[str, Any]], *, limit: int = 6) -> list[dict[str, Any]]:
        if not isinstance(history, list) or not history:
            return []
        out: list[dict[str, Any]] = []
        for item in reversed(history[-40:]):
            if not isinstance(item, dict):
                continue
            error = _candidate_text(item.get("error"))
            if not error and bool(item.get("exec_ok", True)):
                continue
            action_raw = item.get("action")
            action = action_raw if isinstance(action_raw, dict) else {}
            out.append(
                {
                    "step": int(item.get("step") or 0),
                    "action_type": _candidate_text(
                        action.get("type"),
                        action.get("name"),
                        action_raw if isinstance(action_raw, str) else "",
                        "unknown",
                    )[:80],
                    "url": _candidate_text(item.get("url"))[:220],
                    "error": error[:220] if error else "execution_failed",
                }
            )
            if len(out) >= max(1, int(limit)):
                break
        return list(reversed(out))

    def _previous_step_verdict(self, history: list[dict[str, Any]], flags: dict[str, Any]) -> dict[str, str]:
        if not isinstance(history, list) or not history:
            return {"status": "n/a", "summary": "No previous step to evaluate."}
        last = history[-1] if isinstance(history[-1], dict) else {}
        if not isinstance(last, dict):
            return {"status": "uncertain", "summary": "Previous step unavailable."}
        err = _candidate_text(last.get("error"))
        if err:
            return {"status": "failure", "summary": f"Previous step error: {err[:180]}"}
        if not bool(last.get("exec_ok", True)):
            return {
                "status": "failure",
                "summary": "Previous step execution reported failure.",
            }
        if bool(flags.get("url_changed")) or bool(flags.get("dom_changed")):
            return {
                "status": "success",
                "summary": "Previous step changed page state (url/dom diff detected).",
            }
        if bool(flags.get("no_visual_progress")):
            return {
                "status": "uncertain",
                "summary": "No visual progress detected after previous step.",
            }
        return {
            "status": "uncertain",
            "summary": "Previous step outcome unclear from current snapshot.",
        }

    def _loop_nudges(self, *, state: AgentState, flags: dict[str, Any]) -> list[str]:
        n: list[str] = []
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
        web_project_id: str = "",
        use_case: dict[str, str] | None = None,
        snapshot_html: str = "",
        task_constraints: dict[str, str],
        step_index: int,
        mode: str,
        url: str,
        flags: dict[str, Any],
        state: AgentState,
        history_recent: list[dict[str, Any]],
        history_summary: str,
        verdict: dict[str, str],
        loop_nudges: list[str],
        text_ir: dict[str, Any],
        candidates: list[Candidate],
        state_delta: str = "",
        ir_delta: str = "",
        screenshot_available: bool = False,
    ) -> str:
        text_ir = self._augment_text_ir(text_ir=text_ir, candidates=candidates)
        parts: list[str] = []
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
        indexed_candidates = self._indexed_candidate_obs(candidates, limit=24)
        candidate_partitions = self._partition_candidates(candidates=candidates, state=state)
        recent_failures = self._recent_failures(history_recent, limit=4)
        progress_brief = self._progress_brief(state=state)
        site_knowledge = (
            _build_site_knowledge(
                _candidate_text(web_project_id),
                _normalize_use_case_info(use_case),
                prompt,
                current_url=url,
                snapshot_html=snapshot_html,
                candidates=candidates,
            )
            if _env_bool("FSM_USE_SITE_KNOWLEDGE", False)
            else {}
        )
        active_subgoal = {}
        if state.plan.active_id:
            for sg in state.plan.subgoals:
                if sg.id == state.plan.active_id:
                    active_subgoal = {"id": sg.id, "text": sg.text, "status": sg.status}
                    break
        active_objective = self._active_objective_summary(
            prompt=prompt,
            state=state,
            page_observations=page_observations,
            active_subgoal=active_subgoal,
            active_region=active_region,
            active_group=active_group,
        )
        avoid_repeating = self._avoid_repeating_summary(state=state, candidates=candidates)
        reasoning_trace = state.memory.reasoning_trace if isinstance(state.memory.reasoning_trace, dict) else {}
        working_state = self._working_state_summary(
            prompt=prompt,
            state=state,
            text_ir=text_ir,
            page_observations=page_observations,
            active_region=active_region,
            active_group=active_group,
        )
        local_workflow_closure = self._local_workflow_closure_summary(
            state=state,
            active_region=active_region,
            candidates=candidates,
        )
        typed_recent = [
            str(item.get("text") or "")
            for item in history_recent
            if isinstance(item, dict) and str(item.get("action_type") or "").lower() in {"typeaction", "fillaction"} and str(item.get("text") or "").strip()
        ][:10]
        parts.append(f"TASK: {_candidate_text(prompt)}")
        if task_constraints:
            parts.append("TASK CONSTRAINTS:\n" + json.dumps(task_constraints, ensure_ascii=False))
        parts.append(
            "RUNTIME:\n"
            + "browser-use-like operator runtime\n"
            + f"max_actions_per_step={max(1, min(_env_int('FSM_MAX_ACTIONS_PER_STEP', 3), 5))}\n"
            + "tabs_supported=false\n"
            + "file_tools_supported=false"
        )
        parts.append("CURRENT STATE:\n" + f"step_index={int(step_index)}\n" + f"mode={mode!s}\n" + f"url={_candidate_text(url)}")
        parts.append("AVAILABLE TOOLS:\n" + ", ".join(sorted(_supported_browser_tool_names())))
        parts.append("UNAVAILABLE TOOLS:\n" + ", ".join(_unavailable_browser_tools()) + "\nNever emit unavailable tools.")
        parts.append(
            "TOOL USAGE GUIDE:\n"
            + "- Use browser.click for buttons, links, toggles, tabs, checkboxes, radios, and submit controls.\n"
            + "- Use browser.input only for text-entry fields.\n"
            + "- Use browser.select_dropdown only when a concrete option text is known.\n"
            + "- Use browser.dropdown_options before browser.select_dropdown if the correct option is unclear.\n"
            + "- Use browser.extract when the answer is visible but needs deterministic text extraction.\n"
            + "- Use browser.search only to start a web search, not for in-page site search boxes.\n"
            + "- Prefer browser.done when the page already contains the final answer."
        )
        parts.append("ACTIVE OBJECTIVE (JSON):\n" + json.dumps(active_objective, ensure_ascii=False))
        parts.append("WORKING STATE (JSON):\n" + json.dumps(working_state, ensure_ascii=False))
        if local_workflow_closure:
            parts.append("LOCAL WORKFLOW CLOSURE (JSON):\n" + json.dumps(local_workflow_closure, ensure_ascii=False))
        local_html_context = (
            self._local_html_context(
                snapshot_html=snapshot_html,
                state=state,
                candidates=candidates,
                active_region=active_region,
            )
            if _env_bool("FSM_USE_LOCAL_HTML_CONTEXT", True)
            else {}
        )
        if local_html_context:
            parts.append("LOCAL HTML CONTEXT (JSON):\n" + json.dumps(local_html_context, ensure_ascii=False))
        if site_knowledge:
            parts.append("KNOWN SITE MAP (JSON):\n" + json.dumps(site_knowledge, ensure_ascii=False))
        parts.append("AVOID REPEATING (JSON):\n" + json.dumps(avoid_repeating, ensure_ascii=False))
        if reasoning_trace:
            parts.append("PREVIOUS REASONING TRACE (JSON):\n" + json.dumps(reasoning_trace, ensure_ascii=False))
        if working_state:
            parts.append("WORKING STATE SUMMARY:\n" + _working_state_summary(working_state))
        parts.append(
            "BROWSER SNAPSHOT:\n"
            + str(
                self._browser_state_snapshot(
                    url=url,
                    text_ir=text_ir,
                    page_observations=page_observations,
                    screenshot_available=screenshot_available,
                )
            )
        )
        parts.append(
            "ELEMENT TARGETING GUIDE:\n"
            + "- Each shortlist item has index, role, text, and context.\n"
            + "- Prefer targeting by index.\n"
            + "- Use text and context together; do not rely on index alone if multiple items look similar.\n"
            + "- Avoid reusing the same target after repeated failures unless the page materially changed."
        )
        parts.append("VISIBLE TEXT / PAGE SUMMARY:\n" + page_ir_text)
        parts.append(
            "VISIBLE EVIDENCE (JSON):\n"
            + json.dumps(
                {
                    "likely_answers": (page_observations.get("likely_answers") if isinstance(page_observations.get("likely_answers"), list) else [])[:6],
                    "relevant_lines": (page_observations.get("relevant_lines") if isinstance(page_observations.get("relevant_lines"), list) else [])[:10],
                    "page_stats": page_observations.get("page_stats") if isinstance(page_observations.get("page_stats"), dict) else {},
                },
                ensure_ascii=False,
            )
        )
        parts.append("INTERACTIVE ELEMENTS (indexed):\n" + browser_state_text)
        parts.append("INTERACTIVE ELEMENT SHORTLIST (JSON):\n" + json.dumps(indexed_candidates, ensure_ascii=False))
        parts.append("RECENT ACTIONS AND RESULTS (JSON):\n" + json.dumps(history_recent[-HISTORY_RECENT_LIMIT:], ensure_ascii=False))
        if recent_failures:
            parts.append("RECENT FAILURES (JSON):\n" + json.dumps(recent_failures, ensure_ascii=False))
        parts.append("PAGE OBSERVATIONS (JSON):\n" + json.dumps(page_observations, ensure_ascii=False))
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
        if progress_brief:
            parts.append("PROGRESS LEDGER:\n" + progress_brief)
        parts.append("PREVIOUS STEP VERDICT:\n" + f"status={_candidate_text(verdict.get('status'))}\n" + f"summary={_candidate_text(verdict.get('summary'))}\n")
        if active_region:
            parts.append("FOCUSED REGION SUMMARY (JSON):\n" + json.dumps(active_region, ensure_ascii=False))
        if history_summary:
            parts.append(f"HISTORY SUMMARY:\n{history_summary}")
        parts.append(
            "FORM STATE (JSON):\n"
            + json.dumps(
                {
                    "typed_values_recent": typed_recent[:8],
                    "typed_candidate_ids": state.form_progress.typed_candidate_ids[-12:],
                    "active_group_label": state.form_progress.active_group_label,
                    "active_group_candidate_ids": state.form_progress.active_group_candidate_ids[-12:],
                },
                ensure_ascii=False,
            )
        )
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
        if candidate_partitions.get("global"):
            global_summary = {
                "suppressed_global_count": int(candidate_partitions.get("suppressed_global_count") or 0),
                "global_candidates": self._indexed_candidate_obs(candidate_partitions.get("global") or [], limit=8),
            }
            parts.append("GLOBAL CANDIDATE SUMMARY (JSON):\n" + json.dumps(global_summary, ensure_ascii=False))
        return "\n\n".join(parts)

    def _augment_text_ir(self, *, text_ir: dict[str, Any], candidates: list[Candidate]) -> dict[str, Any]:
        base = dict(text_ir or {}) if isinstance(text_ir, dict) else {}
        groups = self._candidate_groups(candidates)
        existing_groups = base.get("control_groups") if isinstance(base.get("control_groups"), list) else []
        merged_groups: list[dict[str, Any]] = []
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

    def active_group_summary(self, *, state: AgentState, candidates: list[Candidate]) -> dict[str, Any]:
        group_id = str(state.form_progress.active_group_id or "").strip()
        context = _norm_ws(state.form_progress.active_group_context)
        candidate_ids = set(state.form_progress.active_group_candidate_ids)
        if not group_id and not context and not candidate_ids:
            return {}
        items: list[dict[str, Any]] = []
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

    def active_region_summary(self, *, state: AgentState, candidates: list[Candidate]) -> dict[str, Any]:
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
        items: list[dict[str, Any]] = []
        for cand in candidates:
            same_region = False
            if (region_id and cand.region_id and cand.region_id == region_id) or (region_context and _norm_ws(cand.context) == region_context) or (candidate_ids and cand.id in candidate_ids):
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

    def _candidate_node(self, *, soup: Any, candidate: Candidate | None) -> Any | None:
        if soup is None or candidate is None or not isinstance(candidate, Candidate):
            return None
        selector = candidate.selector if isinstance(candidate.selector, dict) else {}
        sel_type = str(selector.get("type") or "").strip()
        value = str(selector.get("value") or "").strip()
        attr = str(selector.get("attribute") or "").strip()
        try:
            if sel_type == "attributeValueSelector" and attr and value:
                return soup.find(attrs={attr: value})
            if sel_type == "xpathSelector" and value:
                m = re.search(
                    r'//([a-zA-Z0-9]+)\[contains\(normalize-space\(\.\), "([^"]+)"\)\]',
                    value,
                )
                if m:
                    tag = m.group(1).lower()
                    text = _norm_ws(m.group(2))
                    for node in soup.find_all(tag, limit=80):
                        if text and text in _norm_ws(node.get_text(" ", strip=True)):
                            return node
        except Exception:
            return None
        return None

    def _nearest_region_container(self, node: Any) -> Any | None:
        cur = node
        for _ in range(8):
            if cur is None:
                return None
            tag = str(getattr(cur, "name", "") or "").lower()
            if tag in {"form", "section", "article", "main", "aside"}:
                return cur
            attrs = cur.attrs if isinstance(getattr(cur, "attrs", None), dict) else {}
            classes = " ".join(str(x) for x in list(attrs.get("class") or []))
            blob = f"{_norm_ws(attrs.get('id'))} {classes}".lower()
            if any(
                token in blob
                for token in (
                    "form",
                    "comment",
                    "reply",
                    "review",
                    "contact",
                    "checkout",
                    "register",
                    "login",
                )
            ):
                return cur
            cur = getattr(cur, "parent", None)
        return node

    def _local_html_context(
        self,
        *,
        snapshot_html: str,
        state: AgentState,
        candidates: list[Candidate],
        active_region: dict[str, Any],
    ) -> dict[str, Any]:
        if not _env_bool("FSM_USE_LOCAL_HTML_CONTEXT", True):
            return {}
        html = str(snapshot_html or "")
        if not html or BeautifulSoup is None:
            return {}
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            return {}
        by_id = {cand.id: cand for cand in candidates if isinstance(cand, Candidate) and cand.id}
        region_item_ids = [str(item.get("id") or "") for item in list(active_region.get("items") or []) if isinstance(item, dict)]
        target_candidates: list[Candidate] = []
        for cand_id in region_item_ids + list(state.form_progress.typed_candidate_ids or []):
            cand = by_id.get(str(cand_id))
            if cand is not None and cand not in target_candidates:
                target_candidates.append(cand)
        if not target_candidates:
            for cand in candidates[:6]:
                if isinstance(cand, Candidate) and cand not in target_candidates:
                    target_candidates.append(cand)
                if len(target_candidates) >= 4:
                    break
        active_form = None
        target_candidate_htmls: list[str] = []
        for cand in target_candidates:
            node = self._candidate_node(soup=soup, candidate=cand)
            if node is None:
                continue
            target_candidate_htmls.append(str(node)[:900])
            form = node.find_parent("form") if hasattr(node, "find_parent") else None
            if form is not None:
                active_form = form
                break
            if active_form is None:
                active_form = self._nearest_region_container(node)
        last_used_html = ""
        last_cand = by_id.get(str(state.last_action_element_id or ""))
        last_node = self._candidate_node(soup=soup, candidate=last_cand) if last_cand is not None else None
        if last_node is not None:
            last_used_html = str(last_node)[:2000]
        commit_htmls: list[str] = []
        search_root = active_form or soup
        try:
            for node in search_root.find_all(["button", "input", "a"], limit=24):
                attrs = node.attrs if isinstance(getattr(node, "attrs", None), dict) else {}
                blob = " ".join(
                    [
                        str(getattr(node, "name", "") or ""),
                        str(attrs.get("type") or ""),
                        _norm_ws(attrs.get("aria-label")),
                        _norm_ws(attrs.get("value")),
                        _norm_ws(node.get_text(" ", strip=True)),
                    ]
                ).lower()
                if re.search(r"\b(save|submit|post|publish|apply|create|send|confirm)\b", blob):
                    commit_htmls.append(str(node)[:1200])
        except Exception:
            pass
        active_form_html = str(active_form)[:2500] if active_form is not None else ""
        target_container_html = ""
        if active_form is None and target_candidates:
            try:
                probe_node = self._candidate_node(soup=soup, candidate=target_candidates[0])
                if probe_node is not None:
                    target_container_html = str(self._nearest_region_container(probe_node) or "")[:1800]
            except Exception:
                target_container_html = ""
        return {
            "active_form_html": active_form_html,
            "target_container_html": target_container_html,
            "target_candidate_htmls": _dedupe_keep_order(target_candidate_htmls, 4),
            "last_used_element_html": last_used_html,
            "commit_candidate_htmls": _dedupe_keep_order(commit_htmls, 3),
        }

    def build_policy_obs(
        self,
        *,
        task_id: str,
        prompt: str,
        web_project_id: str = "",
        use_case: dict[str, str] | None = None,
        snapshot_html: str = "",
        step_index: int,
        url: str,
        mode: str,
        flags: dict[str, Any],
        state: AgentState,
        text_ir: dict[str, Any],
        candidates: list[Candidate],
        history: list[dict[str, Any]],
        screenshot_available: bool = False,
    ) -> dict[str, Any]:
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
        state.memory.strategy_summary = _candidate_text((page_observations.get("capability_gap") if isinstance(page_observations, dict) else {}).get("strategy_summary"))[:320]
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
        plan_items = [{"id": sg.id, "text": sg.text, "status": sg.status} for sg in state.plan.subgoals]
        active_objective = self._active_objective_summary(
            prompt=prompt,
            state=state,
            page_observations=page_observations,
            active_subgoal=active,
            active_region=active_region,
            active_group=active_group,
        )
        avoid_repeating = self._avoid_repeating_summary(state=state, candidates=policy_candidates)
        reasoning_trace = state.memory.reasoning_trace if isinstance(state.memory.reasoning_trace, dict) else {}
        working_state = self._working_state_summary(
            prompt=prompt,
            state=state,
            text_ir=text_ir,
            page_observations=page_observations,
            active_region=active_region,
            active_group=active_group,
        )
        local_workflow_closure = self._local_workflow_closure_summary(
            state=state,
            active_region=active_region,
            candidates=policy_candidates,
        )
        local_html_context = self._local_html_context(
            snapshot_html=snapshot_html,
            state=state,
            candidates=policy_candidates,
            active_region=active_region,
        )
        state.memory.working_state = working_state
        tagged_input = self._tagged_policy_input(
            prompt=prompt,
            web_project_id=web_project_id,
            use_case=use_case,
            snapshot_html=snapshot_html,
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
        indexed_policy_candidates = self._indexed_candidate_obs(policy_candidates, limit=24)
        site_knowledge = (
            _build_site_knowledge(
                _candidate_text(web_project_id),
                _normalize_use_case_info(use_case),
                prompt,
                current_url=url,
                snapshot_html=snapshot_html,
                candidates=policy_candidates,
            )
            if _env_bool("FSM_USE_SITE_KNOWLEDGE", False)
            else {}
        )
        return {
            "task_id": str(task_id or ""),
            "web_project_id": _candidate_text(web_project_id),
            "use_case": _normalize_use_case_info(use_case),
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
            "available_browser_tools": sorted(_supported_browser_tool_names()),
            "unavailable_browser_tools": list(_unavailable_browser_tools()),
            "flags": flags,
            "page_observations": page_observations,
            "previous_step_verdict": verdict,
            "loop_nudges": loop_nudges,
            "active_objective": active_objective,
            "working_state": working_state,
            "score_feedback": state.score_feedback if isinstance(state.score_feedback, dict) else {},
            "state_score": (float((state.score_feedback or {}).get("score")) if isinstance(state.score_feedback, dict) and "score" in state.score_feedback else None),
            "local_workflow_closure": local_workflow_closure,
            "local_html_context": local_html_context,
            "site_knowledge": site_knowledge,
            "avoid_repeating": avoid_repeating,
            "reasoning_trace": reasoning_trace,
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
                "reasoning_trace": reasoning_trace,
                "working_state": working_state,
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
            "candidates": indexed_policy_candidates,
            "candidate_partitions": {
                "local": self._indexed_candidate_obs(candidate_partitions.get("local") or [], limit=10),
                "escape": self._indexed_candidate_obs(candidate_partitions.get("escape") or [], limit=6),
                "global": self._indexed_candidate_obs(candidate_partitions.get("global") or [], limit=8),
                "suppressed_global_count": int(candidate_partitions.get("suppressed_global_count") or 0),
            },
            "policy_input_text": tagged_input,
        }


__all__ = [name for name in globals() if not name.startswith("__")]
