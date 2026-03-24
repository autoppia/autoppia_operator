from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from bs4 import BeautifulSoup

from src.operator.agents.fsm import AgentState

from .utils import (
    _candidate_text,
    _constraint_key_tokens,
    _constraint_value_matches,
    _dedupe_keep_order,
    _focus_terms,
    _norm_ws,
    _safe_url,
    _sanitize_selector,
    _task_constraints,
    _tokenize,
)


@dataclass
class Candidate:
    id: str
    role: str
    type: str
    text: str
    href: str
    context: str
    selector: dict[str, Any]
    dom_path: str
    field_hint: str = ""
    field_kind: str = ""
    input_type: str = ""
    ui_state: str = ""
    region_id: str = ""
    region_kind: str = ""
    region_label: str = ""
    parent_region_id: str = ""
    region_ancestor_ids: list[str] = field(default_factory=list)
    group_id: str = ""
    group_label: str = ""
    disabled: bool = False
    required: bool = False
    readonly: bool = False
    placeholder: str = ""
    aria_label: str = ""
    name_attr: str = ""
    current_value: str = ""
    option_values: list[str] = field(default_factory=list)
    bbox: dict[str, float] | None = None

    def as_obs(self, *, index: int | None = None) -> dict[str, Any]:
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
            "required": bool(self.required),
            "readonly": bool(self.readonly),
            "placeholder": self.placeholder[:160] if self.placeholder else "",
            "aria_label": self.aria_label[:160] if self.aria_label else "",
            "name_attr": self.name_attr[:80] if self.name_attr else "",
            "current_value": self.current_value[:120] if self.current_value else "",
            "option_values": self.option_values[:8] if self.option_values else [],
            "bbox": self.bbox,
        }
        if index is not None:
            out["index"] = int(index)
        return out


class CandidateExtractor:
    def extract(self, *, snapshot_html: str, url: str) -> list[Candidate]:
        html = str(snapshot_html or "")
        if not html or BeautifulSoup is None:
            return []
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            return []
        nodes = list(soup.select("a,button,input,select,textarea,[role='button'],[role='link'],[role='tab'],[role='menuitem'],[role='checkbox'],[role='radio'],[role='switch'],[role='combobox']"))
        id_counts: dict[str, int] = {}
        for node in nodes:
            attrs = node.attrs if isinstance(getattr(node, "attrs", None), dict) else {}
            node_id = _norm_ws(attrs.get("id"))
            if node_id:
                id_counts[node_id] = int(id_counts.get(node_id) or 0) + 1
        out: list[Candidate] = []
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
                    selected_texts: list[str] = []
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
                stable_id = self._stable_id(
                    attrs=attrs,
                    selector=selector,
                    text=text,
                    href=href,
                    dom_path=dom_path,
                )
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
                required = ("required" in attrs) or (str(attrs.get("aria-required") or "").strip().lower() == "true")
                readonly = ("readonly" in attrs) or (str(attrs.get("aria-readonly") or "").strip().lower() == "true")
                placeholder = _norm_ws(attrs.get("placeholder"))
                aria_label = _norm_ws(attrs.get("aria-label"))
                name_attr = _norm_ws(attrs.get("name"))
                option_values: list[str] = []
                if tag == "select":
                    for option in node.find_all("option", limit=16):
                        opt_text = _norm_ws(option.get_text(" ", strip=True))
                        opt_value = _norm_ws(option.get("value"))
                        opt_blob = _candidate_text(opt_text, opt_value)
                        if opt_blob:
                            option_values.append(opt_blob[:80])
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
                        required=bool(required),
                        readonly=bool(readonly),
                        placeholder=placeholder[:160],
                        aria_label=aria_label[:160],
                        name_attr=name_attr[:80],
                        current_value=current_value[:120],
                        option_values=_dedupe_keep_order(option_values, 8),
                        bbox=None,
                    )
                )
            except Exception:
                continue
        dedup: dict[str, Candidate] = {}
        for cand in out:
            dedup[cand.id] = cand
        return list(dedup.values())[:220]

    def _role_name(self, tag: str, role: str, attrs: dict[str, Any]) -> str:
        if tag == "a" or role == "link":
            return "link"
        if tag == "button" or role in {
            "button",
            "tab",
            "menuitem",
            "checkbox",
            "radio",
            "switch",
        }:
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
        parts: list[str] = []
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
        attrs: dict[str, Any],
        text: str,
        href: str,
        raw_href: str,
        dom_path: str,
        id_counts: dict[str, int],
    ) -> dict[str, Any]:
        node_id = _norm_ws(attrs.get("id"))
        if node_id and int(id_counts.get(node_id) or 0) <= 1:
            return {
                "type": "attributeValueSelector",
                "attribute": "id",
                "value": node_id,
                "case_sensitive": False,
            }
        node_name = _norm_ws(attrs.get("name"))
        if node_name and tag in {"input", "textarea", "select"}:
            return {
                "type": "attributeValueSelector",
                "attribute": "name",
                "value": node_name,
                "case_sensitive": False,
            }
        if raw_href and tag == "a":
            return {
                "type": "attributeValueSelector",
                "attribute": "href",
                "value": raw_href,
                "case_sensitive": False,
            }
        if text and tag in {"button", "a"}:
            clean = text.replace('"', "'")[:120]
            xpath = f'//{tag}[contains(normalize-space(.), "{clean}")]'
            return {"type": "xpathSelector", "value": xpath, "case_sensitive": False}
        if dom_path:
            # IWA Selector prepends '//' for xpath values that do not start with '//'.
            # Keep this as a raw DOM path to avoid generating invalid triple-slash selectors.
            return {"type": "xpathSelector", "value": dom_path, "case_sensitive": False}
        return {
            "type": "xpathSelector",
            "value": f"//{tag}[1]",
            "case_sensitive": False,
        }

    def _stable_id(
        self,
        *,
        attrs: dict[str, Any],
        selector: dict[str, Any],
        text: str,
        href: str,
        dom_path: str,
    ) -> str:
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
                if str(getattr(container, "name", "") or "").lower() in {
                    "section",
                    "article",
                    "div",
                    "main",
                    "li",
                }:
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
        classes = " ".join(str(x) for x in (attrs.get("class") or [] if isinstance(attrs.get("class"), list) else [attrs.get("class")])).lower()
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

    def _region_lineage(self, node: Any) -> tuple[str, list[str]]:
        parent_region_id = ""
        ancestor_ids: list[str] = []
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
        attrs: dict[str, Any],
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
        task_constraints: dict[str, str],
    ) -> dict[str, str]:
        if not task_constraints:
            return {}
        blob = " ".join(
            [
                cand.text,
                cand.field_hint,
                cand.context,
                cand.group_label,
                cand.region_label,
                cand.href,
            ]
        ).lower()
        matches: dict[str, str] = {}
        for key, value in task_constraints.items():
            key_tokens = _constraint_key_tokens(key)
            key_match = bool(key_tokens and key_tokens.intersection(_tokenize(blob)))
            value_match = _constraint_value_matches(str(value), cand.text)
            if value_match:
                matches[str(key)] = "value"
            elif key_match:
                matches[str(key)] = "field"
        return matches

    def _candidate_constraint_keys(self, *, cand: Candidate, task_constraints: dict[str, str]) -> set[str]:
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
        if re.search(
            r"\bgenre\b|\bgenres\b|\bcategory\b|\bcategories\b|\btag\b|\btags\b",
            text,
            flags=re.I,
        ):
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
        return bool(re.search(r"\b(tab|section|panel|view)\b", blob) and cand.role == "button")

    def _looks_submit_like(self, cand: Candidate) -> bool:
        if cand.field_kind in {"submit", "account_create", "auth_entry"}:
            return True
        if cand.role == "input" and str(cand.input_type or "").strip().lower() in {
            "submit",
            "button",
            "image",
            "reset",
        }:
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

    def _selector_signature(self, selector: dict[str, Any] | None) -> str:
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
        flags: dict[str, Any],
        candidates: list[Candidate],
        state: AgentState,
        current_url: str = "",
        top_k: int = 30,
    ) -> list[Candidate]:
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
        candidate_tags: dict[str, set[str]] = {cand.id: self._candidate_action_tags(cand) for cand in candidates}
        group_stats: dict[str, dict[str, Any]] = {}
        exact_value_match_keys: dict[str, int] = {}
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
                for key in task_constraints:
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
            if int(stats.get("constraint_hits") or 0) > 0 or (prompt_needs.intersection(set(stats.get("field_kinds") or set())) and int(stats.get("input_like") or 0) > 0)
        }
        has_visible_constraint_match = False
        for cand in candidates:
            blob = " ".join([cand.text, cand.href, cand.context, cand.field_hint]).lower()
            constraint_matches = self._candidate_constraint_match(cand=cand, task_constraints=task_constraints)
            if cand.role in {"select", "input"} and any(match == "value" for match in constraint_matches.values()) and ("current=" in blob or " value=" in blob):
                has_visible_constraint_match = True
                break
        credential_needs = {
            "username",
            "email",
            "password",
            "confirm_password",
        }.intersection(prompt_needs)
        has_relevant_form_group = any(key in relevant_groups and int(stats.get("input_like") or 0) > 0 for key, stats in group_stats.items())
        has_direct_constraint_controls = any(cand.role in {"input", "select"} and cand.field_kind and cand.field_kind in prompt_needs for cand in candidates)
        has_password_input_visible = any(cand.role == "input" and cand.field_kind in {"password", "confirm_password"} for cand in candidates)
        has_non_form_controls = any(cand.role in {"button", "link"} for cand in candidates)
        has_same_region_commit_controls = any(self._looks_submit_like(cand) and not cand.disabled for cand in candidates)
        has_mutation_controls = any(candidate_tags.get(cand.id, set()).intersection(mutation_ops) for cand in candidates if cand.role in {"button", "link", "input"})
        scored: list[tuple[float, Candidate]] = []
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
            current_value_already_matches = cand.role in {"select", "input"} and any(constraint_matches.get(key) == "value" for key in constraint_keys) and ("current=" in blob or " value=" in blob)
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
            if cand.field_kind == "sort" and "year" in prompt_needs and any("year" in set(stats.get("field_kinds") or set()) for stats in group_stats.values()):
                score -= 4.4
            if (
                {"username", "email", "password", "confirm_password"}.intersection(prompt_needs)
                and cand.role == "input"
                and cand.field_kind in {"text", "name", "search"}
                and not ({"username", "email", "password", "confirm_password"} & group_kinds)
            ):
                score -= 4.6
            if group_key in relevant_groups:
                score += 3.8
                if cand.role in {"input", "select", "button"}:
                    score += 1.0
            if {"username", "password"}.issubset(group_kinds) or {
                "email",
                "password",
            }.issubset(group_kinds):
                score += 2.6
            if {"username", "email", "password"}.intersection(group_kinds) and cand.role in {"input", "button"}:
                score += 1.4
            if cand.id in visual_hints:
                score += 4.5
            if cand.id in obs_hints:
                score += 3.2
            same_focus_region = False
            if (
                (focus_region_id and cand.region_id and cand.region_id == focus_region_id)
                or (focus_region_context and cand_context and cand_context == focus_region_context)
                or (focus_region_ids and cand.id in focus_region_ids)
            ):
                same_focus_region = True
            if same_focus_region:
                score += 4.4
                if cand.role in {"button", "input", "select"}:
                    score += 1.8
                if last_action_type in {
                    "typeaction",
                    "selectdropdownoptionaction",
                    "clickaction",
                }:
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
            if (
                (active_group_id and cand.group_id and cand.group_id == active_group_id)
                or (active_group_context and cand_context and cand_context == active_group_context)
                or (active_group_ids and cand.id in active_group_ids)
            ):
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
            relevant_form_field_kinds = {kind for kind in group_kinds if kind in {"username", "email", "password", "confirm_password"} or kind in prompt_needs}
            has_local_multifield_form = int((group_stats.get(group_key) or {}).get("input_like") or 0) >= 2
            remaining_relevant_kinds = {kind for kind in relevant_form_field_kinds if kind not in group_typed_kinds}
            input_already_typed = cand.role == "input" and ((cand.id and cand.id in typed_candidate_ids) or (sel_sig and sel_sig in typed_selector_sigs))
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
                if (
                    cand.role == "input"
                    and cand.field_kind
                    in {
                        "username",
                        "email",
                        "password",
                        "confirm_password",
                    }
                    and cand.field_kind in remaining_relevant_kinds
                ):
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
                elif any(constraint_matches.get(key) == "field" and int(exact_value_match_keys.get(key) or 0) > 0 for key in unmet_constraint_keys) and cand.role in {"button", "link"}:
                    score -= 3.6
            elif constraint_keys:
                if cand.role in {"button", "link"} and cand.field_kind not in {
                    "submit",
                    "auth_entry",
                    "account_create",
                }:
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
                    if cand.role == "link" and int((group_stats.get(group_key) or {}).get("input_like") or 0) == 0 and int((group_stats.get(group_key) or {}).get("context_len") or 0) >= 220:
                        score -= 2.2
            if mode == "POPUP" and any(
                k in blob
                for k in (
                    "accept",
                    "reject",
                    "agree",
                    "close",
                    "dismiss",
                    "continue",
                )
            ):
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


__all__ = [name for name in globals() if not name.startswith("__")]
