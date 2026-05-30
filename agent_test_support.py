from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass, field
from typing import Any

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover
    BeautifulSoup = None


_TASK_STATE: dict[str, dict[str, object]] = {}

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
    "tell",
    "show",
    "website",
    "page",
    "current",
    "value",
    "finish",
    "go",
    "your",
}

_NOISY_TEXT_MARKERS = {
    "error 5",
    "error 52",
    "cloudflare",
    "hosting provider",
    "temporarily unavailable",
    "forbidden",
    "not found",
}


@dataclass
class _Candidate:
    selector: dict[str, Any]
    text: str
    tag: str
    attrs: dict[str, str]
    context: str = ""
    context_raw: str = ""
    group: str = ""
    container_chain: list[str] = field(default_factory=list)


def _norm_ws(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _safe_truncate(value: str, limit: int) -> str:
    text = str(value or "")
    return text if len(text) <= limit else f"{text[: max(0, limit - 3)]}..."


def _extract_focus_terms(text: str, *, max_terms: int = 18) -> set[str]:
    tokens = [t for t in re.findall(r"[a-z0-9]{3,}", str(text or "").lower()) if t not in _TASK_TERM_STOPWORDS]
    freq: dict[str, int] = {}
    for token in tokens:
        freq[token] = freq.get(token, 0) + 1
    ranked = sorted(freq.items(), key=lambda item: (item[1], len(item[0]), item[0]), reverse=True)
    return {token for token, _ in ranked[: max(1, int(max_terms))]}


def _score_text_relevance(text: str, focus_terms: set[str]) -> float:
    if not focus_terms:
        return 0.0
    tokens = set(re.findall(r"[a-z0-9]{3,}", str(text or "").lower()))
    if not tokens:
        return 0.0
    overlap = tokens.intersection(focus_terms)
    return float(len(overlap)) / float(max(1, min(len(focus_terms), 8)))


def _text_quality_score(text: str) -> float:
    raw = _norm_ws(str(text or ""))
    if not raw:
        return -2.0
    lowered = raw.lower()
    if any(marker in lowered for marker in _NOISY_TEXT_MARKERS):
        return -2.0

    tokens = re.findall(r"[A-Za-z0-9_.%+-]+", raw)
    if len(tokens) < 3:
        return -0.8

    alpha = sum(ch.isalpha() for ch in raw)
    digits = sum(ch.isdigit() for ch in raw)
    punct = sum((not ch.isalnum()) and (not ch.isspace()) for ch in raw)
    token_count = len(tokens)
    unique_ratio = len({token.lower() for token in tokens}) / max(1, token_count)
    digit_ratio = digits / max(1, alpha + digits)
    punct_ratio = punct / max(1, len(raw))
    prefix = tokens[:8]
    prefix_numeric = 0
    for token in prefix:
        stripped = token.replace(",", "").replace(".", "", 1).replace("%", "")
        if stripped.isdigit():
            prefix_numeric += 1

    score = 0.0
    score += 1.2 if unique_ratio >= 0.55 else 0.4
    if digit_ratio > 0.45:
        score -= 1.4
    elif digit_ratio > 0.30:
        score -= 0.8
    if punct_ratio > 0.30:
        score -= 0.8
    if prefix and (prefix_numeric / max(1, len(prefix))) >= 0.6:
        score -= 0.9
    if token_count >= 7:
        score += 0.4
    return score


def _strip_numeric_noise_chunks(text: str) -> str:
    value = _norm_ws(str(text or ""))
    if not value:
        return ""
    value = re.sub(r"(?:\b[\d][\d.,%+\-/:]*\b\s*){4,}", " ", value)
    return _norm_ws(value).strip(" |:-")


def _host_label(host: str) -> str:
    raw = str(host or "").strip().lower()
    if not raw:
        return "Unknown host"
    parts = [part for part in raw.split(".") if part and part != "www"]
    if not parts:
        return raw
    suffixes = {"com", "org", "net", "io", "co", "app", "dev", "ai", "gg", "tv", "me"}
    if len(parts) >= 3 and parts[-1] in suffixes and parts[-2] in suffixes:
        core = parts[-3]
    elif len(parts) >= 2:
        core = parts[-2]
    else:
        core = parts[0]
    return core.replace("-", " ").replace("_", " ").strip().title() or raw


def _extract_urls(text: str) -> list[str]:
    found = re.findall(r"https?://[^\s)>\"]+", str(text or ""), flags=re.I)
    out: list[str] = []
    seen: set[str] = set()
    for value in found:
        candidate = value.strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    return out[:6]


def _extract_host_hints(text: str) -> list[str]:
    found = re.findall(r"\b[a-z0-9.-]+\.[a-z]{2,}\b", str(text or "").lower())
    out: list[str] = []
    seen: set[str] = set()
    for value in found:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out[:6]


def _split_task_subgoals(task: str) -> list[str]:
    text = _norm_ws(task)
    if not text:
        return []
    parts = re.split(r"\bthen\b|\band\b|[,;]", text, flags=re.I)
    out: list[str] = []
    for part in parts:
        cleaned = _norm_ws(re.sub(r"^(please|kindly)\s+", "", part, flags=re.I))
        if cleaned:
            out.append(cleaned[:120])
    return out[:6] if out else [text[:120]]


def _ensure_subgoal_memory(task_id: str, task: str) -> dict[str, Any] | None:
    if not task_id:
        return None
    state = _TASK_STATE.setdefault(task_id, {})
    existing = state.get("subgoal_memory")
    if isinstance(existing, dict) and isinstance(existing.get("subgoals"), list):
        return existing

    subgoals = []
    for idx, subgoal_text in enumerate(_split_task_subgoals(task)):
        tokens = [token for token in re.findall(r"[a-z0-9]{3,}", subgoal_text.lower()) if token not in {"then", "open", "goto", "navigate", "click", "finish"}]
        subgoals.append(
            {
                "id": idx,
                "text": subgoal_text,
                "urls": _extract_urls(subgoal_text),
                "hosts": _extract_host_hints(subgoal_text),
                "tokens": tokens[:10],
                "done": False,
                "blocked": False,
                "evidence": "",
                "fail_count": 0,
            }
        )

    memory = {
        "task": _norm_ws(task)[:180],
        "subgoals": subgoals,
        "active_id": 0 if subgoals else -1,
        "last_progress_step": -1,
        "stall_count": 0,
    }
    state["subgoal_memory"] = memory
    return memory


def _typed_values_from_history(history: list[dict[str, Any]] | None) -> list[str]:
    values: list[str] = []
    for item in history or []:
        if not isinstance(item, dict):
            continue
        action = item.get("action") if isinstance(item.get("action"), dict) else {}
        action_type = str(action.get("type") or "")
        if action_type in {"TypeAction", "FillAction", "SelectDropDownOptionAction"}:
            text = str(action.get("text") or action.get("value") or "").strip()
            if text:
                values.append(text)
    return values


def _subgoal_done_by_state(
    subgoal: dict[str, Any],
    *,
    url: str,
    page_ir_text: str,
    page_summary: str,
    history: list[dict[str, Any]] | None,
) -> tuple[bool, str]:
    url_l = str(url or "").lower()
    blob = "\n".join([url_l, str(page_ir_text or "").lower(), str(page_summary or "").lower()])
    subgoal_text = str(subgoal.get("text") or "").lower()
    typed_values = [value.lower() for value in _typed_values_from_history(history)]

    if any(marker in subgoal_text for marker in {"login", "sign in", "signin", "email", "password"}):
        needs_email = "email" in subgoal_text or "@" in subgoal_text
        needs_password = "password" in subgoal_text
        email_ok = (not needs_email) or any("@" in value for value in typed_values)
        password_ok = (not needs_password) or any(value.strip() == "password" for value in typed_values)
        on_auth_page = any(marker in url_l for marker in {"login", "sign-in", "signin", "auth"})
        if email_ok and password_ok and on_auth_page:
            return True, "credential_inputs_typed"
        return False, ""

    for target in subgoal.get("urls") or []:
        value = str(target).lower()
        if value and (value in url_l or value in blob):
            return True, f"matched_url:{target}"
    for host in subgoal.get("hosts") or []:
        value = str(host).lower()
        if value and value in url_l:
            return True, f"matched_host:{host}"
    tokens = [str(token).lower() for token in (subgoal.get("tokens") or []) if str(token).strip()]
    if tokens:
        hits = sum(1 for token in tokens if token in blob)
        if hits >= min(2, max(1, len(tokens))):
            return True, f"token_hits:{hits}"
    return False, ""


def _update_subgoal_memory(
    mem: dict[str, Any] | None,
    *,
    step_index: int,
    url: str,
    page_ir_text: str,
    page_summary: str,
    history: list[dict[str, Any]] | None,
    repeat_count: int,
) -> None:
    if not isinstance(mem, dict):
        return
    subgoals = mem.get("subgoals")
    if not isinstance(subgoals, list) or not subgoals:
        return

    progressed = False
    for subgoal in subgoals:
        if not isinstance(subgoal, dict) or bool(subgoal.get("done")):
            continue
        done, evidence = _subgoal_done_by_state(
            subgoal,
            url=url,
            page_ir_text=page_ir_text,
            page_summary=page_summary,
            history=history,
        )
        if done:
            subgoal["done"] = True
            subgoal["blocked"] = False
            subgoal["evidence"] = evidence
            subgoal["fail_count"] = 0
            progressed = True

    active_id = -1
    for subgoal in subgoals:
        if isinstance(subgoal, dict) and not bool(subgoal.get("done")):
            active_id = int(subgoal.get("id") or 0)
            break
    mem["active_id"] = active_id

    if progressed:
        mem["last_progress_step"] = int(step_index)
        mem["stall_count"] = 0
    else:
        mem["stall_count"] = int(mem.get("stall_count") or 0) + 1

    if repeat_count >= 2 and active_id >= 0:
        for subgoal in subgoals:
            if isinstance(subgoal, dict) and int(subgoal.get("id") or -1) == active_id and not bool(subgoal.get("done")):
                subgoal["fail_count"] = int(subgoal.get("fail_count") or 0) + 1
                if int(subgoal.get("fail_count") or 0) >= 3:
                    subgoal["blocked"] = True
                break


def _all_subgoals_done(mem: dict[str, Any] | None) -> bool:
    if not isinstance(mem, dict):
        return False
    subgoals = mem.get("subgoals")
    if not isinstance(subgoals, list) or not subgoals:
        return False
    return all(isinstance(subgoal, dict) and bool(subgoal.get("done")) for subgoal in subgoals)


def _task_is_info_seeking(task: str) -> bool:
    text = str(task or "").lower()
    markers = {
        "what",
        "which",
        "tell",
        "list",
        "summary",
        "summarize",
        "information",
        "details",
        "find",
    }
    return bool(text) and any(marker in text for marker in markers)


def _task_is_auth_flow(task: str) -> bool:
    text = str(task or "").lower()
    markers = {
        "login",
        "log in",
        "signin",
        "sign in",
        "register",
        "sign up",
        "password",
    }
    return bool(text) and any(marker in text for marker in markers)


def _render_observation_result(task: str, task_id: str) -> str | None:
    if not task_id or not _task_is_info_seeking(task):
        return None
    state = _TASK_STATE.get(task_id)
    if not isinstance(state, dict):
        return None
    observations = state.get("observations")
    if not isinstance(observations, list) or not observations:
        return None

    focus_terms = _extract_focus_terms(task)
    rendered: list[str] = []
    used: set[str] = set()
    for item in [entry for entry in observations if isinstance(entry, dict)][-4:]:
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        facts = [str(value) for value in (item.get("facts") or []) if isinstance(value, str)]
        headings = [str(value) for value in (item.get("headings") or []) if isinstance(value, str)]
        candidates = facts[:3] + headings[:3]
        if title:
            candidates.append(title)
        if isinstance(item.get("summary"), str):
            candidates.append(str(item.get("summary") or ""))

        scored: list[tuple[float, str]] = []
        for candidate in candidates:
            cleaned = _strip_numeric_noise_chunks(candidate)
            if not cleaned:
                continue
            quality = _text_quality_score(cleaned)
            relevance = _score_text_relevance(cleaned, focus_terms)
            score = quality + (2.6 * relevance)
            scored.append((score, cleaned))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        if not scored:
            continue
        best_score, best_text = scored[0]
        if best_score < -0.2:
            continue
        if not _task_is_auth_flow(task) and any(marker in url.lower() for marker in {"login", "sign-in", "signin", "auth"}):
            continue
        normalized = _norm_ws(best_text).lower()
        if normalized in used:
            continue
        if any(marker in normalized for marker in _NOISY_TEXT_MARKERS):
            continue
        used.add(normalized)
        rendered.append(f"- {url[:80] if url else 'current page'}: {best_text[:180]}")

    if not rendered:
        return None
    return ("Findings:\n" + "\n".join(rendered[:4]))[:900]


def _candidate_blob(candidate: _Candidate) -> str:
    attrs = candidate.attrs or {}
    bits = [
        candidate.tag,
        candidate.text or "",
        candidate.context or "",
        attrs.get("id") or "",
        attrs.get("name") or "",
        attrs.get("placeholder") or "",
        attrs.get("aria-label") or "",
        attrs.get("href") or "",
    ]
    return _norm_ws(" ".join(str(value) for value in bits)).lower()


def _pick_fallback_candidate_id(
    *,
    candidates: list[_Candidate],
    action: str,
    decision: dict[str, Any],
    avoid_id: int | None = None,
) -> int | None:
    if not candidates:
        return None
    act = str(action or "").lower().strip()
    want_text = _norm_ws(str(decision.get("text") or "")).lower()
    want_url = _norm_ws(str(decision.get("url") or "")).lower()
    want_tokens = set(re.findall(r"[a-z0-9]{2,}", want_text))

    scored: list[tuple[float, int]] = []
    for idx, candidate in enumerate(candidates):
        if avoid_id is not None and idx == avoid_id:
            continue
        score = 0.0
        attrs = candidate.attrs or {}
        blob = _candidate_blob(candidate)
        if act == "click":
            if candidate.tag in {"a", "button"}:
                score += 4.0
            if attrs.get("href"):
                score += 1.5
            if want_url:
                href = str(attrs.get("href") or "").lower()
                if href and (want_url in href or href in want_url):
                    score += 7.0
            if want_tokens:
                tokens = set(re.findall(r"[a-z0-9]{2,}", blob))
                score += min(6.0, 1.5 * len(tokens.intersection(want_tokens)))
        elif act == "type":
            if candidate.tag in {"input", "textarea"}:
                score += 6.0
            if str(attrs.get("type") or "").lower() not in {
                "hidden",
                "submit",
                "button",
            }:
                score += 1.0
        elif act == "select":
            if candidate.tag == "select":
                score += 8.0
        else:
            continue
        if act in {"type", "select"} and want_tokens:
            tokens = set(re.findall(r"[a-z0-9]{2,}", blob))
            score += min(5.0, 1.3 * len(tokens.intersection(want_tokens)))
        scored.append((score, idx))

    if not scored:
        return None
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return int(scored[0][1]) if scored[0][0] > 0.0 else None


def _tool_visible_text(*, html: str, max_chars: int = 2000) -> dict[str, Any]:
    if BeautifulSoup is None:
        text = _norm_ws(re.sub(r"<[^>]+>", " ", str(html or "")))
        return {"ok": True, "text": _safe_truncate(text, int(max_chars or 0))}
    try:
        soup = BeautifulSoup(html or "", "lxml")
        for node in soup(["script", "style", "noscript"]):
            with contextlib.suppress(Exception):
                node.decompose()
        text = _norm_ws(soup.get_text(" ", strip=True))
        return {"ok": True, "text": _safe_truncate(text, int(max_chars or 0))}
    except Exception as exc:
        return {"ok": False, "error": f"extract text failed: {str(exc)[:160]}"}


def _tool_extract_tables(*, html: str, max_tables: int = 6, max_rows: int = 8, max_cols: int = 8) -> dict[str, Any]:
    if BeautifulSoup is None:
        return {"ok": False, "error": "bs4 not available"}
    try:
        soup = BeautifulSoup(html or "", "lxml")
    except Exception as exc:
        return {"ok": False, "error": f"parse failed: {str(exc)[:160]}"}

    tables: list[dict[str, Any]] = []
    for table in soup.find_all("table")[: int(max_tables or 0)]:
        try:
            headers = [_safe_truncate(_norm_ws(header.get_text(" ", strip=True)), 80) for header in table.find_all("th")[: int(max_cols or 0)] if _norm_ws(header.get_text(" ", strip=True))]
            rows: list[list[str]] = []
            for row in table.find_all("tr")[: int(max_rows or 0)]:
                cells = [_safe_truncate(_norm_ws(cell.get_text(" ", strip=True)), 120) for cell in row.find_all(["td", "th"])[: int(max_cols or 0)]]
                if any(cells):
                    rows.append(cells)
            caption_node = table.find("caption")
            caption = ""
            if caption_node is not None:
                caption = _safe_truncate(_norm_ws(caption_node.get_text(" ", strip=True)), 120)
            tables.append({"caption": caption, "headers": headers, "rows": rows})
        except Exception:
            continue
    return {"ok": True, "count": len(tables), "tables": tables}


def _tool_extract_entities(*, html: str, max_items: int = 50) -> dict[str, Any]:
    visible = _tool_visible_text(html=html, max_chars=20000)
    if not visible.get("ok"):
        return {"ok": False, "error": "visible text extraction failed"}
    text = str(visible.get("text") or "")

    def _uniq(values: list[str], limit: int) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = _norm_ws(value)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            out.append(normalized)
            if len(out) >= limit:
                break
        return out

    limit = int(max_items or 0)
    return {
        "ok": True,
        "entities": {
            "emails": _uniq(
                re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text),
                limit,
            ),
            "phones": _uniq(re.findall(r"(?:\+?\d[\d\-\s().]{7,}\d)", text), limit),
            "urls": _uniq(re.findall(r"https?://[^\s)>\"]+", text), limit),
            "prices": _uniq(re.findall(r"(?:\$|USD\s?)\d+(?:[.,]\d{2})?", text), limit),
            "dates": _uniq(
                re.findall(
                    r"(?:\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}\b)",
                    text,
                    flags=re.I,
                ),
                limit,
            ),
        },
    }
