from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import json
import os
import re
import logging
from types import SimpleNamespace
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit
from html.parser import HTMLParser

from fastapi import Body, FastAPI, HTTPException
from dotenv import load_dotenv

# Default this branch to OpenAI via the validator gateway.
os.environ.setdefault("LLM_PROVIDER", "openai")
# Load local development env file when present.
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)

from llm_gateway import openai_chat_completions, is_sandbox_gateway_base_url
from pricing import estimate_cost_usd
from completion_checker import run_completion_check

try:
    from autoppia_iwa.src.web_agents.classes import IWebAgent
    from autoppia_iwa.src.data_generation.tasks.classes import Task
    from autoppia_iwa.src.execution.actions.base import BaseAction
    import autoppia_iwa.src.execution.actions.actions  # noqa: F401
    _AUTOPPIA_IWA_IMPORT_OK = True
    _AUTOPPIA_IWA_IMPORT_ERROR = ""
except Exception:  # pragma: no cover
    IWebAgent = object  # type: ignore[assignment]
    Task = Any  # type: ignore[assignment]
    BaseAction = Any  # type: ignore[assignment]
    _AUTOPPIA_IWA_IMPORT_OK = False
    _AUTOPPIA_IWA_IMPORT_ERROR = "autoppia_iwa import failed in miner runtime"

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover
    BeautifulSoup = None


app = FastAPI(title="Autoppia Web Agent API")
logger = logging.getLogger("autoppia_operator")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


_LOG_DECISIONS = _env_bool("AGENT_LOG_DECISIONS", False)
_LOG_ERRORS = _env_bool("AGENT_LOG_ERRORS", False)


def _log_trace(message: str) -> None:
    if _LOG_DECISIONS:
        logger.info(f"[AGENT_TRACE] {message}")


if not _AUTOPPIA_IWA_IMPORT_OK:
    logger.error(f"[AGENT_TRACE] autoppia_iwa import failed: {_AUTOPPIA_IWA_IMPORT_ERROR}")
else:
    _log_trace(f"autoppia_iwa import ok; BaseAction module={getattr(BaseAction, '__module__', 'unknown')}")


def _normalize_demo_url(raw_url: str | None) -> str:
    """Normalize URL, with optional demo-mode localhost rewriting.

    By default we preserve real URLs.
    Set `AGENT_FORCE_LOCALHOST_URLS=1` to force all URLs onto localhost for demo-web testing.
    """
    normalized = str(raw_url or "").strip()
    if not normalized:
        return normalized

    force_localhost = _env_bool("AGENT_FORCE_LOCALHOST_URLS", False)
    if not force_localhost:
        return normalized

    try:
        if "://" not in normalized:
            if not normalized.startswith("/"):
                # Treat bare host/path values (for example "84.247.180.192/task") as local host
                # while keeping any path/query/fragment.
                if "." in normalized or ":" in normalized:
                    parsed = urlsplit(f"http://{normalized}")
                    path = parsed.path or ""
                    if not path:
                        return "http://localhost"
                    return urlunsplit(("http", "localhost", path, parsed.query, parsed.fragment))
                normalized = f"/{normalized}"
            return f"http://localhost{normalized}"
        parsed = urlsplit(normalized)
        return urlunsplit(("http", "localhost", parsed.path or "/", parsed.query, parsed.fragment))
    except Exception:
        return "http://localhost/"


def _is_navigate_action_type(action_type: Any) -> bool:
    value = str(action_type or "").strip().lower()
    return value in {"navigateaction", "navigate"}


def _sanitize_action_payload(action_payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(action_payload or {})
    if _is_navigate_action_type(payload.get("type")):
        payload["url"] = _normalize_demo_url(payload.get("url"))
    return payload


# Per-task loop detection cache (process-local).
_TASK_STATE: dict[str, dict[str, object]] = {}


@app.get("/health", summary="Health check")
async def health() -> Dict[str, str]:
    return {"status": "healthy"}


# -----------------------------
# IWA Selector helpers
# -----------------------------

def _sel_attr(attribute: str, value: str, case_sensitive: bool = False) -> Dict[str, Any]:
    return {
        "type": "attributeValueSelector",
        "attribute": attribute,
        "value": value,
        "case_sensitive": case_sensitive,
    }


def _sel_text(value: str, case_sensitive: bool = False) -> Dict[str, Any]:
    return {
        "type": "tagContainsSelector",
        "value": value,
        "case_sensitive": case_sensitive,
    }


def _sel_custom(value: str, case_sensitive: bool = False) -> Dict[str, Any]:
    return {
        "type": "attributeValueSelector",
        "attribute": "custom",
        "value": value,
        "case_sensitive": case_sensitive,
    }




def _sel_xpath(value: str) -> Dict[str, Any]:
    return {
        "type": "xpathSelector",
        "attribute": None,
        "value": value,
        "case_sensitive": False,
    }

def _selector_repr(selector: Dict[str, Any]) -> str:
    t = selector.get("type")
    a = selector.get("attribute")
    v = selector.get("value")
    if t == "attributeValueSelector":
        vv = str(v)
        if len(vv) > 80:
            vv = vv[:77] + "..."
        return f"attr[{a}]={vv}"
    if t == "tagContainsSelector":
        return f"text~={v}"
    return str(selector)


# -----------------------------
# Candidate extraction
# -----------------------------

def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


class _Candidate:
    def __init__(
        self,
        selector: Dict[str, Any],
        text: str,
        tag: str,
        attrs: Dict[str, str],
        *,
        text_selector: Optional[Dict[str, Any]] = None,
        context: str = "",
        context_raw: str = "",
        group: str = "",
        container_chain: list[str] | None = None,
    ):
        self.selector = selector
        self.text_selector = text_selector
        self.text = text
        self.tag = tag
        self.attrs = attrs
        self.context = context
        self.context_raw = context_raw
        self.group = group
        self.container_chain = container_chain or []

    def click_selector(self) -> Dict[str, Any]:
        """Selector for click-like actions.

        Prefer stable attribute selectors. Avoid class-based selectors because IWA converts them to `.class` CSS
        and tailwind-style tokens often include `/` or `:` which breaks CSS parsing in Playwright.

        Note: keep this logic generic (no site-specific button text shortcuts).
        """
        if isinstance(self.selector, dict) and self.selector.get("type") == "attributeValueSelector":
            attr = str(self.selector.get("attribute") or "")
            if attr in {"id", "href", "data-testid", "name", "aria-label", "placeholder", "title"}:
                return self.selector

        # If the primary selector isn't a safe attribute selector, try to derive one from attrs.
        for a in ("id", "data-testid", "href", "aria-label", "name", "placeholder", "title"):
            v = (self.attrs or {}).get(a)
            if v:
                return _sel_attr(a, v)

        # Fall back to the element text selector (can be ambiguous, but generic).
        # Generic refinement: if we only have element text, prefer a Playwright :has-text() selector
        # scoped to the tag (button/a). This reduces ambiguity without hardcoding any website logic.
        try:
            t = (self.text or '').strip()
            if t and self.tag in {'button', 'a'}:
                return _sel_custom(f"{self.tag}:has-text({json.dumps(t)})")
        except Exception:
            pass

        if self.text_selector:
            return self.text_selector

        return self.selector

    def type_selector(self) -> Dict[str, Any]:
        """Selector for type/select actions.

        Avoid class selectors for the same reason as click_selector().
        """
        if isinstance(self.selector, dict) and self.selector.get("type") == "attributeValueSelector":
            attr = str(self.selector.get("attribute") or "")
            if attr and attr != "class":
                return self.selector

        for a in ("id", "data-testid", "name", "aria-label", "placeholder", "title"):
            v = (self.attrs or {}).get(a)
            if v:
                return _sel_attr(a, v)

        return _sel_custom(self.tag)


class _CandidateExtractor(HTMLParser):
    """Fallback extractor when BeautifulSoup isn't available."""

    def __init__(self) -> None:
        super().__init__()
        self._current_text: List[str] = []
        self._last_tag: Optional[str] = None
        self._last_attrs: Dict[str, str] = {}
        self.candidates: List[_Candidate] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attr_map = {k: (v or "") for k, v in attrs}
        self._last_tag = tag
        self._last_attrs = attr_map

        if tag in {"button", "a", "input", "textarea", "select"} or attr_map.get("role") in {"button", "link"}:
            label = attr_map.get("aria-label") or attr_map.get("placeholder") or attr_map.get("title") or ""
            selector = _build_selector(tag, attr_map, text=label)
            group = 'FORM' if tag in {'input','textarea','select'} else ('LINKS' if tag=='a' else 'BUTTONS')
            self.candidates.append(_Candidate(selector, label, tag, attr_map, context="", group=group, container_chain=[group]))

    def handle_data(self, data: str) -> None:
        if self._last_tag in {"button", "a"} and data.strip():
            self._current_text.append(data.strip())

    def handle_endtag(self, tag: str) -> None:
        if tag == self._last_tag and self._current_text and self.candidates:
            text = " ".join(self._current_text)[:120]
            c = self.candidates[-1]
            c.text = text or c.text
            if c.tag in {"button", "a"} and c.text:
                c.text_selector = _sel_text(c.text, case_sensitive=False)
        self._current_text = []


def _attrs_to_str_map(attrs: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in (attrs or {}).items():
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            out[k] = " ".join(str(x) for x in v if x is not None).strip()
        else:
            out[k] = str(v)
    return out


def _build_selector(tag: str, attrs: Dict[str, str], *, text: str) -> Dict[str, Any]:
    # Prefer attributes that map directly to IWA Selector.to_playwright_selector().
    if attrs.get("id"):
        return _sel_attr("id", attrs["id"])
    if attrs.get("data-testid"):
        return _sel_attr("data-testid", attrs["data-testid"])
    if tag == "a" and attrs.get("href") and not attrs["href"].lower().startswith("javascript:"):
        return _sel_attr("href", attrs["href"])
    if attrs.get("aria-label"):
        return _sel_attr("aria-label", attrs["aria-label"])
    if attrs.get("name"):
        return _sel_attr("name", attrs["name"])
    if attrs.get("placeholder"):
        return _sel_attr("placeholder", attrs["placeholder"])
    if attrs.get("title"):
        return _sel_attr("title", attrs["title"])
    if text and tag in {"button", "a"}:
        return _sel_text(text, case_sensitive=False)
    return _sel_custom(tag)


def _extract_label_from_bs4(soup, el, attr_map: Dict[str, str]) -> str:
    tag = str(getattr(el, "name", "") or "")

    if tag in {"a", "button"}:
        t = _norm_ws(el.get_text(" ", strip=True))
        if t:
            return t[:120]

    for key in ("aria-label", "placeholder", "title"):
        if attr_map.get(key):
            return _norm_ws(attr_map[key])[:120]

    if attr_map.get("aria-labelledby"):
        lid = attr_map["aria-labelledby"].split()[0]
        if lid:
            lab = soup.find(id=lid)
            if lab is not None:
                t = _norm_ws(lab.get_text(" ", strip=True))
                if t:
                    return t[:120]

    if attr_map.get("id"):
        lab = soup.find("label", attrs={"for": attr_map["id"]})
        if lab is not None:
            t = _norm_ws(lab.get_text(" ", strip=True))
            if t:
                return t[:120]

    parent_label = el.find_parent("label")
    if parent_label is not None:
        t = _norm_ws(parent_label.get_text(" ", strip=True))
        if t:
            return t[:120]

    return ""







def _pick_context_container_bs4(el) -> object | None:
    """Pick a small, card-like container for an element.

    Structural and generic: aims to capture per-item context (e.g., list rows/cards) rather than whole panels.
    """
    try:
        candidates = []
        cur = el
        for _depth in range(8):
            if cur is None:
                break
            try:
                cur = cur.parent
            except Exception:
                break
            if cur is None:
                break
            tag = str(getattr(cur, "name", "") or "")
            if tag not in {"li", "tr", "article", "section", "div", "td"}:
                continue

            try:
                txt_raw = cur.get_text("\n", strip=True)
            except Exception:
                txt_raw = ""
            L = len(txt_raw or "")
            if L <= 0:
                continue

            try:
                n_inter = len(cur.find_all(["a", "button", "input", "select", "textarea"]))
            except Exception:
                n_inter = 0

            candidates.append((L, n_inter, cur))

        if not candidates:
            return None

        best = None
        best_key = None
        for L, n_inter, node in candidates:
            if not (50 <= L <= 900):
                continue
            if n_inter <= 0 or n_inter > 12:
                continue
            key = (L, n_inter)
            if best is None or key < (best_key or key):
                best = node
                best_key = key
        if best is not None:
            return best

        candidates.sort(key=lambda t: (t[0], t[1]))
        return candidates[0][2]
    except Exception:
        return None


def _container_chain_from_el(soup, el) -> list[str]:
    """Return a short container path for an element to render a simplified DOM tree."""
    chain: list[str] = []
    try:
        # Limit depth to keep prompts small.
        ancestors = list(el.parents) if hasattr(el, 'parents') else []
        # BeautifulSoup yields [element, ..., document]; reverse to go top-down.
        for a in reversed(ancestors):
            try:
                tag = str(getattr(a, 'name', '') or '')
                if not tag or tag in {'[document]', 'html', 'body'}:
                    continue
                if tag not in {'header', 'nav', 'main', 'form', 'section', 'article', 'aside', 'footer', 'ul', 'ol', 'table', 'div'}:
                    continue

                aid = ''
                try:
                    aid = str(a.get('id') or a.get('name') or '').strip()
                except Exception:
                    aid = ''

                role = ''
                try:
                    role = str(a.get('role') or '').strip()
                except Exception:
                    role = ''

                # Try to pull a nearby heading (h1-h3) for more semantic labeling.
                heading = ''
                try:
                    h = a.find(['h1', 'h2', 'h3'])
                    if h is not None:
                        heading = _norm_ws(h.get_text(' ', strip=True))
                except Exception:
                    heading = ''

                label_bits = [tag]
                if aid:
                    label_bits.append(f"#{aid}")
                if role and role not in {'presentation'}:
                    label_bits.append(f"role={role}")
                if heading:
                    label_bits.append(heading[:50])

                label = ' '.join([b for b in label_bits if b])
                label = _norm_ws(label)
                if label and (not chain or chain[-1] != label):
                    chain.append(label)
                if len(chain) >= 4:
                    break
            except Exception:
                continue
    except Exception:
        return chain

    # Keep last 3 containers for focus.
    return chain[-3:]



def _extract_candidates_bs4(html: str, *, max_candidates: int) -> List[_Candidate]:
    soup = BeautifulSoup(html, "lxml")

    selectors = [
        "button",
        "a[href]",
        "input",
        "textarea",
        "select",
        "[role='button']",
        "[role='link']",
    ]

    els = []
    for sel in selectors:
        els.extend(soup.select(sel))

    seen: set[tuple[str, str, str]] = set()
    out: List[_Candidate] = []

    for el in els:
        tag = str(getattr(el, "name", "") or "")
        attr_map = _attrs_to_str_map(getattr(el, "attrs", {}) or {})

        group = 'PAGE'
        try:
            # Group by semantic containers for a more browser-use-like state view.
            if el.find_parent('nav') is not None:
                group = 'NAV'
            elif el.find_parent('header') is not None:
                group = 'HEADER'
            elif el.find_parent('footer') is not None:
                group = 'FOOTER'
            elif el.find_parent('form') is not None:
                form = el.find_parent('form')
                fid = ''
                try:
                    fid = str(form.get('id') or form.get('name') or '').strip()
                except Exception:
                    fid = ''
                group = f"FORM:{fid}" if fid else 'FORM'
        except Exception:
            group = group

        # Skip obvious non-interactives.
        if tag == "input" and attr_map.get("type", "").lower() == "hidden":
            continue
        if attr_map.get("disabled") is not None or attr_map.get("aria-disabled", "").lower() == "true":
            continue
        if _is_hidden_candidate_attr(attr_map):
            continue

        label = _extract_label_from_bs4(soup, el, attr_map)

        dom_label = label
        context = ""
        context_raw = ""
        title = ""
        try:
            parent = _pick_context_container_bs4(el) or el.find_parent(["li", "tr", "article", "section", "div"])
            if parent is not None:
                # Preserve line breaks for card-like metadata extraction.
                context_raw = parent.get_text("\n", strip=True)
                context = _norm_ws(context_raw)
                # Try to pull a nearby title so identical buttons become distinguishable.
                h = parent.find(["h1", "h2", "h3", "h4"])
                if h is not None:
                    title = _norm_ws(h.get_text(" ", strip=True))
                if not title:
                    t = parent.find(attrs={"class": re.compile(r"title", re.I)})
                    if t is not None:
                        title = _norm_ws(t.get_text(" ", strip=True))
        except Exception:
            context = ""
            context_raw = ""
            title = ""


        if context and len(context) > 180:
            context = context[:177] + "..."

        # Build a selector. Use dom_label for text-based fallbacks to avoid including long meta.
        primary = _build_selector(tag, attr_map, text=(dom_label or label))

        # Improve selectorability + promptability for <select> elements that lack stable attributes.
        if tag == "select":
            # Capture options for prompting and build a selector that uniquely identifies the select.
            opts: list[tuple[str, str]] = []
            try:
                tmp: list[tuple[str, str]] = []
                for o in el.find_all("option")[:12]:
                    t = ""
                    v = ""
                    try:
                        t = o.get_text(" ", strip=True)
                        v = str(o.get("value") or "").strip()
                    except Exception:
                        t = ""
                        v = ""
                    if t:
                        tmp.append((t, v))
                opts = tmp
            except Exception:
                opts = []

            if isinstance(primary, dict) and primary.get("type") == "attributeValueSelector" and str(primary.get("attribute") or "") == "custom" and str(primary.get("value") or "") == "select":
                first_opt = ""
                try:
                    if opts:
                        first_opt = str(opts[0][0] or "").strip()
                except Exception:
                    first_opt = ""
                if first_opt:
                    safe = first_opt.replace("\"", "'")
                    # This is typically unique and avoids strict-mode ambiguity.
                    primary = _sel_custom(f'select:has(option:has-text("{safe}"))')

            if opts:
                show: list[str] = []
                for t, v in opts[:8]:
                    if v and v != t:
                        show.append(f"{t} (value={v})")
                    else:
                        show.append(t)
                opt_preview = ", ".join(show)
                label = (dom_label or "select") + f" options=[{opt_preview}]"
                label = label[:200]

        container_chain = []
        try:
            container_chain = _container_chain_from_el(soup, el)
        except Exception:
            container_chain = []

        text_sel = None
        if tag in {"a", "button"} and dom_label:
            # Click by DOM text, even if we augmented label for prompting.
            text_sel = _sel_text(dom_label, case_sensitive=False)

        sig = (
            str(primary.get("type") or ""),
            str(primary.get("attribute") or ""),
            str(primary.get("value") or ""),
        )
        if sig in seen:
            continue
        seen.add(sig)

        out.append(_Candidate(primary, label, tag, attr_map, text_selector=text_sel, context=context, context_raw=context_raw, group=group, container_chain=container_chain))
        if len(out) >= max_candidates:
            break

    return out


def _extract_candidates(html: str, max_candidates: int = 30) -> List[_Candidate]:
    if not html:
        return []

    if BeautifulSoup is not None:
        try:
            return _extract_candidates_bs4(html, max_candidates=max_candidates)
        except Exception:
            pass

    parser = _CandidateExtractor()
    try:
        parser.feed(html)
    except Exception:
        return []
    return parser.candidates[:max_candidates]


def _summarize_html(html: str, limit: int = 1200) -> str:
    if not html:
        return ""

    if BeautifulSoup is not None:
        try:
            soup = BeautifulSoup(html, "lxml")
            text = _norm_ws(soup.get_text(" ", strip=True))
            return text[:limit]
        except Exception:
            pass

    try:
        text = re.sub(r"<[^>]+>", " ", html)
        return _norm_ws(text)[:limit]
    except Exception:
        return ""


def _dom_digest(html: str, limit: int = 1400) -> str:
    # Compact, structured page digest to help the LLM reason without sending full HTML.
    if not html:
        return ""

    if BeautifulSoup is not None:
        try:
            soup = BeautifulSoup(html, "lxml")
            for t in soup(["script", "style", "noscript"]):
                try:
                    t.decompose()
                except Exception:
                    pass

            parts: list[str] = []

            title = ""
            try:
                if soup.title and soup.title.get_text(strip=True):
                    title = _norm_ws(soup.title.get_text(" ", strip=True))
            except Exception:
                title = ""
            if title:
                parts.append(f"TITLE: {title[:160]}")

            heads: list[str] = []
            for h in soup.find_all(["h1", "h2", "h3"], limit=12):
                t = _norm_ws(h.get_text(" ", strip=True))
                if t:
                    heads.append(t[:140])
            if heads:
                parts.append("HEADINGS: " + " | ".join(heads[:10]))

            forms_bits: list[str] = []
            for form in soup.find_all("form", limit=4):
                els = form.find_all(["input", "textarea", "select"], limit=12)
                items: list[str] = []
                for el in els:
                    try:
                        attrs = _attrs_to_str_map(getattr(el, "attrs", {}) or {})
                        itype = (attrs.get("type") or "").lower()
                        label = _extract_label_from_bs4(soup, el, attrs)
                        blob = " ".join([label, attrs.get("name",""), attrs.get("id",""), attrs.get("placeholder",""), attrs.get("aria-label",""), itype]).strip()
                        blob = _norm_ws(blob)
                        if not blob:
                            continue
                        items.append(blob[:140])
                    except Exception:
                        continue
                if items:
                    forms_bits.append("; ".join(items[:8]))
            if forms_bits:
                parts.append("FORMS: " + " || ".join(forms_bits[:3]))

            ctas: list[str] = []
            for el in soup.select("button,a[href],[role='button'],[role='link']"):
                try:
                    if len(ctas) >= 14:
                        break
                    t = _norm_ws(el.get_text(" ", strip=True))
                    if not t:
                        t = _norm_ws(str(el.get("aria-label") or "") or "")
                    if not t:
                        continue
                    t_l = t.lower()
                    if t_l in {"home", "logo"}:
                        continue
                    if t not in ctas:
                        ctas.append(t[:90])
                except Exception:
                    continue
            if ctas:
                parts.append("CTAS: " + " | ".join(ctas[:12]))

            out = "\n".join(parts).strip()
            return out[:limit]
        except Exception:
            pass

    return _summarize_html(html, limit=limit)


def _is_hidden_candidate_attr(attr_map: Dict[str, str]) -> bool:
    try:
        if attr_map.get("hidden") is not None:
            return True
        if str(attr_map.get("aria-hidden") or "").lower() == "true":
            return True
        style = str(attr_map.get("style") or "").lower()
        if "display:none" in style or "visibility:hidden" in style:
            return True
        classes = str(attr_map.get("class") or "").lower()
        if any(tok in classes for tok in ("hidden", "sr-only", "invisible")):
            return True
    except Exception:
        return False
    return False


def _extract_page_ir(*, html: str, url: str, candidates: List[_Candidate], max_forms: int = 4, max_links: int = 20, max_cards: int = 10) -> Dict[str, Any]:
    """Build deterministic, compact page IR to reduce prompt bloat."""
    ir: Dict[str, Any] = {
        "title": "",
        "url_path": "",
        "headings": [],
        "forms": [],
        "ctas": [],
        "links": [],
        "cards": [],
    }
    if not html:
        return ir

    try:
        us = urlsplit(str(url or ""))
        ir["url_path"] = str(us.path or "/")
    except Exception:
        ir["url_path"] = str(url or "")

    if BeautifulSoup is not None:
        try:
            soup = BeautifulSoup(html, "lxml")
            for t in soup(["script", "style", "noscript"]):
                try:
                    t.decompose()
                except Exception:
                    pass

            try:
                if soup.title:
                    ir["title"] = _norm_ws(soup.title.get_text(" ", strip=True))[:120]
            except Exception:
                ir["title"] = ""

            heads: list[str] = []
            for h in soup.find_all(["h1", "h2", "h3"], limit=12):
                tx = _norm_ws(h.get_text(" ", strip=True))
                if tx and tx not in heads:
                    heads.append(tx[:100])
            ir["headings"] = heads[:10]
        except Exception:
            pass

    forms_obj = _tool_extract_forms(html=html, max_forms=max_forms, max_inputs=16)
    if isinstance(forms_obj, dict) and forms_obj.get("ok"):
        forms = forms_obj.get("forms")
        if isinstance(forms, list):
            cleaned_forms = []
            for f in forms[:max_forms]:
                if not isinstance(f, dict):
                    continue
                controls = f.get("controls") if isinstance(f.get("controls"), list) else []
                keep_controls = []
                for c in controls[:12]:
                    if not isinstance(c, dict):
                        continue
                    blob = _norm_ws(
                        " ".join([
                            str(c.get("tag") or ""),
                            str(c.get("type") or ""),
                            str(c.get("name") or ""),
                            str(c.get("id") or ""),
                            str(c.get("placeholder") or ""),
                            str(c.get("aria_label") or ""),
                            str(c.get("text") or ""),
                        ])
                    )
                    if blob:
                        keep_controls.append(blob[:140])
                if keep_controls:
                    cleaned_forms.append({
                        "id": str(f.get("id") or ""),
                        "name": str(f.get("name") or ""),
                        "method": str(f.get("method") or ""),
                        "controls": keep_controls[:8],
                    })
            ir["forms"] = cleaned_forms[:max_forms]

    links_obj = _tool_list_links(html=html, base_url=str(url), max_links=max_links, context_max=150)
    if isinstance(links_obj, dict) and links_obj.get("ok"):
        links = links_obj.get("links")
        if isinstance(links, list):
            clean_links = []
            for l in links[:max_links]:
                if not isinstance(l, dict):
                    continue
                text = _norm_ws(str(l.get("text") or ""))
                href = _norm_ws(str(l.get("href") or ""))
                ctx = _norm_ws(str(l.get("context") or ""))
                if not text and not href:
                    continue
                clean_links.append({
                    "text": text[:90],
                    "href": href[:150],
                    "ctx": ctx[:140],
                })
            ir["links"] = clean_links[:max_links]

    cards_obj = _tool_list_cards(candidates=candidates, max_cards=max_cards, max_text=380, max_actions_per_card=3)
    if isinstance(cards_obj, dict) and cards_obj.get("ok"):
        cards = cards_obj.get("cards")
        if isinstance(cards, list):
            out_cards = []
            for c in cards[:max_cards]:
                if not isinstance(c, dict):
                    continue
                actions = c.get("actions") if isinstance(c.get("actions"), list) else []
                actions_clean = []
                for a in actions[:3]:
                    if not isinstance(a, dict):
                        continue
                    actions_clean.append({
                        "tag": str(a.get("tag") or ""),
                        "text": _norm_ws(str(a.get("text") or ""))[:90],
                        "href": _norm_ws(str(a.get("href") or ""))[:120],
                    })
                out_cards.append({
                    "facts": [str(x)[:100] for x in (c.get("card_facts") or [])[:4]],
                    "actions": actions_clean,
                    "text": _norm_ws(str(c.get("card_text") or ""))[:260],
                })
            ir["cards"] = out_cards

    ctas: list[str] = []
    for c in candidates:
        if c.tag not in {"button", "a"}:
            continue
        tx = _norm_ws(c.text or "")
        if not tx:
            continue
        tx_l = tx.lower()
        if tx_l in {"home", "logo"}:
            continue
        if tx not in ctas:
            ctas.append(tx[:90])
        if len(ctas) >= 16:
            break
    ir["ctas"] = ctas
    return ir


def _render_page_ir(ir: Dict[str, Any], max_chars: int = 2200) -> str:
    lines: list[str] = []
    title = _norm_ws(str(ir.get("title") or ""))
    path = _norm_ws(str(ir.get("url_path") or ""))
    if title:
        lines.append(f"TITLE: {title[:120]}")
    if path:
        lines.append(f"PATH: {path[:140]}")

    heads = ir.get("headings") if isinstance(ir.get("headings"), list) else []
    if heads:
        lines.append("HEADINGS: " + " | ".join(str(h)[:90] for h in heads[:8]))

    forms = ir.get("forms") if isinstance(ir.get("forms"), list) else []
    if forms:
        lines.append("FORMS:")
        for i, f in enumerate(forms[:4]):
            if not isinstance(f, dict):
                continue
            method = str(f.get("method") or "")
            controls = f.get("controls") if isinstance(f.get("controls"), list) else []
            lines.append(f"- form[{i}] method={method} controls=" + " ; ".join(str(x)[:120] for x in controls[:6]))

    ctas = ir.get("ctas") if isinstance(ir.get("ctas"), list) else []
    if ctas:
        lines.append("CTAS: " + " | ".join(str(c)[:80] for c in ctas[:12]))

    links = ir.get("links") if isinstance(ir.get("links"), list) else []
    if links:
        lines.append("LINKS:")
        for l in links[:8]:
            if not isinstance(l, dict):
                continue
            lines.append(f"- text={str(l.get('text') or '')[:80]} href={str(l.get('href') or '')[:120]} ctx={str(l.get('ctx') or '')[:120]}")

    cards = ir.get("cards") if isinstance(ir.get("cards"), list) else []
    if cards:
        lines.append("CARDS:")
        for i, c in enumerate(cards[:6]):
            if not isinstance(c, dict):
                continue
            facts = c.get("facts") if isinstance(c.get("facts"), list) else []
            lines.append(f"- card[{i}] facts=" + " | ".join(str(x)[:80] for x in facts[:3]))
            acts = c.get("actions") if isinstance(c.get("actions"), list) else []
            for a in acts[:2]:
                if not isinstance(a, dict):
                    continue
                lines.append(f"  action tag={str(a.get('tag') or '')} text={str(a.get('text') or '')[:70]} href={str(a.get('href') or '')[:100]}")

    out = "\n".join(lines)
    return out[:max_chars]


def _compute_ir_delta(*, task_id: str, page_ir: Dict[str, Any]) -> str:
    if not task_id:
        return ""
    try:
        st = _TASK_STATE.get(task_id)
        if not isinstance(st, dict):
            st = {}
            _TASK_STATE[task_id] = st
        prev = st.get("prev_ir")
        if not isinstance(prev, dict):
            prev = {}

        def _set(key: str, d: Dict[str, Any]) -> set[str]:
            v = d.get(key)
            if not isinstance(v, list):
                return set()
            return {str(x)[:120] for x in v if isinstance(x, (str, int, float))}

        prev_cta = _set("ctas", prev)
        cur_cta = _set("ctas", page_ir)

        prev_heads = _set("headings", prev)
        cur_heads = _set("headings", page_ir)

        p_forms = prev.get("forms") if isinstance(prev.get("forms"), list) else []
        c_forms = page_ir.get("forms") if isinstance(page_ir.get("forms"), list) else []
        p_cards = prev.get("cards") if isinstance(prev.get("cards"), list) else []
        c_cards = page_ir.get("cards") if isinstance(page_ir.get("cards"), list) else []

        st["prev_ir"] = page_ir
        return (
            f"forms:{len(p_forms)}->{len(c_forms)}, "
            f"cards:{len(p_cards)}->{len(c_cards)}, "
            f"ctas_added={len(cur_cta - prev_cta)}, ctas_removed={len(prev_cta - cur_cta)}, "
            f"headings_added={len(cur_heads - prev_heads)}, headings_removed={len(prev_heads - cur_heads)}"
        )
    except Exception:
        return ""


# -----------------------------
# Ranking and prompting
# -----------------------------

# -----------------------------
# Structured hints (entity extraction)
# -----------------------------



def _structured_hints(task: str, candidates: List[_Candidate]) -> Dict[str, Any]:
    """Build compact, structured hints to help the LLM disambiguate UI."""
    task_l = (task or '').lower()

    # Inputs
    inputs: List[Dict[str, Any]] = []
    for i, c in enumerate(candidates):
        if c.tag not in {'input', 'textarea', 'select'}:
            continue
        attrs = {k: (c.attrs.get(k) or '') for k in ('type', 'name', 'id', 'placeholder', 'aria-label')}
        label = (c.text or '').strip()
        blob = ' '.join([label, c.context or '', attrs.get('name',''), attrs.get('id',''), attrs.get('placeholder',''), attrs.get('aria-label','')]).lower()

        kind = 'text'
        if 'password' in blob or attrs.get('type','').lower() == 'password':
            kind = 'password'
        elif 'email' in blob:
            kind = 'email'
        elif any(k in blob for k in ['search', 'buscar', 'query', 'find']):
            kind = 'search'
        elif any(k in blob for k in ['user', 'username', 'login']):
            kind = 'username'

        inputs.append({
            'candidate_id': i,
            'kind': kind,
            'label': label[:80],
            'required': bool((c.attrs or {}).get('required') is not None),
            'value_len': len(str((c.attrs or {}).get('value') or '')),
            'attrs': {k: v for k, v in attrs.items() if v},
        })
    return {
        'inputs': inputs[:20],
        'clickables': [
            {
                'candidate_id': i,
                'tag': c.tag,
                'label': (c.text or '')[:90],
                'href': (c.attrs or {}).get('href','') or (c.attrs or {}).get('data-href',''),
                'context': (c.context or '')[:220],
                'attrs': {k: str((c.attrs or {}).get(k) or '') for k in ('id','name','type','placeholder','aria-label','role') if (c.attrs or {}).get(k)},
            }
            for i, c in sorted(
                [(i, c) for i, c in enumerate(candidates) if c.tag in {'a','button'}],
                key=lambda t: len((t[1].context or '').strip()),
                reverse=True,
            )
        ][:25],
    }

def _tokenize(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]{2,}", (s or "").lower())}


def _score_candidate(task: str, c: _Candidate) -> float:
    """Structural scoring only.

    Avoids task-specific string heuristics; prefers stable selectors and form-relevant elements.
    """
    score = 0.0

    if c.tag in {'input', 'textarea', 'select'}:
        score += 6.0
    elif c.tag == 'button':
        score += 4.0
    elif c.tag == 'a':
        score += 2.0

    attrs = c.attrs or {}
    if attrs.get('id'):
        score += 4.0
    if attrs.get('name'):
        score += 2.0
    if attrs.get('aria-label'):
        score += 2.0
    if attrs.get('placeholder'):
        score += 1.0
    if attrs.get('href'):
        score += 1.0
    if attrs.get('role') in {'button','link'}:
        score += 0.5

    if attrs.get('required') is not None and c.tag in {'input','textarea','select'}:
        score += 2.0

    if c.selector.get('attribute') == 'custom' and c.selector.get('value') in {'a','button','input','select','textarea'}:
        score -= 2.0

    if (c.text or '').strip():
        score += 1.0
    if (c.context or '').strip():
        score += 0.5

    return score

def _rank_candidates(task: str, candidates: List[_Candidate], max_candidates: int) -> List[_Candidate]:
    scored = [(i, _score_candidate(task, c), c) for i, c in enumerate(candidates)]
    scored.sort(key=lambda t: (t[1], -t[0]), reverse=True)
    return [c for _, _, c in scored[:max_candidates]]


def _select_candidates_for_llm(task: str, candidates_all: List[_Candidate], current_url: str, max_total: int = 60) -> List[_Candidate]:
    """Pick a diverse, usable candidate set for the LLM.

    Structural selection only (no task keyword heuristics):
    - keep all form controls (input/textarea/select)
    - keep primary buttons
    - keep a slice of anchors/buttons that have non-trivial surrounding context (cards)
    """
    if not candidates_all:
        return []

    controls = []
    primaries = []
    contextual = []
    others = []
    for c in candidates_all:
        # Skip self-links (common in nav) to avoid loops when already on the target page.
        try:
            from urllib.parse import urlparse
            if c.tag == "a":
                href = str((c.attrs or {}).get("href") or "")
                if href:
                    ph = urlparse(href)
                    pc = urlparse(current_url or "")
                    if ph.path and pc.path and ph.path == pc.path:
                        # Same path; let other non-nav elements be considered.
                        continue
        except Exception:
            pass
        if c.tag in {"input", "textarea", "select"}:
            controls.append(c)
            continue
        if c.tag == "button":
            primaries.append(c)
            continue
        if c.tag in {"a", "button"} and (c.context or "").strip():
            if len((c.context or "").strip()) >= 40:
                contextual.append(c)
            else:
                others.append(c)
            continue
        others.append(c)

    picked = []
    seen = set()
    def add_many(arr, limit):
        nonlocal picked
        for c in arr:
            # NOTE: f-string expressions cannot contain unescaped quotes that match the
            # f-string delimiter. Use single quotes inside the expression.
            sig = f"{_selector_repr(c.selector)}|{(c.text or '')[:80]}"
            if sig in seen:
                continue
            seen.add(sig)
            picked.append(c)
            if len(picked) >= max_total or len(picked) >= limit:
                return

    # Order: controls first, then contextual card links, then buttons, then the rest.
    add_many(controls, max_total)
    if len(picked) < max_total:
        add_many(contextual, max_total)
    if len(picked) < max_total:
        add_many(primaries, max_total)
    if len(picked) < max_total:
        add_many(others, max_total)

    return picked[:max_total]



def _parse_llm_json(content: str) -> Dict[str, Any]:
    if not isinstance(content, str):
        raise ValueError(f"LLM returned non-text content type={type(content)}")

    raw = content.strip()
    # Common case: pure JSON.
    try:
        obj = json.loads(raw)
    except Exception:
        # Best-effort recovery: strip code-fences and/or extract the first JSON object.
        s = raw
        if s.startswith("```"):
            # Remove leading/trailing fenced blocks like ```json ... ```
            s2 = s
            if s2.startswith("```json"):
                s2 = s2[len("```json") :]
            elif s2.startswith("```"):
                s2 = s2[len("```") :]
            if s2.endswith("```"):
                s2 = s2[: -len("```")]
            s = s2.strip()
        start = s.find("{")
        end = s.rfind("}")
        if 0 <= start < end:
            try:
                obj = json.loads(s[start : end + 1])
            except Exception as e:
                raise ValueError(f"LLM returned non-JSON: {raw[:200]}") from e
        else:
            raise ValueError(f"LLM returned non-JSON: {raw[:200]}")
    if not isinstance(obj, dict):
        raise ValueError("LLM returned non-object JSON")
    return obj


def _history_hint(history: List[Dict[str, Any]] | None) -> str:
    if not history:
        return ""

    last = history[-6:]
    # Detect simple repetition: same action+cid repeated.
    repeats = 0
    prev = None
    for h in last:
        a = h.get("action")
        a_name = str(a.get("type") or "") if isinstance(a, dict) else str(a or "")
        k = (a_name, h.get("candidate_id"))
        if prev is not None and k == prev and k != ("", None):
            repeats += 1
        prev = k

    if repeats >= 2:
        return "You appear to be repeating the same action. Choose a DIFFERENT candidate or try scroll."

    failed = 0
    for h in last:
        ok = h.get("success")
        if ok is None:
            ok = h.get("exec_ok")
        if ok is False:
            failed += 1
    if failed >= 2:
        return "Recent actions failed repeatedly. Choose a different strategy/candidate."

    return ""


def _history_tail(history: List[Dict[str, Any]] | None, default_window: int = 6) -> List[Dict[str, Any]]:
    if not history:
        return []
    win = default_window
    win = max(1, min(win, 30))
    return history[-win:]


def _should_run_completion_check(
    *,
    completion_only: bool,
    step_index: int,
    history: List[Dict[str, Any]] | None,
    task_state: Dict[str, Any] | None,
) -> bool:
    if completion_only:
        return True
    min_step = 2
    if int(step_index) >= max(0, min_step):
        return True
    rep = 0
    try:
        rep = int((task_state or {}).get("repeat") or 0)
    except Exception:
        rep = 0
    if rep >= 2:
        return True
    if history and len(history) >= 5:
        return True
    return False


def _action_sig_for_loop(decision: Dict[str, Any]) -> str:
    a = str(decision.get("action") or "").lower().strip()
    cid = decision.get("candidate_id")
    if isinstance(cid, str) and cid.isdigit():
        cid = int(cid)
    txt = str(decision.get("text") or "").strip().lower()[:40]
    u = str(decision.get("url") or "").strip().lower()[:120]
    return f"{a}|{cid}|{txt}|{u}"


def _count_repeated_recent_decision(
    *,
    history: List[Dict[str, Any]] | None,
    url: str,
    decision: Dict[str, Any],
) -> int:
    if not history:
        return 0
    cur_sig = _action_sig_for_loop(decision)
    same = 0
    for h in _history_tail(history, default_window=6):
        try:
            h_url = str(h.get("url") or "")
            if h_url and h_url != str(url):
                continue
            h_dec = {
                "action": h.get("action"),
                "candidate_id": h.get("candidate_id"),
                "text": h.get("text"),
                "url": h.get("url_target"),
            }
            if _action_sig_for_loop(h_dec) == cur_sig:
                same += 1
        except Exception:
            continue
    return same


def _is_repeated_recent_decision(
    *,
    history: List[Dict[str, Any]] | None,
    url: str,
    decision: Dict[str, Any],
    threshold: int = 2,
) -> bool:
    return _count_repeated_recent_decision(history=history, url=url, decision=decision) >= threshold


def _candidate_blob(c: _Candidate) -> str:
    attrs = c.attrs or {}
    bits = [
        c.tag,
        c.text or "",
        c.context or "",
        attrs.get("id") or "",
        attrs.get("name") or "",
        attrs.get("placeholder") or "",
        attrs.get("aria-label") or "",
        attrs.get("href") or "",
    ]
    return _norm_ws(" ".join(str(x) for x in bits)).lower()


def _pick_fallback_candidate_id(
    *,
    candidates: List[_Candidate],
    action: str,
    decision: Dict[str, Any],
    avoid_id: int | None = None,
) -> int | None:
    if not candidates:
        return None
    act = str(action or "").lower().strip()
    want_text = _norm_ws(str(decision.get("text") or "")).lower()
    want_url = _norm_ws(str(decision.get("url") or "")).lower()
    want_tokens = set(re.findall(r"[a-z0-9]{2,}", want_text))

    scored: list[tuple[float, int]] = []
    for i, c in enumerate(candidates):
        if avoid_id is not None and i == avoid_id:
            continue
        s = 0.0
        blob = _candidate_blob(c)
        attrs = c.attrs or {}

        if act == "click":
            if c.tag in {"a", "button"}:
                s += 4.0
            if attrs.get("href"):
                s += 1.5
            if want_url:
                href = str(attrs.get("href") or "").lower()
                if href and (want_url in href or href in want_url):
                    s += 7.0
            if want_tokens:
                tok = set(re.findall(r"[a-z0-9]{2,}", blob))
                s += min(6.0, 1.5 * len(tok.intersection(want_tokens)))

        elif act == "type":
            if c.tag in {"input", "textarea"}:
                s += 6.0
            if str(attrs.get("type") or "").lower() not in {"hidden", "submit", "button"}:
                s += 1.0
            if want_tokens:
                tok = set(re.findall(r"[a-z0-9]{2,}", blob))
                s += min(5.0, 1.3 * len(tok.intersection(want_tokens)))

        elif act == "select":
            if c.tag == "select":
                s += 8.0
            if want_tokens:
                tok = set(re.findall(r"[a-z0-9]{2,}", blob))
                s += min(5.0, 1.3 * len(tok.intersection(want_tokens)))

        else:
            continue

        scored.append((s, i))

    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_i = scored[0]
    if best_score <= 0.0:
        return None
    return int(best_i)


def _task_risk_hint(task: str, step_index: int, candidates: List[_Candidate]) -> str:
    t = str(task or "").lower()
    risk_words = ("delete", "remove", "edit", "update", "add film", "registration", "contact")
    if not any(w in t for w in risk_words):
        return ""
    n = len(candidates or [])
    if step_index <= 1 and n >= 12:
        return (
            "High-risk task detected. Before destructive or irreversible clicks, identify the exact target via "
            "find_card/list_cards/search_text using task constraints, then click the matched action."
        )
    return (
        "For high-risk actions, verify target attributes in context first and avoid generic repeated clicks."
    )


def _split_task_subgoals(task: str) -> list[str]:
    t = _norm_ws(task)
    if not t:
        return []
    # Keep this deterministic and lightweight.
    parts = re.split(r"\bthen\b|\band\b|[,;]", t, flags=re.I)
    out: list[str] = []
    for p in parts:
        pp = _norm_ws(p)
        if not pp:
            continue
        # Normalize imperative noise.
        pp = re.sub(r"^(please|kindly)\s+", "", pp, flags=re.I).strip()
        out.append(pp[:120])
    if not out:
        out = [t[:120]]
    return out[:6]


def _extract_urls(text: str) -> list[str]:
    try:
        vals = re.findall(r"https?://[^\s)>\"]+", str(text or ""), flags=re.I)
    except Exception:
        vals = []
    out: list[str] = []
    seen: set[str] = set()
    for v in vals:
        vv = v.strip()
        if vv and vv not in seen:
            seen.add(vv)
            out.append(vv)
    return out[:6]


def _extract_host_hints(text: str) -> list[str]:
    # Also detect host-like tokens without scheme, e.g. autoppia.com
    vals = re.findall(r"\b[a-z0-9.-]+\.[a-z]{2,}\b", str(text or "").lower())
    out: list[str] = []
    seen: set[str] = set()
    for v in vals:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out[:6]


def _ensure_subgoal_memory(task_id: str, task: str) -> dict[str, Any] | None:
    if not task_id:
        return None
    st = _TASK_STATE.get(task_id)
    if not isinstance(st, dict):
        st = {}
        _TASK_STATE[task_id] = st
    mem = st.get("subgoal_memory")
    if isinstance(mem, dict) and isinstance(mem.get("subgoals"), list):
        return mem

    raw_subgoals = _split_task_subgoals(task)
    subgoals: list[dict[str, Any]] = []
    for i, s in enumerate(raw_subgoals):
        urls = _extract_urls(s)
        hosts = _extract_host_hints(s)
        tokens = [x for x in re.findall(r"[a-z0-9]{3,}", s.lower()) if x not in {"then", "open", "goto", "navigate", "click"}]
        subgoals.append(
            {
                "id": i,
                "text": s,
                "urls": urls,
                "hosts": hosts,
                "tokens": tokens[:10],
                "done": False,
                "blocked": False,
                "evidence": "",
                "fail_count": 0,
            }
        )
    mem = {
        "task": _norm_ws(task)[:180],
        "subgoals": subgoals,
        "active_id": 0 if subgoals else -1,
        "last_progress_step": -1,
        "stall_count": 0,
    }
    st["subgoal_memory"] = mem
    return mem


def _typed_values_from_history(history: List[Dict[str, Any]] | None) -> list[str]:
    out: list[str] = []
    for h in (history or []):
        if not isinstance(h, dict):
            continue
        action = h.get("action") if isinstance(h.get("action"), dict) else {}
        at = str(action.get("type") or "")
        if at in {"TypeAction", "FillAction", "SelectDropDownOptionAction"}:
            txt = str(action.get("text") or action.get("value") or "").strip()
            if txt:
                out.append(txt)
    return out


def _history_has_click_overlay_intercept(history: List[Dict[str, Any]] | None) -> bool:
    for h in _history_tail(history, default_window=8):
        if not isinstance(h, dict):
            continue
        a = h.get("action") if isinstance(h.get("action"), dict) else {}
        a_type = str(a.get("type") or "")
        if a_type != "ClickAction":
            continue
        ok = h.get("success")
        if ok is None:
            ok = h.get("exec_ok")
        if ok is True:
            continue
        err = str(h.get("error") or "").lower()
        if "intercepts pointer events" in err or "overlay-backdrop" in err:
            return True
    return False


def _subgoal_done_by_state(
    sg: dict[str, Any],
    *,
    url: str,
    page_ir_text: str,
    page_summary: str,
    history: List[Dict[str, Any]] | None,
) -> tuple[bool, str]:
    u = str(url or "").lower()
    ir = str(page_ir_text or "").lower()
    ps = str(page_summary or "").lower()
    blob = f"{u}\n{ir}\n{ps}"
    sg_text = str(sg.get("text") or "").lower()
    typed_vals = [x.lower() for x in _typed_values_from_history(history)]

    # Credential-related subgoals require evidence from typed values.
    if any(k in sg_text for k in {"login", "sign in", "signin", "email", "password"}):
        needs_email = ("email" in sg_text) or ("@" in sg_text)
        needs_password = ("password" in sg_text)
        email_ok = (not needs_email) or any("@" in v for v in typed_vals)
        password_ok = (not needs_password) or any(v.strip() == "password" for v in typed_vals)
        on_auth_page = any(k in u for k in {"login", "sign-in", "signin", "auth"})
        if email_ok and password_ok and on_auth_page:
            return True, "credential_inputs_typed"
        # Avoid false-positive done on auth subgoals via token-only matches.
        return False, ""

    for target in sg.get("urls") or []:
        t = str(target).lower()
        if t and (t in u or t in blob):
            return True, f"matched_url:{target}"
    for host in sg.get("hosts") or []:
        h = str(host).lower()
        if h and h in u:
            return True, f"matched_host:{host}"
    toks = [str(t).lower() for t in (sg.get("tokens") or []) if str(t).strip()]
    if toks:
        hit = sum(1 for t in toks if t in blob)
        if hit >= min(2, max(1, len(toks))):
            return True, f"token_hits:{hit}"
    return False, ""


def _update_subgoal_memory(
    mem: dict[str, Any] | None,
    *,
    step_index: int,
    url: str,
    page_ir_text: str,
    page_summary: str,
    history: List[Dict[str, Any]] | None,
    repeat_count: int,
) -> None:
    if not isinstance(mem, dict):
        return
    sgs = mem.get("subgoals")
    if not isinstance(sgs, list) or not sgs:
        return
    progressed = False
    for sg in sgs:
        if not isinstance(sg, dict) or bool(sg.get("done")):
            continue
        done, evidence = _subgoal_done_by_state(
            sg,
            url=url,
            page_ir_text=page_ir_text,
            page_summary=page_summary,
            history=history,
        )
        if done:
            sg["done"] = True
            sg["blocked"] = False
            sg["evidence"] = evidence
            sg["fail_count"] = 0
            progressed = True

    active_id = -1
    for sg in sgs:
        if isinstance(sg, dict) and not bool(sg.get("done")):
            active_id = int(sg.get("id") or 0)
            break
    mem["active_id"] = active_id

    if progressed:
        mem["last_progress_step"] = int(step_index)
        mem["stall_count"] = 0
    else:
        mem["stall_count"] = int(mem.get("stall_count") or 0) + 1

    if repeat_count >= 2 and active_id >= 0:
        for sg in sgs:
            if isinstance(sg, dict) and int(sg.get("id") or -1) == active_id and not bool(sg.get("done")):
                sg["fail_count"] = int(sg.get("fail_count") or 0) + 1
                if int(sg.get("fail_count") or 0) >= 3:
                    sg["blocked"] = True
                break


def _all_subgoals_done(mem: dict[str, Any] | None) -> bool:
    if not isinstance(mem, dict):
        return False
    sgs = mem.get("subgoals")
    if not isinstance(sgs, list) or not sgs:
        return False
    return all(isinstance(sg, dict) and bool(sg.get("done")) for sg in sgs)


def _subgoal_hint(mem: dict[str, Any] | None) -> str:
    if not isinstance(mem, dict):
        return ""
    sgs = mem.get("subgoals")
    if not isinstance(sgs, list) or not sgs:
        return ""
    active_id = int(mem.get("active_id") or -1)
    active = None
    for sg in sgs:
        if isinstance(sg, dict) and int(sg.get("id") or -1) == active_id:
            active = sg
            break
    done_n = sum(1 for sg in sgs if isinstance(sg, dict) and bool(sg.get("done")))
    total_n = len(sgs)
    blocked_n = sum(1 for sg in sgs if isinstance(sg, dict) and bool(sg.get("blocked")))
    if active is None:
        return f"SUBGOALS: {done_n}/{total_n} complete; blocked={blocked_n}. Task may be complete."
    return (
        f"SUBGOALS: {done_n}/{total_n} complete; blocked={blocked_n}. "
        f"ACTIVE_SUBGOAL[{active_id}]: {str(active.get('text') or '')[:140]}"
    )


def _compact_subgoal_metrics(mem: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(mem, dict):
        return {}
    sgs = mem.get("subgoals")
    if not isinstance(sgs, list):
        return {}
    out_sg = []
    for sg in sgs[:6]:
        if not isinstance(sg, dict):
            continue
        out_sg.append(
            {
                "id": int(sg.get("id") or 0),
                "text": str(sg.get("text") or "")[:120],
                "done": bool(sg.get("done")),
                "blocked": bool(sg.get("blocked")),
                "evidence": str(sg.get("evidence") or "")[:80],
            }
        )
    return {
        "active_id": int(mem.get("active_id") or -1),
        "stall_count": int(mem.get("stall_count") or 0),
        "subgoals": out_sg,
    }




def _format_browser_state(*, candidates: List[_Candidate], prev_sig_set: set[str] | None) -> str:
    """Browser-use-like state view: numbered interactives, with a simplified DOM tree."""

    # Build a tree based on container_chain (preferred) or group fallback.
    class _TNode:
        __slots__ = ("name", "children", "items")

        def __init__(self, name: str) -> None:
            self.name = name
            self.children: dict[str, _TNode] = {}
            self.items: list[tuple[int, _Candidate]] = []

    root = _TNode('ROOT')

    def _chain_for(c: _Candidate) -> list[str]:
        ch = []
        try:
            ch = list(getattr(c, 'container_chain', []) or [])
        except Exception:
            ch = []
        if not ch:
            g = (getattr(c, 'group', '') or 'PAGE').strip() or 'PAGE'
            ch = [g]
        # keep it small
        return [str(x)[:80] for x in ch if str(x).strip()][:3]

    # Insert candidates
    for i, c in enumerate(candidates):
        node = root
        for part in _chain_for(c):
            if part not in node.children:
                node.children[part] = _TNode(part)
            node = node.children[part]
        node.items.append((i, c))

    def _render(node: _TNode, indent: str = '') -> list[str]:
        lines: list[str] = []
        # render items first within this container
        for i, c in node.items:
            label = (c.text or '').strip() or (c.attrs or {}).get('placeholder', '') or (c.attrs or {}).get('aria-label', '')
            label = str(label).strip()

            sig = f"{_selector_repr(c.selector)}|{(c.text or '')[:80]}"
            is_new = bool(prev_sig_set) and (sig not in (prev_sig_set or set()))
            star = '* ' if is_new else ''

            attrs_bits: list[str] = []
            for k in ('id','name','type','placeholder','aria-label','href','role'):
                v = (c.attrs or {}).get(k)
                if v:
                    vv = str(v)
                    if len(vv) > 60:
                        vv = vv[:57] + '...'
                    attrs_bits.append(f"{k}={vv}")
            attrs_str = (' | ' + ', '.join(attrs_bits)) if attrs_bits else ''

            ctx = ''
            try:
                if c.tag in {'a','button'} and (c.context or '').strip():
                    # Help disambiguate repeated CTAs like 'View Details' without site-specific parsing.
                    ctx = ' :: ' + _norm_ws(c.context)[:120]
            except Exception:
                ctx = ''

            lines.append(f"{indent}{star}[{i}]<{c.tag}>{label}</{c.tag}>{attrs_str}{ctx}")

        # then render child containers
        for name, child in node.children.items():
            lines.append(f"{indent}{name}:")
            lines.extend(_render(child, indent + "	"))

        return lines

    rendered = _render(root, '')
    return "\n".join(rendered)



def _resolve_url(url: str, base_url: str) -> str:
    """Resolve possibly-relative URL against a base URL."""
    try:
        from urllib.parse import urljoin
        u = str(url or "").strip()
        b = str(base_url or "").strip()
        if not u:
            return ""
        # urljoin handles absolute u (returns u unchanged).
        return urljoin(b, u) if b else u
    except Exception:
        return str(url or "").strip()


def _path_query(url: str, base_url: str = "") -> tuple[str, str]:
    try:
        from urllib.parse import urlparse
        resolved = _resolve_url(url, base_url)
        pu = urlparse(resolved or "")
        return (pu.path or ""), (pu.query or "")
    except Exception:
        s = (url or "").strip()
        return s, ""


def _same_path_query(a: str, b: str, *, base_a: str = "", base_b: str = "") -> bool:
    """Compare (path,query) for URLs, resolving relatives against provided bases."""
    try:
        return _path_query(a, base_a) == _path_query(b, base_b)
    except Exception:
        return (a or "").strip() == (b or "").strip()

def _preserve_seed_url(target_url: str, current_url: str) -> str:
    """If current_url has a seed param, ensure target_url keeps it.

    Demo webs are seeded; the validator expects the seed to stay consistent across navigations.
    """
    try:
        from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
        cur = urlparse(current_url or "")
        tgt = urlparse(target_url or "")
        cur_seed = (parse_qs(cur.query).get("seed") or [None])[0]
        if not cur_seed:
            return target_url
        q = parse_qs(tgt.query)
        if (q.get("seed") or [None])[0] == str(cur_seed):
            return target_url
        q["seed"] = [str(cur_seed)]
        new_q = urlencode(q, doseq=True)
        fixed = tgt._replace(query=new_q)
        if not fixed.scheme and not fixed.netloc:
            return urlunparse(("", "", fixed.path, fixed.params, fixed.query, fixed.fragment))
        return urlunparse(fixed)
    except Exception:
        return target_url



# -----------------------------
# HTML Tools (for LLM-assisted inspection)
# -----------------------------

def _safe_truncate(s: str, n: int) -> str:
    s = str(s or "")
    return s if len(s) <= n else (s[: max(0, n - 3)] + "...")


def _tool_search_text(*, html: str, query: str, regex: bool = False, case_sensitive: bool = False, max_matches: int = 20, context_chars: int = 80) -> Dict[str, Any]:
    """Search raw HTML text and return small context snippets.

    Generic tool: does not assume any site structure.
    """
    q = str(query or "")
    if not q:
        return {"ok": False, "error": "missing query"}

    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        if regex:
            pat = re.compile(q, flags)
        else:
            pat = re.compile(re.escape(q), flags)
    except Exception as e:
        return {"ok": False, "error": f"invalid pattern: {str(e)[:120]}"}

    hay = str(html or "")
    out = []
    for m in pat.finditer(hay):
        if len(out) >= int(max_matches or 0):
            break
        a = max(0, m.start() - int(context_chars))
        b = min(len(hay), m.end() + int(context_chars))
        out.append({
            "start": int(m.start()),
            "end": int(m.end()),
            "snippet": _safe_truncate(hay[a:b].replace("\n", " ").replace("\r", " "), 2 * int(context_chars) + 40),
        })

    return {"ok": True, "matches": out, "count": len(out)}


def _tool_css_select(*, html: str, selector: str, max_nodes: int = 25) -> Dict[str, Any]:
    """Run a CSS selector over the DOM (via BeautifulSoup) and return summaries."""
    if BeautifulSoup is None:
        return {"ok": False, "error": "bs4 not available"}
    sel = str(selector or "").strip()
    if not sel:
        return {"ok": False, "error": "missing selector"}

    try:
        soup = BeautifulSoup(html or "", "lxml")
        nodes = soup.select(sel)
    except Exception as e:
        return {"ok": False, "error": f"css select failed: {str(e)[:160]}"}

    out = []
    for n in nodes[: int(max_nodes or 0)]:
        try:
            tag = str(getattr(n, "name", "") or "")
            attrs = _attrs_to_str_map(getattr(n, "attrs", {}) or {})
            text = _norm_ws(n.get_text(" ", strip=True))
            out.append({
                "tag": tag,
                "attrs": {k: _safe_truncate(v, 120) for k, v in list(attrs.items())[:12]},
                "text": _safe_truncate(text, 240),
            })
        except Exception:
            continue

    return {"ok": True, "count": len(nodes), "nodes": out}


def _tool_extract_forms(*, html: str, max_forms: int = 10, max_inputs: int = 25) -> Dict[str, Any]:
    """Extract forms and their controls in a structured way (generic)."""
    if BeautifulSoup is None:
        return {"ok": False, "error": "bs4 not available"}

    try:
        soup = BeautifulSoup(html or "", "lxml")
    except Exception as e:
        return {"ok": False, "error": f"parse failed: {str(e)[:160]}"}

    forms = []
    for f in soup.find_all("form")[: int(max_forms or 0)]:
        try:
            f_attrs = _attrs_to_str_map(getattr(f, "attrs", {}) or {})
            inputs = []
            for el in f.find_all(["input", "textarea", "select", "button"])[: int(max_inputs or 0)]:
                try:
                    tag = str(getattr(el, "name", "") or "")
                    a = _attrs_to_str_map(getattr(el, "attrs", {}) or {})
                    t = _norm_ws(el.get_text(" ", strip=True))
                    inputs.append({
                        "tag": tag,
                        "type": (a.get("type") or "").lower(),
                        "id": a.get("id") or "",
                        "name": a.get("name") or "",
                        "placeholder": a.get("placeholder") or "",
                        "aria_label": a.get("aria-label") or "",
                        "value": _safe_truncate(a.get("value") or "", 120),
                        "text": _safe_truncate(t, 160),
                    })
                except Exception:
                    continue
            forms.append({
                "id": f_attrs.get("id") or "",
                "name": f_attrs.get("name") or "",
                "action": f_attrs.get("action") or "",
                "method": (f_attrs.get("method") or "").upper(),
                "controls": inputs,
            })
        except Exception:
            continue

    return {"ok": True, "forms": forms, "count": len(forms)}




def _tool_xpath_select(*, html: str, xpath: str, max_nodes: int = 25) -> Dict[str, Any]:
    """Run an XPath selector over the DOM (via lxml) and return summaries."""
    xp = str(xpath or "").strip()
    if not xp:
        return {"ok": False, "error": "missing xpath"}
    try:
        from lxml import html as lxml_html  # type: ignore
    except Exception:
        return {"ok": False, "error": "lxml not available"}

    try:
        doc = lxml_html.fromstring(html or "")
        nodes = doc.xpath(xp)
    except Exception as e:
        return {"ok": False, "error": f"xpath failed: {str(e)[:160]}"}

    out = []
    for n in nodes[: int(max_nodes or 0)]:
        try:
            # lxml may return strings/attrs too.
            if not hasattr(n, 'tag'):
                out.append({"value": _safe_truncate(str(n), 240)})
                continue
            tag = str(getattr(n, 'tag', '') or '')
            attrs = {k: _safe_truncate(str(v), 120) for k, v in list(getattr(n, 'attrib', {}) or {}).items()[:12]}
            text = _norm_ws(' '.join(n.itertext()))
            out.append({"tag": tag, "attrs": attrs, "text": _safe_truncate(text, 240)})
        except Exception:
            continue

    return {"ok": True, "count": len(nodes), "nodes": out}


def _tool_visible_text(*, html: str, max_chars: int = 2000) -> Dict[str, Any]:
    """Extract visible-ish text from the page (best-effort)."""
    if BeautifulSoup is None:
        # Fallback: strip tags very crudely.
        txt = re.sub(r"<[^>]+>", " ", str(html or ""))
        txt = _norm_ws(txt)
        return {"ok": True, "text": _safe_truncate(txt, int(max_chars or 0))}

    try:
        soup = BeautifulSoup(html or "", "lxml")
        for t in soup(["script", "style", "noscript"]):
            try:
                t.decompose()
            except Exception:
                pass
        txt = _norm_ws(soup.get_text(" ", strip=True))
        return {"ok": True, "text": _safe_truncate(txt, int(max_chars or 0))}
    except Exception as e:
        return {"ok": False, "error": f"extract text failed: {str(e)[:160]}"}


def _tool_extract_tables(*, html: str, max_tables: int = 6, max_rows: int = 8, max_cols: int = 8) -> Dict[str, Any]:
    """Extract structured table previews (generic, no site assumptions)."""
    if BeautifulSoup is None:
        return {"ok": False, "error": "bs4 not available"}
    try:
        soup = BeautifulSoup(html or "", "lxml")
    except Exception as e:
        return {"ok": False, "error": f"parse failed: {str(e)[:160]}"}

    out: list[dict[str, Any]] = []
    for t in soup.find_all("table")[: int(max_tables or 0)]:
        try:
            headers: list[str] = []
            hs = t.find_all("th")
            for h in hs[: int(max_cols or 0)]:
                tx = _norm_ws(h.get_text(" ", strip=True))
                if tx:
                    headers.append(_safe_truncate(tx, 80))

            rows: list[list[str]] = []
            tr_nodes = t.find_all("tr")
            for tr in tr_nodes[: int(max_rows or 0)]:
                cells = tr.find_all(["td", "th"])
                row = []
                for c in cells[: int(max_cols or 0)]:
                    row.append(_safe_truncate(_norm_ws(c.get_text(" ", strip=True)), 120))
                if row:
                    rows.append(row)

            caption = ""
            cap = t.find("caption")
            if cap is not None:
                caption = _safe_truncate(_norm_ws(cap.get_text(" ", strip=True)), 120)
            out.append({"caption": caption, "headers": headers, "rows": rows})
        except Exception:
            continue
    return {"ok": True, "count": len(out), "tables": out}


def _tool_extract_entities(*, html: str, max_items: int = 50) -> Dict[str, Any]:
    """Extract common entities from visible text: emails, phones, urls, prices, dates."""
    txt_obj = _tool_visible_text(html=html, max_chars=20000)
    if not isinstance(txt_obj, dict) or not txt_obj.get("ok"):
        return {"ok": False, "error": "visible text extraction failed"}
    txt = str(txt_obj.get("text") or "")

    def _uniq(vals: list[str], n: int) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for v in vals:
            vv = _norm_ws(v)
            if not vv or vv in seen:
                continue
            seen.add(vv)
            out.append(vv)
            if len(out) >= n:
                break
        return out

    emails = _uniq(re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", txt), int(max_items or 0))
    phones = _uniq(re.findall(r"(?:\+?\d[\d\-\s().]{7,}\d)", txt), int(max_items or 0))
    urls = _uniq(re.findall(r"https?://[^\s)>\"]+", txt), int(max_items or 0))
    prices = _uniq(re.findall(r"(?:\$|USD\s?)\d+(?:[.,]\d{2})?", txt), int(max_items or 0))
    dates = _uniq(
        re.findall(
            r"(?:\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}\b)",
            txt,
            flags=re.I,
        ),
        int(max_items or 0),
    )
    return {
        "ok": True,
        "entities": {
            "emails": emails,
            "phones": phones,
            "urls": urls,
            "prices": prices,
            "dates": dates,
        },
    }

def _tool_list_candidates(*, candidates: List["_Candidate"], max_n: int = 80) -> Dict[str, Any]:
    out = []
    for i, c in enumerate((candidates or [])[: int(max_n or 0)]):
        out.append({
            "id": i,
            "tag": c.tag,
            "group": c.group,
            "text": _safe_truncate(c.text or "", 140),
            "context": _safe_truncate(c.context or "", 200),
            "selector": _selector_repr(c.selector) if isinstance(c.selector, dict) else str(c.selector),
            "click": _selector_repr(c.click_selector()),
        })
    return {"ok": True, "count": len(candidates or []), "candidates": out}


def _tool_list_links(
    *,
    html: str,
    base_url: str,
    max_links: int = 60,
    context_max: int = 260,
    href_regex: str = "",
    text_regex: str = "",
) -> Dict[str, Any]:
    """Extract links (href) and nearby container text.

    Generic tool that helps the LLM choose a navigation target without depending on a candidate_id.
    """
    if BeautifulSoup is None:
        return {"ok": False, "error": "bs4 not available"}

    try:
        soup = BeautifulSoup(html or "", "lxml")
    except Exception as e:
        return {"ok": False, "error": f"parse failed: {str(e)[:160]}"}

    href_pat = None
    text_pat = None
    try:
        if href_regex:
            href_pat = re.compile(str(href_regex), re.I)
        if text_regex:
            text_pat = re.compile(str(text_regex), re.I)
    except Exception as e:
        return {"ok": False, "error": f"invalid regex: {str(e)[:160]}"}

    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    for a in soup.select("a[href]"):
        try:
            href = str(a.get("href") or "").strip()
            if not href or href.lower().startswith("javascript:"):
                continue
            if href_pat and not href_pat.search(href):
                continue

            text = _norm_ws(a.get_text(" ", strip=True))
            if not text:
                text = _norm_ws(str(a.get("aria-label") or "") or "")
            if text_pat and not text_pat.search(text):
                continue

            container = _pick_context_container_bs4(a)
            ctx_raw = ""
            if container is not None:
                try:
                    ctx_raw = container.get_text("\n", strip=True)
                except Exception:
                    ctx_raw = ""
            ctx = _safe_truncate(_norm_ws(ctx_raw) if ctx_raw else "", int(context_max or 0))

            resolved = _resolve_url(href, str(base_url or ""))
            resolved = _preserve_seed_url(resolved, str(base_url or ""))

            sig = (resolved or href) + "|" + (text or "")
            if sig in seen:
                continue
            seen.add(sig)

            out.append({
                "href": _safe_truncate(href, 260),
                "url": _safe_truncate(resolved, 320),
                "text": _safe_truncate(text, 160),
                "context": ctx,
            })
            if len(out) >= int(max_links or 0):
                break
        except Exception:
            continue

    return {"ok": True, "count": len(out), "links": out}


def _tool_list_cards(*, candidates: List["_Candidate"], max_cards: int = 25, max_text: int = 900, max_actions_per_card: int = 6) -> Dict[str, Any]:
    """Group candidates into card-like clusters using their extracted container context.

    Generic: clusters clickables (a/button or href-selectable) by context_raw/context. Returns surrounding text plus actions.
    """
    groups: dict[str, dict[str, Any]] = {}

    for i, c in enumerate(candidates or []):
        try:
            # Only cluster around clickables to avoid dumping huge filter panels.
            if c.tag not in {"a", "button"}:
                sel = c.click_selector()
                if not (isinstance(sel, dict) and sel.get("type") == "attributeValueSelector" and str(sel.get("attribute") or "") == "href"):
                    continue

            key = (c.context_raw or c.context or "").strip()
            if not key:
                key = "(no_context)"

            g = groups.get(key)
            if g is None:
                facts = []
                try:
                    lines = [ln.strip() for ln in str(key or '').splitlines() if ln.strip()]
                    facts = [ln for ln in lines if any(ch.isdigit() for ch in ln)][:6]
                except Exception:
                    facts = []
                g = {"card_text": _safe_truncate(key, int(max_text or 0)), "card_facts": facts, "candidate_ids": [], "actions": []}
                groups[key] = g

            g["candidate_ids"].append(i)
            if len(g["actions"]) < int(max_actions_per_card or 0):
                sel = c.click_selector()
                href = ""
                try:
                    if isinstance(sel, dict) and sel.get("type") == "attributeValueSelector" and str(sel.get("attribute") or "") == "href":
                        href = str(sel.get("value") or "").strip()
                except Exception:
                    href = ""

                g["actions"].append({
                    "candidate_id": i,
                    "tag": c.tag,
                    "text": _safe_truncate(c.text or "", 140),
                    "click": _selector_repr(sel),
                    "href": _safe_truncate(href, 240) if href else "",
                })
        except Exception:
            continue

    ranked = []
    for _k, g in groups.items():
        txt = str(g.get("card_text") or "")
        n_actions = len(g.get("actions") or [])
        L = len(txt)
        penalty = 0
        if L < 40:
            penalty += 400
        if L > 900:
            penalty += min(1200, L - 900)
        score = (1000 - penalty + min(L, 700), n_actions)
        ranked.append((score, g))

    ranked.sort(key=lambda x: x[0], reverse=True)
    cards = [g for _, g in ranked[: int(max_cards or 0)]]
    return {"ok": True, "count": len(cards), "cards": cards}


def _tool_find_card(*, candidates: List["_Candidate"], query: str, max_cards: int = 10, max_text: int = 900, max_actions_per_card: int = 6) -> Dict[str, Any]:
    """Find card-like groups whose text/facts/actions match a query."""
    q = _norm_ws(str(query or "")).lower()
    if not q:
        return {"ok": False, "error": "missing query"}
    base = _tool_list_cards(candidates=candidates, max_cards=max_cards * 4, max_text=max_text, max_actions_per_card=max_actions_per_card)
    if not isinstance(base, dict) or not base.get("ok"):
        return base
    cards = base.get("cards") if isinstance(base.get("cards"), list) else []
    out = []
    for c in cards:
        if not isinstance(c, dict):
            continue
        blob_parts = [str(c.get("card_text") or "")]
        facts = c.get("card_facts") if isinstance(c.get("card_facts"), list) else []
        blob_parts.extend(str(x) for x in facts[:8])
        acts = c.get("actions") if isinstance(c.get("actions"), list) else []
        for a in acts[:6]:
            if isinstance(a, dict):
                blob_parts.append(str(a.get("text") or ""))
                blob_parts.append(str(a.get("href") or ""))
        blob = _norm_ws(" ".join(blob_parts)).lower()
        if q in blob:
            out.append(c)
        if len(out) >= int(max_cards or 0):
            break
    return {"ok": True, "query": q, "count": len(out), "cards": out}


_TOOL_REGISTRY = {
    "search_text": _tool_search_text,
    "visible_text": _tool_visible_text,
    "extract_tables": _tool_extract_tables,
    "extract_entities": _tool_extract_entities,
    "css_select": _tool_css_select,
    "xpath_select": _tool_xpath_select,
    "extract_forms": _tool_extract_forms,
    "list_links": _tool_list_links,
    "list_candidates": _tool_list_candidates,
    "list_cards": _tool_list_cards,
    "find_card": _tool_find_card,
}


def _run_tool(tool: str, args: Dict[str, Any], *, html: str, url: str, candidates: List["_Candidate"]) -> Dict[str, Any]:
    t = str(tool or "").strip()
    fn = _TOOL_REGISTRY.get(t)
    if fn is None:
        return {"ok": False, "error": f"unknown tool: {t}", "known": sorted(_TOOL_REGISTRY.keys())}

    a = args if isinstance(args, dict) else {}
    # Inject shared state for tools that need it.
    if t == "list_candidates":
        return fn(candidates=candidates, **{k: v for k, v in a.items() if k in {"max_n"}})
    if t == "list_cards":
        return fn(candidates=candidates, **{k: v for k, v in a.items() if k in {"max_cards", "max_text", "max_actions_per_card"}})
    if t == "find_card":
        return fn(candidates=candidates, **{k: v for k, v in a.items() if k in {"query", "max_cards", "max_text", "max_actions_per_card"}})
    if t == "list_links":
        return fn(html=html, base_url=str(url or ""), **{k: v for k, v in a.items() if k in {"max_links", "context_max", "href_regex", "text_regex"}})
    if t in {"extract_tables"}:
        return fn(html=html, **{k: v for k, v in a.items() if k in {"max_tables", "max_rows", "max_cols"}})
    if t in {"extract_entities"}:
        return fn(html=html, **{k: v for k, v in a.items() if k in {"max_items"}})
    if t in {"search_text", "visible_text", "css_select", "xpath_select", "extract_forms"}:
        return fn(html=html, **a)

    return {"ok": False, "error": f"tool not wired: {t}"}

def _llm_decide(
    *,
    task_id: str,
    task: str,
    step_index: int,
    url: str,
    candidates: List[_Candidate],
    page_summary: str,
    dom_digest: str,
    html_snapshot: str,
    history: List[Dict[str, Any]] | None,
    page_ir_text: str = "",
    extra_hint: str = "",
    target_hint: str = "",
    state_delta: str = "",
    ir_delta: str = "",
    prev_sig_set: set[str] | None = None,
    model_override: str = "",
    include_reasoning: bool = False,
) -> Dict[str, Any]:
    browser_state = _format_browser_state(candidates=candidates, prev_sig_set=prev_sig_set)
    system_msg = (
        "You are a web automation agent. Given the task, step number, state, history, and state diff, choose ONE next action. "
        "Return JSON only (no markdown). "
        "Do NOT provide detailed chain-of-thought. "
        "Return a JSON object with keys: action, candidate_id, text, url. "
        "Preserve the current URL query parameters (e.g., seed) unless the task requires changing them. "
        "action must be one of: click,type,select,navigate,scroll_down,scroll_up,done. "
        "Constraints: for click/type/select, candidate_id must be an integer index into the BROWSER_STATE list (the number inside [..]). "
        "For type/select, text must be non-empty. "
        "Return done immediately when the requested objective is already satisfied. "
        "Do not explore unrelated links after the objective is met. "
        "If the task requires choosing a specific item that matches multiple attributes, first inspect the page using list_cards or list_links, then click/navigate to the matching item. "
        "You may optionally request an HTML inspection tool instead of an action by returning JSON with keys: tool, args. "
        "Available tools: search_text(args: {query, regex?, case_sensitive?, max_matches?, context_chars?}); "
        "visible_text(args: {max_chars?}); extract_tables(args: {max_tables?, max_rows?, max_cols?}); "
        "extract_entities(args: {max_items?}); css_select(args: {selector, max_nodes?}); xpath_select(args: {xpath, max_nodes?}); "
        "extract_forms(args: {max_forms?, max_inputs?}); list_links(args: {max_links?, context_max?, href_regex?, text_regex?}); "
        "list_candidates(args: {max_n?}); list_cards(args: {max_cards?, max_text?, max_actions_per_card?}); "
        "find_card(args: {query, max_cards?, max_text?, max_actions_per_card?}). "
        "After a tool result is returned, pick the next action. Prefer at most 2 tool calls per step."
    )
    if include_reasoning:
        system_msg += (
            " Add a short 'reasoning' string (max 20 words) to the final action JSON. "
            "Do not include chain-of-thought."
        )

    history_lines: List[str] = []
    for h in _history_tail(history, default_window=6):
        step = h.get("step", "?")
        action = h.get("action", "")
        cid = h.get("candidate_id")
        text = h.get("text", "")
        ok = h.get('exec_ok', True)
        err = h.get('error')
        suffix = 'OK' if ok else f"FAILED err={str(err)[:80]}"
        history_lines.append(f"{step}. {action} cid={cid} text={text} [{suffix}]")

    hint = _history_hint(history)

    structured = _structured_hints(task, candidates)

    cards_preview = ""
    cards_enabled = True
    cards_preview_max_chars = 1800
    try:
        if cards_enabled:
            cards_obj = _tool_list_cards(candidates=candidates, max_cards=12, max_text=420, max_actions_per_card=3)
            if isinstance(cards_obj, dict) and cards_obj.get("ok") and cards_obj.get("cards"):
                cards_preview = json.dumps(cards_obj.get("cards"), ensure_ascii=True)
                if len(cards_preview) > cards_preview_max_chars:
                    cards_preview = cards_preview[: max(0, cards_preview_max_chars - 3)] + "..."
    except Exception:
        cards_preview = ""
    user_msg = (
        f"You have a task and must decide the next single browser action.\n"
        f"TASK: {task}\n"
        f"STEP: {int(step_index)}\n"
        f"URL: {url}\n\n"
        + (f"PAGE IR (PRIMARY STRUCTURED STATE):\n{page_ir_text}\n\n" if page_ir_text else "")
        + f"CURRENT STATE (TEXT SUMMARY):\n{page_summary}\n\n"
        + (f"DOM DIGEST (STRUCTURED):\n{dom_digest}\n\n" if dom_digest else "")
        + (f"CARDS (GROUPED CLICKABLE CONTEXTS JSON):\n{cards_preview}\n\n" if cards_preview else "")
        + f"STRUCTURED STATE (JSON):\n{json.dumps(structured, ensure_ascii=True)}\n\n"
        + (f"HISTORY (last steps):\n{chr(10).join(history_lines)}\n\n" if history_lines else "")
        + (f"STATE HINT: {extra_hint}\n\n" if extra_hint else "")
        + (f"TARGETING HINT: {target_hint}\n\n" if target_hint else "")
        + (f"STATE DELTA (prev -> current): {state_delta}\n\n" if state_delta else "")
        + (f"PAGE IR DELTA (prev -> current): {ir_delta}\n\n" if ir_delta else "")
        + "BROWSER_STATE (interactive elements):\n" + browser_state + "\n\n"
        + "Instructions:\n"
        + "- Output JSON only.\n"
        + "- Return ONE action for this step (no multi-step sequences).\n"
        + "- Prefer done over exploratory clicks once the objective is satisfied.\n"
        + "- For single-destination tasks (for example 'go to <site>'), stop when destination is reached.\n"
        + "- If you need to do a multi-step procedure (login/register/contact), pick the best next step only.\n"
        + "- Use candidate_id for click/type/select and ensure it is in-range.\n"
        + "- Use navigate with a full URL when you need to change pages (prefer preserving existing query params like seed).\n"
        + "- For type/select, include non-empty text.\n"
        + "- For delete/remove/edit tasks: confirm the target item with find_card/list_cards before clicking destructive actions.\n"
        + "- If CREDENTIALS are provided, use those exact values when typing.\n"
    )

    # Default to validator gateway default model.
    model = str(model_override or os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "350"))

    usages: List[Dict[str, Any]] = []
    tool_calls = 0
    max_tool_calls = int(os.getenv("AGENT_MAX_TOOL_CALLS", "2"))
    enable_llm_repair = False

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    def _call(extra_system: str = "") -> Dict[str, Any]:
        sys_msg = system_msg + (" " + extra_system if extra_system else "")
        # Keep system message authoritative even after tool results.
        msgs = [{"role": "system", "content": sys_msg}] + [m for m in messages if m.get("role") != "system"]
        resp = openai_chat_completions(
            task_id=task_id,
            messages=msgs,
            model=str(model),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            u = resp.get("usage")
            if isinstance(u, dict):
                usages.append(u)
        except Exception:
            pass
        content = resp["choices"][0]["message"]["content"]
        obj = _parse_llm_json(content)
        try:
            obj["_meta"] = {"llm_calls": len(usages), "llm_usages": usages, "model": str(model), "tool_calls": tool_calls}
        except Exception:
            pass
        return obj

    def _valid_action(obj: Dict[str, Any]) -> bool:
        a = (obj.get("action") or "").lower()
        if a not in {"click", "type", "select", "navigate", "scroll_down", "scroll_up", "done"}:
            return False
        if a == "navigate":
            u = obj.get("url")
            if not isinstance(u, str) or not u.strip():
                return False
            try:
                if _same_path_query(str(u).strip(), str(url).strip(), base_a=str(url).strip(), base_b=""):
                    return False
            except Exception:
                if str(u).strip() == str(url).strip():
                    return False
            return True
        if a in {"click", "type", "select"}:
            cid = obj.get("candidate_id")
            if isinstance(cid, str) and cid.isdigit():
                cid = int(cid)
            if not isinstance(cid, int) or not (0 <= cid < len(candidates)):
                return False
            if a in {"type", "select"}:
                t = obj.get("text")
                if not isinstance(t, str) or not t.strip():
                    return False
        return True

    def _is_tool(obj: Dict[str, Any]) -> bool:
        t = obj.get("tool")
        if not isinstance(t, str) or not t.strip():
            return False
        # Tool response should not mix action.
        if obj.get("action"):
            return False
        return True

    # Tool-aware loop.
    last_obj: Dict[str, Any] = {}
    for _ in range(max_tool_calls + 2):
        try:
            obj = _call()
        except Exception:
            obj = _call("Return ONLY valid JSON. No markdown. No commentary.")

        last_obj = obj

        if _is_tool(obj) and tool_calls < max_tool_calls:
            tool = str(obj.get("tool") or "").strip()
            args = obj.get("args") if isinstance(obj.get("args"), dict) else {}
            tool_calls += 1
            try:
                result = _run_tool(tool, args, html=html_snapshot, url=str(url), candidates=candidates)
            except Exception as e:
                result = {"ok": False, "error": str(e)[:200]}

            # IMPORTANT: tools must inspect snapshot_html, not dom_digest. We'll attach snapshot_html via closure.
            # This placeholder is replaced below.
            messages.append({"role": "assistant", "content": json.dumps({"tool": tool, "args": args}, ensure_ascii=True)})
            messages.append({"role": "user", "content": "TOOL_RESULT " + tool + ": " + json.dumps(result, ensure_ascii=True)})
            continue

        if _valid_action(obj):
            try:
                obj["_meta"] = {"llm_calls": len(usages), "llm_usages": usages, "model": str(model), "tool_calls": tool_calls}
            except Exception:
                pass
            return obj

        if enable_llm_repair:
            obj = _call(
                "Your previous JSON was invalid. Fix it. "
                f"candidate_id must be an integer in [0, {len(candidates) - 1}]. "
                "If action is type/select you must include non-empty text. "
                "If stuck, scroll_down."
            )
            if _valid_action(obj):
                try:
                    obj["_meta"] = {"llm_calls": len(usages), "llm_usages": usages, "model": str(model), "tool_calls": tool_calls}
                except Exception:
                    pass
                return obj
        else:
            # Deterministic low-latency fallback for invalid outputs.
            fallback = {"action": "scroll_down", "candidate_id": None, "_meta": {"llm_calls": len(usages), "llm_usages": usages, "model": str(model), "tool_calls": tool_calls}}
            return fallback

    return last_obj


def _update_task_state(task_id: str, url: str, sig: str) -> None:
    if not task_id:
        return
    try:
        st = _TASK_STATE.get(task_id)
        if not isinstance(st, dict):
            st = {}
            _TASK_STATE[task_id] = st
        last_sig = str(st.get("last_sig") or "")
        last_url = str(st.get("last_url") or "")
        if sig and sig == last_sig and str(url) == last_url:
            st["repeat"] = int(st.get("repeat") or 0) + 1
        else:
            st["repeat"] = 0
        st["last_sig"] = str(sig)
        st["last_url"] = str(url)
    except Exception:
        return




def _compute_state_delta(
    *,
    task_id: str,
    url: str,
    page_summary: str,
    dom_digest: str,
    html_snapshot: str,
    candidates: List[_Candidate],
) -> str:
    """Compute a compact diff signal between current and previous observed state."""
    if not task_id:
        return ""

    try:
        st = _TASK_STATE.get(task_id)
        if not isinstance(st, dict):
            st = {}
            _TASK_STATE[task_id] = st

        prev_url = str(st.get("prev_url") or "")
        prev_summary = str(st.get("prev_summary") or "")
        prev_digest = str(st.get("prev_digest") or "")
        prev_sig_set = set(st.get("prev_sig_set") or [])

        cur_sig_set = set()
        for c in candidates[:30]:
            sig = f"{_selector_repr(c.selector)}|{(c.text or '')[:80]}"
            cur_sig_set.add(sig)

        added = len(cur_sig_set - prev_sig_set) if prev_sig_set else len(cur_sig_set)
        removed = len(prev_sig_set - cur_sig_set) if prev_sig_set else 0
        unchanged = len(cur_sig_set & prev_sig_set) if prev_sig_set else 0

        # Simple summary change heuristic.
        ps = _norm_ws(prev_summary)
        cs = _norm_ws(page_summary)
        pd = _norm_ws(prev_digest)
        cd = _norm_ws(dom_digest)

        same_summary = bool(ps and cs and ps[:240] == cs[:240])
        same_digest = bool(pd and cd and pd[:240] == cd[:240])

        # Persist current state for next step.
        st["prev_url"] = str(url)
        st["prev_summary"] = str(page_summary)
        st["prev_digest"] = str(dom_digest)
        st["prev_sig_set"] = list(cur_sig_set)

        parts = [
            f"url_changed={str(prev_url != str(url)).lower()}" if prev_url else "url_changed=unknown",
            f"summary_changed={str(not same_summary).lower()}" if (ps and cs) else "summary_changed=unknown",
            f"digest_changed={str(not same_digest).lower()}" if (pd and cd) else "digest_changed=unknown",
            f"candidate_added={added}",
            f"candidate_removed={removed}",
            f"candidate_unchanged={unchanged}",
        ]
        return ", ".join(parts)
    except Exception:
        return ""


class ApifiedWebAgent(IWebAgent):
    """Core operator implementing IWA's IWebAgent interface."""

    def __init__(self, id: str = "1", name: str = "AutoppiaOperator") -> None:
        self.id = str(id)
        self.name = str(name)

    async def act(
        self,
        *,
        task: Task,
        snapshot_html: str,
        screenshot: str | bytes | None = None,
        url: str,
        step_index: int,
        history: list[dict[str, Any]] | None = None,
    ) -> list[BaseAction]:
        task_id = str(getattr(task, "id", "") or "")
        prompt = str(getattr(task, "prompt", "") or "")
        create_action_fn = getattr(BaseAction, "create_action", None)
        if not callable(create_action_fn):
            logger.error(
                f"[AGENT_TRACE] BaseAction.create_action missing "
                f"task_id={task_id} step_index={int(step_index)} "
                f"BaseAction={repr(BaseAction)} import_ok={_AUTOPPIA_IWA_IMPORT_OK}"
            )
        payload = {
            "task_id": task_id,
            "prompt": prompt,
            "snapshot_html": snapshot_html,
            "screenshot": screenshot,
            "url": url,
            "step_index": int(step_index),
            "history": history or [],
        }
        resp = await self.act_from_payload(payload)
        actions = resp.get("actions") if isinstance(resp, dict) else []
        _log_trace(
            f"act() raw actions task_id={task_id} step_index={int(step_index)} "
            f"count={len(actions) if isinstance(actions, list) else 0}"
        )
        out: list[BaseAction] = []
        for a in actions if isinstance(actions, list) else []:
            if not isinstance(a, dict):
                continue
            try:
                ac = create_action_fn(a) if callable(create_action_fn) else None
                if ac is not None:
                    out.append(ac)
            except Exception as exc:
                logger.error(
                    f"[AGENT_TRACE] create_action failed task_id={task_id} step_index={int(step_index)} "
                    f"action_type={str(a.get('type') or '')} err={str(exc)} "
                    f"payload={json.dumps(a, ensure_ascii=True)[:500]}"
                )
                continue
        if isinstance(actions, list) and actions and not out:
            logger.error(
                f"[AGENT_TRACE] all actions dropped during conversion task_id={task_id} "
                f"step_index={int(step_index)} "
                f"raw_types={[str(x.get('type') or '') for x in actions if isinstance(x, dict)]}"
            )
        return out

    async def act_from_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        task_id = str(payload.get("task_id") or "")
        task = payload.get("prompt") or payload.get("task_prompt") or ""
        model_override = str(payload.get("model") or "").strip()
        url = _normalize_demo_url(str(payload.get("url") or ""))
        step_index = int(payload.get("step_index") or 0)
        return_metrics = os.getenv("AGENT_RETURN_METRICS", "0").lower() in {"1", "true", "yes"}
        include_reasoning = str(payload.get("include_reasoning") or payload.get("return_reasoning") or "").strip().lower() in {"1", "true", "yes"}
        completion_only = str(payload.get("completion_only") or "").strip().lower() in {"1", "true", "yes"}
        html = payload.get("snapshot_html") or ""
        history = payload.get("history") if isinstance(payload.get("history"), list) else None
        page_summary = _summarize_html(html)
        dom_digest = _dom_digest(html)
        task = str(task or "")

        def _build_reasoning(actions: list[dict[str, Any]], metrics: dict[str, Any] | None) -> str:
            if isinstance(metrics, dict):
                mr = metrics.get("reasoning")
                if isinstance(mr, str) and mr.strip():
                    return " ".join(mr.strip().split())[:200]
            first_type = ""
            if actions and isinstance(actions[0], dict):
                first_type = str(actions[0].get("type") or "")
            decision = str((metrics or {}).get("decision") or "").strip().lower() if isinstance(metrics, dict) else ""
            if first_type == "DoneAction" or decision == "done":
                return "Task appears complete from current page state."
            if first_type == "NavigateAction" or decision == "navigate":
                return "Navigating to the page most relevant to the task."
            if first_type == "ClickAction" or decision.startswith("click"):
                return "Clicking the most relevant element to make progress."
            if first_type == "TypeAction" or decision == "type":
                return "Typing required input for the next step."
            if first_type == "SelectDropDownOptionAction" or decision == "select":
                return "Selecting the best matching option for this step."
            if first_type == "ScrollAction" or decision.startswith("scroll"):
                return "Scrolling to reveal additional relevant content."
            if first_type == "WaitAction" or decision.endswith("wait"):
                return "Waiting briefly due to low-confidence next action."
            return "Choosing the next best action from current state."

        def _build_action_rationale(action: dict[str, Any], metrics: dict[str, Any] | None) -> str:
            action_type = str((action or {}).get("type") or "")
            if isinstance(metrics, dict):
                mr = metrics.get("reasoning")
                if isinstance(mr, str) and mr.strip():
                    return " ".join(mr.strip().split())[:200]
            if action_type == "DoneAction":
                return "No further useful action was detected from current state."
            if action_type == "NavigateAction":
                return "This URL is the most likely next page for completing the task."
            if action_type == "ClickAction":
                return "This target looks like the highest-value clickable next step."
            if action_type == "TypeAction":
                return "Typing is required to provide missing input for progress."
            if action_type == "SelectDropDownOptionAction":
                return "Selecting this option is needed for the requested flow."
            if action_type == "ScrollAction":
                return "Scrolling should reveal relevant content not yet visible."
            if action_type == "WaitAction":
                return "Waiting briefly is safer than a low-confidence action."
            return "Chosen as the best next action from current context."

        def _usage_cost_from_metrics(metrics: dict[str, Any] | None) -> tuple[dict[str, int] | None, float | None, str | None]:
            if not isinstance(metrics, dict):
                return None, None, None
            llm = metrics.get("llm") if isinstance(metrics.get("llm"), dict) else {}
            model = str(llm.get("model") or metrics.get("model") or os.getenv("OPENAI_MODEL", "")).strip() or None
            usages = llm.get("llm_usages") if isinstance(llm.get("llm_usages"), list) else []
            pt = 0
            ct = 0
            for u in usages:
                if not isinstance(u, dict):
                    continue
                try:
                    pt += int(u.get("prompt_tokens") or 0)
                except Exception:
                    pass
                try:
                    ct += int(u.get("completion_tokens") or 0)
                except Exception:
                    pass
            if pt <= 0 and ct <= 0:
                return None, None, model
            usage = {"prompt_tokens": int(pt), "completion_tokens": int(ct), "total_tokens": int(pt + ct)}
            est_cost = None
            if model:
                try:
                    est_cost, _ = estimate_cost_usd(model, usage)
                except Exception:
                    est_cost = None
            return usage, est_cost, model

        def _resp(actions: list[dict[str, Any]], metrics: dict[str, Any] | None = None) -> Dict[str, Any]:
            sanitized_actions: list[dict[str, Any]] = []
            for action in actions:
                if isinstance(action, dict):
                    sanitized_actions.append(_sanitize_action_payload(action))
                else:
                    sanitized_actions.append({})

            out: Dict[str, Any] = {"actions": sanitized_actions}
            if return_metrics and metrics is not None:
                out["metrics"] = metrics
            usage, est_cost, model = _usage_cost_from_metrics(metrics)
            if usage is not None:
                out["usage"] = usage
                out["total_tokens"] = int(usage.get("total_tokens") or 0)
            if est_cost is not None:
                out["estimated_cost_usd"] = float(est_cost)
            if model:
                out["model"] = model
            if include_reasoning:
                out["reasoning"] = _build_reasoning(sanitized_actions, metrics)
                out["action_rationales"] = [
                    _build_action_rationale(a, metrics) for a in sanitized_actions
                ]
            return out

        extract_max_candidates = 80
        llm_max_candidates = 50
        extract_max_candidates = max(20, min(extract_max_candidates, 200))
        llm_max_candidates = max(10, min(llm_max_candidates, extract_max_candidates))

        candidates = _extract_candidates(html, max_candidates=extract_max_candidates)
        candidates_all = list(candidates)
        candidates = _select_candidates_for_llm(task, candidates_all, current_url=str(url), max_total=llm_max_candidates)
        page_ir = _extract_page_ir(html=html, url=str(url), candidates=candidates)
        page_ir_text = _render_page_ir(page_ir, max_chars=int(os.getenv("AGENT_PAGE_IR_MAX_CHARS", "2200")))
        completion_done = False
        completion_reason = ""
        completion_model = ""
        completion_usage_list: list[dict[str, Any]] = []
        completion_meta: dict[str, Any] = {}

        if task_id == "check":
            if candidates:
                return _resp([{"type": "ClickAction", "selector": candidates[0].click_selector()}], {"decision": "check_click", "candidate_id": 0})
            return _resp([{"type": "WaitAction", "time_seconds": 0.1}], {"decision": "check_wait"})

        # Deterministic recovery: if prior sign-in clicks failed due overlay interception
        # and task asks for retry password flow, force retyping password once.
        try:
            if not completion_only:
                task_l = task.lower()
                typed_vals = [v.lower().strip() for v in _typed_values_from_history(history)]
                asked_retry = ("retry" in task_l) and ("password" in task_l)
                has_wrong = any(v == "wrongpass" for v in typed_vals)
                has_right = any(v == "password" for v in typed_vals)
                blocked_click = _history_has_click_overlay_intercept(history)
                if asked_retry and has_wrong and (not has_right) and blocked_click:
                    pw_cid = None
                    for i, c in enumerate(candidates):
                        attrs = c.attrs or {}
                        if c.tag != "input":
                            continue
                        typ = str(attrs.get("type") or "").lower()
                        blob = _candidate_blob(c)
                        if typ == "password" or "password" in blob:
                            pw_cid = i
                            break
                    if isinstance(pw_cid, int):
                        sel = candidates[pw_cid].type_selector()
                        _update_task_state(task_id, str(url), f"type_retry_password:{_selector_repr(sel)}")
                        return _resp(
                            [{"type": "TypeAction", "selector": sel, "text": "password"}],
                            {"decision": "retry_password_after_click_blocked", "candidate_id": pw_cid},
                        )
        except Exception:
            pass

        # Independent completion check (small model) gated by step/repeat to control latency/cost.
        try:
            completion_enabled = os.getenv("AGENT_ENABLE_COMPLETION_CHECK", "1").lower() in {"1", "true", "yes"}
            completion_min_conf = float(os.getenv("AGENT_COMPLETION_MIN_CONFIDENCE", "0.82"))
            st = _TASK_STATE.get(task_id) if task_id else None
            should_completion_check = _should_run_completion_check(
                completion_only=completion_only,
                step_index=int(step_index),
                history=history,
                task_state=st if isinstance(st, dict) else None,
            )
            if completion_enabled and should_completion_check:
                cc, _ = run_completion_check(
                    task_id=task_id,
                    task=task,
                    url=str(url),
                    page_summary=page_summary,
                    history=history,
                    page_ir_text=page_ir_text,
                )
                if cc.is_complete and float(cc.confidence) >= completion_min_conf:
                    completion_done = True
                    completion_reason = cc.reason or "Task complete"
                    completion_model = cc.model
                    completion_usage_list = [cc.usage] if isinstance(cc.usage, dict) else []
                    completion_meta = {
                        "is_complete": cc.is_complete,
                        "confidence": cc.confidence,
                        "reason": cc.reason,
                        "model": cc.model,
                    }
                if completion_only:
                    if completion_done:
                        return _resp(
                            [{"type": "DoneAction", "reason": completion_reason or "Task complete"}],
                            {
                                "decision": "done_completion_check",
                                "reasoning": completion_reason or "Task complete",
                                "model": completion_model,
                                "llm": {
                                    "model": completion_model,
                                    "llm_usages": completion_usage_list,
                                    "llm_calls": 1,
                                    "tool_calls": 0,
                                },
                                "completion_check": completion_meta,
                            },
                        )
                    return _resp(
                        [],
                        {
                            "decision": "not_done_completion_check",
                            "model": completion_model,
                            "llm": {
                                "model": completion_model,
                                "llm_usages": completion_usage_list,
                                "llm_calls": 1,
                                "tool_calls": 0,
                            },
                            "completion_check": completion_meta or {"is_complete": False},
                        },
                    )
        except Exception:
            if completion_only:
                return _resp([], {"decision": "completion_check_error"})

        st = _TASK_STATE.get(task_id) if task_id else None
        effective_url = str(url)
        try:
            if isinstance(st, dict):
                eu = str(st.get("effective_url") or "").strip()
                if eu:
                    effective_url = eu
        except Exception:
            effective_url = str(url)
        extra_hint = ""
        target_hint = ""
        prev_sig_set = None
        try:
            if isinstance(st, dict):
                prev = st.get('prev_sig_set')
                if isinstance(prev, list):
                    prev_sig_set = set(str(x) for x in prev)
        except Exception:
            prev_sig_set = None

        # RAM subgoal memory (per task_id) to maintain multi-step intent.
        subgoal_mem = _ensure_subgoal_memory(task_id=task_id, task=task)
        try:
            repeat_now = int((st or {}).get("repeat") or 0) if isinstance(st, dict) else 0
        except Exception:
            repeat_now = 0
        _update_subgoal_memory(
            subgoal_mem,
            step_index=int(step_index),
            url=str(url),
            page_ir_text=page_ir_text,
            page_summary=page_summary,
            history=history,
            repeat_count=repeat_now,
        )

        state_delta = _compute_state_delta(task_id=task_id, url=str(url), page_summary=page_summary, dom_digest=dom_digest, html_snapshot=html, candidates=candidates)
        ir_delta = _compute_ir_delta(task_id=task_id, page_ir=page_ir)
        try:
            if isinstance(st, dict):
                last_url = str(st.get("last_url") or "")
                repeat = int(st.get("repeat") or 0)
                if last_url and last_url == str(url) and repeat >= 2:
                    extra_hint = "You appear stuck on the same URL after repeating an action. Choose a different element or scroll."
        except Exception:
            extra_hint = ""
        try:
            sg_hint = _subgoal_hint(subgoal_mem)
            if sg_hint:
                extra_hint = ((extra_hint + " ") if extra_hint else "") + sg_hint
        except Exception:
            pass
        try:
            risk_hint = _task_risk_hint(task=task, step_index=int(step_index), candidates=candidates)
            if risk_hint:
                extra_hint = ((extra_hint + " ") if extra_hint else "") + risk_hint
        except Exception:
            pass
        try:
            target_hint = str(payload.get("target_hint") or os.getenv("AGENT_TARGET_HINT", "")).strip()
        except Exception:
            target_hint = ""

        try:
            base_url = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
            if not os.getenv("OPENAI_API_KEY") and not is_sandbox_gateway_base_url(base_url):
                raise RuntimeError("OPENAI_API_KEY not set")
            decision = _llm_decide(
                task_id=task_id,
                task=task,
                step_index=step_index,
                url=effective_url,
                candidates=candidates,
                page_summary=page_summary,
                dom_digest=dom_digest,
                html_snapshot=html,
                history=history,
                page_ir_text=page_ir_text,
                extra_hint=extra_hint,
                target_hint=target_hint,
                state_delta=state_delta,
                ir_delta=ir_delta,
                prev_sig_set=prev_sig_set,
                model_override=model_override,
                include_reasoning=include_reasoning,
            )
        except Exception as e:
            if _LOG_ERRORS:
                logger.exception(
                    f"[AGENT_TRACE] llm_decide exception task_id={task_id} "
                    f"step_index={int(step_index)} url={str(url)} err={str(e)}"
                )
            if os.getenv("AGENT_DEBUG_ERRORS", "0").lower() in {"1", "true", "yes"}:
                raise HTTPException(status_code=500, detail=str(e)[:400])
            return _resp([{"type": "WaitAction", "time_seconds": 1.0}], {"decision": "error_wait"})

        action = (decision.get("action") or "").lower()
        cid = decision.get("candidate_id")
        text = decision.get("text")
        if isinstance(cid, str) and cid.isdigit():
            cid = int(cid)
        original_cid = cid if isinstance(cid, int) else None

        # Deterministic low-cost validation/repair before mapping to executable actions.
        valid_actions = {"click", "type", "select", "navigate", "scroll_down", "scroll_up", "done"}
        if action not in valid_actions:
            action = "scroll_down"
            decision["action"] = action
            cid = None
        if action in {"click", "type", "select"} and not (isinstance(cid, int) and 0 <= cid < len(candidates)):
            fallback_cid = _pick_fallback_candidate_id(
                candidates=candidates,
                action=action,
                decision=decision,
                avoid_id=None,
            )
            if isinstance(fallback_cid, int):
                cid = fallback_cid
                decision["candidate_id"] = fallback_cid
            else:
                action = "scroll_down"
                decision["action"] = action
                cid = None
        if action in {"type", "select"} and (not isinstance(text, str) or not text.strip()):
            action = "scroll_down"
            decision["action"] = action
            cid = None
        if action == "navigate":
            nav_url_chk = str(decision.get("url") or "").strip()
            if not nav_url_chk:
                action = "scroll_down"
                decision["action"] = action
            else:
                try:
                    if _same_path_query(nav_url_chk, effective_url, base_a=effective_url, base_b=""):
                        action = "scroll_down"
                        decision["action"] = action
                except Exception:
                    pass

        # Loop guard: if same decision repeats on same URL recently, force a different move.
        if _is_repeated_recent_decision(history=history, url=str(url), decision=decision, threshold=2):
            if action in {"click", "type", "select"}:
                alt_cid = _pick_fallback_candidate_id(
                    candidates=candidates,
                    action=action,
                    decision=decision,
                    avoid_id=original_cid if isinstance(original_cid, int) else None,
                )
                if isinstance(alt_cid, int):
                    cid = alt_cid
                    decision["candidate_id"] = alt_cid
                else:
                    action = "scroll_down"
                    decision["action"] = action
                    cid = None

        # Hard stop when same URL + same decision pattern keeps repeating.
        repeat_done_threshold = 4
        if repeat_done_threshold > 0:
            repeat_count = _count_repeated_recent_decision(history=history, url=str(url), decision=decision)
            if repeat_count >= repeat_done_threshold:
                action = "done"
                decision["action"] = "done"
                decision["reason"] = "No further progress detected"

        out: Dict[str, Any]
        if action == "navigate":
            nav_url_raw = str(decision.get("url") or "").strip()
            if not nav_url_raw:
                out = _resp([{"type": "WaitAction", "time_seconds": 1.0}], {"decision": "navigate_missing_url"})
            else:
                nav_url = _resolve_url(nav_url_raw, effective_url or str(url))
                if _same_path_query(nav_url, effective_url, base_a=effective_url, base_b=""):
                    _update_task_state(task_id, str(url), "navigate_same_url_scroll")
                    out = _resp([{ "type": "ScrollAction", "down": True, "up": False}], {"decision": "scroll_override"})
                else:
                    _update_task_state(task_id, str(url), f"navigate:{nav_url}")
                    try:
                        if task_id and isinstance(_TASK_STATE.get(task_id), dict):
                            _TASK_STATE[task_id]["effective_url"] = str(nav_url)
                    except Exception:
                        pass
                    out = _resp([{"type": "NavigateAction", "url": nav_url, "go_back": False, "go_forward": False}], {"decision": "navigate", "url": nav_url, "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})})
        elif action in {"scroll_down", "scroll_up"}:
            _update_task_state(task_id, str(url), f"{action}")
            out = _resp([{"type": "ScrollAction", "down": action == "scroll_down", "up": action == "scroll_up"}], {"decision": decision.get("action"), "candidate_id": int(cid) if isinstance(cid,int) else None, "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})})
        elif action == "done":
            _update_task_state(task_id, str(url), "done")
            out = _resp(
                [{"type": "DoneAction", "reason": str(decision.get("reason") or "Task complete")}],
                {
                    "decision": "done",
                    "candidate_id": int(cid) if isinstance(cid, int) else None,
                    "model": decision.get("_meta", {}).get("model"),
                    "llm": decision.get("_meta", {}),
                },
            )
        elif action in {"click", "type", "select"} and isinstance(cid, int) and 0 <= cid < len(candidates):
            c = candidates[cid]
            if action == "click":
                selector = c.click_selector()
                try:
                    if isinstance(selector, dict) and selector.get("type") == "attributeValueSelector" and selector.get("attribute") == "href":
                        href = str(selector.get("value") or "")
                        fixed = _preserve_seed_url(href, effective_url or str(url))
                        if fixed and fixed != href:
                            fixed_abs = _resolve_url(fixed, effective_url or str(url))
                            if _same_path_query(fixed_abs, effective_url, base_a=effective_url, base_b=""):
                                _update_task_state(task_id, str(url), "navigate_seed_fix_same_url_scroll")
                                out = _resp([{ "type": "ScrollAction", "down": True, "up": False}], {"decision": "scroll_override"})
                            else:
                                _update_task_state(task_id, str(url), f"navigate_seed_fix:{fixed_abs}")
                                try:
                                    if task_id and isinstance(_TASK_STATE.get(task_id), dict):
                                        _TASK_STATE[task_id]["effective_url"] = str(fixed_abs)
                                except Exception:
                                    pass
                                out = _resp([{ "type": "NavigateAction", "url": fixed_abs, "go_back": False, "go_forward": False}], {"decision": "navigate", "url": fixed_abs, "candidate_id": int(cid) if isinstance(cid,int) else None, "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})})
                        else:
                            _update_task_state(task_id, str(url), f"click:{_selector_repr(selector)}")
                            out = _resp([{"type": "ClickAction", "selector": selector}], {"decision": "click", "candidate_id": int(cid) if isinstance(cid,int) else None, "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})})
                    else:
                        _update_task_state(task_id, str(url), f"click:{_selector_repr(selector)}")
                        out = _resp([{"type": "ClickAction", "selector": selector}], {"decision": "click", "candidate_id": int(cid) if isinstance(cid,int) else None, "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})})
                except Exception:
                    _update_task_state(task_id, str(url), f"click:{_selector_repr(selector)}")
                    out = _resp([{"type": "ClickAction", "selector": selector}], {"decision": "click", "candidate_id": int(cid) if isinstance(cid,int) else None, "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})})
            elif action == "type":
                if not text:
                    raise HTTPException(status_code=400, detail="type action missing text")
                selector = c.type_selector()
                _update_task_state(task_id, str(url), f"type:{_selector_repr(selector)}")
                out = _resp([{"type": "TypeAction", "selector": selector, "text": str(text)}], {"decision": "type", "candidate_id": int(cid) if isinstance(cid,int) else None, "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})})
            else:
                if not text:
                    raise HTTPException(status_code=400, detail="select action missing text")
                selector = c.type_selector()
                _update_task_state(task_id, str(url), f"select:{_selector_repr(selector)}")
                out = _resp([{"type": "SelectDropDownOptionAction", "selector": selector, "text": str(text), "timeout_ms": int(os.getenv("AGENT_SELECT_TIMEOUT_MS", "4000"))}], {"decision": "select", "candidate_id": int(cid) if isinstance(cid,int) else None, "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})})
        else:
            if candidates and step_index < 5:
                selector = candidates[0].click_selector()
                _update_task_state(task_id, str(url), f"fallback_click:{_selector_repr(selector)}")
                out = _resp([{"type": "ClickAction", "selector": selector}], {"decision": "click_override", "candidate_id": 0 if candidates else None, "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})})
            else:
                _update_task_state(task_id, str(url), "fallback_wait")
                out = _resp([{"type": "WaitAction", "time_seconds": 2.0}], {"decision": "fallback_wait", "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})})

        try:
            threshold = int(os.getenv("AGENT_REPEAT_DONE_THRESHOLD", "4"))
            st_after = _TASK_STATE.get(task_id) if task_id else None
            repeat_after = int((st_after or {}).get("repeat") or 0) if isinstance(st_after, dict) else 0
            if threshold > 0 and repeat_after >= threshold:
                out = _resp(
                    [{"type": "DoneAction", "reason": "No further progress detected"}],
                    {
                        "decision": "done_stuck",
                        "repeat": repeat_after,
                        "model": decision.get("_meta", {}).get("model"),
                        "llm": decision.get("_meta", {}),
                    },
                )
        except Exception:
            pass

        # If all inferred subgoals are done, force final DoneAction.
        try:
            st_after = _TASK_STATE.get(task_id) if task_id else None
            mem_after = st_after.get("subgoal_memory") if isinstance(st_after, dict) else None
            if _all_subgoals_done(mem_after):
                actions_out = out.get("actions") if isinstance(out, dict) and isinstance(out.get("actions"), list) else []
                has_done = any(isinstance(a, dict) and str(a.get("type") or "") in {"DoneAction", "FinishAction"} for a in actions_out)
                if not has_done:
                    actions_out.append({"type": "DoneAction", "reason": "All subgoals completed"})
                    out["actions"] = actions_out
                if isinstance(out.get("metrics"), dict):
                    m = dict(out.get("metrics") or {})
                    m["decision"] = "done_subgoals"
                    out["metrics"] = m
        except Exception:
            pass

        # Attach compact subgoal memory for observability.
        try:
            st_after = _TASK_STATE.get(task_id) if task_id else None
            mem_after = st_after.get("subgoal_memory") if isinstance(st_after, dict) else None
            if isinstance(out, dict):
                if isinstance(out.get("metrics"), dict):
                    m = dict(out.get("metrics") or {})
                    m["subgoal_memory"] = _compact_subgoal_metrics(mem_after)
                    out["metrics"] = m
        except Exception:
            pass

        # If completion checker is confident, append DoneAction at the end so
        # any planned action for this step is still executed first.
        try:
            if completion_done and isinstance(out, dict):
                actions_out = out.get("actions") if isinstance(out.get("actions"), list) else []
                has_done = any(
                    isinstance(a, dict) and str(a.get("type") or "") in {"DoneAction", "FinishAction"}
                    for a in actions_out
                )
                if not has_done:
                    actions_out.append({"type": "DoneAction", "reason": completion_reason or "Task complete"})
                    out["actions"] = actions_out
                if isinstance(out.get("metrics"), dict):
                    m = dict(out.get("metrics") or {})
                    m["completion_check"] = completion_meta
                    llm = dict(m.get("llm") or {})
                    if completion_usage_list:
                        merged_usages = list(llm.get("llm_usages") or [])
                        merged_usages.extend(completion_usage_list)
                        llm["llm_usages"] = merged_usages
                        llm["llm_calls"] = int(llm.get("llm_calls") or 0) + 1
                    if completion_model:
                        llm["completion_model"] = completion_model
                    m["llm"] = llm
                    out["metrics"] = m
        except Exception:
            pass

        if include_reasoning:
            llm_reasoning = decision.get("reasoning")
            if isinstance(llm_reasoning, str) and llm_reasoning.strip():
                llm_text = " ".join(llm_reasoning.strip().split())[:200]
                out["reasoning"] = llm_text
                actions_out = out.get("actions") if isinstance(out.get("actions"), list) else []
                if actions_out:
                    out["action_rationales"] = [llm_text for _ in actions_out]

        try:
            action_types = [str(a.get("type") or "") for a in out.get("actions", []) if isinstance(a, dict)]
        except Exception:
            action_types = []
        _log_trace(
            f"act_from_payload task_id={task_id} step_index={int(step_index)} "
            f"decision={str(action)} candidate_id={str(cid)} out_actions={action_types}"
        )
        return out

# -----------------------------
# HTTP entrypoint
# -----------------------------

AutoppiaOperator = ApifiedWebAgent
OPERATOR = AutoppiaOperator(id=os.getenv("WEB_AGENT_ID", "1"), name="AutoppiaOperator")


def _task_from_payload(payload: Dict[str, Any]) -> Task:
    """Build a minimal Task object from /act payload for the IWebAgent interface."""
    task_payload = {
        "id": str(payload.get("task_id") or ""),
        "url": _normalize_demo_url(str(payload.get("url") or "")),
        "prompt": str(payload.get("prompt") or payload.get("task_prompt") or ""),
        "web_project_id": payload.get("web_project_id"),
    }
    try:
        if isinstance(Task, type):
            return Task(**task_payload)
    except Exception:
        pass
    return SimpleNamespace(**task_payload)


@app.post("/act", summary="Decide next agent actions")
async def act(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    task_id = str(payload.get("task_id") or "")
    step_index = int(payload.get("step_index") or 0)
    url = _normalize_demo_url(str(payload.get("url") or ""))
    _log_trace(
        f"/act start task_id={task_id} step_index={step_index} url={url} "
        f"prompt_len={len(str(payload.get('prompt') or payload.get('task_prompt') or ''))} "
        f"html_len={len(str(payload.get('snapshot_html') or ''))} "
        f"history_len={len(payload.get('history') or []) if isinstance(payload.get('history'), list) else 0}"
    )
    raw_resp = await OPERATOR.act_from_payload(payload)
    actions = raw_resp.get("actions") if isinstance(raw_resp, dict) else []
    normalized = []
    for action in actions if isinstance(actions, list) else []:
        try:
            if isinstance(action, dict):
                normalized.append(_sanitize_action_payload(action))
                continue
            action_payload = action.model_dump(exclude_none=True)
            normalized.append(_sanitize_action_payload(action_payload))
        except Exception as exc:
            logger.error(
                f"[AGENT_TRACE] /act action normalization failed task_id={task_id} "
                f"step_index={step_index} err={str(exc)} raw={str(action)[:500]}"
            )
            continue
    _log_trace(
        f"/act end task_id={task_id} step_index={step_index} "
        f"raw_count={len(actions) if isinstance(actions, list) else 0} out_count={len(normalized)} "
        f"types={[str(a.get('type') or '') for a in normalized if isinstance(a, dict)]}"
    )
    out: Dict[str, Any] = {"actions": normalized}
    if isinstance(raw_resp, dict) and "metrics" in raw_resp:
        out["metrics"] = raw_resp.get("metrics")
    if isinstance(raw_resp, dict) and isinstance(raw_resp.get("reasoning"), str):
        out["reasoning"] = str(raw_resp.get("reasoning")).strip()[:200]
    if isinstance(raw_resp, dict) and isinstance(raw_resp.get("action_rationales"), list):
        vals: list[str] = []
        for x in raw_resp.get("action_rationales", []):
            if isinstance(x, str) and x.strip():
                vals.append(" ".join(x.strip().split())[:200])
        if vals:
            out["action_rationales"] = vals
    if isinstance(raw_resp, dict) and isinstance(raw_resp.get("usage"), dict):
        out["usage"] = raw_resp.get("usage")
    if isinstance(raw_resp, dict) and raw_resp.get("total_tokens") is not None:
        out["total_tokens"] = raw_resp.get("total_tokens")
    if isinstance(raw_resp, dict) and raw_resp.get("estimated_cost_usd") is not None:
        out["estimated_cost_usd"] = raw_resp.get("estimated_cost_usd")
    if isinstance(raw_resp, dict) and isinstance(raw_resp.get("model"), str):
        out["model"] = str(raw_resp.get("model"))
    return out


@app.post("/step", summary="Alias for /act")
async def step(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return await act(payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
