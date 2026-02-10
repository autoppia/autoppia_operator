from typing import Any, Dict, List, Optional, Tuple

import os
import json
import re
import urllib.request
import urllib.error
from html.parser import HTMLParser

from fastapi import Body, FastAPI, HTTPException


FIXED_AUTBOOKS_URL = os.getenv(
    "FIXED_AUTBOOKS_URL",
    "http://84.247.180.192:8001/books/book-original-002?seed=36",
)

app = FastAPI(title="Autoppia Web Agent API")


@app.get("/health", summary="Health check")
async def health() -> Dict[str, str]:
    return {"status": "healthy"}


class _Candidate:
    def __init__(self, selector: str, text: str, tag: str, attrs: Dict[str, str]):
        self.selector = selector
        self.text = text
        self.tag = tag
        self.attrs = attrs


class _CandidateExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._stack: List[str] = []
        self._current_text: List[str] = []
        self._last_tag: Optional[str] = None
        self._last_attrs: Dict[str, str] = {}
        self.candidates: List[_Candidate] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attr_map = {k: (v or "") for k, v in attrs}
        self._stack.append(tag)
        self._last_tag = tag
        self._last_attrs = attr_map
        if tag in {"button", "a", "input", "textarea", "select"} or attr_map.get("role") in {"button", "link"}:
            selector = _build_selector(tag, attr_map)
            text = attr_map.get("aria-label") or attr_map.get("placeholder") or ""
            self.candidates.append(_Candidate(selector, text, tag, attr_map))

    def handle_data(self, data: str) -> None:
        if self._last_tag in {"button", "a"} and data.strip():
            self._current_text.append(data.strip())

    def handle_endtag(self, tag: str) -> None:
        if tag == self._last_tag and self._current_text:
            text = " ".join(self._current_text)[:120]
            if self.candidates:
                self.candidates[-1].text = text or self.candidates[-1].text
        self._current_text = []
        if self._stack:
            self._stack.pop()


def _build_selector(tag: str, attrs: Dict[str, str]) -> str:
    if attrs.get("id"):
        return f"#{_css_escape(attrs['id'])}"
    if attrs.get("name"):
        return f'{tag}[name="{_css_escape(attrs["name"])}"]'
    if attrs.get("aria-label"):
        return f'[aria-label="{_css_escape(attrs["aria-label"])}"]'
    if attrs.get("placeholder"):
        return f'[placeholder="{_css_escape(attrs["placeholder"])}"]'
    return tag


def _css_escape(s: str) -> str:
    return re.sub(r'([!"#$%&\'()*+,./:;<=>?@\[\\\]^`{|}~])', r"\\\1", s)


def _extract_candidates(html: str, max_candidates: int = 30) -> List[_Candidate]:
    if not html:
        return []
    parser = _CandidateExtractor()
    try:
        parser.feed(html)
    except Exception:
        return []
    return parser.candidates[:max_candidates]


class LLMGateway:
    """Minimal gateway-style client that adds IWA-Task-Id for subnet tracking."""

    def __init__(self) -> None:
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.api_key = os.getenv("OPENAI_API_KEY", "")

    def predict(self, *, task_id: str, messages: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        body = {
            "model": model,
            "messages": messages,
            "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
            "max_tokens": 300,
            "response_format": {"type": "json_object"},
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "IWA-Task-ID": task_id,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"OpenAI error: {e.read().decode('utf-8')}")
        return json.loads(raw)


_llm_gateway = LLMGateway()


def _call_openai(task_id: str, messages: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return _llm_gateway.predict(task_id=task_id, messages=messages, model=model)


def _llm_decide(task_id: str, task: str, url: str, candidates: List[_Candidate]) -> Dict[str, Any]:
    items = []
    for i, c in enumerate(candidates):
        label = (c.text or "").strip()
        if not label:
            label = c.attrs.get("placeholder", "") or c.attrs.get("aria-label", "")
        items.append(f"{i}: <{c.tag}> '{label}' selector={c.selector}")

    system_msg = (
        "You are a web automation agent. Choose ONE action. "
        "Return JSON: {\"action\":\"click|type|select|scroll_down|scroll_up|wait|done\", "
        "\"candidate_id\":int|null, \"text\":string|null}."
    )
    user_msg = (
        f"TASK: {task}\nURL: {url}\n\nCANDIDATES:\n" + "\n".join(items[:30]) +
        "\n\nRules: candidate_id required for click/type/select. text required for type/select."
    )
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    resp = _call_openai(
        task_id=task_id,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        model=model,
    )
    content = resp["choices"][0]["message"]["content"]
    return json.loads(content)


@app.post("/act", summary="Decide next agent actions")
async def act(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    Basic LLM-driven CUA endpoint.

    Uses OpenAI (via HTTP) to select ONE action based on a small list of
    candidates parsed from the HTML snapshot. Falls back to a fixed
    NavigateAction if LLM is not configured.
    """
    task_id = str(payload.get("task_id") or "")
    task = payload.get("prompt") or payload.get("task_prompt") or ""
    url = payload.get("url") or ""
    html = payload.get("snapshot_html") or ""

    candidates = _extract_candidates(html)

    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set")
        decision = _llm_decide(task_id, task, url, candidates)
    except Exception:
        return {
            "actions": [
                {
                    "type": "NavigateAction",
                    "url": FIXED_AUTBOOKS_URL,
                }
            ]
        }

    action = (decision.get("action") or "").lower()
    cid = decision.get("candidate_id")
    text = decision.get("text")

    if action in {"scroll_down", "scroll_up"}:
        return {"actions": [{"type": "ScrollAction", "down": action == "scroll_down", "up": action == "scroll_up"}]}
    if action == "wait":
        return {"actions": [{"type": "WaitAction", "time_seconds": 2.0}]}
    if action == "done":
        return {"actions": []}

    if action in {"click", "type", "select"} and isinstance(cid, int) and 0 <= cid < len(candidates):
        selector = {
            "type": "attributeValueSelector",
            "attribute": "custom",
            "value": candidates[cid].selector,
            "case_sensitive": False,
        }
        if action == "click":
            return {"actions": [{"type": "ClickAction", "selector": selector}]}
        if action == "type":
            if not text:
                raise HTTPException(status_code=400, detail="type action missing text")
            return {"actions": [{"type": "TypeAction", "selector": selector, "text": str(text)}]}
        if action == "select":
            if not text:
                raise HTTPException(status_code=400, detail="select action missing text")
            return {"actions": [{"type": "SelectDropDownOptionAction", "selector": selector, "text": str(text)}]}

    return {"actions": [{"type": "WaitAction", "time_seconds": 2.0}]}


@app.post("/step", summary="Alias for /act")
async def step(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return await act(payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
