import asyncio
from types import SimpleNamespace

import pytest

import agent


@pytest.fixture(autouse=True)
def clear_task_state() -> None:
    agent._TASK_STATE.clear()
    yield
    agent._TASK_STATE.clear()


def test_url_helpers_normalize_resolve_and_preserve_seed() -> None:
    """Url helpers normalize resolve and preserve seed."""
    assert agent._normalize_demo_url("https://demo.example/path?q=1#frag") == "http://localhost/path?q=1#frag"
    assert agent._normalize_demo_url("/login") == "http://localhost/login"
    assert agent._resolve_url("/products", "http://localhost/shop?seed=7") == "http://localhost/products"
    assert agent._path_query("/products?x=1", "http://localhost/base") == ("/products", "x=1")
    assert agent._same_path_query("/products?x=1", "http://localhost/products?x=1", base_a="http://localhost/base") is True
    assert agent._preserve_seed_url("/checkout", "http://localhost/shop?seed=7") == "/checkout?seed=7"


def test_extract_candidates_and_structured_hints(sample_html: str) -> None:
    """Extract candidates and structured hints."""
    candidates = agent._extract_candidates(sample_html, max_candidates=20)

    assert candidates
    assert {"input", "select", "a", "button"}.issubset({c.tag for c in candidates})
    assert all((c.attrs or {}).get("type") != "hidden" for c in candidates)

    inputs = agent._structured_inputs(candidates)
    clickables = agent._structured_clickables(candidates)
    selected = agent._select_candidates_for_llm("buy a camera", candidates, current_url="http://localhost/home?seed=7", max_total=6)

    assert any(item["kind"] == "search" for item in inputs)
    assert any(item["label"] == "View camera" for item in clickables)
    assert len(selected) <= 6
    assert all(not (c.tag == "a" and (c.attrs or {}).get("href") == "/home?seed=7") for c in selected)


def test_page_ir_rendering_and_delta_tracking(sample_html: str) -> None:
    """Page IR rendering and delta tracking."""
    candidates = agent._extract_candidates(sample_html, max_candidates=20)

    ir = agent._extract_page_ir(
        html=sample_html,
        url="http://localhost/catalog?seed=7",
        candidates=candidates,
    )
    rendered = agent._render_page_ir(ir)
    first_delta = agent._compute_ir_delta(task_id="task-ir", page_ir=ir)

    mutated_ir = dict(ir)
    mutated_ir["ctas"] = ["Search", "Buy now", "Checkout"]
    second_delta = agent._compute_ir_delta(task_id="task-ir", page_ir=mutated_ir)

    assert ir["title"] == "Demo Shop"
    assert ir["url_path"] == "/catalog"
    assert "Featured products" in ir["headings"]
    assert ir["forms"]
    assert ir["links"]
    assert ir["cards"]
    assert "TITLE: Demo Shop" in rendered
    assert "FORMS:" in rendered
    assert "LINKS:" in rendered
    assert "forms:0->1" in first_delta
    assert "ctas_added=1" in second_delta


def test_dom_tools_and_tool_runner(sample_html: str) -> None:
    """Dom tools and tool runner."""
    candidates = agent._extract_candidates(sample_html, max_candidates=20)

    search = agent._tool_search_text(html=sample_html, query="Camera 3000")
    forms = agent._tool_extract_forms(html=sample_html)
    links = agent._tool_list_links(html=sample_html, base_url="http://localhost/catalog?seed=7", href_regex="products")
    cards = agent._tool_list_cards(candidates=candidates)
    found_card = agent._tool_find_card(candidates=candidates, query="camera")
    via_runner = agent._run_tool(
        "list_links",
        {"text_regex": "view"},
        html=sample_html,
        url="http://localhost/catalog?seed=7",
        candidates=candidates,
    )

    assert search["ok"] is True
    assert search["count"] >= 1
    assert forms["ok"] is True
    assert forms["forms"][0]["id"] == "search-form"
    assert links["ok"] is True
    assert any(link["url"].endswith("/products/camera?seed=7") for link in links["links"])
    assert cards["ok"] is True
    assert cards["count"] >= 1
    assert found_card["ok"] is True
    assert found_card["count"] >= 1
    assert via_runner["ok"] is True
    assert via_runner["count"] >= 2


def test_parse_llm_json_and_validation_helpers(sample_html: str) -> None:
    """Parse LLM json and validation helpers."""
    candidates = agent._extract_candidates(sample_html, max_candidates=20)

    parsed = agent._parse_llm_json('```json\n{"action":"click","candidate_id":"1"}\n```')

    assert parsed == {"action": "click", "candidate_id": "1"}
    assert agent._llm_is_tool({"tool": "list_links"}) is True
    assert agent._llm_valid_navigate({"url": "/checkout?seed=7"}, "http://localhost/catalog?seed=7") is True
    assert agent._llm_valid_navigate({"url": "/catalog?seed=7"}, "http://localhost/catalog?seed=7") is False
    assert agent._llm_valid_action({"action": "click", "candidate_id": 0}, "http://localhost/catalog?seed=7", candidates) is True
    assert agent._llm_valid_action({"action": "type", "candidate_id": 0}, "http://localhost/catalog?seed=7", candidates) is False


def test_act_from_payload_check_mode_returns_click_action(sample_html: str, monkeypatch) -> None:
    """Act from payload check mode returns click action."""
    monkeypatch.setenv("AGENT_RETURN_METRICS", "1")
    operator = agent.ApifiedWebAgent()

    resp = asyncio.run(
        operator.act_from_payload(
            {
                "task_id": "check",
                "prompt": "open the homepage",
                "url": "https://demo.example/catalog?seed=7",
                "snapshot_html": sample_html,
                "step_index": 0,
                "history": [],
            }
        )
    )

    assert resp["actions"]
    assert resp["actions"][0]["type"] in {"ClickAction", "WaitAction"}
    assert resp["metrics"]["decision"].startswith("check_")


def test_act_from_payload_uses_llm_decision_and_falls_back_on_error(monkeypatch) -> None:
    """Act from payload uses LLM decision and falls back on error."""
    operator = agent.ApifiedWebAgent()
    payload = {
        "task_id": "task-live",
        "prompt": "buy now",
        "url": "http://localhost/product?seed=7",
        "snapshot_html": "<html><body><button id='buy'>Buy now</button></body></html>",
        "step_index": 1,
        "history": [],
    }

    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:9000/openai/v1")

    def fake_llm_decide(**kwargs):
        assert kwargs["task_id"] == "task-live"
        assert kwargs["task"] == "buy now"
        return {"action": "click", "candidate_id": 0, "reason": "primary CTA"}

    monkeypatch.setattr(agent, "_llm_decide", fake_llm_decide)

    success = asyncio.run(operator.act_from_payload(payload))

    assert success["actions"] == [
        {"type": "ClickAction", "selector": {"type": "attributeValueSelector", "attribute": "id", "value": "buy", "case_sensitive": False}}
    ]

    def broken_llm_decide(**kwargs):
        raise RuntimeError("provider failed")

    monkeypatch.setattr(agent, "_llm_decide", broken_llm_decide)

    fallback = asyncio.run(operator.act_from_payload(payload))

    assert fallback["actions"] == [{"type": "WaitAction", "time_seconds": 1.0}]


def test_selector_and_candidate_methods() -> None:
    """Selector and candidate methods."""
    assert agent._selector_repr(agent._sel_attr("id", "buy")) == "attr[id]=buy"
    assert agent._selector_repr(agent._sel_text("Buy now")) == "text~=Buy now"
    assert agent._sel_xpath("//a")["type"] == "xpathSelector"

    candidate = agent._Candidate(
        selector=agent._sel_custom("button"),
        text="Buy now",
        tag="button",
        attrs={"title": "Buy"},
        text_selector=agent._sel_text("Buy now"),
    )
    assert candidate.click_selector()["attribute"] == "title"
    assert candidate.type_selector()["attribute"] == "custom"


def test_fallback_candidate_extractor_without_bs4(monkeypatch) -> None:
    """Fallback candidate extractor without bs4."""
    monkeypatch.setattr(agent, "BeautifulSoup", None)

    html = "<html><body><a href='/next'>Next</a><button>Buy</button></body></html>"
    candidates = agent._extract_candidates(html, max_candidates=10)
    digest = agent._dom_digest(html)

    assert [c.tag for c in candidates] == ["a", "button"]
    assert digest


def test_tool_error_paths_and_state_helpers(monkeypatch) -> None:
    """Tool error paths and state helpers."""
    assert agent._tool_search_text(html="<html></html>", query="", regex=False)["ok"] is False
    assert agent._run_tool("unknown", {}, html="", url="", candidates=[])["ok"] is False
    assert agent._state_delta_strings_equal("abc", "abc") is True
    assert agent._state_delta_strings_equal("a" * 250, "a" * 250 + "b") is True
    assert agent._act_effective_url_from_state({"effective_url": "http://localhost/next"}, "http://localhost/current") == "http://localhost/next"
    assert agent._act_prev_sig_set_from_state({"prev_sig_set": ["a", "b"]}) == {"a", "b"}

    monkeypatch.setattr(agent, "_task_risk_hint", lambda task, step_index, candidates: "Avoid checkout")
    hint = agent._act_extra_hint_from_state(
        {"last_url": "http://localhost/page", "repeat": 2},
        "http://localhost/page",
        "buy now",
        3,
        [],
    )
    assert "You appear stuck" in hint
    assert "Avoid checkout" in hint


def test_dom_selection_tools_and_task_payload(sample_html: str) -> None:
    """Dom selection tools and task payload."""
    candidates = agent._extract_candidates(sample_html, max_candidates=20)

    css = agent._tool_css_select(html=sample_html, selector="article h2")
    xpath = agent._tool_xpath_select(html=sample_html, xpath="//article//h2/text()")
    visible = agent._tool_visible_text(html=sample_html, max_chars=200)
    listed = agent._tool_list_candidates(candidates=candidates, max_n=3)
    task = agent._task_from_payload({"task_id": "t1", "url": "/page", "prompt": "Buy camera"})
    delta1 = agent._compute_state_delta(
        task_id="state-1",
        url="http://localhost/page?seed=7",
        page_summary="Page summary",
        dom_digest="Digest one",
        _html_snapshot=sample_html,
        candidates=candidates,
    )
    delta2 = agent._compute_state_delta(
        task_id="state-1",
        url="http://localhost/page2?seed=7",
        page_summary="Other summary",
        dom_digest="Digest two",
        _html_snapshot=sample_html,
        candidates=candidates[:2],
    )

    assert css["ok"] is True and css["count"] == 2
    assert xpath["ok"] is True and xpath["nodes"][0]["value"] == "Camera 3000"
    assert visible["ok"] is True and "Featured products" in visible["text"]
    assert listed["ok"] is True and len(listed["candidates"]) == 3
    assert task.id == "t1"
    assert task.url == "http://localhost/page"
    assert "url_changed=unknown" in delta1
    assert "url_changed=true" in delta2


def test_history_risk_and_browser_state_helpers(sample_html: str) -> None:
    """History risk and browser state helpers."""
    candidates = agent._extract_candidates(sample_html, max_candidates=20)

    history_hint = agent._history_hint(
        [
            {"action": "click", "candidate_id": 1},
            {"action": "click", "candidate_id": 1},
            {"action": "click", "candidate_id": 1},
        ]
    )
    risk_hint = agent._task_risk_hint("Delete the old product", 0, candidates)
    browser_state = agent._format_browser_state(candidates=candidates[:4], prev_sig_set=None)

    assert "repeating the same action" in history_hint
    assert "high-risk" in risk_hint.lower()
    assert "[0]" in browser_state
    assert "<" in browser_state


def test_llm_decide_supports_valid_action_tool_and_retry(sample_html: str, monkeypatch) -> None:
    """Llm decide supports valid action tool and retry."""
    candidates = agent._extract_candidates(sample_html, max_candidates=20)

    responses = iter(
        [
            {"choices": [{"message": {"content": '{"action":"click","candidate_id":0}'}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
        ]
    )
    monkeypatch.setattr(agent, "openai_chat_completions", lambda **kwargs: next(responses))

    direct = agent._llm_decide(
        task_id="llm-1",
        task="Buy now",
        step_index=0,
        url="http://localhost/catalog?seed=7",
        candidates=candidates,
        page_summary="summary",
        dom_digest="digest",
        _html_snapshot=sample_html,
        history=[],
        opts={"page_ir_text": "ir"},
    )

    assert direct["action"] == "click"
    assert direct["_meta"]["llm_calls"] == 1

    tool_responses = iter(
        [
            {"choices": [{"message": {"content": '{"tool":"list_links","args":{"text_regex":"view"}}'}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
            {"choices": [{"message": {"content": '{"action":"navigate","url":"/products/camera?seed=7"}'}}], "usage": {"prompt_tokens": 2, "completion_tokens": 1}},
        ]
    )
    monkeypatch.setattr(agent, "openai_chat_completions", lambda **kwargs: next(tool_responses))

    with_tool = agent._llm_decide(
        task_id="llm-2",
        task="Open the camera page",
        step_index=1,
        url="http://localhost/catalog?seed=7",
        candidates=candidates,
        page_summary="summary",
        dom_digest="digest",
        _html_snapshot=sample_html,
        history=[],
        opts={"page_ir_text": "ir"},
    )

    assert with_tool["action"] == "navigate"
    assert with_tool["_meta"]["tool_calls"] == 1
    assert with_tool["_meta"]["llm_calls"] == 2

    retry_responses = iter(
        [
            {"choices": [{"message": {"content": '{"action":"type","candidate_id":0}'}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
            {"choices": [{"message": {"content": '{"action":"scroll_down"}'}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
        ]
    )
    monkeypatch.setattr(agent, "openai_chat_completions", lambda **kwargs: next(retry_responses))

    retried = agent._llm_decide(
        task_id="llm-3",
        task="Keep exploring",
        step_index=2,
        url="http://localhost/catalog?seed=7",
        candidates=candidates,
        page_summary="summary",
        dom_digest="digest",
        _html_snapshot=sample_html,
        history=[],
        opts={"page_ir_text": "ir"},
    )

    assert retried["action"] == "scroll_down"
    assert retried["_meta"]["llm_calls"] == 2


def test_act_response_dispatch_and_agent_act(monkeypatch) -> None:
    """Act response dispatch and agent act."""
    cand = agent._Candidate(
        selector=agent._sel_attr("href", "/checkout"),
        text="Checkout",
        tag="a",
        attrs={"href": "/checkout"},
    )

    def resp(actions, metrics=None):
        out = {"actions": actions}
        if metrics is not None:
            out["metrics"] = metrics
        return out

    scroll = agent._act_response_for_action(
        "navigate",
        None,
        None,
        {"url": "/catalog?seed=7"},
        [cand],
        "task-1",
        "http://localhost/catalog?seed=7",
        "http://localhost/catalog?seed=7",
        0,
        resp,
    )
    done = agent._act_response_for_action(
        "done",
        None,
        None,
        {},
        [cand],
        "task-1",
        "http://localhost/catalog?seed=7",
        "http://localhost/catalog?seed=7",
        0,
        resp,
    )
    click = agent._act_response_for_action(
        "click",
        0,
        None,
        {"reason": "go checkout"},
        [cand],
        "task-1",
        "http://localhost/catalog?seed=7",
        "http://localhost/catalog?seed=7",
        0,
        resp,
    )
    fallback = agent._act_response_for_action(
        "unknown",
        None,
        None,
        {},
        [cand],
        "task-1",
        "http://localhost/catalog?seed=7",
        "http://localhost/catalog?seed=7",
        0,
        resp,
    )

    assert scroll["actions"][0]["type"] == "ScrollAction"
    assert done["actions"] == []
    assert click["actions"][0]["type"] in {"ClickAction", "NavigateAction"}
    assert fallback["actions"][0]["type"] == "ClickAction"

    created = []

    class FakeBaseAction:
        @staticmethod
        def create_action(raw):
            created.append(raw)
            if raw.get("type") == "BrokenAction":
                raise RuntimeError("bad")
            return {"created": raw["type"]}

    monkeypatch.setattr(agent, "BaseAction", FakeBaseAction)

    operator = agent.ApifiedWebAgent()

    async def fake_act_from_payload(payload):
        return {
            "actions": [
                {"type": "ClickAction"},
                {"type": "BrokenAction"},
                "skip-me",
            ]
        }

    monkeypatch.setattr(operator, "act_from_payload", fake_act_from_payload)

    task = type("Task", (), {"id": "t1", "prompt": "Do it"})()
    out = asyncio.run(operator.act(task=task, snapshot_html="<html></html>", url="http://localhost", step_index=0))

    assert out == [{"created": "ClickAction"}]
    assert created[0]["type"] == "ClickAction"


def test_act_handlers_and_tool_error_branches(monkeypatch) -> None:
    """Act handlers and tool error branches."""
    def resp(actions, metrics=None):
        out = {"actions": actions}
        if metrics is not None:
            out["metrics"] = metrics
        return out

    wait = agent._act_handle_navigate({}, "http://localhost/page", "http://localhost/page", "task-x", resp)
    same_url_scroll = agent._act_handle_click_href(
        agent._sel_attr("href", "/page?seed=7"),
        0,
        "/page",
        "http://localhost/page?seed=7",
        "http://localhost/page?seed=7",
        "task-x",
        {},
        resp,
    )

    c = agent._Candidate(
        selector=agent._sel_custom("input"),
        text="",
        tag="input",
        attrs={"id": "email", "type": "email"},
    )
    type_action = agent._act_handle_type_or_select("type", c, 0, "foo@example.com", "task-x", "http://localhost/page", {}, resp)
    select_action = agent._act_handle_type_or_select("select", c, 0, "Option A", "task-x", "http://localhost/page", {}, resp)

    assert wait["actions"][0]["type"] == "WaitAction"
    assert same_url_scroll["actions"][0]["type"] == "ScrollAction"
    assert type_action["actions"][0]["type"] == "TypeAction"
    assert select_action["actions"][0]["type"] == "SelectDropDownOptionAction"

    with pytest.raises(agent.HTTPException):
        agent._act_handle_type_or_select("type", c, 0, "", "task-x", "http://localhost/page", {}, resp)

    assert agent._tool_css_select(html="<html></html>", selector="")["ok"] is False
    assert agent._tool_xpath_select(html="<html></html>", xpath="")["ok"] is False

    monkeypatch.setattr(agent, "BeautifulSoup", None)
    assert agent._tool_visible_text(html="<div>Hello <b>world</b></div>", max_chars=50)["text"] == "Hello world"


def test_run_tool_dispatchers_and_form_entry(sample_html: str) -> None:
    """Run tool dispatchers and form entry."""
    candidates = agent._extract_candidates(sample_html, max_candidates=20)
    forms_without_bs4 = None

    listed = agent._run_tool("list_candidates", {"max_n": 2}, html=sample_html, url="http://localhost/catalog?seed=7", candidates=candidates)
    cards = agent._run_tool("list_cards", {"max_cards": 2}, html=sample_html, url="http://localhost/catalog?seed=7", candidates=candidates)
    found = agent._run_tool("find_card", {"query": "camera", "max_cards": 1}, html=sample_html, url="http://localhost/catalog?seed=7", candidates=candidates)

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(agent, "BeautifulSoup", None)
        forms_without_bs4 = agent._tool_extract_forms(html=sample_html)
    finally:
        monkeypatch.undo()

    assert listed["ok"] is True and listed["count"] >= 2
    assert cards["ok"] is True and cards["count"] >= 1
    assert found["ok"] is True and found["count"] == 1
    assert forms_without_bs4["ok"] is False


def test_misc_agent_scoring_and_parsing_helpers(sample_html: str) -> None:
    """Misc agent scoring and parsing helpers."""
    parsed = type("Parsed", (), {"path": "", "query": "", "fragment": ""})()
    candidates = agent._extract_candidates(sample_html, max_candidates=20)

    assert agent._env_bool("NON_EXISTENT_FLAG", True) is True
    assert agent._normalize_url_parsed(parsed) == "http://localhost"
    assert agent._tokenize("Buy now! 2026") == {"buy", "now", "2026"}
    assert agent._score_tag("input") > agent._score_tag("a")
    assert agent._score_attrs({"id": "x", "required": ""}, "input") >= 6.0
    assert agent._score_candidate("buy", candidates[0]) >= 0.0
    assert agent._rank_candidates("buy", candidates, max_candidates=3)
    assert isinstance(agent._bucket_candidates(candidates, "http://localhost/catalog?seed=7"), tuple)
    assert agent._llm_json_recover("```json\n{\"action\":\"done\"}\n```") == {"action": "done"}

    with pytest.raises(ValueError):
        agent._parse_llm_json(123)  # type: ignore[arg-type]


def test_bs4_label_and_container_helpers(sample_html: str) -> None:
    """Bs4 label and container helpers."""
    soup = agent.BeautifulSoup(sample_html, "lxml")
    labelled = soup.find("input", {"id": "search-box"})
    button = soup.find("button", {"aria-label": "Add camera"})

    labelled_text = agent._extract_label_from_bs4(soup, labelled, agent._attrs_to_str_map(labelled.attrs))
    button_text = agent._extract_label_from_bs4(soup, button, agent._attrs_to_str_map(button.attrs))
    container_info = agent._container_node_info(button.find_parent("article"))
    candidates = agent._collect_container_candidates(button)
    best = agent._best_container_from_candidates(candidates)
    picked = agent._pick_context_container_bs4(button)
    chain = agent._container_chain_from_el(button)

    assert labelled_text == "Search catalog"
    assert button_text == "Add to cart"
    assert container_info is not None
    assert candidates
    assert best is not None
    assert picked is not None
    assert chain


def test_group_skip_context_and_select_helpers(sample_html: str) -> None:
    """Group skip context and select helpers."""
    soup = agent.BeautifulSoup(sample_html, "lxml")
    nav_link = soup.find("nav").find("a")
    form_input = soup.find("input", {"id": "search-box"})
    footer_html = "<footer><a href='/help'>Help</a></footer>"
    footer_soup = agent.BeautifulSoup(footer_html, "lxml")
    footer_link = footer_soup.find("a")
    select_el = soup.find("select")

    assert agent._group_for_el(nav_link) == "NAV"
    assert agent._group_for_el(form_input).startswith("FORM")
    assert agent._group_for_el(footer_link) == "FOOTER"
    assert agent._should_skip_candidate("input", {"type": "hidden"}) is True
    assert agent._should_skip_candidate("button", {"disabled": ""}) is True
    assert agent._should_skip_candidate("button", {"aria-disabled": "true"}) is True

    context, context_raw, title = agent._context_and_title_for_el(nav_link)
    opts = agent._select_options_from_el(select_el)
    primary, label = agent._select_primary_and_label(agent._sel_custom("select"), "Category", opts)

    assert isinstance(context, str)
    assert isinstance(context_raw, str)
    assert isinstance(title, str)
    assert opts == [("Books", "books"), ("Games", "games")]
    assert "option:has-text" in primary["value"]
    assert "options=[" in label


def test_make_candidate_deduplicates_and_dom_digest_helpers(sample_html: str) -> None:
    """Make candidate deduplicates and dom digest helpers."""
    soup = agent.BeautifulSoup(sample_html, "lxml")
    button = soup.find("button")
    seen = set()

    first = agent._make_candidate_from_el(button, soup, seen)
    second = agent._make_candidate_from_el(button, soup, seen)
    digest = agent._dom_digest(sample_html, limit=500)
    title = agent._dom_digest_title(soup)
    headings = agent._dom_digest_headings(soup)
    forms_bits = agent._dom_digest_forms_bits(soup)
    ctas = agent._dom_digest_ctas(soup)

    assert first is not None
    assert second is None
    assert title == "Demo Shop"
    assert "Featured products" in headings
    assert forms_bits
    assert ctas
    assert "TITLE: Demo Shop" in digest


def test_ir_cleanup_helpers_with_mixed_payloads() -> None:
    """Ir cleanup helpers with mixed payloads."""
    form_entry = agent._ir_cleaned_form_entry(
        {
            "id": "f1",
            "name": "search",
            "method": "POST",
            "controls": [{"tag": "input", "type": "text", "name": "q", "id": "search", "placeholder": "Search", "aria_label": "", "text": ""}],
        }
    )
    forms = agent._ir_cleaned_forms({"ok": True, "forms": ["bad", {"controls": []}, {"controls": [{"tag": "input", "name": "q"}], "method": "GET"}]}, 5)
    links = agent._ir_cleaned_links({"ok": True, "links": ["bad", {"text": "Go", "href": "/go", "context": "ctx"}, {"text": "", "href": "", "context": ""}]}, 5)
    cards = agent._ir_cleaned_cards({"ok": True, "cards": ["bad", {"card_facts": ["Price 1"], "actions": [{"tag": "a", "text": "Open", "href": "/x"}], "card_text": "Card"}]}, 5)

    assert form_entry is not None
    assert forms
    assert links == [{"text": "Go", "href": "/go", "ctx": "ctx"}]
    assert cards[0]["facts"] == ["Price 1"]


def test_summary_and_ir_cleanup_fallbacks(monkeypatch) -> None:
    """Summary and IR cleanup fallbacks."""
    monkeypatch.setattr(agent, "BeautifulSoup", None)

    broken_html = "<div>Hello <span>world"
    assert agent._extract_candidates("", max_candidates=5) == []
    assert "world" in agent._strip_tags_simple(broken_html)
    assert agent._summarize_html(broken_html)
    assert agent._dom_digest("", limit=10) == ""
    assert agent._ir_title_headings_from_soup("<html></html>") == ("", [])
    assert agent._ir_cleaned_form_entry({"controls": []}) is None
    assert agent._ir_cleaned_forms({"ok": False}, 2) == []
    assert agent._ir_cleaned_links({"ok": False}, 2) == []
    assert agent._ir_cleaned_cards({"ok": False}, 2) == []


def test_css_xpath_visible_text_error_paths(monkeypatch) -> None:
    """Css XPath visible text error paths."""
    class BrokenSoup:
        def __init__(self, html, parser):
            raise RuntimeError("parse boom")

    monkeypatch.setattr(agent, "BeautifulSoup", BrokenSoup)
    css = agent._tool_css_select(html="<html></html>", selector="a")
    forms = agent._tool_extract_forms(html="<html></html>")
    visible = agent._tool_visible_text(html="<html></html>")

    assert css["ok"] is False
    assert forms["ok"] is False
    assert visible["ok"] is False


def test_ir_and_card_helpers(sample_html: str) -> None:
    """Ir and card helpers."""
    candidates = agent._extract_candidates(sample_html, max_candidates=20)
    card = agent._tool_list_cards(candidates=candidates, max_cards=1)["cards"][0]
    blob = agent._card_blob_for_query(card)
    facts = agent._card_facts_from_key("Price 10\nNo digits here\nStock 5")
    meta = agent._act_decision_meta({"_meta": {"model": "gpt-5-mini", "llm_calls": 1}})

    assert "camera" in blob.lower() or "console" in blob.lower()
    assert facts == ["Price 10", "Stock 5"]
    assert meta["model"] == "gpt-5-mini"


def test_state_and_task_helpers_edge_cases() -> None:
    """State and task helpers edge cases."""
    assert agent._compute_state_delta(task_id="", url="u", page_summary="s", dom_digest="d", _html_snapshot="", candidates=[]) == ""
    assert agent._compute_ir_delta(task_id="", page_ir={}) == ""
    assert agent._act_effective_url_from_state(None, "http://localhost") == "http://localhost"
    assert agent._act_prev_sig_set_from_state(None) is None
    task = agent._task_from_payload({"task_id": "x", "url": "site.com", "task_prompt": "Go"})
    assert task.prompt == "Go"


def test_small_edge_branches(monkeypatch) -> None:
    """Small edge branches."""
    monkeypatch.setattr(agent, "urlsplit", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("boom")))
    assert agent._normalize_demo_url("bad:url") == "http://localhost/"
    assert agent._selector_repr({"type": "xpathSelector", "value": "//a"}) == "{'type': 'xpathSelector', 'value': '//a'}"

    c_text = agent._Candidate(selector={"type": "other"}, text="Buy", tag="button", attrs={}, text_selector=agent._sel_text("Buy"))
    c_sel = agent._Candidate(selector={"type": "other"}, text="", tag="div", attrs={})
    c_type = agent._Candidate(selector=agent._sel_attr("name", "email"), text="", tag="input", attrs={})

    assert c_text.click_selector()["type"] == "attributeValueSelector"
    assert c_sel.click_selector() == {"type": "other"}
    assert c_type.type_selector()["attribute"] == "name"


def test_selector_builder_and_label_resolution_branches() -> None:
    """Selector builder and label resolution branches."""
    assert agent._build_selector("button", {"data-testid": "cta"}, text="")["attribute"] == "data-testid"
    assert agent._build_selector("a", {"href": "/next"}, text="")["attribute"] == "href"
    assert agent._build_selector("input", {"aria-label": "Email"}, text="")["attribute"] == "aria-label"
    assert agent._build_selector("input", {"name": "email"}, text="")["attribute"] == "name"
    assert agent._build_selector("input", {"placeholder": "Search"}, text="")["attribute"] == "placeholder"
    assert agent._build_selector("input", {"title": "Title"}, text="")["attribute"] == "title"
    assert agent._build_selector("button", {}, text="Save")["type"] == "tagContainsSelector"

    html = """
    <html><body>
      <span id="user-label">User Name</span>
      <input aria-labelledby="user-label" />
      <label><input type="checkbox" /> Accept terms</label>
    </body></html>
    """
    soup = agent.BeautifulSoup(html, "lxml")
    labelled = soup.find("input", {"aria-labelledby": "user-label"})
    wrapped = soup.find("input", {"type": "checkbox"})

    assert agent._extract_label_from_bs4(soup, labelled, agent._attrs_to_str_map(labelled.attrs)) == "User Name"
    assert agent._extract_label_from_bs4(soup, wrapped, agent._attrs_to_str_map(wrapped.attrs)) == "Accept terms"


def test_candidate_methods_and_action_handlers_cover_real_paths() -> None:
    """Candidate methods and action handlers cover real paths."""
    def resp(actions, metrics=None):
        out = {"actions": actions}
        if metrics is not None:
            out["metrics"] = metrics
        return out

    text_candidate = agent._Candidate(selector={"type": "other"}, text="Buy now", tag="button", attrs={}, text_selector=agent._sel_text("Buy now"))
    href_candidate = agent._Candidate(selector=agent._sel_attr("href", "/details"), text="Details", tag="a", attrs={"href": "/details"})

    assert text_candidate.click_selector()["type"] == "attributeValueSelector"
    assert href_candidate.click_selector()["attribute"] == "href"
    assert agent._Candidate(selector={"type": "other"}, text="", tag="textarea", attrs={}).type_selector()["value"] == "textarea"

    nav = agent._act_handle_navigate(
        {"url": "/checkout?seed=7", "_meta": {"model": "gpt-5-mini"}},
        "http://localhost/catalog?seed=7",
        "http://localhost/catalog?seed=7",
        "task-nav",
        resp,
    )
    click_same = agent._act_handle_click_href(
        agent._sel_attr("href", "/catalog?seed=7"),
        0,
        "/catalog?seed=7",
        "http://localhost/catalog?seed=7",
        "http://localhost/catalog?seed=7",
        "task-click",
        {"_meta": {"model": "gpt-5-mini"}},
        resp,
    )
    click_direct = agent._act_handle_click_href(
        agent._sel_attr("href", "/details"),
        0,
        "/details",
        "http://localhost/catalog",
        "http://localhost/catalog",
        "task-click2",
        {"_meta": {"model": "gpt-5-mini"}},
        resp,
    )

    assert nav["actions"][0]["type"] == "NavigateAction"
    assert click_same["actions"][0]["type"] == "ClickAction"
    assert click_direct["actions"][0]["type"] == "ClickAction"


def test_operator_act_edge_paths(monkeypatch) -> None:
    """Operator act edge paths."""
    class NoCreateAction:
        pass

    monkeypatch.setattr(agent, "BaseAction", NoCreateAction)
    operator = agent.ApifiedWebAgent()

    async def fake_empty_actions(payload):
        return {"actions": [{"type": "ClickAction"}]}

    monkeypatch.setattr(operator, "act_from_payload", fake_empty_actions)
    task = type("Task", (), {"id": "t1", "prompt": "Do it"})()
    out = asyncio.run(operator.act(task=task, snapshot_html="<html></html>", url="http://localhost", step_index=0))

    assert out == []


def test_act_from_payload_error_and_empty_candidate_check(monkeypatch) -> None:
    """Act from payload error and empty candidate check."""
    operator = agent.ApifiedWebAgent()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    error_resp = asyncio.run(
        operator.act_from_payload(
            {
                "task_id": "task-error",
                "prompt": "buy",
                "url": "http://demo.example/page",
                "snapshot_html": "<html><body><button>Buy</button></body></html>",
                "step_index": 0,
            }
        )
    )

    check_wait = asyncio.run(
        operator.act_from_payload(
            {
                "task_id": "check",
                "prompt": "noop",
                "url": "http://demo.example/page",
                "snapshot_html": "",
                "step_index": 0,
            }
        )
    )

    assert error_resp["actions"] == [{"type": "WaitAction", "time_seconds": 1.0}]
    assert check_wait["actions"] == [{"type": "WaitAction", "time_seconds": 0.1}]
