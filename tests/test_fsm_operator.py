from __future__ import annotations

import json
from typing import Any, Dict

from fsm_operator import (
    FSMOperator,
    MAX_INTERNAL_META_STEPS,
    AgentFormProgress,
    AgentState,
    Candidate,
    CandidateExtractor,
    CandidateRanker,
    FlagDetector,
    ObsBuilder,
)


def _dummy_llm_invalid(**_: Any) -> Dict[str, Any]:
    return {"choices": [{"message": {"content": "not-json"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


def _dummy_llm_meta_loop(**_: Any) -> Dict[str, Any]:
    return {
        "choices": [{"message": {"content": '{"type":"meta","meta_tool":{"name":"META.REPLAN","arguments":{}}}'}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _dummy_llm_final(**_: Any) -> Dict[str, Any]:
    return {
        "choices": [{"message": {"content": '{"type":"final","done":true,"content":"Treasury value found: T 399,29"}'}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
        "model": "gpt-4o-mini",
    }


def _dummy_llm_type_login_username(**_: Any) -> Dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "content": (
                        '{"type":"browser","tool_call":{"name":"browser.type","arguments":{'
                        '"selector":{"type":"attributeValueSelector","attribute":"id","value":"login-username","case_sensitive":false},'
                        '"text":"<username>"}}}'
                    )
                }
            }
        ],
        "usage": {"prompt_tokens": 6, "completion_tokens": 5, "total_tokens": 11},
        "model": "gpt-5.2",
    }


def _dummy_vision_llm_apply(**_: Any) -> Dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "answer": "The filter panel is visible and the Apply button is the best next target.",
                            "element_ids": ["el_apply"],
                            "signals": ["filter panel visible", "apply button visible"],
                            "confidence": "high",
                        }
                    )
                }
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        "model": "gpt-4o-mini",
    }


def _base_payload() -> Dict[str, Any]:
    return {
        "task_id": "fsm-test",
        "prompt": "Go to example.com and find pricing information",
        "url": "https://example.com",
        "snapshot_html": "<html><body><h1>Pricing</h1><a href='/pricing'>Pricing</a></body></html>",
        "step_index": 0,
        "history": [],
        "allowed_tools": [
            {"name": "browser.navigate"},
            {"name": "browser.click"},
            {"name": "browser.type"},
            {"name": "browser.scroll"},
            {"name": "browser.back"},
            {"name": "browser.wait"},
        ],
    }


def test_state_out_roundtrip_without_process_local_state() -> None:
    engine1 = FSMOperator(llm_call=_dummy_llm_invalid)
    first = engine1.run(payload=_base_payload())
    st = first.get("state_out")
    assert isinstance(st, dict)
    assert st.get("visited", {}).get("urls") == ["https://example.com"]

    # New instance: same decision context must still continue from state_in.
    engine2 = FSMOperator(llm_call=_dummy_llm_invalid)
    payload = dict(_base_payload())
    payload["step_index"] = 1
    payload["state_in"] = st
    second = engine2.run(payload=payload)
    st2 = second.get("state_out")
    assert isinstance(st2, dict)
    assert "https://example.com" in (st2.get("visited", {}).get("urls") or [])


def test_fsm_emits_at_most_one_browser_action_per_step() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    out = engine.run(payload=_base_payload())
    actions = out.get("actions")
    assert isinstance(actions, list)
    assert len(actions) <= 1


def test_meta_tool_loop_is_capped(monkeypatch: Any) -> None:
    monkeypatch.setenv("FSM_DIRECT_LOOP", "0")
    monkeypatch.setenv("FSM_ALLOW_CONTROL_META_TOOLS", "1")
    engine = FSMOperator(llm_call=_dummy_llm_meta_loop)
    payload = _base_payload()
    payload["allowed_tools"] = list(payload["allowed_tools"]) + [{"name": "META.REPLAN"}]
    out = engine.run(payload=payload)
    st = out.get("state_out") or {}
    counters = st.get("counters") if isinstance(st.get("counters"), dict) else {}
    assert int(counters.get("meta_steps_used") or 0) == MAX_INTERNAL_META_STEPS


def test_stuck_recovery_triggers_with_loop_signals(monkeypatch: Any) -> None:
    monkeypatch.setenv("FSM_DIRECT_LOOP", "0")
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    payload = _base_payload()
    payload["state_in"] = {
        "mode": "NAV",
        "counters": {"stall_count": 3, "repeat_action_count": 2, "meta_steps_used": 0},
        "last_action_element_id": "el_repeat",
        "blocklist": {"element_ids": [], "until_step": 0},
    }
    out = engine.run(payload=payload)
    actions = out.get("actions") if isinstance(out.get("actions"), list) else []
    assert len(actions) == 1
    assert actions[0].get("type") in {"NavigateAction", "WaitAction", "ScrollAction"}
    st = out.get("state_out") or {}
    blocked = st.get("blocklist", {}).get("element_ids") if isinstance(st.get("blocklist"), dict) else []
    assert isinstance(blocked, list)


def test_done_and_content_emitted_without_report_result_action() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_final)
    payload = _base_payload()
    payload["step_index"] = 2
    payload["state_in"] = {
        "mode": "REPORT",
        "memory": {"facts": ["Treasury value found: T 399,29"], "checkpoints": []},
    }
    payload["include_reasoning"] = True
    out = engine.run(payload=payload)
    assert out.get("done") is True
    assert isinstance(out.get("content"), str) and out.get("content")
    assert out.get("actions") == []


def test_early_final_is_blocked_on_first_step_without_facts(monkeypatch: Any) -> None:
    monkeypatch.setenv("FSM_DIRECT_LOOP", "0")
    engine = FSMOperator(llm_call=_dummy_llm_final)
    payload = _base_payload()
    payload["step_index"] = 0
    payload["history"] = []
    out = engine.run(payload=payload)
    assert out.get("done") is False
    actions = out.get("actions")
    assert isinstance(actions, list)
    assert len(actions) <= 1


def test_allowed_tools_parses_function_definitions_shape() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    payload = _base_payload()
    payload["allowed_tools"] = [
        {"type": "function", "function": {"name": "navigate"}},
        {"type": "function", "function": {"name": "wait"}},
    ]
    payload["state_in"] = {
        "mode": "NAV",
        "counters": {"stall_count": 3, "repeat_action_count": 2, "meta_steps_used": 0},
    }
    out = engine.run(payload=payload)
    actions = out.get("actions") if isinstance(out.get("actions"), list) else []
    assert len(actions) == 1
    # STUCK ladder should recognize browser.navigate/browser.wait from function defs.
    assert actions[0].get("type") in {"NavigateAction", "WaitAction"}


def test_browser_scroll_accepts_page_amount_keyword() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    action = engine._browser_action_from_tool_call(
        tool_call={"name": "browser.scroll", "arguments": {"direction": "down", "amount": "page"}},
        prompt="Scroll down.",
        ranked_candidates=[],
        state=AgentState(),
        current_url="https://example.com",
        allowed={"browser.scroll"},
    )
    assert action == {
        "type": "ScrollAction",
        "direction": "down",
        "up": False,
        "down": True,
        "amount": 650,
    }


def test_browser_scroll_invalid_amount_falls_back_to_default() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    action = engine._browser_action_from_tool_call(
        tool_call={"name": "browser.scroll", "arguments": {"direction": "up", "amount": "nonsense"}},
        prompt="Scroll up.",
        ranked_candidates=[],
        state=AgentState(),
        current_url="https://example.com",
        allowed={"browser.scroll"},
    )
    assert action == {
        "type": "ScrollAction",
        "direction": "up",
        "up": True,
        "down": False,
        "amount": 650,
    }


def test_prompt_domain_does_not_force_external_navigation() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    payload = _base_payload()
    payload["prompt"] = "Go to gmail.com and login"
    payload["url"] = "http://84.247.180.192:8000/?seed=123"
    payload["snapshot_html"] = "<html><body><h1>Home</h1></body></html>"
    out = engine.run(payload=payload)
    actions = out.get("actions") if isinstance(out.get("actions"), list) else []
    assert len(actions) <= 1
    if actions:
        action = actions[0]
        if action.get("type") == "NavigateAction":
            assert "gmail.com" not in str(action.get("url") or "")


def test_obs_builder_compacts_history_and_provides_tagged_input() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    payload = _base_payload()
    long_history = []
    for i in range(35):
        long_history.append(
            {
                "step": i,
                "url": f"https://example.com/page/{i}",
                "action": {"type": "ScrollAction" if i % 2 == 0 else "ClickAction"},
                "exec_ok": bool(i % 7 != 0),
                "error": "timed out" if i % 11 == 0 else "",
            }
        )
    payload["history"] = long_history
    out = engine.run(payload=payload)
    st = out.get("state_out") or {}
    mem = st.get("memory") if isinstance(st.get("memory"), dict) else {}
    summary = str(mem.get("history_summary") or "")
    assert summary
    assert "history_total=" in summary


def test_meta_search_text_and_find_elements_update_state() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    state = AgentState()
    candidates = [
        Candidate(
            id="el_search",
            role="input",
            type="input",
            text="Search films",
            href="",
            context="Search bar",
            selector={"type": "attributeValueSelector", "attribute": "id", "value": "search", "case_sensitive": False},
            dom_path="html/body/input[1]",
            bbox=None,
        )
    ]
    text_ir = {"visible_text": "Current portfolio value is T 399,29 and APY 2.01%."}
    r1 = engine.meta.execute(
        task_id="meta-search",
        tool_name="META.SEARCH_TEXT",
        args={"query": "portfolio value"},
        state=state,
        prompt="find value",
        text_ir=text_ir,
        candidates=candidates,
        url="https://example.com",
    )
    r2 = engine.meta.execute(
        task_id="meta-find",
        tool_name="META.FIND_ELEMENTS",
        args={"role": "input", "text": "search", "limit": 5},
        state=state,
        prompt="find input",
        text_ir=text_ir,
        candidates=candidates,
        url="https://example.com",
    )
    assert bool(r1.get("ok")) is True
    assert bool(r2.get("ok")) is True
    assert any("portfolio value" in f.lower() for f in state.memory.facts)
    assert "el_search" in state.frontier.pending_elements


def test_candidate_extractor_uses_dom_href_for_link_selector() -> None:
    extractor = CandidateExtractor()
    html = """
    <html><body>
      <a href="/login?seed=123">Login</a>
    </body></html>
    """
    candidates = extractor.extract(snapshot_html=html, url="http://84.247.180.192:8000/?seed=456")
    assert candidates
    login = next((c for c in candidates if c.role == "link"), None)
    assert login is not None
    assert login.selector.get("attribute") == "href"
    assert login.selector.get("value") == "/login?seed=123"
    assert str(login.href).endswith("/login?seed=123")


def test_candidate_extractor_uses_dom_path_selector_for_duplicated_input_ids() -> None:
    extractor = CandidateExtractor()
    html = """
    <html><body>
      <label for="entry-field">Title</label><input id="entry-field" />
      <label for="entry-field">Rating</label><input id="entry-field" />
    </body></html>
    """
    candidates = extractor.extract(snapshot_html=html, url="https://example.com")
    inputs = [c for c in candidates if c.role == "input"]
    assert len(inputs) >= 2
    assert all(c.selector.get("type") == "xpathSelector" for c in inputs)
    assert all(str(c.selector.get("value") or "").startswith("html") for c in inputs)


def test_auth_flow_is_not_forced_by_pre_actions() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    payload = {
        "task_id": "auth-test",
        "prompt": "Authenticate with username equals '<username>' and password equals '<password>'",
        "url": "http://84.247.180.192:8000/login?seed=111",
        "snapshot_html": """
            <html><body>
              <form>
                <input id="username" name="username" />
                <input id="password" name="password" type="password" />
                <button>Login</button>
              </form>
            </body></html>
        """,
        "step_index": 0,
        "history": [],
        "state_in": {"mode": "NAV"},
        "allowed_tools": [{"name": "browser.type"}, {"name": "browser.click"}],
    }
    out = engine.run(payload=payload)
    actions = out.get("actions") if isinstance(out.get("actions"), list) else []
    assert actions and actions[0].get("type") in {"ClickAction", "TypeAction"}
    reasoning = str(out.get("reasoning") or "")
    assert "PRE:type_identifier" not in reasoning
    assert "PRE:type_password" not in reasoning


def test_typed_values_from_history_supports_string_action_shape() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    vals = engine._typed_values_from_history(
        [
            {"step": 1, "action": "TypeAction", "text": "user1"},
            {"step": 2, "action": {"type": "TypeAction", "raw": {"text": "p4ss"}}},
        ]
    )
    assert "user1" in vals
    assert "p4ss" in vals


def test_coerce_type_text_replaces_low_quality_model_value() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    state = AgentState()
    candidate = Candidate(
        id="el_pass",
        role="input",
        type="input",
        text="Password",
        href="",
        context="Register form",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "password-input", "case_sensitive": False},
        dom_path="html/body/form/input[1]",
        field_kind="password",
        input_type="password",
    )
    coerced = engine._coerce_type_text(
        text="''",
        prompt="Please register an account using username equals '', email equals '' which ends with '@gmail.com', and password equals ''.",
        candidate=candidate,
        state=state,
    )
    assert coerced == "<signup_password>"


def test_candidate_with_low_quality_typed_value_is_not_considered_filled() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    state = AgentState(
        form_progress=AgentFormProgress(
            typed_candidate_ids=["el_pass"],
            typed_values_by_candidate={"el_pass": "''"},
        )
    )
    candidate = Candidate(
        id="el_pass",
        role="input",
        type="input",
        text="Password",
        href="",
        context="Register form",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "password-input", "case_sensitive": False},
        dom_path="html/body/form/input[1]",
        field_kind="password",
        input_type="password",
    )
    assert not engine._candidate_has_usable_typed_value(candidate=candidate, history=[], state=state)


def test_coerce_type_text_preserves_better_remembered_value() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    candidate = Candidate(
        id="el_user",
        role="input",
        type="input",
        text="Username",
        href="",
        context="Register form",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "register-username-field", "case_sensitive": False},
        dom_path="html/body/form/input[1]",
        field_kind="username",
    )
    state = AgentState(
        form_progress=AgentFormProgress(
            typed_values_by_candidate={"el_user": "<signup_username>"},
            typed_candidate_ids=["el_user"],
        )
    )
    coerced = engine._coerce_type_text(
        text="''",
        prompt="Please register an account using username equals '', email equals '' which ends with '@gmail.com', and password equals ''.",
        candidate=candidate,
        state=state,
    )
    assert coerced == "<signup_username>"


def test_coerce_type_text_prefers_prompt_anchored_identity_value_over_model_guess() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    state = AgentState()
    username = Candidate(
        id="el_user",
        role="input",
        type="input",
        text="Username",
        href="",
        context="Register form",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "register-username-field", "case_sensitive": False},
        dom_path="html/body/form/input[1]",
        field_kind="username",
    )
    password = Candidate(
        id="el_pass",
        role="input",
        type="input",
        text="Password",
        href="",
        context="Register form",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "password-input", "case_sensitive": False},
        dom_path="html/body/form/input[2]",
        field_kind="password",
        input_type="password",
    )
    prompt = "Please register an account using username equals '', email equals '' which ends with '@gmail.com', and password equals ''."
    assert engine._coerce_type_text(text="user7", prompt=prompt, candidate=username, state=state) == "<signup_username>"
    assert engine._coerce_type_text(text="Passw0rd!", prompt=prompt, candidate=password, state=state) == "<signup_password>"


def test_non_auth_prompt_on_login_page_does_not_trigger_auth_pre_actions() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    payload = {
        "task_id": "non-auth-login-page",
        "prompt": "Delete a film where duration is not 142 minutes.",
        "url": "http://84.247.180.192:8000/login?seed=111",
        "snapshot_html": """
            <html><body>
              <form>
                <input id="login-username" />
                <input id="login-password" type="password" />
                <button id="login-btn">Login</button>
              </form>
            </body></html>
        """,
        "step_index": 1,
        "history": [],
        "state_in": {"mode": "NAV"},
        "allowed_tools": [{"name": "browser.click"}, {"name": "browser.type"}, {"name": "browser.wait"}],
    }
    out = engine.run(payload=payload)
    reasoning = str(out.get("reasoning") or "")
    assert "PRE:submit_login" not in reasoning
    assert "PRE:type_identifier" not in reasoning


def test_extract_credentials_ignores_empty_quoted_values() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    ids, pwds = engine._extract_credentials(
        "Please register using username equals '', email equals '' and password equals ''."
    )
    assert ids == []
    assert pwds == []


def test_registration_flow_is_not_forced_by_pre_actions() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    base = {
        "task_id": "reg-flow",
        "prompt": "Please register using username equals '', email equals '' which ends with '@gmail.com', and password equals ''.",
        "url": "http://84.247.180.192:8000/register?seed=1",
        "snapshot_html": """
            <html><body>
              <form>
                <input id="register-username" name="username" />
                <input id="register-email" name="email" />
                <input id="register-password" name="password" type="password" />
                <button id="register-button">Register</button>
              </form>
            </body></html>
        """,
        "history": [],
        "allowed_tools": [{"name": "browser.type"}, {"name": "browser.click"}],
    }
    out = engine.run(payload={**base, "step_index": 0, "include_reasoning": True})
    actions = out.get("actions") if isinstance(out.get("actions"), list) else []
    assert actions and actions[0].get("type") in {"ClickAction", "TypeAction"}
    reasoning = str(out.get("reasoning") or "")
    assert "PRE:type_signup_username" not in reasoning
    assert "PRE:type_signup_email" not in reasoning
    assert "PRE:type_signup_password" not in reasoning


def test_infer_input_text_generates_missing_registration_values_generically() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    prompt = "Please register using username equals '', email equals '' which ends with '@gmail.com', and password equals ''."
    username = Candidate(
        id="u1",
        role="input",
        type="input",
        text="",
        href="",
        context="Registration form",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "register-username", "case_sensitive": False},
        dom_path="html/body/form/input[1]",
        field_hint="Username",
        field_kind="username",
    )
    email = Candidate(
        id="e1",
        role="input",
        type="input",
        text="",
        href="",
        context="Registration form",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "register-email", "case_sensitive": False},
        dom_path="html/body/form/input[2]",
        field_hint="Email",
        field_kind="email",
        input_type="email",
    )
    password = Candidate(
        id="p1",
        role="input",
        type="input",
        text="",
        href="",
        context="Registration form",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "register-password", "case_sensitive": False},
        dom_path="html/body/form/input[3]",
        field_hint="Password",
        field_kind="password",
        input_type="password",
    )
    username_value = engine._infer_input_text(prompt=prompt, candidate=username)
    email_value = engine._infer_input_text(prompt=prompt, candidate=email)
    password_value = engine._infer_input_text(prompt=prompt, candidate=password)
    assert username_value == "<signup_username>"
    assert email_value == "<signup_email>"
    assert password_value == "<signup_password>"


def test_ranker_prefers_navigation_link_over_search_input_when_required_fields_missing() -> None:
    ranker = CandidateRanker()
    state = AgentState()
    candidates = [
        Candidate(
            id="search-input",
            role="input",
            type="input",
            text="Search films",
            href="",
            context="Hero search",
            selector={"type": "attributeValueSelector", "attribute": "id", "value": "text-input", "case_sensitive": False},
            dom_path="html/body/main/input[1]",
            field_hint="Search",
            field_kind="search",
            group_id="g-search",
            group_label="Hero search",
        ),
        Candidate(
            id="login-link",
            role="link",
            type="a",
            text="Login",
            href="https://example.com/login",
            context="Header navigation",
            selector={"type": "attributeValueSelector", "attribute": "href", "value": "/login", "case_sensitive": False},
            dom_path="html/body/header/nav/a[1]",
            field_kind="link",
            group_id="g-nav",
            group_label="Header navigation",
        ),
    ]
    ranked = ranker.rank(
        task="Authenticate with username equals '<username>' and password equals '<password>'",
        mode="NAV",
        flags={"search_box": True},
        candidates=candidates,
        state=state,
        current_url="https://example.com",
        top_k=10,
    )
    assert ranked[0].id == "login-link"


def test_browser_select_rejects_non_select_candidate_and_falls_back_to_real_select() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    state = AgentState()
    ranked = [
        Candidate(
            id="sort-button",
            role="button",
            type="button",
            text="Year desc",
            href="",
            context="Sort controls",
            selector={"type": "xpathSelector", "value": "//button[contains(normalize-space(.), 'year desc')]"},
            dom_path="html/body/div/button[1]",
            field_kind="button",
            group_id="g-sort",
            group_label="Sort controls",
        ),
        Candidate(
            id="year-select",
            role="select",
            type="select",
            text="All years 2023 2022 2004 2003 2002",
            href="",
            context="Filter panel",
            selector={"type": "attributeValueSelector", "attribute": "id", "value": "year-filter", "case_sensitive": False},
            dom_path="html/body/div/select[1]",
            field_kind="year",
            group_id="g-filter",
            group_label="Filter panel",
        ),
    ]
    action = engine._browser_action_from_tool_call(
        tool_call={
            "name": "browser.select",
            "arguments": {
                "selector": {"type": "xpathSelector", "value": "//button[contains(normalize-space(.), 'year desc')]"},
                "_element_id": "sort-button",
                "value": "2003",
            },
        },
        ranked_candidates=ranked,
        state=state,
        prompt="Filter movies by year equals 2003",
        allowed={"browser.select"},
        current_url="https://example.com/search",
    )
    assert action is not None
    assert action["type"] == "SelectDropDownOptionAction"
    assert action["selector"] == ranked[1].selector
    assert action["_element_id"] == "year-select"


def test_browser_type_blank_text_is_inferred_from_candidate_and_prompt() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    state = AgentState()
    ranked = [
        Candidate(
            id="register-username",
            role="input",
            type="input",
            text="Username",
            href="",
            context="Register form",
            selector={"type": "attributeValueSelector", "attribute": "id", "value": "register-username-field", "case_sensitive": False},
            dom_path="html/body/form/input[1]",
            field_kind="username",
            group_id="g-register",
            group_label="Register form",
        )
    ]
    action = engine._browser_action_from_tool_call(
        tool_call={
            "name": "browser.type",
            "arguments": {
                "selector": {"type": "attributeValueSelector", "attribute": "id", "value": "register-username-field", "case_sensitive": False},
                "text": "",
            },
        },
        ranked_candidates=ranked,
        state=state,
        prompt="Please register an account using username equals '', email equals '' which ends with '@gmail.com', and password equals ''.",
        allowed={"browser.type"},
        current_url="https://example.com/register",
    )
    assert isinstance(action, dict)
    assert action.get("type") == "TypeAction"
    assert action.get("text") == "<signup_username>"


def test_registration_intent_opens_signup_page_before_submit() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    out = engine.run(
        payload={
            "task_id": "reg-open-page",
            "prompt": "Please register an account.",
            "url": "https://example.com",
            "snapshot_html": "<html><body><a href='/register'>Register</a></body></html>",
            "step_index": 0,
            "history": [],
            "include_reasoning": True,
            "allowed_tools": [{"name": "browser.click"}],
        }
    )
    actions = out.get("actions") if isinstance(out.get("actions"), list) else []
    assert actions and actions[0].get("type") == "ClickAction"


def test_browser_action_mapping_attaches_candidate_element_id_from_selector() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    state = AgentState()
    selector = {"type": "attributeValueSelector", "attribute": "id", "value": "message-button", "case_sensitive": False}
    ranked = [
        Candidate(
            id="el_message",
            role="button",
            type="button",
            text="Send",
            href="",
            context="contact form",
            selector=selector,
            dom_path="html/body/form/button[1]",
            bbox=None,
        )
    ]
    action = engine._browser_action_from_tool_call(
        tool_call={"name": "browser.click", "arguments": {"selector": selector}},
        ranked_candidates=ranked,
        state=state,
        prompt="Send message",
        allowed={"browser.click"},
    )
    assert isinstance(action, dict)
    assert action.get("type") == "ClickAction"
    assert action.get("_element_id") == "el_message"


def test_browser_action_mapping_accepts_underscore_element_id_alias() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    state = AgentState()
    selector = {"type": "attributeValueSelector", "attribute": "id", "value": "entry-field", "case_sensitive": False}
    ranked = [
        Candidate(
            id="el_rating",
            role="input",
            type="input",
            text="Rating",
            href="",
            context="Rating",
            selector=selector,
            dom_path="html/body/form/input[2]",
            bbox=None,
        )
    ]
    action = engine._browser_action_from_tool_call(
        tool_call={"name": "browser.type", "arguments": {"selector": selector, "_element_id": "el_rating", "text": "4.1"}},
        ranked_candidates=ranked,
        state=state,
        prompt="Set rating to 4.1",
        allowed={"browser.type"},
    )
    assert isinstance(action, dict)
    assert action.get("type") == "TypeAction"
    assert action.get("_element_id") == "el_rating"
    assert action.get("text") == "4.1"


def test_repeated_same_element_is_added_to_blocklist() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    payload = {
        "task_id": "repeat-block",
        "prompt": "Click submit",
        "url": "https://example.com",
        "snapshot_html": "<html><body><button id='submit-btn'>Submit</button></body></html>",
        "step_index": 0,
        "history": [],
        "allowed_tools": [{"name": "browser.click"}],
    }
    out1 = engine.run(payload=payload)
    payload2 = dict(payload)
    payload2["step_index"] = 1
    payload2["state_in"] = out1.get("state_out")
    out2 = engine.run(payload=payload2)
    payload3 = dict(payload)
    payload3["step_index"] = 2
    payload3["state_in"] = out2.get("state_out")
    out3 = engine.run(payload=payload3)
    st3 = out3.get("state_out") or {}
    blocklist = st3.get("blocklist") if isinstance(st3.get("blocklist"), dict) else {}
    ids = blocklist.get("element_ids") if isinstance(blocklist.get("element_ids"), list) else []
    assert isinstance(ids, list)
    assert ids


def test_repeated_click_on_input_is_promoted_to_type_action() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    cand = Candidate(
        id="el_name",
        role="input",
        type="input",
        text="Name",
        href="",
        context="Contact form name",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "contact-name", "case_sensitive": False},
        dom_path="html/body/form/input[1]",
        bbox=None,
    )
    state = AgentState()
    state.counters.repeat_action_count = 1
    promoted = engine._promote_click_input_to_type(
        action={"type": "ClickAction", "selector": cand.selector, "_element_id": "el_name"},
        prompt="Submit contact form with a name that is NOT 'TestUser'.",
        ranked_candidates=[cand],
        state=state,
    )
    assert isinstance(promoted, dict)
    assert promoted.get("type") == "TypeAction"
    assert promoted.get("text") in {"AutoUser", "ExampleUser", "Auto User"}


def test_submit_click_is_guarded_with_missing_form_inputs() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    state = AgentState()
    submit = Candidate(
        id="el_submit",
        role="button",
        type="button",
        text="Sign Up",
        href="",
        context="Register form",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "signup-button", "case_sensitive": False},
        dom_path="html/body/form/button[1]",
        bbox=None,
    )
    username = Candidate(
        id="el_user",
        role="input",
        type="input",
        text="Username",
        href="",
        context="Register form username",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "register-username", "case_sensitive": False},
        dom_path="html/body/form/input[1]",
        bbox=None,
    )
    guarded = engine._guard_submit_without_inputs(
        action={"type": "ClickAction", "selector": submit.selector, "_element_id": "el_submit"},
        prompt="Create account to continue.",
        history=[],
        ranked_candidates=[submit, username],
        state=state,
    )
    assert isinstance(guarded, dict)
    assert guarded.get("type") == "TypeAction"
    assert guarded.get("_element_id") == "el_user"


def test_group_guard_finishes_missing_required_input_before_select() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    state = AgentState()
    state.form_progress.active_group_id = "g-register"
    state.form_progress.active_group_candidate_ids = ["el_user", "el_email", "el_pass", "el_confirm", "el_year"]
    state.form_progress.typed_candidate_ids = ["el_user", "el_pass", "el_confirm"]
    username = Candidate(
        id="el_user",
        role="input",
        type="input",
        text="Username",
        href="",
        context="Register form",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "register-username", "case_sensitive": False},
        dom_path="html/body/form/input[1]",
        field_kind="username",
        group_id="g-register",
        group_label="Register form",
    )
    email = Candidate(
        id="el_email",
        role="input",
        type="input",
        text="Email",
        href="",
        context="Register form",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "register-email", "case_sensitive": False},
        dom_path="html/body/form/input[2]",
        field_kind="email",
        input_type="email",
        group_id="g-register",
        group_label="Register form",
    )
    password = Candidate(
        id="el_pass",
        role="input",
        type="input",
        text="Password",
        href="",
        context="Register form",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "register-password", "case_sensitive": False},
        dom_path="html/body/form/input[3]",
        field_kind="password",
        input_type="password",
        group_id="g-register",
        group_label="Register form",
    )
    confirm = Candidate(
        id="el_confirm",
        role="input",
        type="input",
        text="Confirm password",
        href="",
        context="Register form",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "register-confirm-password", "case_sensitive": False},
        dom_path="html/body/form/input[4]",
        field_kind="confirm_password",
        input_type="password",
        group_id="g-register",
        group_label="Register form",
    )
    year_select = Candidate(
        id="el_year",
        role="select",
        type="select",
        text="1917 2020 2024",
        href="",
        context="Register form",
        selector={"type": "xpathSelector", "value": "html/body/form/select[1]"},
        dom_path="html/body/form/select[1]",
        field_kind="year",
        group_id="g-register",
        group_label="Register form",
    )
    guarded = engine._guard_missing_group_inputs(
        action={"type": "SelectDropDownOptionAction", "selector": year_select.selector, "text": "1917", "_element_id": "el_year"},
        prompt="Please register an account using username equals '', email equals '' which ends with '@gmail.com', and password equals ''.",
        history=[],
        ranked_candidates=[username, email, password, confirm, year_select],
        state=state,
    )
    assert isinstance(guarded, dict)
    assert guarded.get("type") == "TypeAction"
    assert guarded.get("_element_id") == "el_email"
    assert guarded.get("text") == "<signup_email>"


def test_group_guard_prefers_submit_over_optional_select_after_required_fields() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    state = AgentState()
    state.form_progress.active_group_id = "g-register"
    state.form_progress.active_group_candidate_ids = ["el_user", "el_pass", "el_confirm", "el_year", "el_submit"]
    state.form_progress.typed_candidate_ids = ["el_user", "el_pass", "el_confirm"]
    username = Candidate(
        id="el_user",
        role="input",
        type="input",
        text="Username",
        href="",
        context="Register form",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "register-username", "case_sensitive": False},
        dom_path="html/body/form/input[1]",
        field_kind="username",
        group_id="g-register",
        group_label="Register form",
    )
    password = Candidate(
        id="el_pass",
        role="input",
        type="input",
        text="Password",
        href="",
        context="Register form",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "register-password", "case_sensitive": False},
        dom_path="html/body/form/input[2]",
        field_kind="password",
        input_type="password",
        group_id="g-register",
        group_label="Register form",
    )
    confirm = Candidate(
        id="el_confirm",
        role="input",
        type="input",
        text="Confirm password",
        href="",
        context="Register form",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "register-confirm-password", "case_sensitive": False},
        dom_path="html/body/form/input[3]",
        field_kind="confirm_password",
        input_type="password",
        group_id="g-register",
        group_label="Register form",
    )
    year_select = Candidate(
        id="el_year",
        role="select",
        type="select",
        text="1917 2020 2024",
        href="",
        context="Register form",
        selector={"type": "xpathSelector", "value": "html/body/form/select[1]"},
        dom_path="html/body/form/select[1]",
        field_kind="year",
        group_id="g-register",
        group_label="Register form",
    )
    submit = Candidate(
        id="el_submit",
        role="button",
        type="button",
        text="Create account",
        href="",
        context="Register form",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "register-action", "case_sensitive": False},
        dom_path="html/body/form/button[1]",
        field_kind="account_create",
        input_type="submit",
        group_id="g-register",
        group_label="Register form",
    )
    guarded = engine._guard_missing_group_inputs(
        action={"type": "SelectDropDownOptionAction", "selector": year_select.selector, "text": "1917", "_element_id": "el_year"},
        prompt="Please register an account using username equals '', and password equals ''.",
        history=[],
        ranked_candidates=[username, password, confirm, year_select, submit],
        state=state,
    )
    assert isinstance(guarded, dict)
    assert guarded.get("type") == "ClickAction"
    assert guarded.get("_element_id") == "el_submit"


def test_submit_guard_uses_history_candidate_ids_to_avoid_retyping_same_field() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    state = AgentState()
    submit = Candidate(
        id="el_submit",
        role="button",
        type="button",
        text="Login",
        href="",
        context="Sign in form",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "signin-control", "case_sensitive": False},
        dom_path="html/body/form/button[1]",
        bbox=None,
    )
    username = Candidate(
        id="el_user",
        role="input",
        type="input",
        text="Username",
        href="",
        context="Sign in username",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "login-username", "case_sensitive": False},
        dom_path="html/body/form/input[1]",
        bbox=None,
    )
    password = Candidate(
        id="el_pass",
        role="input",
        type="input",
        text="Password",
        href="",
        context="Sign in password",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "password-entry-field", "case_sensitive": False},
        dom_path="html/body/form/input[2]",
        bbox=None,
    )
    guarded = engine._guard_submit_without_inputs(
        action={"type": "ClickAction", "selector": submit.selector, "_element_id": "el_submit"},
        prompt="Login to continue.",
        history=[{"action": "TypeAction", "candidate_id": "el_user", "text": "user1"}],
        ranked_candidates=[submit, username, password],
        state=state,
    )
    assert isinstance(guarded, dict)
    assert guarded.get("type") == "TypeAction"
    assert guarded.get("_element_id") == "el_pass"


def test_text_selector_click_is_resolved_to_candidate_selector() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    state = AgentState()
    button = Candidate(
        id="el_send",
        role="button",
        type="button",
        text="Enviar Mensaje",
        href="",
        context="contact form submit",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "message-button", "case_sensitive": False},
        dom_path="html/body/form/button[1]",
        bbox=None,
    )
    action = engine._browser_action_from_tool_call(
        tool_call={"name": "browser.click", "arguments": {"selector": {"type": "text", "value": "Enviar Mensaje"}}},
        ranked_candidates=[button],
        state=state,
        prompt="Send message",
        allowed={"browser.click"},
    )
    assert isinstance(action, dict)
    assert action.get("type") == "ClickAction"
    assert action.get("selector", {}).get("attribute") == "id"
    assert action.get("_element_id") == "el_send"


def test_ranker_prefers_create_targets_for_create_tasks() -> None:
    ranker = CandidateRanker()
    state = AgentState()
    candidates = [
        Candidate(
            id="el_detail",
            role="link",
            type="a",
            text="View details",
            href="/movies/123",
            context="Movie card",
            selector={"type": "attributeValueSelector", "attribute": "href", "value": "/movies/123", "case_sensitive": False},
            dom_path="html/body/a[1]",
            bbox=None,
        ),
        Candidate(
            id="el_add",
            role="button",
            type="button",
            text="Add Movie",
            href="",
            context="Create new movie",
            selector={"type": "attributeValueSelector", "attribute": "id", "value": "add-movie", "case_sensitive": False},
            dom_path="html/body/button[1]",
            bbox=None,
        ),
    ]
    ranked = ranker.rank(
        task="Add a film with rating 4.1",
        mode="NAV",
        flags={},
        candidates=candidates,
        state=state,
        top_k=10,
    )
    assert ranked
    assert ranked[0].id == "el_add"


def test_redundant_type_action_uses_state_roundtrip_and_advances_input(monkeypatch: Any) -> None:
    monkeypatch.setenv("FSM_DIRECT_LOOP", "0")
    engine = FSMOperator(llm_call=_dummy_llm_type_login_username)
    html = """
    <html><body>
      <form id="login-form">
        <label for="login-username">Username</label>
        <input id="login-username" name="username" />
        <label for="password-entry-field">Password</label>
        <input id="password-entry-field" name="password" type="password" />
        <button id="signin-control">Sign In</button>
      </form>
    </body></html>
    """
    payload = {
        "task_id": "fsm-login-typed-progress",
        "prompt": "Login to continue",
        "url": "https://example.com/login",
        "snapshot_html": html,
        "step_index": 0,
        "history": [],
        "allowed_tools": [
            {"name": "browser.click"},
            {"name": "browser.type"},
            {"name": "browser.wait"},
            {"name": "browser.back"},
            {"name": "browser.scroll"},
        ],
    }
    first = engine.run(payload=payload)
    first_actions = first.get("actions") if isinstance(first.get("actions"), list) else []
    assert len(first_actions) == 1
    assert first_actions[0].get("type") == "TypeAction"
    first_selector = first_actions[0].get("selector") if isinstance(first_actions[0].get("selector"), dict) else {}
    assert first_selector.get("value") == "login-username"
    state = AgentState.from_state_in(first.get("state_out"), prompt="Login to continue")
    ranked = engine.ranker.rank(
        task="Login to continue",
        mode="NAV",
        flags={},
        candidates=engine.extractor.extract(snapshot_html=html, url="https://example.com/login"),
        state=state,
        current_url="https://example.com/login",
        top_k=20,
    )
    repeated = engine._browser_action_from_tool_call(
        tool_call={
            "name": "browser.type",
            "arguments": {
                "selector": {
                    "type": "attributeValueSelector",
                    "attribute": "id",
                    "value": "login-username",
                    "case_sensitive": False,
                },
                "text": "<username>",
            },
        },
        ranked_candidates=ranked,
        state=state,
        prompt="Login to continue",
        allowed={"browser.type", "browser.click"},
    )
    assert isinstance(repeated, dict)
    guarded = engine._guard_redundant_type_action(
        action=repeated,
        prompt="Login to continue",
        history=[],
        ranked_candidates=ranked,
        state=state,
    )
    assert isinstance(guarded, dict)
    assert guarded.get("type") in {"TypeAction", "ClickAction"}
    guarded_selector = guarded.get("selector") if isinstance(guarded.get("selector"), dict) else {}
    if guarded.get("type") == "TypeAction":
        assert guarded_selector.get("value") != "login-username"


def test_click_href_with_conflicting_session_query_is_normalized_to_navigate() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    state = AgentState()
    state.session_query = {"seed": "85854"}
    add_link = Candidate(
        id="el_add",
        role="link",
        type="a",
        text="Add Movies",
        href="http://84.247.180.192:8000/search?seed=999",
        context="Profile nav",
        selector={"type": "attributeValueSelector", "attribute": "href", "value": "/search?seed=999", "case_sensitive": False},
        dom_path="html/body/a[1]",
        bbox=None,
    )
    action = engine._browser_action_from_tool_call(
        tool_call={"name": "browser.click", "arguments": {"element_id": "el_add"}},
        ranked_candidates=[add_link],
        state=state,
        prompt="Open Add Movies",
        allowed={"browser.click", "browser.navigate"},
        current_url="http://84.247.180.192:8000/profile?seed=85854",
    )
    assert isinstance(action, dict)
    assert action.get("type") == "ClickAction"
    assert action.get("_element_id") == "el_add"


def test_navigate_url_with_conflicting_session_query_is_pinned() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    state = AgentState()
    state.session_query = {"seed": "123"}
    action = engine._browser_action_from_tool_call(
        tool_call={"name": "browser.navigate", "arguments": {"url": "http://84.247.180.192:8000/search?seed=999"}},
        ranked_candidates=[],
        state=state,
        prompt="Navigate to search",
        allowed={"browser.navigate"},
        current_url="http://84.247.180.192:8000/?seed=123",
    )
    assert isinstance(action, dict)
    assert action.get("type") == "NavigateAction"
    assert str(action.get("url") or "").endswith("seed=123")


def test_navigate_same_site_target_uses_visible_link_click_when_available() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    state = AgentState()
    current_url = "http://84.247.180.192:8001/?seed=7"
    register_link = Candidate(
        id="el_register",
        role="link",
        type="a",
        text="Register",
        href="http://84.247.180.192:8001/signup?seed=7",
        context="Home Search Register Login",
        selector={"type": "attributeValueSelector", "attribute": "href", "value": "/signup?seed=7", "case_sensitive": False},
        dom_path="html/body/a[1]",
        bbox=None,
    )
    action = engine._browser_action_from_tool_call(
        tool_call={"name": "browser.navigate", "arguments": {"url": "http://84.247.180.192:8001/signup?seed=7"}},
        ranked_candidates=[register_link],
        state=state,
        prompt="Open register page",
        allowed={"browser.navigate", "browser.click"},
        current_url=current_url,
    )
    assert isinstance(action, dict)
    assert action.get("type") == "ClickAction"
    assert action.get("_element_id") == "el_register"


def test_navigate_same_site_target_uses_visible_link_click_when_query_differs() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    state = AgentState()
    current_url = "http://84.247.180.192:8001/?seed=7"
    register_link = Candidate(
        id="el_register",
        role="link",
        type="a",
        text="Register",
        href="http://84.247.180.192:8001/signup",
        context="Home Search Register Login",
        selector={"type": "attributeValueSelector", "attribute": "href", "value": "/signup", "case_sensitive": False},
        dom_path="html/body/a[1]",
        bbox=None,
    )
    action = engine._browser_action_from_tool_call(
        tool_call={"name": "browser.navigate", "arguments": {"url": "http://84.247.180.192:8001/signup?seed=7"}},
        ranked_candidates=[register_link],
        state=state,
        prompt="Open register page",
        allowed={"browser.navigate", "browser.click"},
        current_url=current_url,
    )
    assert isinstance(action, dict)
    assert action.get("type") == "ClickAction"
    assert action.get("_element_id") == "el_register"


def test_click_href_selector_with_conflicting_session_query_is_pinned_without_candidate_match() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    state = AgentState()
    state.session_query = {"seed": "777"}
    action = engine._browser_action_from_tool_call(
        tool_call={
            "name": "browser.click",
            "arguments": {
                "selector": {
                    "type": "attributeValueSelector",
                    "attribute": "href",
                    "value": "/login?seed=999",
                    "case_sensitive": False,
                }
            },
        },
        ranked_candidates=[],
        state=state,
        prompt="Open login page",
        allowed={"browser.click", "browser.navigate"},
        current_url="http://84.247.180.192:8000/profile?seed=777",
    )
    assert isinstance(action, dict)
    assert action.get("type") == "ClickAction"


def test_click_visible_same_site_link_keeps_click_when_query_differs() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    state = AgentState()
    state.session_query = {"seed": "7"}
    current_url = "http://84.247.180.192:8001/?seed=7"
    register_link = Candidate(
        id="el_register",
        role="link",
        type="a",
        text="Register",
        href="http://84.247.180.192:8001/signup",
        context="Home Search Register Login",
        selector={"type": "attributeValueSelector", "attribute": "href", "value": "/signup", "case_sensitive": False},
        dom_path="html/body/a[1]",
        bbox=None,
    )
    action = engine._browser_action_from_tool_call(
        tool_call={
            "name": "browser.click",
            "arguments": {
                "selector": {"type": "attributeValueSelector", "attribute": "href", "value": "/signup", "case_sensitive": False},
                "element_id": "el_register",
            },
        },
        ranked_candidates=[register_link],
        state=state,
        prompt="Open register page",
        allowed={"browser.click", "browser.navigate"},
        current_url=current_url,
    )
    assert isinstance(action, dict)
    assert action.get("type") == "ClickAction"
    assert action.get("_element_id") == "el_register"


def test_extractor_dom_path_xpath_is_emitted_without_leading_slash() -> None:
    extractor = CandidateExtractor()
    html = """
    <html><body>
      <div><form><input value="x" /></form></div>
    </body></html>
    """
    candidates = extractor.extract(snapshot_html=html, url="https://example.com")
    assert candidates
    xpath_candidates = [c for c in candidates if str((c.selector or {}).get("type")) == "xpathSelector"]
    assert xpath_candidates
    assert all(not str(c.selector.get("value") or "").startswith("/") for c in xpath_candidates)


def test_browser_type_rejects_non_input_selector_and_falls_back_to_input_candidate() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    state = AgentState()
    candidates = [
        Candidate(
            id="el_div_panel",
            role="button",
            type="div",
            text="Panel",
            href="",
            context="Container",
            selector={"type": "attributeValueSelector", "attribute": "id", "value": "panel", "case_sensitive": False},
            dom_path="html/body/div[1]",
            bbox=None,
        ),
        Candidate(
            id="el_cast_input",
            role="input",
            type="input",
            text="Cast",
            href="",
            context="Cast (comma separated)",
            selector={"type": "attributeValueSelector", "attribute": "id", "value": "cast-input", "case_sensitive": False},
            dom_path="html/body/input[1]",
            bbox=None,
        ),
    ]
    action = engine._browser_action_from_tool_call(
        tool_call={
            "name": "browser.type",
            "arguments": {
                "selector": {"type": "attributeValueSelector", "attribute": "id", "value": "panel", "case_sensitive": False},
                "text": "cosmic",
            },
        },
        ranked_candidates=candidates,
        state=state,
        prompt="Type cosmic into cast input",
        allowed={"browser.type"},
        current_url="https://example.com/profile",
    )
    assert isinstance(action, dict)
    assert action.get("type") == "TypeAction"
    selector = action.get("selector") if isinstance(action.get("selector"), dict) else {}
    assert selector.get("value") == "cast-input"


def test_flag_detector_not_found_phrase_does_not_trigger_error_page_false_positive() -> None:
    detector = FlagDetector()
    state = AgentState()
    html = """
    <html><body>
      <h1>Add New Film</h1>
      <p>Dataset entry not found for this seed.</p>
    </body></html>
    """
    flags = detector.detect(snapshot_html=html, url="https://example.com/profile", history=[], state=state)
    assert bool(flags.get("error_page")) is False


def test_obs_builder_extracts_structured_forms_with_select_options() -> None:
    builder = ObsBuilder()
    html = """
    <html><body>
      <h1>Search Movies</h1>
      <form id="filters">
        <label for="year">Year</label>
        <select id="year" name="year">
          <option value="">Any</option>
          <option value="2002">2002</option>
          <option value="2003">2003</option>
        </select>
        <label for="query">Title</label>
        <input id="query" name="query" placeholder="Search title" />
        <button id="apply">Apply Filters</button>
      </form>
    </body></html>
    """
    text_ir = builder.build_text_ir(html)
    assert text_ir.get("title") == ""
    forms = text_ir.get("forms") if isinstance(text_ir.get("forms"), list) else []
    assert forms
    first = forms[0] if isinstance(forms[0], dict) else {}
    controls = first.get("controls") if isinstance(first.get("controls"), list) else []
    assert any(isinstance(c, dict) and c.get("label") == "Year" for c in controls)
    select_control = next((c for c in controls if isinstance(c, dict) and c.get("tag") == "select"), {})
    assert "2003" in (select_control.get("options") or [])
    control_groups = text_ir.get("control_groups") if isinstance(text_ir.get("control_groups"), list) else []
    assert control_groups


def test_policy_obs_includes_grouped_item_cards() -> None:
    builder = ObsBuilder()
    state = AgentState()
    candidates = [
        Candidate(
            id="el_card_title",
            role="link",
            type="a",
            text="Interstellar",
            href="/movies/1",
            context="Interstellar 2014 169 min Matthew McConaughey View Delete",
            selector={"type": "attributeValueSelector", "attribute": "href", "value": "/movies/1", "case_sensitive": False},
            dom_path="html/body/div[1]/a[1]",
            bbox=None,
        ),
        Candidate(
            id="el_card_delete",
            role="button",
            type="button",
            text="Delete",
            href="",
            context="Interstellar 2014 169 min Matthew McConaughey View Delete",
            selector={"type": "attributeValueSelector", "attribute": "id", "value": "delete-1", "case_sensitive": False},
            dom_path="html/body/div[1]/button[1]",
            bbox=None,
        ),
        Candidate(
            id="el_filter_year",
            role="select",
            type="select",
            text="",
            href="",
            context="Filters Year Genre Apply",
            selector={"type": "attributeValueSelector", "attribute": "id", "value": "year", "case_sensitive": False},
            dom_path="html/body/form/select[1]",
            bbox=None,
        ),
        Candidate(
            id="el_filter_apply",
            role="button",
            type="button",
            text="Apply",
            href="",
            context="Filters Year Genre Apply",
            selector={"type": "attributeValueSelector", "attribute": "id", "value": "apply", "case_sensitive": False},
            dom_path="html/body/form/button[1]",
            bbox=None,
        ),
    ]
    policy_obs = builder.build_policy_obs(
        task_id="grouped-obs",
        prompt="Delete the film Interstellar after filtering by year equals 2014",
        step_index=1,
        url="https://example.com/movies",
        mode="NAV",
        flags={},
        state=state,
        text_ir={"title": "Movies", "visible_text": "Interstellar 2014 Matthew McConaughey", "headings": ["Movies"], "forms": []},
        candidates=candidates,
        history=[],
    )
    text_ir = policy_obs.get("text_ir") if isinstance(policy_obs.get("text_ir"), dict) else {}
    cards = text_ir.get("cards") if isinstance(text_ir.get("cards"), list) else []
    control_groups = text_ir.get("control_groups") if isinstance(text_ir.get("control_groups"), list) else []
    assert cards
    assert any("Interstellar" in json_blob for json_blob in [str(card) for card in cards])
    assert control_groups
    assert any(str(group.get("kind") or "") == "controls" for group in control_groups if isinstance(group, dict))
    assert "ITEM GROUPS:" in str(policy_obs.get("page_ir_text") or "")


def test_meta_vision_qa_updates_state_visual_memory() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid, vision_call=_dummy_vision_llm_apply)
    state = AgentState()
    candidates = [
        Candidate(
            id="el_apply",
            role="button",
            type="button",
            text="Apply",
            href="",
            context="Filters Year Genre Apply",
            selector={"type": "attributeValueSelector", "attribute": "id", "value": "apply-btn", "case_sensitive": False},
            dom_path="html/body/form/button[1]",
            bbox=None,
        )
    ]
    result = engine.meta.execute(
        task_id="vision-meta",
        tool_name="META.VISION_QA",
        args={"question": "Which visible control should be used next?"},
        state=state,
        prompt="Filter results by year",
        text_ir={"headings": ["Movies"], "control_groups": [{"label": "Filters"}], "cards": []},
        candidates=candidates,
        url="https://example.com/movies",
        screenshot="aGVsbG8=",
    )
    assert bool(result.get("ok")) is True
    assert "el_apply" in state.memory.visual_element_hints
    assert state.memory.visual_notes
    assert state.frontier.pending_elements and "el_apply" in state.frontier.pending_elements


def test_auto_vision_on_loop_boosts_visual_target_for_fallback(monkeypatch: Any) -> None:
    monkeypatch.setenv("FSM_DIRECT_LOOP", "0")
    html = """
    <html><body>
      <a id="related" href="/related">Related Movie</a>
      <form id="filters">
        <label for="year">Year</label>
        <select id="year"><option>2003</option></select>
        <button id="apply-btn">Apply</button>
      </form>
    </body></html>
    """
    probe_engine = FSMOperator(llm_call=_dummy_llm_invalid, vision_call=_dummy_vision_llm_apply)
    extracted = probe_engine.extractor.extract(snapshot_html=html, url="https://example.com/movies")
    apply_candidate = next(c for c in extracted if c.text == "Apply")

    def _vision_llm(**_: Any) -> Dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "answer": "Use the Apply button in the filter form.",
                                "element_ids": [apply_candidate.id],
                                "signals": ["filter controls visible", "apply button visible"],
                                "confidence": "high",
                            }
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 6, "total_tokens": 15},
            "model": "gpt-4o-mini",
        }

    engine = FSMOperator(llm_call=_dummy_llm_invalid, vision_call=_vision_llm)
    out = engine.run(
        payload={
            "task_id": "auto-vision-loop",
            "prompt": "Filter movies by year equals 2003",
            "url": "https://example.com/movies",
            "snapshot_html": html,
            "screenshot": "aGVsbG8=",
            "step_index": 2,
            "history": [],
            "state_in": {
                "mode": "NAV",
                "counters": {"stall_count": 2, "repeat_action_count": 1, "meta_steps_used": 0},
                "last_url": "https://example.com/movies",
                "last_dom_hash": "samehash",
                "last_action_sig": "ClickAction|same",
            },
            "allowed_tools": [{"name": "browser.click"}, {"name": "browser.wait"}],
        }
    )
    state_out = out.get("state_out") if isinstance(out.get("state_out"), dict) else {}
    memory = state_out.get("memory") if isinstance(state_out.get("memory"), dict) else {}
    assert apply_candidate.id in (memory.get("visual_element_hints") or [])
    actions = out.get("actions") if isinstance(out.get("actions"), list) else []
    assert actions
    assert actions[0].get("_element_id") == apply_candidate.id


def test_policy_obs_includes_active_control_group() -> None:
    builder = ObsBuilder()
    state = AgentState(
        form_progress=AgentFormProgress(
            active_group_label="Filters Year Genre Apply",
            active_group_context="Filters Year Genre Apply",
            active_group_candidate_ids=["el_year", "el_apply"],
        )
    )
    candidates = [
        Candidate(
            id="el_year",
            role="select",
            type="select",
            text="Year",
            href="",
            context="Filters Year Genre Apply",
            selector={"type": "attributeValueSelector", "attribute": "id", "value": "year", "case_sensitive": False},
            dom_path="html/body/form/select[1]",
            bbox=None,
        ),
        Candidate(
            id="el_apply",
            role="button",
            type="button",
            text="Apply",
            href="",
            context="Filters Year Genre Apply",
            selector={"type": "attributeValueSelector", "attribute": "id", "value": "apply", "case_sensitive": False},
            dom_path="html/body/form/button[1]",
            bbox=None,
        ),
    ]
    policy_obs = builder.build_policy_obs(
        task_id="active-group",
        prompt="Filter by year and apply the filter",
        step_index=2,
        url="https://example.com/movies",
        mode="NAV",
        flags={},
        state=state,
        text_ir={"title": "Movies", "visible_text": "Movies", "headings": ["Movies"], "forms": []},
        candidates=candidates,
        history=[],
    )
    text_ir = policy_obs.get("text_ir") if isinstance(policy_obs.get("text_ir"), dict) else {}
    active_group = text_ir.get("active_group") if isinstance(text_ir.get("active_group"), dict) else {}
    assert active_group
    assert active_group.get("label") == "Filters Year Genre Apply"
    assert "ACTIVE CONTROL GROUP:" in str(policy_obs.get("page_ir_text") or "")


def test_policy_obs_includes_main_style_deltas_and_grouped_browser_state() -> None:
    builder = ObsBuilder()
    state = AgentState(last_url="https://example.com/movies")
    state.memory.prev_page_summary = "Movies ; Old heading ; stale summary"
    state.memory.prev_page_ir_text = "TITLE: Movies\nHEADINGS: Old heading"
    state.memory.prev_candidate_sigs = ["attributeValueSelector|id|old-link|Old|link"]
    candidates = [
        Candidate(
            id="el_year",
            role="select",
            type="select",
            text="Year",
            href="",
            context="Filters Year Genre Apply",
            selector={"type": "attributeValueSelector", "attribute": "id", "value": "year", "case_sensitive": False},
            dom_path="html/body/form/select[1]",
            group_label="Filters",
            field_kind="year",
        ),
        Candidate(
            id="el_apply",
            role="button",
            type="button",
            text="Apply",
            href="",
            context="Filters Year Genre Apply",
            selector={"type": "attributeValueSelector", "attribute": "id", "value": "apply", "case_sensitive": False},
            dom_path="html/body/form/button[1]",
            group_label="Filters",
            field_kind="submit",
        ),
        Candidate(
            id="el_movie",
            role="link",
            type="link",
            text="Interstellar",
            href="https://example.com/movies/1",
            context="Interstellar 2014 View details",
            selector={"type": "attributeValueSelector", "attribute": "href", "value": "/movies/1", "case_sensitive": False},
            dom_path="html/body/main/a[1]",
            group_label="Featured Movies",
        ),
    ]
    policy_obs = builder.build_policy_obs(
        task_id="deltas",
        prompt="Filter movies by year equals 2014",
        step_index=1,
        url="https://example.com/movies",
        mode="NAV",
        flags={},
        state=state,
        text_ir={"title": "Movies", "visible_text": "Interstellar 2014", "headings": ["Movies"], "forms": []},
        candidates=candidates,
        history=[],
    )
    assert "summary_changed=" in str(policy_obs.get("state_delta") or "")
    assert "ir_changed=" in str(policy_obs.get("ir_delta") or "")
    browser_state = str(policy_obs.get("browser_state_text") or "")
    assert "Filters" in browser_state
    assert "[el_apply]" in browser_state
    assert "Featured Movies" in browser_state


def test_policy_obs_includes_focused_region_and_prioritizes_local_candidates() -> None:
    builder = ObsBuilder()
    state = AgentState(
        focus_region={
            "region_id": "region-form",
            "region_kind": "form",
            "region_label": "Login form",
            "region_context": "Login form Email Password Sign in",
            "candidate_ids": ["email", "submit"],
            "recent_region_ids": ["region-form"],
        }
    )
    candidates = [
        Candidate(
            id="nav-home",
            role="link",
            type="a",
            text="Home",
            href="/",
            context="Top navigation",
            selector={"type": "attributeValueSelector", "attribute": "href", "value": "/", "case_sensitive": False},
            dom_path="html/body/nav/a[1]",
            region_id="region-nav",
            region_kind="nav",
            region_label="Top navigation",
        ),
        Candidate(
            id="email",
            role="input",
            type="input",
            text="",
            href="",
            context="Login form Email Password Sign in",
            selector={"type": "attributeValueSelector", "attribute": "id", "value": "email", "case_sensitive": False},
            dom_path="html/body/main/form/input[1]",
            field_kind="email",
            input_type="email",
            region_id="region-form",
            region_kind="form",
            region_label="Login form",
        ),
        Candidate(
            id="submit",
            role="button",
            type="button",
            text="Sign in",
            href="",
            context="Login form Email Password Sign in",
            selector={"type": "xpathSelector", "value": "//button[contains(normalize-space(.), \"Sign in\")]", "case_sensitive": False},
            dom_path="html/body/main/form/button[1]",
            field_kind="submit",
            region_id="region-form",
            region_kind="form",
            region_label="Login form",
        ),
    ]
    policy_obs = builder.build_policy_obs(
        task_id="focused-region",
        prompt="Log in with email and password",
        step_index=2,
        url="https://example.com/login",
        mode="NAV",
        flags={"url_changed": False, "dom_changed": True},
        state=state,
        text_ir={"title": "Login", "visible_text": "Login form", "headings": ["Login"], "forms": [], "control_groups": [], "cards": [], "html_excerpt": "<form></form>"},
        candidates=candidates,
        history=[],
        screenshot_available=False,
    )
    active_region = policy_obs["text_ir"]["active_region"]
    assert active_region["region_id"] == "region-form"
    assert "FOCUSED REGION" in str(policy_obs.get("page_ir_text") or "")
    assert policy_obs["candidates"][0]["id"] in {"email", "submit"}


def test_progress_ledger_records_no_effect_and_releases_blocked_region() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    state = AgentState(
        mode="NAV",
        focus_region={
            "region_id": "region-form",
            "region_kind": "form",
            "region_label": "Login form",
            "region_context": "Login form",
            "candidate_ids": ["email"],
        },
        counters={"stall_count": 2, "repeat_action_count": 1, "meta_steps_used": 0},
    )
    history = [
        {
            "step": 0,
            "action": {"type": "ClickAction", "_element_id": "email"},
            "exec_ok": False,
            "error": "timeout",
            "url": "https://example.com/login",
        }
    ]
    flags = {"no_visual_progress": True, "url_changed": False, "dom_changed": False, "loop_level": "high"}
    engine._record_progress_effect(step_index=1, history=history, state=state, flags=flags)
    assert state.progress.last_effect == "BLOCKED"
    assert state.progress.no_progress_score >= 5
    engine._apply_stagnation_policy(state=state, flags=flags)
    assert state.focus_region.region_id == ""


def test_ranker_prefers_focus_region_candidates_over_global_nav() -> None:
    ranker = CandidateRanker()
    state = AgentState(
        focus_region={
            "region_id": "region-form",
            "region_kind": "form",
            "region_label": "Login form",
            "region_context": "Login form Email Password Sign in",
            "candidate_ids": ["email", "submit"],
        }
    )
    candidates = [
        Candidate(
            id="home",
            role="link",
            type="a",
            text="Home",
            href="/",
            context="Top navigation",
            selector={"type": "attributeValueSelector", "attribute": "href", "value": "/", "case_sensitive": False},
            dom_path="html/body/nav/a[1]",
            region_id="region-nav",
            region_kind="nav",
            region_label="Top navigation",
        ),
        Candidate(
            id="email",
            role="input",
            type="input",
            text="",
            href="",
            context="Login form Email Password Sign in",
            selector={"type": "attributeValueSelector", "attribute": "id", "value": "email", "case_sensitive": False},
            dom_path="html/body/form/input[1]",
            field_kind="email",
            input_type="email",
            region_id="region-form",
            region_kind="form",
            region_label="Login form",
        ),
    ]
    ranked = ranker.rank(
        task="Log in with email and password",
        mode="NAV",
        flags={"dom_changed": True},
        candidates=candidates,
        state=state,
        current_url="https://example.com/login",
        top_k=10,
    )
    assert ranked[0].id == "email"


def test_fsm_done_defaults_content_and_respects_reasoning_flag() -> None:
    def _llm_empty_final(**_: Any) -> Dict[str, Any]:
        return {
            "choices": [{"message": {"content": '{"type":"final","done":true,"content":""}'}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            "model": "gpt-5.2",
        }

    engine = FSMOperator(llm_call=_llm_empty_final)
    payload = _base_payload()
    payload["step_index"] = 2
    payload["state_in"] = {"mode": "REPORT", "memory": {"facts": ["done"]}}
    out = engine.run(payload=payload)
    assert out.get("done") is True
    assert isinstance(out.get("content"), str) and str(out.get("content")).strip()
    assert out.get("reasoning") is None


def test_wait_only_flow_completes_after_successful_wait(monkeypatch: Any) -> None:
    monkeypatch.setenv("FSM_DIRECT_LOOP", "0")
    def _llm_wait(**_: Any) -> Dict[str, Any]:
        return {
            "choices": [{"message": {"content": '{"type":"browser","tool_call":{"name":"browser.wait","arguments":{"time_seconds":1}}}'}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "model": "gpt-5.2",
        }

    engine = FSMOperator(llm_call=_llm_wait)
    base = {
        "task_id": "wait-only",
        "prompt": "Wait briefly and then finish.",
        "url": "https://example.com",
        "snapshot_html": "<html><body><h1>Example Domain</h1></body></html>",
        "allowed_tools": [{"name": "browser.wait"}],
    }
    first = engine.run(payload={**base, "step_index": 0, "history": []})
    second = engine.run(
        payload={
            **base,
            "step_index": 1,
            "state_in": first.get("state_out"),
            "history": [{"step": 0, "action": first.get("actions", [None])[0], "exec_ok": True, "url": "https://example.com"}],
        }
    )
    assert second.get("done") is True
    assert isinstance(second.get("content"), str) and second.get("content")
    assert second.get("actions") == []


def test_popup_pre_action_prefers_escape_after_overlay_intercept(monkeypatch: Any) -> None:
    monkeypatch.setenv("FSM_DIRECT_LOOP", "0")
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    out = engine.run(
        payload={
            "task_id": "popup-escape",
            "prompt": "Close the popup and continue",
            "url": "https://example.com/auth",
            "snapshot_html": "<html><body><div role='dialog'>Modal<button id='cancel'>Cancel</button></div></body></html>",
            "step_index": 2,
            "history": [
                {
                    "step": 1,
                    "action": {"type": "ClickAction"},
                    "exec_ok": False,
                    "error": "intercepts pointer events",
                    "url": "https://example.com/auth",
                }
            ],
            "state_in": {"last_action_sig": "ClickAction|same", "mode": "NAV"},
            "allowed_tools": [{"name": "browser.send_keys"}, {"name": "browser.click"}],
        }
    )
    actions = out.get("actions") if isinstance(out.get("actions"), list) else []
    assert actions
    assert actions[0].get("type") == "SendKeysIWAAction"
    assert actions[0].get("keys") == "Escape"


def test_flag_detector_marks_modal_auth_form_as_interactive_not_popup() -> None:
    flags = FlagDetector().detect(
        snapshot_html=(
            "<html><body>"
            "<div role='dialog' aria-modal='true'>"
            "<form>"
            "<input id='email' type='email' aria-label='Email' />"
            "<input id='password' type='password' aria-label='Password' />"
            "<button type='submit'>Sign in</button>"
            "</form>"
            "</div>"
            "</body></html>"
        ),
        url="https://example.com/login",
        history=[],
        state=AgentState(),
    )
    assert flags.get("interactive_modal_form") is True
    assert flags.get("modal_dialog") is False


def test_flag_detector_marks_modal_auth_panel_as_interactive_not_popup() -> None:
    flags = FlagDetector().detect(
        snapshot_html=(
            "<html><body>"
            "<div role='dialog' aria-modal='true'>"
            "<a href='/auth/sign-in'>Sign in</a>"
            "<a href='/auth/sign-up'>Sign up</a>"
            "</div>"
            "</body></html>"
        ),
        url="https://example.com/auth",
        history=[],
        state=AgentState(),
    )
    assert flags.get("interactive_modal_form") is True
    assert flags.get("modal_dialog") is False


def test_popup_solver_skips_disabled_buttons_and_form_groups() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    popup_result = engine.skills.solve_popups(
        candidates=[
            Candidate(
                id="email",
                role="input",
                type="input",
                text="",
                href="",
                context="Create your free account",
                selector={"type": "attributeValueSelector", "attribute": "id", "value": "email"},
                dom_path="html/body/div/form/input[1]",
                field_kind="email",
                group_id="auth-group",
                group_label="Create your free account",
            ),
            Candidate(
                id="password",
                role="input",
                type="input",
                text="",
                href="",
                context="Create your free account",
                selector={"type": "attributeValueSelector", "attribute": "id", "value": "password"},
                dom_path="html/body/div/form/input[2]",
                field_kind="password",
                group_id="auth-group",
                group_label="Create your free account",
            ),
            Candidate(
                id="create",
                role="button",
                type="button",
                text="Create your free account",
                href="",
                context="Create your free account",
                selector={"type": "attributeValueSelector", "attribute": "id", "value": "create"},
                dom_path="html/body/div/form/button[1]",
                group_id="auth-group",
                group_label="Create your free account",
                disabled=True,
            ),
            Candidate(
                id="cancel",
                role="button",
                type="button",
                text="Cancel",
                href="",
                context="Cookie dialog overlay",
                selector={"type": "attributeValueSelector", "attribute": "id", "value": "cancel"},
                dom_path="html/body/div/button[1]",
                group_id="popup-group",
                group_label="Cookie dialog",
            ),
        ]
    )
    assert popup_result.get("primary_element_id") == "cancel"


def test_router_replans_after_repeated_popup_failures() -> None:
    router = FSMOperator(llm_call=_dummy_llm_invalid).router
    state = AgentState(mode="NAV", counters={"repeat_action_count": 1, "stall_count": 0, "meta_steps_used": 0})
    mode, reason = router.next_mode(
        step_index=3,
        state=state,
        flags={"modal_dialog": True, "interactive_modal_form": False, "cookie_banner": False, "captcha_suspected": False, "loop_level": "low"},
    )
    assert mode == "PLAN"
    assert reason == "popup_stalled_replan"


def test_router_collapses_extract_like_modes_back_to_nav() -> None:
    router = FSMOperator(llm_call=_dummy_llm_invalid).router
    state = AgentState(mode="EXTRACT")
    mode, reason = router.next_mode(
        step_index=4,
        state=state,
        flags={"modal_dialog": False, "cookie_banner": False, "captcha_suspected": False, "loop_level": "none"},
    )
    assert mode == "NAV"
    assert reason == "continue_navigation"


def test_type_action_on_checkbox_is_converted_to_click() -> None:
    def _llm_type_checkbox(**_: Any) -> Dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"type":"browser","tool_call":{"name":"browser.type","arguments":{'
                            '"element_id":"checkbox-1","text":"autoppia"}}}'
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "model": "gpt-5.2",
        }

    engine = FSMOperator(llm_call=_llm_type_checkbox)
    out = engine.run(
        payload={
            "task_id": "checkbox-type",
            "prompt": "Accept the checkbox",
            "url": "https://example.com",
            "snapshot_html": "<html><body><input id='checkbox-1' type='checkbox' /><button>Continue</button></body></html>",
            "step_index": 1,
            "state_in": {"mode": "NAV"},
            "allowed_tools": [{"name": "browser.type"}, {"name": "browser.click"}],
        }
    )
    actions = out.get("actions") if isinstance(out.get("actions"), list) else []
    assert actions
    assert actions[0].get("type") == "ClickAction"
    assert isinstance(actions[0].get("_element_id"), str) and actions[0].get("_element_id")


def test_password_retry_uses_next_prompt_password() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    candidate = Candidate(
        id="pwd",
        role="input",
        type="input",
        text="",
        href="",
        context="Login form",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "password"},
        dom_path="html/body/form/input[2]",
        field_kind="password",
        input_type="password",
    )
    state = AgentState(
        form_progress=AgentFormProgress(
            typed_values_by_candidate={"pwd": "wrongpass"},
            typed_candidate_ids=["pwd"],
        )
    )
    value = engine._coerce_type_text(
        text="wrongpass",
        prompt="Try password wrongpass first, then retry with password password.",
        candidate=candidate,
        state=state,
    )
    assert value == "password"


def test_ranker_prefers_auth_entry_when_mutation_task_has_no_local_mutation_controls() -> None:
    ranker = CandidateRanker()
    state = AgentState()
    movie_link = Candidate(
        id="el_movie",
        role="link",
        type="link",
        text="Nightmare Alley",
        href="/movies/real-movie-120?seed=7",
        context="Featured movies duration 120 minutes",
        selector={"type": "attributeValueSelector", "attribute": "href", "value": "/movies/real-movie-120?seed=7", "case_sensitive": False},
        dom_path="html/body/main/a[1]",
    )
    register_link = Candidate(
        id="el_register",
        role="link",
        type="link",
        text="Register",
        href="/register?seed=7",
        context="Header navigation create account",
        selector={"type": "attributeValueSelector", "attribute": "href", "value": "/register?seed=7", "case_sensitive": False},
        dom_path="html/body/header/nav/a[1]",
    )
    login_link = Candidate(
        id="el_login",
        role="link",
        type="link",
        text="Login",
        href="/login?seed=7",
        context="Header navigation sign in",
        selector={"type": "attributeValueSelector", "attribute": "href", "value": "/login?seed=7", "case_sensitive": False},
        dom_path="html/body/header/nav/a[2]",
    )
    ranked = ranker.rank(
        task="Delete a film whose duration is NOT '142' minutes.",
        mode="NAV",
        flags={},
        candidates=[movie_link, register_link, login_link],
        state=state,
        current_url="https://example.com/",
        top_k=3,
    )
    assert ranked[0].id == "el_register"


def test_page_observations_expose_capability_gap_for_read_only_mutation_page() -> None:
    builder = ObsBuilder()
    state = AgentState()
    movie_link = Candidate(
        id="el_movie",
        role="link",
        type="link",
        text="Nightmare Alley",
        href="/movies/real-movie-120?seed=7",
        context="Featured movies duration 120 minutes",
        selector={"type": "attributeValueSelector", "attribute": "href", "value": "/movies/real-movie-120?seed=7", "case_sensitive": False},
        dom_path="html/body/main/a[1]",
    )
    register_link = Candidate(
        id="el_register",
        role="link",
        type="link",
        text="Register",
        href="/register?seed=7",
        context="Header navigation create account",
        selector={"type": "attributeValueSelector", "attribute": "href", "value": "/register?seed=7", "case_sensitive": False},
        dom_path="html/body/header/nav/a[1]",
    )
    policy_obs = builder.build_policy_obs(
        task_id="delete-read-only",
        prompt="Delete a film whose duration is NOT '142' minutes.",
        step_index=0,
        url="https://example.com/",
        mode="NAV",
        flags={},
        state=state,
        text_ir={"title": "Movies", "visible_text": "Movies", "headings": ["Movies"], "forms": []},
        candidates=[movie_link, register_link],
        history=[],
    )
    page_obs = policy_obs.get("page_observations") if isinstance(policy_obs.get("page_observations"), dict) else {}
    capability_gap = page_obs.get("capability_gap") if isinstance(page_obs.get("capability_gap"), dict) else {}
    assert capability_gap.get("read_only_for_task") is True
    assert capability_gap.get("preferred_transition") == "register"
    assert "delete" in (capability_gap.get("missing_task_operations") or [])
    assert "read-only" in str(policy_obs.get("page_ir_text") or "").lower()
    memory = policy_obs.get("memory") if isinstance(policy_obs.get("memory"), dict) else {}
    assert "registration" in str(memory.get("strategy_summary") or "").lower()


def test_ranker_prefers_local_mutation_control_over_unrelated_profile_fields() -> None:
    ranker = CandidateRanker()
    state = AgentState()
    delete_button = Candidate(
        id="el_delete",
        role="button",
        type="button",
        text="Delete Film",
        href="",
        context="Profile movie card manage actions",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "delete-film", "case_sensitive": False},
        dom_path="html/body/main/section/button[1]",
    )
    profile_name = Candidate(
        id="el_name",
        role="input",
        type="input",
        text="",
        href="",
        context="Profile settings update your account",
        field_hint="Name",
        field_kind="name",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "profile-name", "case_sensitive": False},
        dom_path="html/body/main/form/input[1]",
    )
    ranked = ranker.rank(
        task="Delete a film whose duration is NOT '142' minutes.",
        mode="NAV",
        flags={},
        candidates=[profile_name, delete_button],
        state=state,
        current_url="https://example.com/profile",
        top_k=2,
    )
    assert ranked[0].id == "el_delete"


def test_ranker_prefers_non_form_controls_on_delete_only_task_after_auth() -> None:
    ranker = CandidateRanker()
    state = AgentState()
    movie_button = Candidate(
        id="el_movie_button",
        role="button",
        type="button",
        text="Movies",
        href="",
        context="Profile movies manage entries",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "movie-button", "case_sensitive": False},
        dom_path="html/body/main/section/button[1]",
    )
    profile_email = Candidate(
        id="el_profile_email",
        role="input",
        type="input",
        text="",
        href="",
        context="Profile settings",
        field_hint="Email",
        field_kind="email",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "profile-email-field", "case_sensitive": False},
        dom_path="html/body/main/form/input[1]",
    )
    ranked = ranker.rank(
        task="Delete a film whose duration is NOT '142' minutes.",
        mode="NAV",
        flags={},
        candidates=[profile_email, movie_button],
        state=state,
        current_url="https://example.com/profile",
        top_k=2,
    )
    assert ranked[0].id == "el_movie_button"


def test_ranker_prefers_section_switch_over_profile_form_on_delete_task() -> None:
    ranker = CandidateRanker()
    state = AgentState()
    movies_tab = Candidate(
        id="el_movies_tab",
        role="button",
        type="button",
        text="Movies",
        href="",
        context="tablist profile movies add film",
        ui_state="inactive",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "radix-_r_0_-trigger-movies", "case_sensitive": False},
        dom_path="html/body/main/div/button[2]",
    )
    profile_tab = Candidate(
        id="el_profile_tab",
        role="button",
        type="button",
        text="Editar Perfil",
        href="",
        context="tablist profile movies add film",
        ui_state="active",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "radix-_r_0_-trigger-profile", "case_sensitive": False},
        dom_path="html/body/main/div/button[1]",
    )
    save_profile = Candidate(
        id="el_save_profile",
        role="button",
        type="button",
        text="Save Profile",
        href="",
        context="Edit Profile form",
        field_hint="Save Profile",
        field_kind="submit",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "save-profile", "case_sensitive": False},
        dom_path="html/body/main/form/button[1]",
    )
    profile_email = Candidate(
        id="el_profile_email",
        role="input",
        type="input",
        text="",
        href="",
        context="Profile settings",
        field_hint="Email",
        field_kind="email",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "profile-email-field", "case_sensitive": False},
        dom_path="html/body/main/form/input[1]",
    )
    ranked = ranker.rank(
        task="Delete a film whose duration is NOT '142' minutes.",
        mode="NAV",
        flags={},
        candidates=[profile_email, save_profile, profile_tab, movies_tab],
        state=state,
        current_url="https://example.com/profile",
        top_k=4,
    )
    assert ranked[0].id == "el_movies_tab"


def test_delete_guard_redirects_unrelated_type_to_click() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    movie_button = Candidate(
        id="el_movie_button",
        role="button",
        type="button",
        text="Movies",
        href="",
        context="Profile movies manage entries",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "movie-button", "case_sensitive": False},
        dom_path="html/body/main/section/button[1]",
    )
    profile_email = Candidate(
        id="el_profile_email",
        role="input",
        type="input",
        text="",
        href="",
        context="Profile settings",
        field_hint="Email",
        field_kind="email",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "profile-email-field", "case_sensitive": False},
        dom_path="html/body/main/form/input[1]",
    )
    action = {
        "type": "TypeAction",
        "selector": profile_email.selector,
        "text": "autoppia@example.com",
        "_element_id": "el_profile_email",
    }
    guarded = engine._guard_delete_task_against_unrelated_form_edits(
        action=action,
        prompt="Delete a film whose duration is NOT '142' minutes.",
        ranked_candidates=[movie_button, profile_email],
        state=AgentState(),
    )
    assert guarded is not None
    assert guarded.get("type") == "ClickAction"
    assert guarded.get("_element_id") == "el_movie_button"


def test_browser_action_from_tool_call_redirects_delete_only_type_to_click() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    profile_tab = Candidate(
        id="el_profile_tab",
        role="button",
        type="button",
        text="Editar Perfil",
        href="",
        context="tablist profile movies add film",
        ui_state="active",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "radix-_r_0_-trigger-profile", "case_sensitive": False},
        dom_path="html/body/main/div/button[1]",
    )
    movie_button = Candidate(
        id="el_movie_button",
        role="button",
        type="button",
        text="Movies",
        href="",
        context="tablist profile movies add film",
        ui_state="inactive",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "radix-_r_0_-trigger-movies", "case_sensitive": False},
        dom_path="html/body/main/div/button[2]",
    )
    profile_email = Candidate(
        id="el_profile_email",
        role="input",
        type="input",
        text="",
        href="",
        context="Profile settings",
        field_hint="Email",
        field_kind="email",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "profile-email-field", "case_sensitive": False},
        dom_path="html/body/main/form/input[1]",
    )
    action = engine._browser_action_from_tool_call(
        tool_call={
            "name": "browser.type",
            "arguments": {"element_id": "el_profile_email", "text": "autoppia@example.com"},
        },
        ranked_candidates=[profile_tab, movie_button, profile_email],
        state=AgentState(),
        prompt="Delete a film whose duration is NOT '142' minutes.",
        allowed={"browser.type", "browser.click"},
        current_url="https://example.com/profile",
    )
    assert action is not None
    assert action.get("type") == "ClickAction"
    assert action.get("_element_id") == "el_movie_button"


def test_read_only_mutation_page_promotes_plan_mode(monkeypatch: Any) -> None:
    monkeypatch.setenv("FSM_DIRECT_LOOP", "0")
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    out = engine.run(
        payload={
            **_base_payload(),
            "prompt": "Delete a film whose duration is NOT '142' minutes.",
            "url": "https://example.com/movies/real-movie-120?seed=7",
            "snapshot_html": (
                "<html><body><h1>Nightmare Alley</h1>"
                "<a href='/register?seed=7'>Register</a>"
                "<button id='watchlist-action'>Add to Watchlist</button>"
                "</body></html>"
            ),
            "step_index": 1,
            "state_in": {"mode": "NAV"},
        }
    )
    state_out = out.get("state_out") if isinstance(out.get("state_out"), dict) else {}
    assert state_out.get("mode") in {"PLAN", "NAV", "DONE"}
    reasoning = str(out.get("reasoning") or "")
    assert "capability_gap_model_replan" in reasoning or state_out.get("memory", {}).get("strategy_summary")


def test_missing_group_guard_does_not_override_section_switch() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    state = AgentState(
        form_progress=AgentFormProgress(
            active_group_id="profile-group",
            active_group_label="Edit Profile",
            active_group_context="Edit Profile",
            active_group_candidate_ids=["el_profile_email", "el_profile_bio"],
        )
    )
    movies_tab = Candidate(
        id="el_movies_tab",
        role="button",
        type="button",
        text="Movies",
        href="",
        context="tablist profile movies add film",
        ui_state="inactive",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "radix-_r_0_-trigger-movies", "case_sensitive": False},
        dom_path="html/body/main/div/button[2]",
    )
    profile_email = Candidate(
        id="el_profile_email",
        role="input",
        type="input",
        text="",
        href="",
        context="Edit Profile",
        field_hint="Email",
        field_kind="email",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "profile-email-field", "case_sensitive": False},
        dom_path="html/body/main/form/input[1]",
        group_id="profile-group",
    )
    action = {"type": "ClickAction", "selector": movies_tab.selector, "_element_id": "el_movies_tab"}
    guarded = engine._guard_missing_group_inputs(
        action=action,
        prompt="Delete a film whose duration is NOT '142' minutes.",
        history=[],
        ranked_candidates=[movies_tab, profile_email],
        state=state,
    )
    assert guarded == action


def test_redundant_select_prefers_same_group_submit() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    select_candidate = Candidate(
        id="el_year",
        role="select",
        type="select",
        text="Year",
        href="",
        context="Filters Year Genre Apply",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "year", "case_sensitive": False},
        dom_path="html/body/form/select[1]",
        bbox=None,
    )
    apply_candidate = Candidate(
        id="el_apply",
        role="button",
        type="button",
        text="Apply",
        href="",
        context="Filters Year Genre Apply",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "apply", "case_sensitive": False},
        dom_path="html/body/form/button[1]",
        bbox=None,
    )
    action = engine._guard_redundant_select_action(
        action={
            "type": "SelectDropDownOptionAction",
            "selector": select_candidate.selector,
            "text": "2003",
            "_element_id": select_candidate.id,
        },
        history=[
            {
                "action": {
                    "type": "SelectDropDownOptionAction",
                    "selector": select_candidate.selector,
                    "_element_id": select_candidate.id,
                }
            }
        ],
        ranked_candidates=[select_candidate, apply_candidate],
        state=AgentState(),
    )
    assert action is not None
    assert action.get("type") == "ClickAction"
    assert action.get("_element_id") == "el_apply"


def test_default_vision_question_narrows_after_select_loop() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid, vision_call=_dummy_vision_llm_apply)
    question = engine._default_vision_question(
        prompt="Filter movies by year equals 2003",
        state=AgentState(
            form_progress=AgentFormProgress(active_group_label="Filters Year Genre Apply"),
            last_action_sig="SelectDropDownOptionAction|same",
        ),
        text_ir={"control_groups": [{"label": "Filters"}], "cards": []},
    )
    assert "avoid repeating the same select" in question
    assert "Filters Year Genre Apply" in question


def test_extractor_adds_region_lineage_for_nested_form_controls() -> None:
    extractor = CandidateExtractor()
    html = """
    <html><body>
      <section>
        <h2>Profile</h2>
        <form id="profile-form">
          <label for="email">Email</label>
          <input id="email" type="email" />
          <button id="save-btn">Save</button>
        </form>
      </section>
    </body></html>
    """
    candidates = extractor.extract(snapshot_html=html, url="https://example.com/profile")
    assert candidates
    email = next(c for c in candidates if c.id and c.role == "input")
    assert email.region_id
    assert email.region_kind in {"form", "group"}
    assert isinstance(email.region_ancestor_ids, list)


def test_fallback_prefers_local_escape_candidate_before_global_back() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    policy_obs = {
        "candidates": [
            {
                "id": "global-link",
                "role": "link",
                "selector": {"type": "attributeValueSelector", "attribute": "href", "value": "/elsewhere", "case_sensitive": False},
            }
        ],
        "candidate_partitions": {
            "local": [],
            "escape": [
                {
                    "id": "save-btn",
                    "role": "button",
                    "text": "Save",
                    "selector": {"type": "attributeValueSelector", "attribute": "id", "value": "save-btn", "case_sensitive": False},
                }
            ],
            "global": [],
            "suppressed_global_count": 4,
        },
        "memory": {"typed_candidate_ids": [], "visual_element_hints": []},
        "flags": {"loop_level": "high"},
        "counters": {"stall_count": 4, "repeat_action_count": 2},
    }
    out = engine.policy._fallback(
        prompt="Save the current form",
        mode="PLAN",
        policy_obs=policy_obs,
        allowed_tools={"browser.click", "browser.back"},
    )
    assert out["type"] == "browser"
    assert out["tool_call"]["name"] == "browser.click"
    assert out["tool_call"]["arguments"]["element_id"] == "save-btn"


def test_ranker_demotes_local_pager_when_submit_is_available() -> None:
    ranker = CandidateRanker()
    state = AgentState()
    state.focus_region.region_id = "wizard"
    state.focus_region.region_kind = "group"
    state.focus_region.region_label = "Wizard"
    state.focus_region.candidate_ids = ["next-btn", "save-btn"]
    next_btn = Candidate(
        id="next-btn",
        role="button",
        type="button",
        text="Next",
        href="",
        context="Wizard controls",
        field_kind="pager",
        region_id="wizard",
        region_kind="group",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "next-btn", "case_sensitive": False},
        dom_path="html/body/div/button[1]",
    )
    save_btn = Candidate(
        id="save-btn",
        role="button",
        type="button",
        text="Save",
        href="",
        context="Wizard controls",
        field_kind="submit",
        region_id="wizard",
        region_kind="group",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "save-btn", "case_sensitive": False},
        dom_path="html/body/div/button[2]",
    )
    ranked = ranker.rank(
        task="Save the current form",
        mode="NAV",
        flags={},
        candidates=[next_btn, save_btn],
        state=state,
        current_url="https://example.com/wizard",
        top_k=5,
    )
    assert ranked[0].id == "save-btn"


def test_ranker_prefers_remaining_relevant_form_inputs_before_submit() -> None:
    ranker = CandidateRanker()
    state = AgentState()
    state.form_progress.active_group_id = "auth-group"
    state.form_progress.active_group_candidate_ids = ["user", "pass", "confirm", "submit"]
    state.form_progress.typed_candidate_ids = ["user"]
    state.form_progress.typed_values_by_candidate = {"user": "<signup_username>"}
    candidates = [
        Candidate(
            id="user",
            role="input",
            type="input",
            text="Username",
            href="",
            context="Create account Username Password Confirm Password Create account",
            selector={"type": "attributeValueSelector", "attribute": "id", "value": "username-input", "case_sensitive": False},
            dom_path="html/body/form/input[1]",
            field_kind="username",
            group_id="auth-group",
            group_label="Create account",
        ),
        Candidate(
            id="pass",
            role="input",
            type="input",
            text="Password",
            href="",
            context="Create account Username Password Confirm Password Create account",
            selector={"type": "attributeValueSelector", "attribute": "id", "value": "password-input", "case_sensitive": False},
            dom_path="html/body/form/input[2]",
            field_kind="password",
            input_type="password",
            group_id="auth-group",
            group_label="Create account",
        ),
        Candidate(
            id="confirm",
            role="input",
            type="input",
            text="Confirm Password",
            href="",
            context="Create account Username Password Confirm Password Create account",
            selector={"type": "attributeValueSelector", "attribute": "id", "value": "confirm-password-input", "case_sensitive": False},
            dom_path="html/body/form/input[3]",
            field_kind="confirm_password",
            input_type="password",
            group_id="auth-group",
            group_label="Create account",
        ),
        Candidate(
            id="submit",
            role="button",
            type="button",
            text="Create account",
            href="",
            context="Create account Username Password Confirm Password Create account",
            selector={"type": "attributeValueSelector", "attribute": "id", "value": "signup-submit-button", "case_sensitive": False},
            dom_path="html/body/form/button[1]",
            field_kind="account_create",
            group_id="auth-group",
            group_label="Create account",
        ),
    ]
    ranked = ranker.rank(
        task="Register with username <signup_username> and password <signup_password>",
        mode="NAV",
        flags={},
        candidates=candidates,
        state=state,
        current_url="https://example.com/signup",
        top_k=10,
    )
    assert ranked
    ids = [cand.id for cand in ranked]
    assert ids[0] in {"pass", "confirm"}
    assert ids.index("submit") > ids.index("pass")


def test_constraint_progress_demotes_already_satisfied_choice_candidates() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    state = AgentState()
    memoir_button = Candidate(
        id="memoir-btn",
        role="button",
        type="button",
        text="Memoir",
        href="",
        context="Genre choices",
        group_label="Genres",
        region_label="Genres",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "memoir-btn", "case_sensitive": False},
        dom_path="html/body/div/button[1]",
    )
    author_input = Candidate(
        id="author-input",
        role="input",
        type="input",
        text="",
        href="",
        context="Author field",
        field_hint="Author",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "author-input", "case_sensitive": False},
        dom_path="html/body/input[1]",
    )
    engine._remember_form_progress_from_action(
        prompt="genres equals 'Memoir' and author equals 'thunder'",
        action={"type": "ClickAction", "selector": memoir_button.selector, "_element_id": memoir_button.id},
        ranked_candidates=[memoir_button, author_input],
        state=state,
    )
    assert "genres" in state.progress.satisfied_constraints
    ranked = engine.ranker.rank(
        task="genres equals 'Memoir' and author equals 'thunder'",
        mode="NAV",
        flags={},
        candidates=[memoir_button, author_input],
        state=state,
        current_url="https://example.com/add",
        top_k=5,
    )
    assert ranked[0].id == "author-input"


def test_ranker_prefers_exact_constraint_value_choice_over_wrong_choice() -> None:
    ranker = CandidateRanker()
    state = AgentState()
    memoir = Candidate(
        id="memoir",
        role="button",
        type="button",
        text="Memoir",
        href="",
        context="Genres chooser",
        group_label="Genres",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "memoir", "case_sensitive": False},
        dom_path="html/body/div/button[1]",
    )
    mystery = Candidate(
        id="mystery",
        role="button",
        type="button",
        text="Mystery",
        href="",
        context="Genres chooser",
        group_label="Genres",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "mystery", "case_sensitive": False},
        dom_path="html/body/div/button[2]",
    )
    ranked = ranker.rank(
        task="genres equals 'Memoir'",
        mode="NAV",
        flags={},
        candidates=[mystery, memoir],
        state=state,
        current_url="https://example.com/add",
        top_k=5,
    )
    assert ranked[0].id == "memoir"


def test_select_candidates_for_policy_keeps_local_shortlist_and_limits_global_noise() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    state = AgentState()
    state.focus_region.region_id = "editor"
    state.focus_region.region_kind = "form"
    state.focus_region.candidate_ids = ["title", "save"]
    local_input = Candidate(
        id="title",
        role="input",
        type="input",
        text="",
        href="",
        context="Edit form",
        field_hint="Title",
        region_id="editor",
        region_kind="form",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "title", "case_sensitive": False},
        dom_path="html/body/form/input[1]",
    )
    local_save = Candidate(
        id="save",
        role="button",
        type="button",
        text="Save",
        href="",
        context="Edit form",
        field_kind="submit",
        region_id="editor",
        region_kind="form",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "save", "case_sensitive": False},
        dom_path="html/body/form/button[1]",
    )
    global_candidates = [
        Candidate(
            id=f"nav-{i}",
            role="link",
            type="link",
            text=f"Section {i}",
            href=f"https://example.com/section-{i}",
            context="Main navigation links",
            selector={"type": "attributeValueSelector", "attribute": "id", "value": f"nav-{i}", "case_sensitive": False},
            dom_path=f"html/body/nav/a[{i+1}]",
        )
        for i in range(20)
    ]
    selected = engine.obs_builder._select_candidates_for_policy(
        candidates=[local_input, local_save, *global_candidates],
        current_url="https://example.com/edit",
        state=state,
        max_total=24,
    )
    ids = [cand.id for cand in selected]
    assert ids[:2] == ["title", "save"]
    assert len(selected) < len([local_input, local_save, *global_candidates])
    assert len(selected) <= 24


def test_build_policy_obs_exposes_short_action_list_not_large_candidate_dump() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    state = AgentState()
    candidates = [
        Candidate(
            id=f"cand-{i}",
            role="link" if i % 2 else "button",
            type="link" if i % 2 else "button",
            text=f"Candidate {i}",
            href=f"https://example.com/item-{i}" if i % 2 else "",
            context="Page controls" if i < 5 else "Global navigation",
            selector={"type": "attributeValueSelector", "attribute": "id", "value": f"cand-{i}", "case_sensitive": False},
            dom_path=f"html/body/div[{i+1}]",
        )
        for i in range(40)
    ]
    policy_obs = engine.obs_builder.build_policy_obs(
        task_id="t1",
        prompt="Open the relevant item",
        step_index=1,
        url="https://example.com",
        mode="NAV",
        flags={},
        state=state,
        text_ir={"title": "Example", "visible_text": "Example page", "headings": []},
        candidates=candidates,
        history=[],
        screenshot_available=False,
    )
    assert len(policy_obs["candidates"]) <= 24
    assert "ACTION SHORTLIST (JSON):" in policy_obs["policy_input_text"]


def test_store_expected_effect_for_submit_click() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    state = AgentState()
    submit = Candidate(
        id="save",
        role="button",
        type="button",
        text="Save",
        href="",
        context="Edit form",
        field_kind="submit",
        selector={"type": "attributeValueSelector", "attribute": "id", "value": "save", "case_sensitive": False},
        dom_path="html/body/form/button[1]",
    )
    action = {"type": "ClickAction", "_element_id": "save", "selector": submit.selector}
    engine._store_expected_effect(action=action, ranked_candidates=[submit], state=state)
    assert state.progress.pending_expected_effect == "submit_effect"
    assert state.progress.pending_expected_target_id == "save"


def test_record_progress_effect_marks_expected_effect_miss() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    state = AgentState()
    state.progress.pending_expected_effect = "navigation"
    state.progress.pending_expected_action_type = "ClickAction"
    state.progress.pending_expected_target_id = "go"
    history = [{"action": {"type": "ClickAction", "_element_id": "go"}, "exec_ok": True}]
    engine._record_progress_effect(
        step_index=2,
        history=history,
        state=state,
        flags={"url_changed": False, "dom_changed": False, "no_visual_progress": True},
    )
    effect = state.progress.recent_effects[-1]
    assert effect.expected_effect == "navigation"
    assert effect.expected_effect_met is False
    assert "clickaction_expected_navigation_miss" in state.progress.failed_patterns
    assert state.progress.consecutive_no_effect_steps >= 1


def test_obs_extract_uses_small_model_and_caches_by_dom_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def _obs_llm(*, task_id: str, messages: list[dict], model: str, temperature: float, max_tokens: int) -> dict:
        calls["n"] += 1
        assert model == "gpt-4o-mini"
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"page_kind":"form","summary":"Filter form with apply button","regions":[{"kind":"form","label":"Filters","candidate_ids":["apply-btn"]}],"forms":[{"label":"Filters","fields":["genre","author"],"commit_ids":["apply-btn"]}],"facts":[],"primary_candidate_ids":["apply-btn"]}'
                    }
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "model": "gpt-4o-mini",
        }

    monkeypatch.setenv("FSM_OBS_EXTRACT_MODE", "auto")
    monkeypatch.setenv("OPENAI_OBS_MODEL", "gpt-4o-mini")
    engine = FSMOperator(llm_call=_obs_llm)
    state = AgentState()
    flags = {"dom_hash": "abc123", "dom_changed": True, "url_changed": False}
    text_ir = {
        "title": "Books",
        "headings": ["Filters"],
        "visible_text": "Filters Apply",
        "html_excerpt": "<form><button id='apply-btn'>Apply</button></form>",
        "forms": [],
        "cards": [],
    }
    candidates = [
        Candidate(
            id="apply-btn",
            role="button",
            type="button",
            text="Apply",
            href="",
            context="Filters Apply",
            selector={"type": "attributeValueSelector", "attribute": "id", "value": "apply-btn", "case_sensitive": False},
            dom_path="html/body/form/button[1]",
        )
    ]
    first = engine._maybe_extract_observation(
        task_id="obs-1",
        prompt="Filter books",
        url="https://example.com",
        flags=flags,
        state=state,
        text_ir=text_ir,
        candidates=candidates,
    )
    second = engine._maybe_extract_observation(
        task_id="obs-1",
        prompt="Filter books",
        url="https://example.com",
        flags=flags,
        state=state,
        text_ir=text_ir,
        candidates=candidates,
    )
    assert first.get("summary") == "Filter form with apply button"
    assert second.get("summary") == "Filter form with apply button"
    assert "apply-btn" in state.memory.obs_candidate_hints
    assert calls["n"] == 1


def test_build_text_ir_extracts_generic_page_facts() -> None:
    builder = ObsBuilder()
    text_ir = builder.build_text_ir(
        snapshot_html="""
        <html><body>
          <section>
            <h1>Treasury Details</h1>
            <div><span>Total Treasury</span><span>2.8K</span></div>
            <div><span>Total Value</span><span>124</span></div>
          </section>
        </body></html>
        """
    )
    page_facts = text_ir.get("page_facts") if isinstance(text_ir.get("page_facts"), list) else []
    assert any("Total Treasury" in str(fact) and "2.8K" in str(fact) for fact in page_facts)


def test_build_text_ir_extracts_visible_value_lines() -> None:
    builder = ObsBuilder()
    text_ir = builder.build_text_ir(
        snapshot_html="""
        <html><body>
          <section>
            <h1>Treasury Details</h1>
            <div>Total Treasury</div>
            <div>2.8K τ</div>
            <div>Wallet Count</div>
            <div>12</div>
          </section>
        </body></html>
        """
    )
    value_lines = text_ir.get("value_lines") if isinstance(text_ir.get("value_lines"), list) else []
    assert any("Total Treasury" in str(line) and "2.8K" in str(line) for line in value_lines)
    assert any("Wallet Count" in str(line) and "12" in str(line) for line in value_lines)


def test_page_ir_text_includes_visible_value_lines_for_informational_task() -> None:
    builder = ObsBuilder()
    text_ir = builder.build_text_ir(
        snapshot_html="""
        <html><body>
          <h1>Treasury</h1>
          <div>Total Treasury</div>
          <div>2.8K τ</div>
          <div>Total Value Locked</div>
          <div>2847 τ</div>
        </body></html>
        """
    )
    page_ir = builder._page_ir_text(prompt="Tell me the treasury total value", text_ir=text_ir, candidates=[])
    assert "VISIBLE VALUE LINES:" in page_ir
    assert "RELEVANT VISIBLE TEXT:" in page_ir
    assert "2.8K" in page_ir or "2847" in page_ir


def test_build_text_ir_extracts_required_form_fields_and_commit_controls() -> None:
    builder = ObsBuilder()
    text_ir = builder.build_text_ir(
        snapshot_html="""
        <html><body>
          <form id="signup-form">
            <label for="user">Username</label>
            <input id="user" name="username" required />
            <label for="mail">Email</label>
            <input id="mail" type="email" name="email" required />
            <label for="pass">Password</label>
            <input id="pass" type="password" name="password" required />
            <button id="signup-submit">Create account</button>
          </form>
        </body></html>
        """
    )
    forms = text_ir.get("forms") if isinstance(text_ir.get("forms"), list) else []
    assert forms
    signup = forms[0]
    controls = signup.get("controls") if isinstance(signup.get("controls"), list) else []
    assert any(bool(control.get("required")) and str(control.get("label") or "") == "Username" for control in controls if isinstance(control, dict))
    assert any(bool(control.get("required")) and str(control.get("label") or "") == "Email" for control in controls if isinstance(control, dict))
    commit_controls = signup.get("commit_controls") if isinstance(signup.get("commit_controls"), list) else []
    assert any("create account" in str(item).lower() for item in commit_controls)


def test_page_ir_text_includes_form_commit_controls() -> None:
    builder = ObsBuilder()
    text_ir = builder.build_text_ir(
        snapshot_html="""
        <html><body>
          <form id="login-form">
            <label for="login-user">Username</label>
            <input id="login-user" name="username" required />
            <label for="login-password">Password</label>
            <input id="login-password" type="password" required />
            <button id="signin-submit">Sign in</button>
          </form>
        </body></html>
        """
    )
    page_ir = builder._page_ir_text(prompt="Log in with the provided credentials", text_ir=text_ir, candidates=[])
    assert "FORMS:" in page_ir
    assert "required" in page_ir.lower()
    assert "commits=" in page_ir
    assert "sign in" in page_ir.lower()


def test_build_text_ir_extracts_selected_value_for_select_controls() -> None:
    builder = ObsBuilder()
    text_ir = builder.build_text_ir(
        snapshot_html="""
        <html><body>
          <form>
            <label>Genre
              <select name="genre">
                <option value="">All genres</option>
                <option value="Allegory" selected>Allegory</option>
                <option value="Fantasy">Fantasy</option>
              </select>
            </label>
          </form>
        </body></html>
        """
    )
    forms = text_ir.get("forms") if isinstance(text_ir.get("forms"), list) else []
    assert forms
    controls = forms[0].get("controls") if isinstance(forms[0].get("controls"), list) else []
    genre = next(control for control in controls if isinstance(control, dict) and str(control.get("tag") or "") == "select")
    assert genre.get("value") == "Allegory"


def test_candidate_extractor_includes_current_selected_value_for_select() -> None:
    extractor = CandidateExtractor()
    candidates = extractor.extract(
        url="https://example.com/search",
        snapshot_html="""
        <html><body>
          <main>
            <section>
              <label for="genre-filter">Genre</label>
              <select id="genre-filter" name="genre">
                <option value="">All genres</option>
                <option value="Allegory" selected>Allegory</option>
                <option value="Fantasy">Fantasy</option>
              </select>
            </section>
          </main>
        </body></html>
        """,
    )
    genre = next(c for c in candidates if c.role == "select")
    assert "current=Allegory" in genre.text




def test_prompt_field_needs_and_field_kind_detect_genre() -> None:
    ranker = CandidateRanker()
    assert "genre" in ranker._prompt_field_needs("Show me books where the genres equal Allegory")


def test_augment_text_ir_merges_form_and_candidate_control_groups() -> None:
    builder = ObsBuilder()
    form_payload = {
        "id": "form_1",
        "controls": [
            {"tag": "input", "type": "text", "label": "Username", "options": [], "value": ""},
        ],
        "text": "Create account",
        "commit_controls": ["Sign up"],
    }
    text_ir = {
        "visible_text": "",
        "visible_lines": [],
        "headings": [],
        "title": "",
        "forms": [form_payload],
        "control_groups": builder._control_groups_from_forms([form_payload]),
    }
    candidates = [
        Candidate(
            id="el_1",
            role="select",
            type="select",
            text="Allegory",
            field_hint="Genre",
            href="",
            dom_path="html/body/main/section/div/select[1]",
            selector={"type": "attributeValueSelector", "attribute": "id", "value": "genre-filter", "case_sensitive": False},
            context="Living catalog Curated books Genre Allegory Clear filters",
            group_id="group_1",
            group_label="Living catalog",
            region_id="region_1",
            region_label="Living catalog",
            region_kind="group",
        ),
        Candidate(
            id="el_2",
            role="button",
            type="clickable",
            text="Clear filters",
            field_hint="",
            href="",
            dom_path="html/body/main/section/div/button[1]",
            selector={"type": "xpathSelector", "value": "//button[contains(normalize-space(.), \"Clear filters\")]", "case_sensitive": False},
            context="Living catalog Curated books Genre Allegory Clear filters",
            group_id="group_1",
            group_label="Living catalog",
            region_id="region_1",
            region_label="Living catalog",
            region_kind="group",
        ),
    ]
    augmented = builder._augment_text_ir(text_ir=text_ir, candidates=candidates)
    groups = augmented.get("control_groups") if isinstance(augmented.get("control_groups"), list) else []
    assert len(groups) >= 2


def test_build_text_ir_extracts_standalone_control_panel() -> None:
    builder = ObsBuilder()
    text_ir = builder.build_text_ir(
        snapshot_html="""
        <html><body>
          <section id="library">
            <h2>Living catalog</h2>
            <select name="genre">
              <option value="">All genres</option>
              <option value="Allegory" selected>Allegory</option>
              <option value="Fantasy">Fantasy</option>
            </select>
            <select name="year">
              <option value="">All years</option>
              <option value="1953" selected>1953</option>
            </select>
            <button type="button">Clear filters</button>
          </section>
        </body></html>
        """
    )
    groups = text_ir.get("control_groups") if isinstance(text_ir.get("control_groups"), list) else []
    assert groups
    controls_blob = " | ".join(" ; ".join(group.get("controls") or []) for group in groups if isinstance(group, dict))
    assert "Allegory" in controls_blob
    assert "Clear filters" in controls_blob


def test_pre_done_verification_allows_informational_answer_from_page_evidence() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    ok, reason = engine._pre_done_verification(
        step_index=2,
        state=AgentState(),
        prompt="Go to the site and tell me the total treasury value",
        text_ir={"page_facts": ["Total Treasury: 2.8K", "Wallet Count: 12"]},
        content="Total Treasury: 2.8K",
        flags={},
    )
    assert ok is True
    assert reason == "ok"


def test_legacy_run_auto_finalizes_informational_task_when_page_fact_is_visible(monkeypatch: Any) -> None:
    monkeypatch.setenv("FSM_DIRECT_LOOP", "0")
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    out = engine.run(
        payload={
            "task_id": "info-1",
            "prompt": "Go to the website and tell me the total treasury value",
            "step_index": 1,
            "url": "https://example.com/treasury",
            "snapshot_html": """
            <html><body>
              <main>
                <h1>Treasury Details</h1>
                <div><span>Total Treasury</span><span>2.8K</span></div>
              </main>
            </body></html>
            """,
            "state_in": {},
            "allowed_tools": [{"name": "browser.navigate"}, {"name": "browser.click"}],
            "history": [],
        }
    )
    assert out.get("done") is True
    assert "2.8K" in str(out.get("content") or "")


def test_pre_done_verification_rejects_vague_informational_answer() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    ok, reason = engine._pre_done_verification(
        step_index=2,
        state=AgentState(),
        prompt="Go to the site and tell me the treasury total value",
        text_ir={"page_facts": ["Total Treasury: 2.8K"]},
        content="The user is on the treasury page, which likely displays the total value.",
        flags={},
    )
    assert ok is False
    assert reason == "vague_informational_answer"


def test_completion_only_returns_done_only_with_concrete_page_evidence() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    incomplete = engine.run(
        payload={
            "task_id": "completion-info-1",
            "prompt": "Go to the site and tell me the total treasury value",
            "step_index": 1,
            "completion_only": True,
            "url": "https://example.com/",
            "snapshot_html": "<html><body><h1>Treasury Details</h1><p>Overview page</p></body></html>",
            "state_in": {},
        }
    )
    assert incomplete.get("done") is False
    assert incomplete.get("content") is None

    complete = engine.run(
        payload={
            "task_id": "completion-info-2",
            "prompt": "Go to the site and tell me the total treasury value",
            "step_index": 1,
            "completion_only": True,
            "url": "https://example.com/treasury",
            "snapshot_html": """
            <html><body>
              <h1>Treasury Details</h1>
              <div><span>Total Treasury</span><span>2.8K</span></div>
            </body></html>
            """,
            "state_in": {},
        }
    )
    assert complete.get("done") is True
    assert "2.8K" in str(complete.get("content") or "")


def test_completion_only_requires_page_context_overlap_for_informational_answer() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    out = engine.run(
        payload={
            "task_id": "completion-info-3",
            "prompt": "Go to the site and tell me the total treasury value",
            "step_index": 1,
            "completion_only": True,
            "url": "https://example.com/",
            "snapshot_html": """
            <html><body>
              <h1>Dashboard</h1>
              <div><span>Total Value Locked</span><span>2844</span></div>
            </body></html>
            """,
            "state_in": {},
        }
    )
    assert out.get("done") is False


def test_completion_only_does_not_finish_on_root_page_even_with_numeric_fact() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    out = engine.run(
        payload={
            "task_id": "completion-info-4",
            "prompt": "Go to the site and tell me the treasury total value",
            "step_index": 1,
            "completion_only": True,
            "url": "https://example.com/",
            "snapshot_html": """
            <html><body>
              <h1>Homepage</h1>
              <div><span>Total Treasury</span><span>2.8K</span></div>
            </body></html>
            """,
            "state_in": {},
        }
    )
    assert out.get("done") is False


def test_browser_end_tool_call_normalizes_to_final_content() -> None:
    def _llm_end(**_: Any) -> Dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"type":"browser","tool_call":{"name":"browser.end","arguments":{"content":"Total Treasury: 2.8K"}}}'
                    }
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "model": "gpt-5.2",
        }

    engine = FSMOperator(llm_call=_llm_end)
    out = engine.run(
        payload={
            "task_id": "browser-end",
            "prompt": "Tell me the treasury total value",
            "step_index": 1,
            "url": "https://example.com/treasury",
            "snapshot_html": """
            <html><body>
              <h1>Treasury</h1>
              <div><span>Total Treasury</span><span>2.8K</span></div>
            </body></html>
            """,
            "state_in": {},
            "include_reasoning": True,
            "allowed_tools": [{"name": "browser.end"}],
        }
    )
    assert out.get("done") is True
    assert out.get("content") == "Total Treasury: 2.8K"


def test_direct_loop_final_reasoning_uses_final_content(monkeypatch: Any) -> None:
    monkeypatch.setenv("FSM_DIRECT_LOOP", "1")

    def _llm_end(**_: Any) -> Dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"type":"final","done":true,"content":"Total Treasury: 2.8K"}'
                    }
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "model": "gpt-5.2",
        }

    engine = FSMOperator(llm_call=_llm_end)
    out = engine.run(
        payload={
            "task_id": "direct-reasoning-final",
            "prompt": "Tell me the treasury total value",
            "step_index": 1,
            "url": "https://example.com/treasury",
            "snapshot_html": """
            <html><body>
              <h1>Treasury</h1>
              <div><span>Total Treasury</span><span>2.8K</span></div>
            </body></html>
            """,
            "state_in": {},
            "include_reasoning": True,
            "allowed_tools": [{"name": "browser.end"}],
        }
    )
    reasoning = str(out.get("reasoning") or "")
    assert out.get("done") is True
    assert "Total Treasury: 2.8K" in reasoning
    assert "Current page answer not clear yet" not in reasoning


def test_direct_loop_does_not_auto_finalize_from_page_evidence(monkeypatch: Any) -> None:
    monkeypatch.setenv("FSM_DIRECT_LOOP", "1")
    engine = FSMOperator(llm_call=_dummy_llm_invalid)
    out = engine.run(
        payload={
            "task_id": "direct-no-autofinal",
            "prompt": "Tell me the treasury total value",
            "step_index": 1,
            "url": "https://example.com/treasury",
            "snapshot_html": """
            <html><body>
              <h1>Treasury</h1>
              <div><span>Total Treasury</span><span>2.8K</span></div>
            </body></html>
            """,
            "state_in": {},
            "include_reasoning": True,
            "allowed_tools": [{"name": "browser.end"}],
        }
    )
    assert out.get("done") is False
    assert out.get("content") is None
