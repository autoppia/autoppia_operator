from __future__ import annotations

from src.operator import fsm_operator, step_engine
from src.operator.agents import StepEngine
from src.operator.agents.step_engine import CanonicalBrowserState
from src.operator.agents.step_engine.candidates import Candidate, CandidateExtractor
from src.operator.agents.step_engine.state import AgentState
from src.operator.agents.step_engine.utils import clean_snapshot_html_for_llm


def test_step_engine_is_canonical_runtime() -> None:
    assert step_engine.StepEngine is StepEngine
    assert step_engine._STEP_ENGINE is not None


def test_fsm_runtime_is_compatibility_alias() -> None:
    assert fsm_operator.FSMOperator is step_engine.StepEngine
    assert fsm_operator._FSM_OPERATOR is step_engine._STEP_ENGINE


def test_canonical_browser_state_serializes_cleanly() -> None:
    state = CanonicalBrowserState(
        task_id="task-1",
        prompt="log in",
        url="https://example.com/login",
        snapshot_html="<html></html>",
        step_index=2,
        include_reasoning=True,
        history=[{"tool_calls": [{"name": "browser.click", "arguments": {"index": 1}}]}],
    )
    payload = state.to_payload()
    assert payload["task_id"] == "task-1"
    assert payload["step_index"] == 2
    assert payload["include_reasoning"] is True


def test_guard_redundant_type_action_advances_to_submit_when_field_is_already_filled() -> None:
    engine = StepEngine(llm_call=lambda **_: {})
    selector_username = {
        "type": "attributeValueSelector",
        "attribute": "id",
        "value": "login-username",
        "case_sensitive": False,
    }
    selector_submit = {
        "type": "attributeValueSelector",
        "attribute": "id",
        "value": "signin-control",
        "case_sensitive": False,
    }
    username = Candidate(
        id="username-input",
        role="input",
        type="input",
        text="Username",
        href="",
        context="Login form",
        selector=selector_username,
        dom_path="html/body/form/input[1]",
        field_hint="Username",
        field_kind="username",
        current_value="user1",
        group_id="login-form",
        group_label="Login",
    )
    submit = Candidate(
        id="login-submit",
        role="button",
        type="button",
        text="Sign in",
        href="",
        context="Login form",
        selector=selector_submit,
        dom_path="html/body/form/button[1]",
        field_kind="auth_entry",
        group_id="login-form",
        group_label="Login",
    )
    state = AgentState()
    state.form_progress.typed_candidate_ids = ["username-input"]
    state.form_progress.typed_values_by_candidate = {"username-input": "user1"}
    action = {
        "type": "TypeAction",
        "selector": selector_username,
        "text": "user1",
        "_element_id": "username-input",
    }

    guarded = engine._guard_redundant_type_action(
        action=action,
        prompt="Log in using username equals <username> and password equals <password>.",
        history=[],
        ranked_candidates=[username, submit],
        state=state,
    )

    assert guarded is not None
    assert guarded["type"] == "ClickAction"
    assert guarded["_element_id"] == "login-submit"


def test_candidate_extractor_prefers_stable_attribute_selectors_before_xpath() -> None:
    extractor = CandidateExtractor()
    html = """
    <html><body>
      <button aria-label="Open account menu">Menu</button>
      <input placeholder="Search films" />
    </body></html>
    """

    candidates = extractor.extract(snapshot_html=html, url="https://example.com")

    button = next(c for c in candidates if c.role == "button")
    field = next(c for c in candidates if c.role == "input")
    assert button.selector["type"] == "attributeValueSelector"
    assert button.selector["attribute"] == "aria-label"
    assert button.selector["value"] == "Open account menu"
    assert field.selector["type"] == "attributeValueSelector"
    assert field.selector["attribute"] == "placeholder"
    assert field.selector["value"] == "Search films"


def test_direct_loop_login_prefers_seeded_login_navigation() -> None:
    engine = StepEngine(llm_call=lambda **_: {})
    out = engine.run(
        payload={
            "task_id": "login-hard-seed",
            "prompt": "First, authenticate with username 'user1' and password 'Passw0rd!' to log in successfully.",
            "web_project_id": "autocinema",
            "use_case": {"name": "LOGIN"},
            "url": "http://84.247.180.192:8000/?seed=252",
            "snapshot_html": "<html><body><a id='featured-movie-view-details-btn-2' href='/movies/real-movie-064?seed=252'>View Details</a></body></html>",
            "step_index": 0,
            "history": [],
            "include_reasoning": True,
        }
    )
    actions = out.get("actions") or []
    assert actions
    first = actions[0]
    assert first["type"] == "NavigateAction"
    assert first["url"].endswith("/login?seed=252")


def test_blank_page_fallback_failure_sets_error_and_failure_reason() -> None:
    engine = StepEngine(llm_call=lambda **_: {})
    out = engine.run(
        payload={
            "task_id": "blank-fail",
            "prompt": "Continue with the task.",
            "url": "about:blank",
            "snapshot_html": "<html><body></body></html>",
            "step_index": 0,
            "history": [],
            "include_reasoning": True,
        }
    )
    assert out["done"] is True
    assert out["error"] == "blank_page_no_navigation_target"
    assert out["failure_reason"] == "blank_page_no_navigation_target"
    assert "blank page" in str(out["content"]).lower()


def test_clean_snapshot_html_strips_scripts_styles_and_comments() -> None:
    raw = """<!-- meta -->
    <html><head><style>.x{color:red}</style></head><body>
    <script>alert(1)</script><p id="ok">Hi</p></body></html>"""
    out = clean_snapshot_html_for_llm(raw)
    assert "alert" not in out
    assert ".x{" not in out
    assert "<!--" not in out
    assert 'id="ok"' in out


def test_clean_snapshot_html_truncates_long_data_uri_in_src() -> None:
    blob = "data:image/png;base64," + ("A" * 1200)
    raw = f'<html><body><img src="{blob}" /><button id="go">Go</button></body></html>'
    out = clean_snapshot_html_for_llm(raw)
    assert "[data-uri-truncated]" in out
    assert "AAAA" not in out
    assert 'id="go"' in out


def test_clean_snapshot_html_truncates_total_length(monkeypatch) -> None:
    monkeypatch.setenv("FSM_SNAPSHOT_HTML_MAX_CHARS", "9000")
    raw = "<html><body>" + ("x" * 20_000) + "</body></html>"
    out = clean_snapshot_html_for_llm(raw)
    assert len(out) <= 9500
    assert "fsm:snapshot_html_truncated" in out
