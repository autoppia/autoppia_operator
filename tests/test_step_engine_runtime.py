from __future__ import annotations

from src.operator import fsm_operator, step_engine
from src.operator.agents import StepEngine
from src.operator.agents.step_engine.candidates import Candidate
from src.operator.agents.step_engine.candidates import CandidateExtractor
from src.operator.agents.step_engine import CanonicalBrowserState
from src.operator.agents.step_engine.state import AgentState


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
