from __future__ import annotations

from typing import Any, Dict

from fsm_operator import FSMOperator, MAX_INTERNAL_META_STEPS


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


def test_meta_tool_loop_is_capped() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_meta_loop)
    out = engine.run(payload=_base_payload())
    st = out.get("state_out") or {}
    counters = st.get("counters") if isinstance(st.get("counters"), dict) else {}
    assert int(counters.get("meta_steps_used") or 0) == MAX_INTERNAL_META_STEPS


def test_stuck_recovery_triggers_with_loop_signals() -> None:
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
    assert actions[0].get("type") == "ScrollAction"
    st = out.get("state_out") or {}
    blocked = st.get("blocklist", {}).get("element_ids") if isinstance(st.get("blocklist"), dict) else []
    assert "el_repeat" in (blocked or [])


def test_done_and_content_emitted_without_report_result_action() -> None:
    engine = FSMOperator(llm_call=_dummy_llm_final)
    payload = _base_payload()
    payload["include_reasoning"] = True
    out = engine.run(payload=payload)
    assert out.get("done") is True
    assert isinstance(out.get("content"), str) and out.get("content")
    assert out.get("actions") == []
