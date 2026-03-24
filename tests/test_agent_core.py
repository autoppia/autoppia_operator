from __future__ import annotations

from agent_test_support import (
    _TASK_STATE,
    _all_subgoals_done,
    _Candidate,
    _ensure_subgoal_memory,
    _extract_focus_terms,
    _host_label,
    _pick_fallback_candidate_id,
    _render_observation_result,
    _split_task_subgoals,
    _text_quality_score,
    _tool_extract_entities,
    _tool_extract_tables,
    _update_subgoal_memory,
)


def test_split_task_subgoals_creates_multiple_steps() -> None:
    task = "Go to autoppia.com and open Studio, then finish."
    out = _split_task_subgoals(task)
    assert len(out) >= 2
    assert any("studio" in x.lower() for x in out)


def test_subgoal_memory_marks_done_from_url_and_text() -> None:
    mem = _ensure_subgoal_memory("t-subgoal-1", "Go to autoppia.com and open studio")
    assert mem is not None
    _update_subgoal_memory(
        mem,
        step_index=0,
        url="https://autoppia.com/",
        page_ir_text="TITLE: Autoppia",
        page_summary="Home page",
        history=[],
        repeat_count=0,
    )
    # First subgoal should be done after reaching host.
    done_count = sum(1 for sg in (mem.get("subgoals") or []) if isinstance(sg, dict) and sg.get("done"))
    assert done_count >= 1

    _update_subgoal_memory(
        mem,
        step_index=1,
        url="https://app.autoppia.com/studio",
        page_ir_text="PATH: /studio",
        page_summary="Studio page",
        history=[],
        repeat_count=0,
    )
    assert _all_subgoals_done(mem) is True


def test_extract_tables_and_entities_tools() -> None:
    html = """
    <html><body>
      <table>
        <tr><th>Name</th><th>Price</th></tr>
        <tr><td>Item A</td><td>$12.50</td></tr>
      </table>
      Contact support@example.com on 2026-02-21
    </body></html>
    """
    tables = _tool_extract_tables(html=html, max_tables=2, max_rows=3, max_cols=3)
    assert tables.get("ok") is True
    assert tables.get("count", 0) >= 1
    first = (tables.get("tables") or [])[0]
    assert "headers" in first and "rows" in first

    ents = _tool_extract_entities(html=html, max_items=10)
    assert ents.get("ok") is True
    entities = ents.get("entities") or {}
    assert any("support@example.com" in e for e in (entities.get("emails") or []))
    assert any("$12.50" in p for p in (entities.get("prices") or []))


def test_pick_fallback_candidate_prefers_matching_click_target() -> None:
    c0 = _Candidate(
        selector={
            "type": "attributeValueSelector",
            "attribute": "href",
            "value": "https://autoppia.com/docs",
            "case_sensitive": False,
        },
        text="Docs",
        tag="a",
        attrs={"href": "https://autoppia.com/docs"},
    )
    c1 = _Candidate(
        selector={
            "type": "attributeValueSelector",
            "attribute": "href",
            "value": "https://autoppia.com/studio",
            "case_sensitive": False,
        },
        text="Studio",
        tag="a",
        attrs={"href": "https://autoppia.com/studio"},
    )
    decision = {
        "action": "click",
        "url": "https://autoppia.com/studio",
        "text": "open studio",
    }
    picked = _pick_fallback_candidate_id(candidates=[c0, c1], action="click", decision=decision)
    assert picked == 1


def test_text_quality_score_penalizes_numeric_noise() -> None:
    noisy = "2259 0.004253 0.00% 13/129 10.1% 7674808 315.20% 129 96 99 54 63 62 71"
    clean = "The current portfolio value is shown in the header as T 399,29."
    assert _text_quality_score(noisy) < _text_quality_score(clean)


def test_host_label_is_generic_without_domain_hardcoding() -> None:
    assert _host_label("docs.example-platform.io") == "Example Platform"
    assert _host_label("app.service.co") == "Service"


def test_render_observation_result_prefers_relevant_clean_evidence() -> None:
    task_id = "t-observation-generic"
    _TASK_STATE[task_id] = {
        "observations": [
            {
                "url": "https://example.com/dashboard",
                "title": "Dashboard",
                "headings": ["Treasury", "Portfolio Overview"],
                "facts": [
                    "2259 0.004253 0.00% 13/129 10.1% 7674808 315.20%",
                    "Treasury value is T 399,29 at current block.",
                ],
                "summary": "Numbers table",
            }
        ]
    }
    out = _render_observation_result("Go to example.com and tell me portfolio treasury value", task_id)
    assert isinstance(out, str)
    assert "treasury value" in out.lower()
    assert "2259 0.004253" not in out.lower()
    _TASK_STATE.pop(task_id, None)


def test_extract_focus_terms_removes_common_stopwords() -> None:
    terms = _extract_focus_terms("Please go to the website and tell me the portfolio treasury value")
    assert "portfolio" in terms
    assert "treasury" in terms
    assert "please" not in terms
