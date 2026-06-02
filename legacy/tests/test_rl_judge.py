from training.rl.judge import _parse_json_obj, judge_step_progress


def test_parse_json_obj_handles_plain_json() -> None:
    obj = _parse_json_obj('{"reward_delta": 0.5, "confidence": 0.8, "reason": "good"}')
    assert obj["reward_delta"] == 0.5
    assert obj["confidence"] == 0.8


def test_parse_json_obj_handles_wrapped_text() -> None:
    obj = _parse_json_obj('text before {"reward_delta": -1, "confidence": 1, "reason": "bad"} text after')
    assert obj["reward_delta"] == -1
    assert obj["reason"] == "bad"


def test_judge_step_progress_returns_zero_when_api_key_missing(monkeypatch) -> None:
    monkeypatch.setenv("CONTACT_RL_JUDGE_ENABLED", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    score = judge_step_progress(
        task_id="t1",
        task_prompt="Go to /contact and submit the form",
        step_index=0,
        before_url="http://example.com/",
        after_url="http://example.com/",
        before_html="<html></html>",
        after_html="<html></html>",
        action={"type": "TypeAction", "text": "David"},
        base_reward=-1.0,
    )
    assert score.reward_delta == 0.0
    assert score.confidence == 0.0
    assert score.reason == "judge_missing_api_key"
