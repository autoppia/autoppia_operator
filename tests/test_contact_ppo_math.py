import json

from training.rl.contact_ppo_trainer import (
    PPOTransition,
    TeacherStep,
    _compute_gae,
    _load_teacher_steps,
    _teacher_reward_delta,
)


def test_compute_gae_shapes_returns() -> None:
    transitions = [
        PPOTransition(seed=1, step_idx=0, prompt_ids=[1], completion_ids=[2], old_logprob=-1.0, value=0.2, reward=0.0, done=False, before_url="", after_url="", chosen_action=None),
        PPOTransition(seed=1, step_idx=1, prompt_ids=[1], completion_ids=[2], old_logprob=-1.0, value=0.1, reward=1.0, done=True, before_url="", after_url="", chosen_action=None),
    ]
    adv, rets = _compute_gae(transitions, gamma=0.99, lam=0.95)
    assert len(adv) == 2
    assert len(rets) == 2
    assert rets[1] > 0.9


def test_teacher_reward_delta_prefers_matching_route() -> None:
    teacher = TeacherStep(seed=1, step_idx=0, tool_name="browser.navigate", route_hint="/contact?seed=1")
    delta, reason = _teacher_reward_delta(
        teacher,
        {
            "type": "ClickAction",
            "selector": {"attribute": "href", "value": "/contact?seed=1"},
        },
    )
    assert delta > 0.0
    assert "match" in reason


def test_load_teacher_steps_uses_ordinal_when_step_idx_missing(tmp_path) -> None:
    payload = {
        "episodes": [
            {
                "guided_execution": [
                    {"policy_tool_call": {"name": "browser.navigate", "arguments": {"url": "http://x/contact?seed=1"}}},
                    {"policy_tool_call": {"name": "browser.input", "arguments": {"index": 3, "text": "David"}}},
                ]
            }
        ]
    }
    p = tmp_path / "seed_0001_fake.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    class DummyTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [1, 2, 3]}

    teacher = _load_teacher_steps([str(p)], DummyTokenizer())
    assert teacher[(1, 0)].tool_name == "browser.navigate"
    assert teacher[(1, 1)].tool_name == "browser.input"
