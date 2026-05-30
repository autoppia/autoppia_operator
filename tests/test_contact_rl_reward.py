from training.rl.reward import compute_step_reward


def test_reward_penalizes_repeat_and_off_route() -> None:
    action = {"type": "ClickAction", "_element_id": "a1", "selector": {"type": "attributeValueSelector", "attribute": "id", "value": "foo"}}
    breakdown = compute_step_reward(
        prev_score=0.0,
        current_score=0.0,
        success=False,
        exec_ok=True,
        current_url="http://84.247.180.192:8000/register?seed=10",
        chosen_action=action,
        previous_action=action,
    )
    assert breakdown.repeat_penalty > 0.0
    assert breakdown.off_route_penalty > 0.0


def test_reward_adds_success_and_progress() -> None:
    breakdown = compute_step_reward(
        prev_score=0.2,
        current_score=1.0,
        success=True,
        exec_ok=True,
        current_url="http://84.247.180.192:8000/contact?seed=10",
        chosen_action={"type": "ClickAction"},
        previous_action=None,
    )
    assert breakdown.terminal_success == 1.0
    assert breakdown.score_delta > 0.0
    assert breakdown.total > 1.0


def test_reward_uses_task_prompt_hints() -> None:
    task_prompt = (
        "Navigate directly to /contact. "
        "Use canonical field ids: contact-name-input, contact-email-input, send-message-button. "
        "Do not visit registration or unrelated pages."
    )
    good_breakdown = compute_step_reward(
        prev_score=0.0,
        current_score=0.0,
        success=False,
        exec_ok=True,
        current_url="http://84.247.180.192:8000/",
        chosen_action={"type": "NavigateAction", "url": "http://84.247.180.192:8000/contact?seed=10"},
        previous_action=None,
        task_prompt=task_prompt,
    )
    bad_breakdown = compute_step_reward(
        prev_score=0.0,
        current_score=0.0,
        success=False,
        exec_ok=True,
        current_url="http://84.247.180.192:8000/",
        chosen_action={"type": "TypeAction", "selector": {"type": "attributeValueSelector", "attribute": "id", "value": "input"}, "text": "David"},
        previous_action=None,
        task_prompt=task_prompt,
    )
    assert good_breakdown.task_hint_bonus > 0.0
    assert bad_breakdown.task_hint_penalty > 0.0
    assert good_breakdown.total > bad_breakdown.total


def test_reward_penalizes_leaving_primary_route() -> None:
    task_prompt = (
        "Follow this route exactly: /contact. "
        "Click the send/submit button once after all fields are filled."
    )
    breakdown = compute_step_reward(
        prev_score=0.0,
        current_score=0.0,
        success=False,
        exec_ok=True,
        current_url="http://84.247.180.192:8000/submit?seed=1",
        previous_url="http://84.247.180.192:8000/contact?seed=1",
        chosen_action={"type": "NavigateAction", "url": "http://84.247.180.192:8000/submit?seed=1"},
        previous_action=None,
        task_prompt=task_prompt,
    )
    assert breakdown.task_hint_bonus == 0.0
    assert breakdown.off_route_penalty >= 0.30


def test_reward_bonus_for_correct_field_value_alignment() -> None:
    task_prompt = (
        "Fill out the contact form with a name that equals 'David', "
        "an email that contains 'user1@site.com', "
        "a subject that does NOT contain 'Information', "
        "and a message that equals 'Please provide me with more information'."
    )
    good = compute_step_reward(
        prev_score=0.0,
        current_score=0.0,
        success=False,
        exec_ok=True,
        current_url="http://84.247.180.192:8000/contact?seed=1",
        before_html="<input id='contact-email-input' value=''>",
        after_html="<input id='contact-email-input' value='user1@site.com'>",
        chosen_action={
            "type": "TypeAction",
            "selector": {"type": "attributeValueSelector", "attribute": "id", "value": "contact-email-input"},
            "text": "user1@site.com",
        },
        previous_action=None,
        task_prompt=task_prompt,
    )
    bad = compute_step_reward(
        prev_score=0.0,
        current_score=0.0,
        success=False,
        exec_ok=True,
        current_url="http://84.247.180.192:8000/contact?seed=1",
        before_html="<input id='contact-subject-input' value=''>",
        after_html="<input id='contact-subject-input' value='user1@site.com'>",
        chosen_action={
            "type": "TypeAction",
            "selector": {"type": "attributeValueSelector", "attribute": "id", "value": "contact-subject-input"},
            "text": "user1@site.com",
        },
        previous_action=None,
        task_prompt=task_prompt,
    )
    assert good.field_alignment_bonus > bad.field_alignment_bonus
    assert good.total > bad.total


def test_reward_prefers_submit_once_form_targets_are_present() -> None:
    task_prompt = (
        "Fill out the contact form with a name that equals 'David', "
        "an email that contains 'user1@site.com', "
        "a subject that does NOT contain 'Information', "
        "and a message that equals 'Please provide me with more information'."
    )
    complete_html = (
        "<input id='contact-name-input' value='David'>"
        "<input id='contact-email-input' value='user1@site.com'>"
        "<input id='contact-subject-input' value='Inquiry'>"
        "<textarea id='contact-message-textarea'>Please provide me with more information</textarea>"
        "<button id='send-message-button'>Send</button>"
    )
    submit = compute_step_reward(
        prev_score=0.0,
        current_score=0.0,
        success=False,
        exec_ok=True,
        current_url="http://84.247.180.192:8000/contact?seed=1",
        before_html=complete_html,
        after_html=complete_html,
        chosen_action={
            "type": "ClickAction",
            "selector": {"type": "attributeValueSelector", "attribute": "id", "value": "send-message-button"},
        },
        previous_action=None,
        task_prompt=task_prompt,
    )
    rewrite = compute_step_reward(
        prev_score=0.0,
        current_score=0.0,
        success=False,
        exec_ok=True,
        current_url="http://84.247.180.192:8000/contact?seed=1",
        before_html=complete_html,
        after_html=complete_html,
        chosen_action={
            "type": "TypeAction",
            "selector": {"type": "attributeValueSelector", "attribute": "id", "value": "contact-message-textarea"},
            "text": "Please provide me with more information",
        },
        previous_action=None,
        task_prompt=task_prompt,
    )
    assert submit.submit_readiness_bonus > 0.0
    assert rewrite.redundant_field_penalty > 0.0
    assert submit.total > rewrite.total


def test_reward_strongly_prefers_primary_route_navigation_early() -> None:
    task_prompt = "Navigate directly to /contact and stay on the contact workflow only."
    navigate = compute_step_reward(
        step_index=0,
        prev_score=0.0,
        current_score=0.0,
        success=False,
        exec_ok=True,
        current_url="http://84.247.180.192:8000/?seed=1",
        chosen_action={"type": "NavigateAction", "url": "http://84.247.180.192:8000/contact?seed=1"},
        previous_action=None,
        task_prompt=task_prompt,
    )
    type_generic = compute_step_reward(
        step_index=0,
        prev_score=0.0,
        current_score=0.0,
        success=False,
        exec_ok=True,
        current_url="http://84.247.180.192:8000/?seed=1",
        chosen_action={
            "type": "TypeAction",
            "selector": {"type": "attributeValueSelector", "attribute": "id", "value": "input"},
            "text": "David",
        },
        previous_action=None,
        task_prompt=task_prompt,
    )
    assert navigate.task_hint_bonus > type_generic.task_hint_bonus
    assert type_generic.task_hint_penalty > navigate.task_hint_penalty
    assert navigate.total > type_generic.total


def test_reward_penalizes_premature_noop_before_primary_route() -> None:
    task_prompt = "Navigate directly to /contact and stay on the contact workflow only."
    noop = compute_step_reward(
        step_index=0,
        prev_score=0.0,
        current_score=0.0,
        success=False,
        exec_ok=True,
        current_url="http://84.247.180.192:8000/?seed=1",
        chosen_action=None,
        previous_action=None,
        task_prompt=task_prompt,
    )
    navigate = compute_step_reward(
        step_index=0,
        prev_score=0.0,
        current_score=0.0,
        success=False,
        exec_ok=True,
        current_url="http://84.247.180.192:8000/?seed=1",
        chosen_action={"type": "NavigateAction", "url": "http://84.247.180.192:8000/contact?seed=1"},
        previous_action=None,
        task_prompt=task_prompt,
    )
    assert noop.noop_penalty >= 0.35
    assert noop.total < navigate.total
