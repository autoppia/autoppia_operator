from __future__ import annotations

from src.operator.agents.step_engine import policy as policy_module


def test_infer_autocinema_use_case_prefers_prompt_signal() -> None:
    use_case = policy_module._infer_autocinema_use_case(
        "Register a new Autocinema user and then continue.",
        {},
    )
    assert use_case == "REGISTRATION"


def test_infer_autocinema_use_case_prefers_specific_watchlist_and_share_signals() -> None:
    assert (
        policy_module._infer_autocinema_use_case(
            "Remove from watchlist the movie titled 'The Matrix'.",
            {},
        )
        == "REMOVE_FROM_WATCHLIST"
    )
    assert (
        policy_module._infer_autocinema_use_case(
            "Share the current movie with a friend.",
            {},
        )
        == "SHARE_MOVIE"
    )


def test_autocinema_example_block_includes_matching_examples(monkeypatch: object) -> None:
    monkeypatch.setattr(
        policy_module,
        "_autocinema_success_examples",
        lambda: [
            {
                "use_case": "LOGIN",
                "url_path": "/login",
                "step_index": 1,
                "prompt": "Log in with the provided credentials.",
                "tool_calls": [
                    {"name": "browser.input", "arguments": {"index": 0, "text": "<username>"}},
                    {"name": "browser.input", "arguments": {"index": 1, "text": "<password>"}},
                ],
            },
            {
                "use_case": "CONTACT",
                "url_path": "/contact",
                "step_index": 0,
                "prompt": "Open the contact form.",
                "tool_calls": [{"name": "browser.click", "arguments": {"index": 0}}],
            },
        ],
    )

    block = policy_module._autocinema_example_block(
        "Please log in with the provided credentials.",
        {
            "url": "https://autocinema.example/login",
            "step_index": 1,
        },
    )

    joined = "\n".join(block)
    assert "RETRIEVED SUCCESSFUL AUTOCINEMA EXAMPLES:" in joined
    assert "use_case=LOGIN" in joined
    assert "<username>" in joined
    assert "use_case=CONTACT" not in joined


def test_obs_candidate_intent_tags_treat_active_watchlist_toggle_as_remove() -> None:
    tags = policy_module._obs_candidate_intent_tags(
        {
            "text": "In Watchlist",
            "field_hint": "Watchlist",
            "ui_state": "active",
            "context": "Watch trailer In Watchlist Share",
        }
    )
    assert "watchlist_remove" in tags
    assert "watchlist_add" not in tags


def test_direct_intent_prefers_local_share_control_over_detail_or_comment() -> None:
    tool_call = policy_module._preferred_direct_intent_action(
        "Share the current movie with a friend.",
        {
            "url": "https://autocinema.example/movies/interstellar?seed=4242",
            "candidates": [
                {
                    "index": 0,
                    "id": "comment-name",
                    "role": "input",
                    "text": "Your name",
                    "field_hint": "Name",
                    "context": "Add a Note Share your thoughts about this film",
                },
                {
                    "index": 1,
                    "id": "view-details",
                    "role": "link",
                    "text": "View details",
                    "href": "/movies/interstellar?seed=4242",
                    "context": "Interstellar View details",
                },
                {
                    "index": 2,
                    "id": "share-movie",
                    "role": "button",
                    "text": "Share",
                    "field_hint": "Share",
                    "context": "Watch trailer Add to watchlist Share",
                },
            ],
        },
        allowed_tools={"browser.click"},
    )
    assert tool_call is not None
    action, chosen = tool_call
    assert action["arguments"]["index"] == 2
    assert chosen["id"] == "share-movie"


def test_preferred_seed_navigation_routes_title_focused_task_to_search() -> None:
    action = policy_module._preferred_seed_stable_navigation(
        "Add to wishlist a movie where the name equals 'The Incredibles'",
        {
            "url": "https://autocinema.example/?seed=31000",
            "page_observations": {"capability_gap": {}},
        },
        allowed_tools={"browser.navigate", "browser.click"},
    )
    assert action is not None
    tool_call = action["tool_call"]
    assert tool_call["name"] == "browser.navigate"
    target_url = tool_call["arguments"]["url"]
    assert target_url.startswith("https://autocinema.example/?")
    assert "seed=31000" in target_url
    assert "search=The+Incredibles" in target_url


def test_preferred_seed_navigation_uses_capability_gap_for_mutation_task() -> None:
    action = policy_module._preferred_seed_stable_navigation(
        "Add a film where the cast equals 'cosmic' and the rating equals 4.1.",
        {
            "url": "https://autocinema.example/?seed=1000",
            "page_observations": {
                "capability_gap": {
                    "read_only_for_task": True,
                    "preferred_transition": "login",
                }
            },
        },
        allowed_tools={"browser.navigate", "browser.click"},
    )
    assert action is not None
    tool_call = action["tool_call"]
    assert tool_call["name"] == "browser.navigate"
    assert tool_call["arguments"]["url"].endswith("/login?seed=1000")


def test_preferred_seed_navigation_uses_login_for_watchlist_when_detail_page_is_read_only() -> None:
    action = policy_module._preferred_seed_stable_navigation(
        "Add to wishlist a movie where the name equals 'The Incredibles'",
        {
            "url": "https://autocinema.example/movies/real-movie-050?seed=31000",
            "page_observations": {
                "capability_gap": {
                    "read_only_for_task": True,
                    "preferred_transition": "login",
                }
            },
        },
        allowed_tools={"browser.navigate", "browser.click"},
    )
    assert action is not None
    tool_call = action["tool_call"]
    assert tool_call["name"] == "browser.navigate"
    assert tool_call["arguments"]["url"].endswith("/login?seed=31000")


def test_preferred_title_result_action_anchors_to_matching_movie_card() -> None:
    action = policy_module._preferred_title_result_action(
        "Add to watchlist a movie where the name equals 'The Incredibles'",
        {
            "url": "https://autocinema.example/?seed=31000&search=The+Incredibles",
            "candidates": [
                {
                    "index": 0,
                    "role": "link",
                    "text": "Contact",
                    "href": "#contact",
                    "context": "Header Contact",
                },
                {
                    "index": 1,
                    "role": "link",
                    "text": "The Incredibles",
                    "href": "/movies/the-incredibles?seed=31000",
                    "context": "Movie card View detail",
                },
            ],
        },
        allowed_tools={"browser.click"},
    )
    assert action is not None
    assert action["tool_call"]["name"] == "browser.click"
    assert action["tool_call"]["arguments"]["index"] == 1


def test_preferred_title_result_action_skips_generic_home_and_related_cards() -> None:
    action = policy_module._preferred_title_result_action(
        "Add to watchlist a movie where the name equals 'The Incredibles'",
        {
            "url": "https://autocinema.example/?seed=31000&search=The+Incredibles",
            "candidates": [
                {
                    "index": 0,
                    "role": "link",
                    "text": "Autocinema",
                    "href": "/?seed=31000",
                    "context": "The Incredibles header logo",
                },
                {
                    "index": 1,
                    "id": "related-card-3",
                    "role": "link",
                    "text": "The Incredibles",
                    "href": "/movies/other-title?seed=31000",
                    "context": "Related movie The Incredibles",
                },
                {
                    "index": 2,
                    "role": "link",
                    "text": "The Incredibles",
                    "href": "/movies/the-incredibles?seed=31000",
                    "context": "Movie card View detail",
                },
            ],
        },
        allowed_tools={"browser.click"},
    )
    assert action is not None
    assert action["tool_call"]["arguments"]["index"] == 2


def test_preferred_direct_intent_action_from_markup_uses_detail_controls_when_candidates_are_missing() -> None:
    action = policy_module._preferred_direct_intent_action_from_markup(
        "Add to watchlist the current movie.",
        {
            "url": "https://autocinema.example/movies/interstellar?seed=4242",
            "snapshot_html": """
                <div>
                  <button id="play-trailer">Watch trailer</button>
                  <button id="add-list-btn">Add to watchlist</button>
                  <button id="share-widget">Share</button>
                </div>
            """,
        },
        allowed_tools={"browser.click"},
    )
    assert action is not None
    assert action["tool_call"]["name"] == "browser.click"
    assert action["tool_call"]["arguments"]["selector"]["attribute"] == "id"
    assert action["tool_call"]["arguments"]["selector"]["value"] == "add-list-btn"


def test_extract_prompt_field_targets_prefers_task_section_over_html_noise() -> None:
    prompt = """TASK: Fill out the contact form with a name that equals 'David', an email that contains 'user1@site.com', and a message that equals 'Please provide me with more information'. Navigate directly to /contact and use ids contact-name-input and contact-email-input.

TASK CONSTRAINTS:
{"follow_this_route_exactly": "/contact"}

PAGE GROUPS (JSON):
{"forms":[{"controls":[{"id":"contact-name-input","label":"Name","placeholder":"Your name"},{"id":"contact-email-input","label":"Email","placeholder":"you@example.com"}]}]}

INTERACTIVE ELEMENT SHORTLIST (JSON):
[{"selector":{"value":"contact-subject-input"},"context":"Subject","field_kind":"email","placeholder":"What's this about?"}]
"""
    targets = policy_module._extract_prompt_field_targets(prompt)
    assert targets["name"] == "David"
    assert targets["email"] == "user1@site.com"
    assert targets["message"] == "Please provide me with more information"
    assert "href" not in targets
    assert "id" not in targets


def test_preferred_prompt_form_action_uses_task_targets_before_noisy_shortlist() -> None:
    prompt = """TASK: Fill out the contact form with a name that equals 'David', an email that contains 'user1@site.com', and a message that equals 'Please provide me with more information'. Navigate directly to /contact and use ids contact-name-input, contact-email-input, contact-message-textarea, send-message-button.

TASK CONSTRAINTS:
{"follow_this_route_exactly": "/contact"}
"""
    policy_obs = {
        "url": "https://autocinema.example/contact?seed=4242",
        "page_groups": {
            "forms": [
                {
                    "controls": [
                        {"tag": "input", "type": "", "id": "contact-name-input", "label": "Name", "value": ""},
                        {"tag": "input", "type": "email", "id": "contact-email-input", "label": "Email", "value": ""},
                        {"tag": "input", "type": "", "id": "contact-subject-input", "label": "Subject", "value": ""},
                        {"tag": "textarea", "type": "", "id": "contact-message-textarea", "label": "Message", "value": ""},
                        {"tag": "button", "type": "submit", "id": "send-message-button", "text": "Send Message"},
                    ]
                }
            ]
        },
        "candidates": [
            {
                "index": 0,
                "selector": {"type": "attributeValueSelector", "attribute": "id", "value": "contact-subject-input"},
                "context": "Subject",
                "field_kind": "email",
            },
            {
                "index": 1,
                "selector": {"type": "attributeValueSelector", "attribute": "id", "value": "contact-email-input"},
                "context": "Email",
                "field_kind": "email",
            },
            {
                "index": 2,
                "selector": {"type": "attributeValueSelector", "attribute": "id", "value": "contact-message-textarea"},
                "context": "Message",
                "field_kind": "email",
            },
            {
                "index": 3,
                "selector": {"type": "attributeValueSelector", "attribute": "id", "value": "contact-name-input"},
                "context": "Name",
                "field_kind": "name",
            },
        ],
    }
    action = policy_module._preferred_prompt_form_action(
        prompt,
        policy_obs,
        allowed_tools={"browser.input", "browser.click"},
    )
    assert action == {
        "name": "browser.input",
        "arguments": {
            "text": "David",
            "selector": {
                "type": "attributeValueSelector",
                "attribute": "id",
                "value": "contact-name-input",
                "case_sensitive": False,
            },
        },
    }


def test_preferred_prompt_form_action_login_prefers_username_then_password_then_submit() -> None:
    prompt = "TASK: Navigate directly to /login first. Fill the username field with user1, then fill the password field with Passw0rd!, then click the visible sign in button."
    policy_obs = {
        "url": "https://autocinema.example/login?seed=999",
        "candidates": [
            {
                "selector": {"type": "attributeValueSelector", "attribute": "id", "value": "signin-control", "case_sensitive": False},
                "text": "Sign In",
                "context": "Login form submit button",
            },
            {
                "selector": {"type": "attributeValueSelector", "attribute": "id", "value": "password-entry-field", "case_sensitive": False},
                "text": "Password",
                "context": "Login password field",
                "field_kind": "password",
            },
            {
                "selector": {"type": "attributeValueSelector", "attribute": "id", "value": "login-username", "case_sensitive": False},
                "text": "Username",
                "context": "Login username field",
                "field_kind": "username",
            },
        ],
    }
    first = policy_module._preferred_prompt_form_action(prompt, policy_obs, allowed_tools={"browser.input", "browser.click"})
    assert first == {
        "name": "browser.input",
        "arguments": {
            "text": "user1",
            "selector": {
                "type": "attributeValueSelector",
                "attribute": "id",
                "value": "login-username",
                "case_sensitive": False,
            },
        },
    }
    policy_obs["candidates"][2]["current_value"] = "user1"
    second = policy_module._preferred_prompt_form_action(prompt, policy_obs, allowed_tools={"browser.input", "browser.click"})
    assert second["arguments"]["selector"]["value"] == "password-entry-field"
    policy_obs["candidates"][1]["current_value"] = "Passw0rd!"
    third = policy_module._preferred_prompt_form_action(prompt, policy_obs, allowed_tools={"browser.input", "browser.click"})
    assert third == {
        "name": "browser.click",
        "arguments": {
            "selector": {
                "type": "attributeValueSelector",
                "attribute": "id",
                "value": "signin-control",
                "case_sensitive": False,
            }
        },
    }


def test_preferred_prompt_form_action_contact_prefers_explicit_order_before_submit() -> None:
    prompt = "TASK: Fill out the contact form with a name that equals 'David', an email that contains 'user1@site.com', a subject that does NOT contain 'Information', and a message that equals 'Please provide me with more information'."
    policy_obs = {
        "url": "https://autocinema.example/contact?seed=999",
        "candidates": [
            {"selector": {"type": "attributeValueSelector", "attribute": "id", "value": "send-message-button", "case_sensitive": False}, "text": "Send Message", "context": "Contact form submit"},
            {"selector": {"type": "attributeValueSelector", "attribute": "id", "value": "contact-message-textarea", "case_sensitive": False}, "text": "Message", "context": "Contact form message"},
            {"selector": {"type": "attributeValueSelector", "attribute": "id", "value": "contact-subject-input", "case_sensitive": False}, "text": "Subject", "context": "Contact form subject"},
            {"selector": {"type": "attributeValueSelector", "attribute": "id", "value": "email-field", "case_sensitive": False}, "text": "Email", "context": "Contact form email"},
            {"selector": {"type": "attributeValueSelector", "attribute": "id", "value": "contact-name", "case_sensitive": False}, "text": "Name", "context": "Contact form name"},
        ],
    }
    first = policy_module._preferred_prompt_form_action(prompt, policy_obs, allowed_tools={"browser.input", "browser.click"})
    assert first["arguments"]["selector"]["value"] == "contact-name"
    assert first["arguments"]["text"] == "David"
    policy_obs["candidates"][4]["current_value"] = "David"
    second = policy_module._preferred_prompt_form_action(prompt, policy_obs, allowed_tools={"browser.input", "browser.click"})
    assert second["arguments"]["selector"]["value"] == "email-field"
    policy_obs["candidates"][3]["current_value"] = "user1@site.com"
    third = policy_module._preferred_prompt_form_action(prompt, policy_obs, allowed_tools={"browser.input", "browser.click"})
    assert third["arguments"]["selector"]["value"] == "contact-subject-input"
    policy_obs["candidates"][2]["current_value"] = "Inquiry"
    fourth = policy_module._preferred_prompt_form_action(prompt, policy_obs, allowed_tools={"browser.input", "browser.click"})
    assert fourth["arguments"]["selector"]["value"] == "contact-message-textarea"
    policy_obs["candidates"][1]["current_value"] = "Please provide me with more information"
    fifth = policy_module._preferred_prompt_form_action(prompt, policy_obs, allowed_tools={"browser.input", "browser.click"})
    assert fifth["name"] == "browser.click"
    assert fifth["arguments"]["selector"]["value"] == "send-message-button"
