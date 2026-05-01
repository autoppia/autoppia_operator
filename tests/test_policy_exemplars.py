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


def test_example_block_uses_project_specific_trace_examples(monkeypatch: object) -> None:
    monkeypatch.setattr(
        policy_module,
        "_project_success_examples",
        lambda project_id: [
            {
                "web_project_id": project_id,
                "use_case": "LOGIN_BOOK",
                "url_path": "/login",
                "step_index": 1,
                "prompt": "Log in to the library.",
                "tool_calls": [{"name": "browser.input", "arguments": {"index": 0, "text": "<username>"}}],
            },
            {
                "web_project_id": project_id,
                "use_case": "SEARCH_BOOK",
                "url_path": "/search",
                "step_index": 1,
                "prompt": "Search for a book.",
                "tool_calls": [{"name": "browser.click", "arguments": {"index": 1}}],
            },
        ],
    )

    block = policy_module._autocinema_example_block(
        "Log in to the library with the provided credentials.",
        {
            "web_project_id": "autobooks",
            "url": "https://books.example/login",
            "step_index": 1,
            "use_case": {"name": "LOGIN_BOOK"},
        },
    )

    joined = "\n".join(block)
    assert "RETRIEVED SUCCESSFUL TRACE EXAMPLES:" in joined
    assert "use_case=LOGIN_BOOK" in joined
    assert "use_case=SEARCH_BOOK" not in joined


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


def test_preferred_seed_navigation_uses_site_knowledge_section_route() -> None:
    action = policy_module._preferred_seed_stable_navigation(
        "Log in to the site.",
        {
            "url": "https://autocinema.example/about?seed=42",
            "site_knowledge": {
                "current_task_routing": {"likely_best_section": "auth"},
                "routes": [
                    {"section_id": "info", "path": "/about", "label": "About"},
                    {"section_id": "auth", "path": "/login", "label": "Login"},
                ],
            },
            "page_observations": {"capability_gap": {}},
            "use_case": {"name": "LOGIN_BOOK"},
        },
        allowed_tools={"browser.navigate"},
    )
    assert action is not None
    tool_call = action["tool_call"]
    assert tool_call["name"] == "browser.navigate"
    assert tool_call["arguments"]["url"].rstrip("/") == "https://autocinema.example/login"


def test_preferred_prompt_navigation_opens_domain_from_blank_page() -> None:
    action = policy_module._preferred_prompt_navigation(
        "Open autoppia.com and summarize the homepage.",
        {
            "url": "about:blank",
        },
        allowed_tools={"browser.navigate", "browser.click"},
    )
    assert action is not None
    tool_call = action["tool_call"]
    assert tool_call["name"] == "browser.navigate"
    assert tool_call["arguments"]["url"] == "https://autoppia.com"


def test_preferred_prompt_navigation_explicit_https_without_open_verb() -> None:
    action = policy_module._preferred_prompt_navigation(
        "Read https://docs.example.com/guide.html and list the headings.",
        {"url": "about:blank"},
        allowed_tools={"browser.navigate"},
    )
    assert action is not None
    assert action["tool_call"]["arguments"]["url"] == "https://docs.example.com/guide.html"


def test_preferred_prompt_navigation_named_site_without_open_verb() -> None:
    action = policy_module._preferred_prompt_navigation(
        "Search for alpine marmots on Wikipedia.",
        {"url": "about:blank"},
        allowed_tools={"browser.navigate"},
    )
    assert action is not None
    assert action["tool_call"]["arguments"]["url"] == "https://www.wikipedia.org"


def test_fallback_navigates_from_blank_page_for_general_web_task() -> None:
    policy = policy_module.Policy(llm_call=lambda **_: {})
    action = policy._fallback(
        prompt="Go to google.com and search for weather in Tokyo.",
        mode="DIRECT",
        policy_obs={"url": "about:blank"},
        allowed_tools={"browser.navigate", "browser.click"},
    )
    assert action["type"] == "browser"
    assert action["tool_call"]["name"] == "browser.navigate"
    assert action["tool_call"]["arguments"]["url"] == "https://www.google.com"


def test_fallback_fails_fast_on_blank_page_without_inferable_target() -> None:
    policy = policy_module.Policy(llm_call=lambda **_: {})
    action = policy._fallback(
        prompt="Continue with the task.",
        mode="DIRECT",
        policy_obs={"url": "about:blank"},
        allowed_tools={"browser.navigate", "browser.click"},
    )
    assert action["type"] == "final"
    assert action["done"] is True
    assert action["error"] == "blank_page_no_navigation_target"
    assert action["failure_reason"] == "blank_page_no_navigation_target"


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
