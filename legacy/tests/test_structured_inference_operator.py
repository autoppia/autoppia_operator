from __future__ import annotations

import pytest

from src.operator.agents.operator import StructuredInferenceOperator


@pytest.mark.asyncio
async def test_structured_operator_contact_routes_then_fills_form():
    operator = StructuredInferenceOperator()
    home_payload = {
        "task_id": "t1",
        "prompt": "TASK: Fill out the contact form with a name that equals 'David', an email that contains 'user1@site.com', a subject that does NOT contain 'Information', and a message that equals 'Please provide me with more information'.",
        "url": "http://84.247.180.192:8000/?seed=1",
        "snapshot_html": """
        <html><body>
          <a id="nav-contact" href="/contact?seed=1">Contact</a>
          <input id="entry-field" placeholder="Search" />
        </body></html>
        """,
        "allowed_tools": [{"name": "browser.click"}, {"name": "browser.navigate"}, {"name": "browser.input"}],
        "history": [],
    }
    first = await operator.act_from_payload(home_payload)
    assert first["actions"][0]["type"] in {"ClickAction", "NavigateAction"}

    contact_payload = {
        **home_payload,
        "url": "http://84.247.180.192:8000/contact?seed=1",
        "snapshot_html": """
        <html><body>
          <input id="contact-name-input" />
          <input id="contact-email-input" />
          <input id="contact-subject-input" />
          <textarea id="contact-message-textarea"></textarea>
          <button id="send-message-button">Send Message</button>
        </body></html>
        """,
        "history": [],
    }
    second = await operator.act_from_payload(contact_payload)
    assert second["actions"][0]["type"] == "TypeAction"
    assert second["actions"][0]["selector"]["value"] == "contact-name-input"


@pytest.mark.asyncio
async def test_structured_operator_contact_variant_fields_then_submit():
    operator = StructuredInferenceOperator()
    payload = {
        "task_id": "t3",
        "prompt": "TASK: Submit the contact form with a name that is NOT 'TestUser'.",
        "url": "http://84.247.180.192:8000/contact?seed=2",
        "snapshot_html": """
        <html><body>
          <input id="name-input-field" aria-label="Name" />
          <input id="contact-email-addr" type="email" aria-label="Email" />
          <input id="contact-subject-entry" aria-label="Subject" />
          <textarea id="message-entry-field" aria-label="Message"></textarea>
          <button id="submit-contact-form">Send</button>
        </body></html>
        """,
        "allowed_tools": [{"name": "browser.click"}, {"name": "browser.navigate"}, {"name": "browser.input"}],
        "history": [],
    }
    actions = []
    for _ in range(5):
        out = await operator.act_from_payload(payload)
        action = out["actions"][0]
        actions.append(action)
        payload["history"] = payload.get("history", []) + [{"action": action}]
    assert actions[0]["selector"]["value"] == "name-input-field"
    assert actions[1]["selector"]["value"] == "contact-email-addr"
    assert actions[2]["selector"]["value"] == "contact-subject-entry"
    assert actions[3]["selector"]["value"] == "message-entry-field"
    assert actions[4]["type"] == "ClickAction"


@pytest.mark.asyncio
async def test_structured_operator_contact_full_prompt_fills_all_fields_then_submit():
    operator = StructuredInferenceOperator()
    payload = {
        "task_id": "t3b",
        "prompt": "TASK: Fill out the contact form with a name that equals 'David', an email that contains 'user1@site.com', a subject that does NOT contain 'Information', and a message that equals 'Please provide me with more information'.",
        "url": "http://84.247.180.192:8000/contact?seed=2",
        "snapshot_html": """
        <html><body>
          <input id="name-input-field" aria-label="Name" />
          <input id="contact-email-addr" type="email" aria-label="Email" />
          <input id="contact-subject-entry" aria-label="Subject" />
          <textarea id="message-entry-field" aria-label="Message"></textarea>
          <button id="submit-contact-form">Send</button>
        </body></html>
        """,
        "allowed_tools": [{"name": "browser.click"}, {"name": "browser.navigate"}, {"name": "browser.input"}],
        "history": [],
    }
    actions = []
    for _ in range(5):
        out = await operator.act_from_payload(payload)
        action = out["actions"][0]
        actions.append(action)
        payload["history"] = payload.get("history", []) + [{"action": action}]
    assert actions[0]["selector"]["value"] == "name-input-field"
    assert actions[1]["selector"]["value"] == "contact-email-addr"
    assert actions[2]["selector"]["value"] == "contact-subject-entry"
    assert actions[3]["selector"]["value"] == "message-entry-field"
    assert actions[4]["type"] == "ClickAction"


@pytest.mark.asyncio
async def test_structured_operator_login_advances_to_password_then_submit():
    operator = StructuredInferenceOperator()
    payload = {
        "task_id": "t2",
        "prompt": "TASK: Log in using username equals 'user1' and password equals 'Passw0rd!'.",
        "url": "http://84.247.180.192:8000/login?seed=7",
        "snapshot_html": """
        <html><body>
          <input id="login-username" />
          <input id="login-password" type="password" />
          <button id="login-submit">Login</button>
        </body></html>
        """,
        "allowed_tools": [{"name": "browser.click"}, {"name": "browser.navigate"}, {"name": "browser.input"}],
        "history": [],
    }
    first = await operator.act_from_payload(payload)
    assert first["actions"][0]["type"] == "TypeAction"
    assert first["actions"][0]["selector"]["value"] == "login-username"

    second = await operator.act_from_payload(
        {
            **payload,
            "history": [
                {
                    "action": {
                        "type": "TypeAction",
                        "selector": {"type": "attributeValueSelector", "attribute": "id", "value": "login-username"},
                        "text": "user1",
                    }
                }
            ],
        }
    )
    assert second["actions"][0]["type"] == "TypeAction"
    assert second["actions"][0]["selector"]["value"] == "login-password"


@pytest.mark.asyncio
async def test_structured_operator_contact_prefers_submit_button_over_home_link():
    operator = StructuredInferenceOperator()
    payload = {
        "task_id": "t4",
        "prompt": "TASK: Fill out the contact form with a name that equals 'David', an email that contains 'user1@site.com', a subject that does NOT contain 'Information', and a message that equals 'Please provide me with more information'.",
        "url": "http://84.247.180.192:8000/contact?seed=4",
        "snapshot_html": """
        <html><body>
          <a id="home-link" href="/?seed=4">Home</a>
          <input id="contact-name-input" />
          <input id="email-input-field" />
          <input id="subject-input-field" />
          <textarea id="message-textarea-field"></textarea>
          <button id="send-message-button">Send Message</button>
        </body></html>
        """,
        "allowed_tools": [{"name": "browser.click"}, {"name": "browser.navigate"}, {"name": "browser.input"}],
        "history": [
            {"action": {"type": "TypeAction", "selector": {"type": "attributeValueSelector", "attribute": "id", "value": "contact-name-input"}, "text": "David"}},
            {"action": {"type": "TypeAction", "selector": {"type": "attributeValueSelector", "attribute": "id", "value": "email-input-field"}, "text": "user1@site.com"}},
            {"action": {"type": "TypeAction", "selector": {"type": "attributeValueSelector", "attribute": "id", "value": "subject-input-field"}, "text": "Inquiry"}},
            {"action": {"type": "TypeAction", "selector": {"type": "attributeValueSelector", "attribute": "id", "value": "message-textarea-field"}, "text": "Please provide me with more information"}},
        ],
    }
    out = await operator.act_from_payload(payload)
    action = out["actions"][0]
    assert action["type"] == "ClickAction"
    assert action["selector"]["value"] == "send-message-button"
