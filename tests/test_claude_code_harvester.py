from __future__ import annotations

import json
from pathlib import Path

import training.claude_code_harvester as module
from training.claude_code_harvester import brief_prompt_lines, summarize_attempt_for_claude


def test_brief_prompt_lines_flattens_route_fields_submit_and_pitfalls() -> None:
    payload = {
        "brief": {
            "route": ["/contact", "stay on form"],
            "prompt_lines": ["Fill all required fields."],
            "fields": [
                {
                    "name": "subject",
                    "ids": ["contact-subject-input"],
                    "value_rule": "Use a safe subject like General inquiry.",
                }
            ],
            "submit": {
                "ids": ["send-message-button"],
                "text": ["Send Message"],
                "action": "click submit once",
            },
            "success_signals": {
                "texts": ["Message Sent!"],
                "ids": [],
                "url_contains": [],
            },
            "pitfalls": ["Do not reopen the contact page after filling fields."],
            "action_sketch": ["Navigate to /contact", "Fill fields", "Submit form"],
            "confidence": 0.9,
        }
    }

    lines = brief_prompt_lines(payload)
    text = " ".join(lines)
    assert "Follow this route exactly" in text
    assert "contact-subject-input" in text
    assert "send-message-button" in text
    assert "Message Sent!" in text
    assert "Avoid this mistake" in text
    assert "Action sketch" in text


def test_summarize_attempt_for_claude_includes_recent_trace_steps(tmp_path: Path) -> None:
    trace_file = tmp_path / "episode.json"
    trace_file.write_text(
        json.dumps(
            {
                "step_traces": [
                    {
                        "step_index": 1,
                        "actions": [{"type": "click"}],
                        "execution": {"exec_ok": False, "error": "wrong button"},
                    },
                    {
                        "step_index": 2,
                        "actions": [{"type": "input_text"}],
                        "execution": {"exec_ok": True, "error": ""},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    report = {
        "model": "gpt-5.4-mini",
        "episodes": [
            {
                "success": False,
                "score": 0.0,
                "steps": 2,
                "final_content": "Still on the wrong page",
                "estimated_cost_usd": 0.12,
            }
        ],
    }
    row = {"trace_file": str(trace_file), "final_url": "/contact?seed=1"}

    summary = summarize_attempt_for_claude(attempt_name="claude_01", report=report, row=row)

    assert summary["attempt_name"] == "claude_01"
    assert summary["final_url"] == "/contact?seed=1"
    assert summary["recent_steps"][0]["actions"] == ["click"]
    assert summary["recent_steps"][0]["error"] == "wrong button"


def test_summarize_attempt_for_claude_includes_candidate_actions_and_failure_hints(tmp_path: Path) -> None:
    candidate_file = tmp_path / "candidate.json"
    candidate_file.write_text(
        json.dumps(
            {
                "actions": [
                    {"type": "NavigateAction", "url": "http://example.test/contact"},
                    {"type": "TypeAction", "text": "David"},
                ]
            }
        ),
        encoding="utf-8",
    )
    report = {
        "episodes": [
            {
                "success": False,
                "score": 0.0,
                "steps": 1,
                "final_content": "I am not on /contact yet, navigate to /contact first before filling the contact form and submit.",
            }
        ]
    }
    row = {"candidate_path": str(candidate_file)}
    summary = summarize_attempt_for_claude(attempt_name="claude_01", report=report, row=row)
    assert summary["candidate_actions"][0]["type"] == "NavigateAction"
    assert "not_on_target_page" in summary["failure_hints"]


def test_file_snippets_focus_on_use_case_keywords(tmp_path: Path) -> None:
    component = tmp_path / "ContactSection.tsx"
    component.write_text(
        "\n".join(
            [
                "const unrelated = 'x';",
                "function ContactSection() {",
                "  const [email, setEmail] = useState('');",
                "  const [message, setMessage] = useState('');",
                "  return <form><button>Send</button></form>;",
                "}",
            ]
        ),
        encoding="utf-8",
    )
    variants = tmp_path / "id-variants.json"
    variants.write_text(
        json.dumps(
            {
                "contact-email": ["contact-email"],
                "contact-message": ["contact-message"],
                "search-input": ["search-input"],
            }
        ),
        encoding="utf-8",
    )

    snippets = module._file_snippets([component, variants], use_case="CONTACT")
    by_path = {Path(item["path"]).name: item["content"] for item in snippets}
    assert "email" in by_path["ContactSection.tsx"].lower()
    assert "message" in by_path["ContactSection.tsx"].lower()
    assert "contact-email" in by_path["id-variants.json"]
    assert "search-input" not in by_path["id-variants.json"]


def test_generate_claude_brief_uses_gateway_for_gpt_models(monkeypatch, tmp_path: Path) -> None:
    task_cache = tmp_path / "tasks.json"
    task_cache.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "prompt": "Send a support message.",
                        "use_case": {"name": "CONTACT", "constraints": {"required": ["name", "email"]}},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    seen: dict[str, object] = {}

    def fake_openai_chat_completions(*, task_id, messages, model, temperature=0.2, max_tokens=300):
        seen["task_id"] = task_id
        seen["messages"] = messages
        seen["model"] = model
        seen["temperature"] = temperature
        seen["max_tokens"] = max_tokens
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "use_case": "CONTACT",
                                "seed": 7,
                                "route": ["/contact"],
                                "prompt_lines": ["Open the contact page."],
                                "fields": [],
                                "submit": {"ids": ["send-message-button"], "text": ["Send"], "action": "click once"},
                                "success_signals": {"texts": ["Message Sent!"], "ids": [], "url_contains": []},
                                "steps": [{"type": "NavigateAction", "url": "http://example.test/contact?seed=7"}],
                                "pitfalls": ["Do not leave the page before submitting."],
                                "action_sketch": ["Navigate", "Submit"],
                                "confidence": 0.75,
                            }
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140},
            "model": "gpt-5.4-mini",
        }

    monkeypatch.setattr(module, "_candidate_web_files", lambda use_case: [])
    monkeypatch.setattr(module, "_load_existing_examples", lambda use_case: [])
    monkeypatch.setattr(module, "_file_snippets", lambda paths, use_case: [])
    monkeypatch.setattr(module, "openai_chat_completions", fake_openai_chat_completions)

    payload = module.generate_claude_brief(
        use_case="CONTACT",
        seed=7,
        model="gpt-5.4-mini",
        task_cache_path=task_cache,
        web_project_id="autocinema",
    )

    assert seen["model"] == "gpt-5.4-mini"
    assert str(seen["task_id"]).startswith("harvester-brief-contact-7")
    assert payload["brief"]["route"] == ["/contact"]
    assert payload["brief"]["steps"][0]["type"] == "NavigateAction"
    assert payload["meta"]["model"] == "gpt-5.4-mini"
