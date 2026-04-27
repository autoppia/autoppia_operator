from __future__ import annotations

import json
from pathlib import Path

from training.format_for_sft import BASE_MODEL, StepRequest, StepResponse, export_harvest_to_sft


def test_export_harvest_to_sft_generates_manifest_and_non_empty_split(tmp_path: Path) -> None:
    manifest = export_harvest_to_sft(
        input_path="data/autocinema/contact/gold/episodes.jsonl",
        summary_path="data/autocinema/contact/gold/summary.json",
        output_dir=str(tmp_path / "sft"),
        train_seeds=[29, 52],
        val_seeds=[57],
        seed=36,
        trace_only=False,
    )

    train_path = tmp_path / "sft" / "train.jsonl"
    val_path = tmp_path / "sft" / "val.jsonl"
    manifest_path = tmp_path / "sft" / "manifest.json"

    assert train_path.exists()
    assert val_path.exists()
    assert manifest_path.exists()

    train_rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    val_rows = [json.loads(line) for line in val_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert train_rows
    assert val_rows
    assert train_rows[0]["messages"][0]["role"] == "system"
    assert train_rows[0]["messages"][-1]["role"] == "assistant"
    assert manifest["base_model"] == BASE_MODEL
    assert manifest["train_examples"] == len(train_rows)
    assert manifest["val_examples"] == len(val_rows)
    assert manifest["successful_episodes_used"] == len(manifest["selected_episode_ids"])


def test_export_harvest_to_sft_keeps_richer_runtime_like_observation(tmp_path: Path) -> None:
    export_harvest_to_sft(
        input_path="data/autocinema/contact/gold/episodes.jsonl",
        summary_path="data/autocinema/contact/gold/summary.json",
        output_dir=str(tmp_path / "sft"),
        train_seeds=[29, 52],
        val_seeds=[57],
        seed=36,
        trace_only=False,
    )
    train_path = tmp_path / "sft" / "train.jsonl"
    first = json.loads(train_path.read_text(encoding="utf-8").splitlines()[0])
    user_text = first["messages"][1]["content"]
    assistant_text = first["messages"][2]["content"]

    assert ("Forms:" in user_text) or ("Headings:" in user_text) or ('"mode": "guided_harvester"' in user_text)
    assert ("Candidates:" in user_text) or ('"planned_action"' in user_text)
    assert ("selector=" in user_text) or ('"attempts"' in user_text)
    assert ("browser." in assistant_text) or ('"type":' in assistant_text)


def test_export_harvest_to_sft_trace_only_skips_guided_fallback(tmp_path: Path) -> None:
    trace_dir = tmp_path / "episodes"
    trace_dir.mkdir(parents=True)
    trace_file = trace_dir / "trace.json"
    trace_file.write_text(
        json.dumps(
            {
                "task_prompt": "Login",
                "episode": {"episode_task_id": "ep-trace", "use_case": "LOGIN"},
                "steps": [
                    {
                        "step_index": 0,
                        "before": {"url": "http://example.test/"},
                        "act_request": {
                            "prompt": "Login",
                            "url": "http://example.test/",
                            "snapshot_html": "<html><body><a href='/login'>Login</a></body></html>",
                            "allowed_tools": [],
                        },
                        "act_response": {
                            "tool_calls": [
                                {
                                    "name": "browser.click",
                                    "arguments": {
                                        "selector": {
                                            "type": "attributeValueSelector",
                                            "attribute": "href",
                                            "value": "/login",
                                            "case_sensitive": False,
                                        }
                                    },
                                }
                            ]
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    guided_report = tmp_path / "guided.json"
    guided_report.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "use_case": "CONTACT",
                        "guided_execution": [
                            {
                                "url": "http://example.test/contact",
                                "planned_action": {
                                    "type": "NavigateAction",
                                    "url": "http://example.test/contact",
                                },
                                "attempts": [
                                    {
                                        "action": {
                                            "type": "NavigateAction",
                                            "url": "http://example.test/contact",
                                        },
                                        "success": True,
                                        "error": "",
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "episode_task_id": "ep-trace",
                        "use_case": "LOGIN",
                        "seed": 1,
                        "success": True,
                        "score": 1.0,
                        "trace_file": str(trace_file),
                    }
                ),
                json.dumps(
                    {
                        "episode_task_id": "ep-guided",
                        "use_case": "CONTACT",
                        "seed": 2,
                        "success": True,
                        "score": 1.0,
                        "result_path": str(guided_report),
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({"episodes_total": 2, "successes_total": 2}),
        encoding="utf-8",
    )

    manifest = export_harvest_to_sft(
        input_path=str(episodes),
        summary_path=str(summary),
        output_dir=str(tmp_path / "sft"),
        train_seeds=[1],
        val_seeds=[2],
        seed=7,
        trace_only=True,
    )

    train_rows = [json.loads(line) for line in (tmp_path / "sft" / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(train_rows) == 1
    assert train_rows[0]["metadata"]["use_case"] == "LOGIN"
    assert manifest["source_mode"] == "trace_only"
    assert manifest["guided_fallback_episodes"] == 0
    assert manifest["skipped_without_trace"] == 1


def test_export_harvest_to_sft_skips_self_navigate_and_remaps_click_index(tmp_path: Path) -> None:
    trace_dir = tmp_path / "episodes"
    trace_dir.mkdir(parents=True)
    trace_file = trace_dir / "trace.json"
    snapshot_html = "<html><body><input name='email' placeholder='Email' /><input name='password' placeholder='Password' /><button>Log in</button></body></html>"
    trace_file.write_text(
        json.dumps(
            {
                "task_prompt": "Login",
                "episode": {"episode_task_id": "ep-trace", "use_case": "LOGIN"},
                "steps": [
                    {
                        "step_index": 0,
                        "before": {"url": "http://example.test/login"},
                        "act_request": {
                            "prompt": "Login",
                            "url": "http://example.test/login",
                            "snapshot_html": snapshot_html,
                            "allowed_tools": [],
                        },
                        "act_response": {
                            "tool_calls": [
                                {
                                    "name": "browser.navigate",
                                    "arguments": {"url": "http://example.test/login"},
                                }
                            ]
                        },
                    },
                    {
                        "step_index": 1,
                        "before": {"url": "http://example.test/login"},
                        "act_request": {
                            "prompt": "Login",
                            "url": "http://example.test/login",
                            "snapshot_html": snapshot_html,
                            "allowed_tools": [],
                        },
                        "act_response": {
                            "tool_calls": [
                                {
                                    "name": "browser.click",
                                    "arguments": {"index": 0},
                                }
                            ]
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(
        json.dumps(
            {
                "episode_task_id": "ep-trace",
                "use_case": "LOGIN",
                "seed": 1,
                "success": True,
                "score": 1.0,
                "trace_file": str(trace_file),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"episodes_total": 1, "successes_total": 1}), encoding="utf-8")

    manifest = export_harvest_to_sft(
        input_path=str(episodes),
        summary_path=str(summary),
        output_dir=str(tmp_path / "sft"),
        train_seeds=[1],
        val_seeds=[],
        trace_only=True,
    )

    train_rows = [json.loads(line) for line in (tmp_path / "sft" / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(train_rows) == 1
    assistant = train_rows[0]["messages"][2]["content"]
    assert '"name": "browser.click"' in assistant
    assert '"index": 2' in assistant
    assert manifest["train_examples"] == 1


def test_export_harvest_to_sft_runtime_aligned_uses_policy_prompt_and_browser_wrapper(tmp_path: Path) -> None:
    trace_dir = tmp_path / "episodes"
    trace_dir.mkdir(parents=True)
    trace_file = trace_dir / "trace.json"
    snapshot_html = "<html><body><h1>Login</h1><input name='email' placeholder='Email' /><input name='password' placeholder='Password' /><button>Log in</button></body></html>"
    trace_file.write_text(
        json.dumps(
            {
                "task_prompt": "Login with the visible credentials.",
                "episode": {"episode_task_id": "ep-trace", "use_case": "LOGIN"},
                "steps": [
                    {
                        "step_index": 0,
                        "before": {"url": "http://example.test/login"},
                        "act_request": {
                            "task_id": "task-1",
                            "prompt": "Login with the visible credentials.",
                            "web_project_id": "autocinema",
                            "url": "http://example.test/login",
                            "snapshot_html": snapshot_html,
                            "allowed_tools": ["browser.click", "browser.input"],
                            "state_in": {},
                        },
                        "act_response": {
                            "tool_calls": [
                                {
                                    "name": "browser.click",
                                    "arguments": {"index": 0},
                                }
                            ]
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(
        json.dumps(
            {
                "episode_task_id": "ep-trace",
                "use_case": "LOGIN",
                "seed": 1,
                "success": True,
                "score": 1.0,
                "trace_file": str(trace_file),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"episodes_total": 1, "successes_total": 1}), encoding="utf-8")

    manifest = export_harvest_to_sft(
        input_path=str(episodes),
        summary_path=str(summary),
        output_dir=str(tmp_path / "sft"),
        train_seeds=[1],
        val_seeds=[],
        trace_only=True,
        runtime_aligned=True,
    )

    train_rows = [json.loads(line) for line in (tmp_path / "sft" / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    user_text = train_rows[0]["messages"][1]["content"]
    request = json.loads(user_text)
    assistant = json.loads(train_rows[0]["messages"][2]["content"])

    StepRequest.model_validate(request)
    StepResponse.model_validate(assistant)
    assert set(request).issubset({"protocol_version", "task_id", "prompt", "url", "html", "screenshot", "step_index", "history", "tools", "include_reasoning"})
    assert "snapshot_html" not in request
    assert "allowed_tools" not in request
    assert "web_project_id" not in request
    assert "use_case" not in request
    assert request["protocol_version"] == "1.0"
    assert request["task_id"] == "task-1"
    assert request["html"]
    assert request["include_reasoning"] is False
    assert isinstance(request["tools"], list)
    assert set(assistant).issubset({"protocol_version", "tool_calls", "content", "reasoning", "done", "error"})
    assert assistant["protocol_version"] == "1.0"
    assert assistant["tool_calls"][0]["name"] == "browser.click"
    assert assistant["tool_calls"][0]["arguments"]["index"] == 2
    assert assistant["done"] is False
    assert manifest["runtime_aligned"] is True
    assert manifest["format_version"].endswith(".v3")
