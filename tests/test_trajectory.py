from autoppia_harvester.trajectory import extract_json_object, normalize_trajectory


def test_normalize_trajectory_accepts_iwa_tool_calls():
    calls = normalize_trajectory(
        {
            "trajectory": [
                {"name": "browser.navigate", "arguments": {"url": "http://localhost:8000"}},
                {"name": "browser.input", "arguments": {"text": "abc"}},
            ]
        }
    )
    assert [call.name for call in calls] == ["navigate", "type"]


def test_extract_json_object_accepts_claude_json_wrapper():
    parsed = extract_json_object('{"result": "{\\"success\\": true, \\"trajectory\\": []}"}')
    assert parsed["success"] is True
