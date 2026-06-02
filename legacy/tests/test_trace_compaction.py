from __future__ import annotations

import json
from pathlib import Path

from training.trace_compaction import compact_trace_dir


def test_compact_trace_dir_strips_large_inline_fields(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace" / "episodes"
    trace_dir.mkdir(parents=True, exist_ok=True)
    episode_path = trace_dir / "episode.json"
    episode_path.write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "before": {"screenshot": "AAA", "html": "<html>before</html>"},
                        "after": {"screenshot": "BBB", "html": "<html>after</html>"},
                        "act_request": {"snapshot_html": "<html>snap</html>", "screenshot": "CCC"},
                        "act_response": {"tool_calls": [{"name": "browser.click"}]},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    stats = compact_trace_dir(tmp_path / "trace")

    payload = json.loads(episode_path.read_text(encoding="utf-8"))
    step = payload["steps"][0]
    assert stats.files_rewritten == 1
    assert stats.fields_removed == 4
    assert "screenshot" not in step["before"]
    assert "screenshot" not in step["after"]
    assert "screenshot" not in step["act_request"]
    assert "snapshot_html" not in step["act_request"]
    assert step["before"]["html"] == "<html>before</html>"
