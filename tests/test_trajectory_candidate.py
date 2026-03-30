from __future__ import annotations

from pathlib import Path

from training.trajectory_candidate import TrajectoryCandidate, candidate_path, load_candidate, write_candidate


def test_candidate_roundtrip(tmp_path: Path) -> None:
    candidate = TrajectoryCandidate(
        use_case="CONTACT",
        seed=7,
        attempt_name="claude_01",
        teacher_model="claude-sonnet-4-5",
        generation_mode="claude_code_brief",
        brief_path=str(tmp_path / "brief.json"),
        prompt_lines=("go to contact",),
        actions=({"type": "NavigateAction", "url": "http://example.test/contact?seed=7"},),
        metadata={"x": 1},
    )
    path = candidate_path(output_root=tmp_path, seed=7, attempt_name="claude_01")
    write_candidate(path, candidate)
    loaded = load_candidate(path)
    assert loaded.use_case == "CONTACT"
    assert loaded.seed == 7
    assert loaded.actions[0]["type"] == "NavigateAction"
