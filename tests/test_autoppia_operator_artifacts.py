from pathlib import Path

from training.autoppia_operator import (
    OPERATOR_ARTIFACT_VERSION,
    OperatorArtifactPaths,
    build_operator_manifest,
    is_verified_success,
    operator_run_root,
)


def test_operator_run_root_is_scoped_by_project_use_case_and_seed(tmp_path: Path) -> None:
    root = operator_run_root(base_dir=tmp_path, web_project_id="Auto Cinema", use_case="SEARCH_FILM", seed=7)
    assert root == tmp_path / "operator_runs" / "auto-cinema" / "search_film" / "seed_0007"


def test_operator_manifest_marks_only_iwa_verified_success_as_distillation_ready(tmp_path: Path) -> None:
    paths = OperatorArtifactPaths.for_run(tmp_path / "run")
    manifest = build_operator_manifest(
        web_project_id="autocinema",
        use_case="login",
        seed=1,
        paths=paths,
        final_report={"episodes": [{"success": True, "score": 1.0}]},
    )
    assert manifest["artifact_version"] == OPERATOR_ARTIFACT_VERSION
    assert manifest["verified_success"] is True
    assert manifest["distillation_ready"] is True
    assert manifest["paths"]["final_trajectory"].endswith("trajectory.json")


def test_is_verified_success_rejects_unscored_or_partial_reports() -> None:
    assert is_verified_success({"success": True, "score": 0.5}) is False
    assert is_verified_success({"episodes": [{"success": True, "score": 1.0}]}) is True
    assert is_verified_success({"episodes": [{"success": False, "score": 1.0}]}) is False
