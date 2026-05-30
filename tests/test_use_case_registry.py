from training.use_case_registry import all_use_case_specs, get_use_case_spec


def test_login_spec_exposes_harvester_hints_and_clusters() -> None:
    spec = get_use_case_spec("LOGIN")
    assert spec.name == "LOGIN"
    assert any("/login" in hint for hint in spec.harvester_hints)
    assert spec.harvester_model_ladder == ("gpt-5.4-mini", "gpt-5.4")
    assert "no_progress_loop" in spec.likely_failure_clusters
    assert spec.recommended_gold_target == 500
    assert spec.recommended_holdout_target == 50


def test_registry_covers_all_autocinema_use_cases_with_hints() -> None:
    specs = all_use_case_specs()
    names = {spec.name for spec in specs}
    assert len(specs) == 16
    assert "LOGIN" in names
    assert "ADD_TO_WATCHLIST" in names
    assert "WATCH_TRAILER" in names
    assert all(spec.harvester_hints for spec in specs)
