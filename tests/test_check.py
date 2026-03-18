from pathlib import Path
from types import SimpleNamespace

import pytest

import check
from agent import app


def test_find_route_detects_expected_endpoints() -> None:
    """Find route detects expected endpoints."""
    assert check._find_route(app, "/health", "GET") is True
    assert check._find_route(app, "/act", "POST") is True
    assert check._find_route(app, "/step", "POST") is True
    assert check._find_route(app, "/missing", "GET") is False


def test_validate_actions_shape_accepts_valid_payload() -> None:
    """Validate actions shape accepts valid payload."""
    assert check._validate_actions_shape({"actions": [{"type": "ClickAction"}]}) is None


def test_validate_actions_shape_rejects_invalid_payloads() -> None:
    """Validate actions shape rejects invalid payloads."""
    assert check._validate_actions_shape({}) == "Missing top-level 'actions' key"
    assert check._validate_actions_shape({"actions": "bad"}) == "'actions' must be a list, got str"
    assert check._validate_actions_shape({"actions": [1]}) == "actions[0] must be an object, got int"
    assert check._validate_actions_shape({"actions": [{}]}) == "actions[0].type must be a non-empty string"


def test_parse_requirements_pkgs_handles_comments_extras_and_markers() -> None:
    """Parse requirements pkgs handles comments extras and markers."""
    req_text = """
    # comment
    uvicorn[standard]>=0.23.0
    python-dateutil>=2.9.0.post0; python_version>='3.9'
    fastapi==0.115.0  # inline comment
    """

    assert check._parse_requirements_pkgs(req_text) == {
        "uvicorn",
        "python-dateutil",
        "fastapi",
    }


def test_load_module_imports_python_file(tmp_path: Path) -> None:
    """Load module imports python file."""
    module_path = tmp_path / "tmp_module.py"
    module_path.write_text("VALUE = 42\n", encoding="utf-8")

    module = check._load_module(module_path, "tmp_module")

    assert module.VALUE == 42


def test_call_act_executes_minimal_check_payload() -> None:
    """Call act executes minimal check payload."""
    resp = check._call_act(app)

    assert isinstance(resp, dict)
    assert isinstance(resp.get("actions"), list)
    assert resp["actions"]
    assert resp["actions"][0]["type"] in {"ClickAction", "WaitAction"}


def test_call_act_returns_none_without_matching_route() -> None:
    """Call act returns none without matching route."""
    fake_app = SimpleNamespace(routes=[])
    assert check._call_act(fake_app) is None


def test_call_act_supports_sync_endpoint() -> None:
    """Call act supports sync endpoint."""

    def endpoint(payload):
        return {"actions": [{"type": "WaitAction"}], "payload": payload["task_id"]}

    fake_route = SimpleNamespace(path="/act", endpoint=endpoint)
    fake_app = SimpleNamespace(routes=[fake_route])

    resp = check._call_act(fake_app)

    assert resp == {"actions": [{"type": "WaitAction"}], "payload": "check"}


def test_load_module_missing_file_exits(tmp_path: Path) -> None:
    """Load module missing file exits."""
    with pytest.raises(SystemExit):
        check._load_module(tmp_path / "missing.py", "missing")


def test_iter_repo_files_skips_git_cache_and_data(tmp_path: Path, monkeypatch) -> None:
    """Iter repo files skips git cache and data."""
    monkeypatch.setattr(check, "REPO_ROOT", tmp_path)
    (tmp_path / "keep.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "ignored.txt").write_text("x", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "ignored.pyc").write_bytes(b"x")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "ignored.json").write_text("{}", encoding="utf-8")

    files = check._iter_repo_files()

    assert files == [tmp_path / "keep.py"]


def test_read_text_and_output_helpers(capsys, tmp_path: Path) -> None:
    """Read text and output helpers."""
    missing = check._read_text(tmp_path / "missing.txt")
    check._ok("all good")
    check._warn("careful")

    assert missing == ""
    out = capsys.readouterr().out
    assert "[OK] all good" in out
    assert "[WARN] careful" in out

    with pytest.raises(SystemExit):
        check._fail("stop")


def test_scan_helpers_detect_secrets_pyc_and_env(tmp_path: Path, monkeypatch) -> None:
    """Scan helpers detect secrets pyc and env."""
    monkeypatch.setattr(check, "REPO_ROOT", tmp_path)
    secret_file = tmp_path / "secret.py"
    secret_file.write_text('API_KEY = "sk-proj-1234567890abc"\n', encoding="utf-8")
    pyc_file = tmp_path / "bad.pyc"
    pyc_file.write_bytes(b"\x00")
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=test\n", encoding="utf-8")

    failures = []
    warnings = []
    monkeypatch.setattr(check, "_fail", lambda msg: failures.append(msg))
    monkeypatch.setattr(check, "_warn", lambda msg: warnings.append(msg))
    monkeypatch.setattr(check.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1))

    check._scan_for_secrets()
    check._scan_for_pyc()
    check._check_env_file()

    assert any("Possible secret key found in secret.py" in msg for msg in failures)
    assert any("Found .pyc files" in msg for msg in warnings)
    assert any(".env exists in repo folder" in msg for msg in warnings)


def test_main_happy_path_validates_minimal_repo(tmp_path: Path, monkeypatch, capsys) -> None:
    """Main happy path validates minimal repo."""
    monkeypatch.setattr(check, "REPO_ROOT", tmp_path)

    (tmp_path / "main.py").write_text(
        "from fastapi import FastAPI\n"
        "app = FastAPI()\n"
        "@app.get('/health')\n"
        "async def health():\n"
        "    return {'status': 'healthy'}\n"
        "@app.post('/act')\n"
        "async def act(payload: dict):\n"
        "    return {'actions': [{'type': 'WaitAction', 'time_seconds': 0.1}]}\n"
        "@app.post('/step')\n"
        "async def step(payload: dict):\n"
        "    return await act(payload)\n",
        encoding="utf-8",
    )
    (tmp_path / "agent.py").write_text("# placeholder\n", encoding="utf-8")
    (tmp_path / "llm_gateway.py").write_text('OPENAI_BASE_URL = "x"\nHEADER_NAME = "IWA-Task-ID"\n', encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("\n".join(f"{pkg}\n" for pkg in sorted(check.EXPECTED_SANDBOX_PACKAGES)), encoding="utf-8")

    monkeypatch.setattr(check, "_scan_for_secrets", lambda: None)
    monkeypatch.setattr(check, "_scan_for_pyc", lambda: None)
    monkeypatch.setattr(check, "_check_env_file", lambda: None)

    check.main()

    out = capsys.readouterr().out
    assert "Found main.py" in out
    assert "POST /act route found" in out
    assert "All checks passed." in out


def test_main_warns_when_optional_files_are_missing(tmp_path: Path, monkeypatch) -> None:
    """Main warns when optional files are missing."""
    monkeypatch.setattr(check, "REPO_ROOT", tmp_path)

    (tmp_path / "main.py").write_text(
        "from fastapi import FastAPI\napp = FastAPI()\n@app.get('/health')\nasync def health():\n    return {}\n@app.post('/act')\nasync def act(payload: dict):\n    return {'actions': [{'type': 'WaitAction'}]}\n",
        encoding="utf-8",
    )
    (tmp_path / "requirements.txt").write_text("\n".join(f"{pkg}\n" for pkg in sorted(check.EXPECTED_SANDBOX_PACKAGES)), encoding="utf-8")

    warnings = []
    monkeypatch.setattr(check, "_warn", lambda msg: warnings.append(msg))
    monkeypatch.setattr(check, "_scan_for_secrets", lambda: None)
    monkeypatch.setattr(check, "_scan_for_pyc", lambda: None)
    monkeypatch.setattr(check, "_check_env_file", lambda: None)

    check.main()

    assert any("agent.py not found" in msg for msg in warnings)
    assert any("llm_gateway.py not found" in msg for msg in warnings)
    assert any("POST /step route not found" in msg for msg in warnings)


def test_main_fails_for_missing_route_and_bad_response(tmp_path: Path, monkeypatch) -> None:
    """Main fails for missing route and bad response."""
    monkeypatch.setattr(check, "REPO_ROOT", tmp_path)
    (tmp_path / "main.py").write_text("from fastapi import FastAPI\napp = FastAPI()\n", encoding="utf-8")
    (tmp_path / "agent.py").write_text("# x\n", encoding="utf-8")
    (tmp_path / "llm_gateway.py").write_text('OPENAI_BASE_URL = "x"\nHEADER_NAME = "IWA-Task-ID"\n', encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("\n".join(f"{pkg}\n" for pkg in sorted(check.EXPECTED_SANDBOX_PACKAGES)), encoding="utf-8")

    monkeypatch.setattr(check, "_scan_for_secrets", lambda: None)
    monkeypatch.setattr(check, "_scan_for_pyc", lambda: None)
    monkeypatch.setattr(check, "_check_env_file", lambda: None)

    with pytest.raises(SystemExit):
        check.main()
