import importlib.util
import json
import runpy
import sys
import types
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_compare_eval_parse_run_and_slug() -> None:
    """Compare eval parse run and slug."""
    module = _load_module(
        "compare_eval_module",
        Path(__file__).resolve().parents[1] / "scripts" / "compare_eval.py",
    )

    run = module._parse_run("OpenAI:GPT-5.2")

    assert run.provider == "openai"
    assert run.model == "GPT-5.2"
    assert run.slug == "openai__gpt-5.2"

    with pytest.raises(SystemExit):
        module._parse_run("invalid-run")


def test_compare_eval_main_writes_summary(tmp_path: Path, monkeypatch) -> None:
    """Compare eval main writes summary."""
    module = _load_module(
        "compare_eval_main_module",
        Path(__file__).resolve().parents[1] / "scripts" / "compare_eval.py",
    )

    monkeypatch.setattr(module, "REPO_DIR", tmp_path)
    monkeypatch.setattr(
        module.argparse.ArgumentParser,
        "parse_args",
        lambda self: Namespace(
            runs=[["openai:gpt-5-mini"]],
            num_tasks=2,
            max_steps=3,
            distinct_use_cases=False,
            use_case=None,
            web_project_id=None,
            repeat=1,
            temperature=0.2,
        ),
    )

    def fake_run(cmd, cwd: str, env: dict):
        out_path = Path(cmd[cmd.index("--out") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "provider": "openai",
                    "model": "gpt-5-mini",
                    "num_tasks": 2,
                    "successes": 1,
                    "timing": {"avg_task_seconds": 1.5},
                    "episodes": [
                        {"estimated_cost_usd": 0.1, "total_tokens": 50},
                        {"estimated_cost_usd": 0.2, "total_tokens": 75},
                    ],
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    module.main()

    summary = json.loads((tmp_path / "data" / "compare" / "compare_summary.json").read_text(encoding="utf-8"))

    assert summary["runs"][0]["provider"] == "openai"
    assert summary["runs"][0]["success_rate"] == 0.5
    assert summary["runs"][0]["estimated_cost_usd"] == 0.3


def test_compare_eval_main_handles_optional_flags_and_subprocess_failure(tmp_path: Path, monkeypatch) -> None:
    """Compare eval main handles optional flags and subprocess failure."""
    module = _load_module(
        "compare_eval_failure_module",
        Path(__file__).resolve().parents[1] / "scripts" / "compare_eval.py",
    )

    monkeypatch.setattr(module, "REPO_DIR", tmp_path)
    monkeypatch.setattr(
        module.argparse.ArgumentParser,
        "parse_args",
        lambda self: Namespace(
            runs=[["openai:gpt-5-mini"]],
            num_tasks=2,
            max_steps=3,
            distinct_use_cases=True,
            use_case="LOGIN",
            web_project_id="wp1",
            repeat=1,
            temperature=0.2,
        ),
    )

    captured = {}

    def fake_run(cmd, cwd: str, env: dict):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=9)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with pytest.raises(SystemExit) as exc:
        module.main()

    assert exc.value.code == 9
    assert "--distinct-use-cases" in captured["cmd"]
    assert captured["cmd"][-4:] == ["--use-case", "LOGIN", "--web-project-id", "wp1"]


def test_compare_eval_run_as_main(monkeypatch, tmp_path: Path) -> None:
    """Compare eval run as main."""
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "compare_eval.py"
    monkeypatch.setenv("PYTHON", sys.executable)
    monkeypatch.setattr(
        sys,
        "argv",
        ["compare_eval.py", "--runs", "openai:gpt-5-mini", "--num-tasks", "1"],
    )

    module = _load_module("compare_eval_run_main_module", script_path)
    monkeypatch.setattr(module, "REPO_DIR", tmp_path)
    sys.modules.pop("__main__", None)

    captured = {}
    fake_module = types.ModuleType("__main__")
    fake_module.__dict__.update(module.__dict__)
    fake_module.REPO_DIR = tmp_path
    sys.modules["__main__"] = fake_module

    def fake_run(cmd, cwd: str, env: dict):
        out_path = Path(cmd[cmd.index("--out") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"num_tasks": 1, "successes": 1, "timing": {"avg_task_seconds": 1.0}, "episodes": []}), encoding="utf-8")
        captured["called"] = True
        return SimpleNamespace(returncode=0)

    fake_module.subprocess.run = fake_run
    runpy.run_path(str(script_path), run_name="__main__")

    assert captured["called"] is True


def test_generate_tasks_load_operator_env_calls_dotenv(tmp_path: Path, monkeypatch) -> None:
    """Generate tasks load operator env calls dotenv."""
    module = _load_module(
        "generate_tasks_env_module",
        Path(__file__).resolve().parents[1] / "scripts" / "generate_tasks.py",
    )

    calls = []

    fake_dotenv = SimpleNamespace(
        load_dotenv=lambda path, override=True: calls.append((Path(path), override))
    )
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY=test\n", encoding="utf-8")

    module._load_operator_env(tmp_path)

    assert calls == [(env_path, True)]


def test_generate_tasks_main_writes_cache_file(tmp_path: Path, monkeypatch) -> None:
    """Generate tasks main writes cache file."""
    module = _load_module(
        "generate_tasks_main_module",
        Path(__file__).resolve().parents[1] / "scripts" / "generate_tasks.py",
    )

    out_path = tmp_path / "tasks.json"
    monkeypatch.setattr(
        module.argparse.ArgumentParser,
        "parse_args",
        lambda self: Namespace(
            project_id=["autocinema"],
            project_ids=None,
            prompts_per_use_case=1,
            dynamic=False,
            out=str(out_path),
        ),
    )
    monkeypatch.setattr(module, "_load_operator_env", lambda operator_dir: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    payload = {
        "timestamp": "2026-03-18T00:00:00",
        "projects": [{"project_id": "autocinema", "project_name": "Auto Cinema", "num_tasks": 1}],
        "tasks": [{"id": "task-1"}],
    }

    def fake_asyncio_run(coro):
        coro.close()
        return payload

    monkeypatch.setattr(module.asyncio, "run", fake_asyncio_run)

    module.main()

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["projects"][0]["project_id"] == "autocinema"
    assert data["tasks"] == [{"id": "task-1"}]


def test_generate_tasks_generate_uses_autoppia_modules() -> None:
    """Generate tasks generate uses autoppia modules."""
    module = _load_module(
        "generate_tasks_generate_module",
        Path(__file__).resolve().parents[1] / "scripts" / "generate_tasks.py",
    )

    benchmark_module = types.ModuleType("autoppia_iwa.entrypoints.benchmark.utils.task_generation")
    demo_module = types.ModuleType("autoppia_iwa.src.demo_webs.config")

    class FakeTask:
        def __init__(self, task_id: str) -> None:
            self.task_id = task_id

        def serialize(self) -> dict:
            return {"id": self.task_id}

    async def fake_generate_tasks_for_project(project, prompts_per_use_case, use_cases, dynamic):
        assert project.id == "autocinema"
        assert prompts_per_use_case == 2
        assert use_cases is None
        assert dynamic is True
        return [FakeTask("task-1"), FakeTask("task-2")]

    benchmark_module.generate_tasks_for_project = fake_generate_tasks_for_project
    benchmark_module.get_projects_by_ids = lambda projects, ids: [SimpleNamespace(id="autocinema", name="Auto Cinema")]
    demo_module.demo_web_projects = ["demo-projects"]

    sys.modules["autoppia_iwa.entrypoints.benchmark.utils.task_generation"] = benchmark_module
    sys.modules["autoppia_iwa.src.demo_webs.config"] = demo_module

    payload = module.asyncio.run(module._generate("autocinema", 2, True))

    assert payload["project_id"] == "autocinema"
    assert payload["project_name"] == "Auto Cinema"
    assert payload["tasks"] == [{"id": "task-1"}, {"id": "task-2"}]


def test_generate_tasks_main_exits_without_api_key(tmp_path: Path, monkeypatch) -> None:
    """Generate tasks main exits without API key."""
    module = _load_module(
        "generate_tasks_missing_key_module",
        Path(__file__).resolve().parents[1] / "scripts" / "generate_tasks.py",
    )

    monkeypatch.setattr(
        module.argparse.ArgumentParser,
        "parse_args",
        lambda self: Namespace(
            project_id=[],
            project_ids=None,
            prompts_per_use_case=1,
            dynamic=False,
            out=str(tmp_path / "tasks.json"),
        ),
    )
    monkeypatch.setattr(module, "_load_operator_env", lambda operator_dir: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(SystemExit, match="OPENAI_API_KEY missing"):
        module.main()


def test_generate_tasks_main_uses_default_project_and_fails_on_empty_payload(tmp_path: Path, monkeypatch) -> None:
    """Generate tasks main uses default project and fails on empty payload."""
    module = _load_module(
        "generate_tasks_default_project_module",
        Path(__file__).resolve().parents[1] / "scripts" / "generate_tasks.py",
    )

    out_path = tmp_path / "empty.json"
    monkeypatch.setattr(
        module.argparse.ArgumentParser,
        "parse_args",
        lambda self: Namespace(
            project_id=[],
            project_ids=None,
            prompts_per_use_case=1,
            dynamic=False,
            out=str(out_path),
        ),
    )
    monkeypatch.setattr(module, "_load_operator_env", lambda operator_dir: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    async def fake_generate(project_id: str, prompts_per_use_case: int, dynamic: bool) -> dict:
        assert project_id == "autocinema"
        return {"project_id": project_id, "project_name": "Auto Cinema", "tasks": []}

    monkeypatch.setattr(module, "_generate", fake_generate)

    with pytest.raises(SystemExit, match="Generated 0 tasks"):
        module.main()

    assert out_path.exists()
