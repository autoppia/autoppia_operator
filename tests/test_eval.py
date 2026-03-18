import asyncio
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import importlib.util
import pytest


def _install_fake_autoppia_iwa() -> None:
    root = types.ModuleType("autoppia_iwa")
    src = types.ModuleType("autoppia_iwa.src")
    data_generation = types.ModuleType("autoppia_iwa.src.data_generation")
    tasks_pkg = types.ModuleType("autoppia_iwa.src.data_generation.tasks")
    tasks_classes = types.ModuleType("autoppia_iwa.src.data_generation.tasks.classes")
    evaluation_pkg = types.ModuleType("autoppia_iwa.src.evaluation")
    stateful = types.ModuleType("autoppia_iwa.src.evaluation.stateful_evaluator")
    execution_pkg = types.ModuleType("autoppia_iwa.src.execution")
    actions_pkg = types.ModuleType("autoppia_iwa.src.execution.actions")
    actions_actions = types.ModuleType("autoppia_iwa.src.execution.actions.actions")
    actions_base = types.ModuleType("autoppia_iwa.src.execution.actions.base")

    class FakeTask:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeBaseAction:
        @staticmethod
        def create_action(raw):
            return types.SimpleNamespace(type=raw.get("type", "Unknown"), text=raw.get("text"))

    class FakeAsyncStatefulEvaluator:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        async def reset(self):
            return None

        async def step(self, action):
            return None

        async def close(self):
            return None

    tasks_classes.Task = FakeTask
    actions_base.BaseAction = FakeBaseAction
    stateful.AsyncStatefulEvaluator = FakeAsyncStatefulEvaluator

    sys.modules["autoppia_iwa"] = root
    sys.modules["autoppia_iwa.src"] = src
    sys.modules["autoppia_iwa.src.data_generation"] = data_generation
    sys.modules["autoppia_iwa.src.data_generation.tasks"] = tasks_pkg
    sys.modules["autoppia_iwa.src.data_generation.tasks.classes"] = tasks_classes
    sys.modules["autoppia_iwa.src.evaluation"] = evaluation_pkg
    sys.modules["autoppia_iwa.src.evaluation.stateful_evaluator"] = stateful
    sys.modules["autoppia_iwa.src.execution"] = execution_pkg
    sys.modules["autoppia_iwa.src.execution.actions"] = actions_pkg
    sys.modules["autoppia_iwa.src.execution.actions.actions"] = actions_actions
    sys.modules["autoppia_iwa.src.execution.actions.base"] = actions_base


def _load_eval_module(name: str) -> object:
    _install_fake_autoppia_iwa()
    path = Path(__file__).resolve().parents[1] / "eval.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_eval_helpers_and_load_tasks(tmp_path: Path) -> None:
    """Eval helpers and load tasks."""
    module = _load_eval_module("eval_helpers_module")
    cache_path = tmp_path / "tasks.json"
    cache_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {"id": "1", "prompt": "Login", "url": "http://demo", "web_project_id": "wp1", "use_case": {"name": "LOGIN"}},
                    {"id": "2", "prompt": "Buy", "url": "http://demo", "web_project_id": "wp2", "use_case": {"name": "CHECKOUT"}},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert module._task_row_matches_filters({"id": "1", "use_case": {"name": "LOGIN"}, "web_project_id": "wp1"}, "1", "log", "wp1") is True
    assert module._serialize_screenshot(None) is None
    assert module._serialize_screenshot("abc") == "abc"
    assert module._serialize_screenshot(b"abc") == "YWJj"

    task = SimpleNamespace(url="http://demo/page", prompt="demo")
    seeded, seed = module.inject_seed(task, seed=7)
    tasks = module.load_tasks(cache_path=cache_path, use_case="login", limit=5)

    assert seeded.url == "http://demo/page?seed=7"
    assert seed == 7
    assert len(tasks) == 1
    assert tasks[0].id == "1"


def test_load_tasks_skips_invalid_rows(tmp_path: Path) -> None:
    """Load tasks skips invalid rows."""
    module = _load_eval_module("eval_invalid_rows_module")
    cache_path = tmp_path / "tasks.json"
    cache_path.write_text(json.dumps([{"id": "1"}, {"id": "2", "prompt": "Buy", "url": "http://demo"}]), encoding="utf-8")

    class StrictTask:
        def __init__(self, **kwargs):
            if "prompt" not in kwargs:
                raise ValueError("missing prompt")
            self.__dict__.update(kwargs)

    module.Task = StrictTask
    tasks = module.load_tasks(cache_path=cache_path, limit=5)

    assert len(tasks) == 1
    assert tasks[0].id == "2"


def test_start_agent_server_sync_writes_log_and_invokes_popen(tmp_path: Path, monkeypatch) -> None:
    """Start agent server sync writes log and invokes popen."""
    module = _load_eval_module("eval_server_module")
    captured = {}

    class FakePopen:
        def __init__(self, cmd, cwd, stdout, stderr, env):
            captured["cmd"] = cmd
            captured["cwd"] = cwd
            captured["env"] = env
            self.stdout = stdout

    monkeypatch.setattr(module.subprocess, "Popen", FakePopen)

    proc, log_f = module._start_agent_server_sync(5555, tmp_path / "server.log", {"X": "1"}, tmp_path)
    log_f.close()

    assert captured["cmd"][-1] == "5555"
    assert captured["cwd"] == str(tmp_path)
    assert (tmp_path / "server.log").read_text(encoding="utf-8").startswith("\n=== uvicorn main:app port=5555 ===")
    assert proc is not None


def test_run_evaluation_returns_when_no_tasks(monkeypatch, tmp_path: Path) -> None:
    """Run evaluation returns when no tasks."""
    module = _load_eval_module("eval_empty_module")
    monkeypatch.setattr(module, "SCRIPT_DIR", tmp_path)
    monkeypatch.setattr(module, "TASK_CACHE", tmp_path / "missing.json")
    monkeypatch.setattr(module, "load_tasks", lambda **kwargs: [])
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:9000/openai/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = asyncio.run(module.run_evaluation(num_tasks=1, opts={"task_cache": str(tmp_path / "missing.json")}))

    assert result is None


def test_run_evaluation_exits_without_required_keys(monkeypatch, tmp_path: Path) -> None:
    """Run evaluation exits without required keys."""
    module = _load_eval_module("eval_missing_keys_module")
    monkeypatch.setattr(module, "SCRIPT_DIR", tmp_path)
    monkeypatch.setattr(module, "TASK_CACHE", tmp_path / "missing.json")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    with pytest.raises(SystemExit):
        asyncio.run(module.run_evaluation(provider="openai", num_tasks=1, opts={"task_cache": str(tmp_path / "missing.json")}))

    with pytest.raises(SystemExit):
        asyncio.run(module.run_evaluation(provider="anthropic", num_tasks=1, opts={"task_cache": str(tmp_path / "missing.json")}))


def test_run_evaluation_success_flow(monkeypatch, tmp_path: Path) -> None:
    """Run evaluation success flow."""
    module = _load_eval_module("eval_success_module")
    monkeypatch.setattr(module, "SCRIPT_DIR", tmp_path)
    monkeypatch.setattr(module, "OPERATOR_ROOT", tmp_path)
    monkeypatch.setattr(module, "TASK_CACHE", tmp_path / "tasks.json")
    monkeypatch.setattr(module.asyncio, "sleep", lambda _: asyncio.sleep(0))

    task = module.Task(id="task-1", prompt="Buy camera", url="http://demo", web_project_id="wp1", use_case={"name": "BUY"})
    monkeypatch.setattr(module, "load_tasks", lambda **kwargs: [task])

    class FakeEvaluator:
        def __init__(self, task, web_agent_id):
            self.task = task
            self.web_agent_id = web_agent_id

        async def reset(self):
            return SimpleNamespace(
                snapshot=SimpleNamespace(html="<html><body>start</body></html>", url="http://demo?seed=7", screenshot=b"img"),
                score=SimpleNamespace(raw_score=0.0, success=False),
                action_result=None,
            )

        async def step(self, action):
            return SimpleNamespace(
                snapshot=SimpleNamespace(html="<html><body>done</body></html>", url="http://demo/next?seed=7", screenshot=None),
                score=SimpleNamespace(raw_score=1.0, success=True),
                action_result=SimpleNamespace(successfully_executed=True, error=None),
            )

        async def close(self):
            return None

    monkeypatch.setattr(module, "AsyncStatefulEvaluator", FakeEvaluator)
    monkeypatch.setenv("START_AGENT_SERVER", "0")
    monkeypatch.setenv("AGENT_BASE_URL", "http://agent.test")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:9000/openai/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    class FakeResponse:
        async def json(self):
            return {
                "actions": [{"type": "ClickAction", "text": "Buy now"}],
                "metrics": {
                    "decision": "click",
                    "candidate_id": 0,
                    "llm": {
                        "model": "gpt-5-mini",
                        "llm_calls": 1,
                        "llm_usages": [{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}],
                    },
                },
            }

        def raise_for_status(self):
            return None

    class FakePostContext:
        async def __aenter__(self):
            return FakeResponse()

        async def __aexit__(self, exc_type, exc, tb):
            return None

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        def post(self, url, json):
            return FakePostContext()

    sys.modules["aiohttp"] = types.SimpleNamespace(ClientSession=FakeSession)

    result = asyncio.run(
        module.run_evaluation(
            provider="openai",
            model="gpt-5-mini",
            num_tasks=1,
            max_steps=2,
            opts={"task_cache": str(tmp_path / "tasks.json"), "out_path": str(tmp_path / "out.json")},
        )
    )

    assert result["successes"] == 1
    assert result["failures"] == 0
    assert result["errors"] == 0
    assert result["episodes"][0]["total_tokens"] == 15
    assert (tmp_path / "out.json").exists()


def test_run_evaluation_model_mismatch_records_error(monkeypatch, tmp_path: Path) -> None:
    """Run evaluation model mismatch records error."""
    module = _load_eval_module("eval_mismatch_module")
    monkeypatch.setattr(module, "SCRIPT_DIR", tmp_path)
    monkeypatch.setattr(module, "OPERATOR_ROOT", tmp_path)
    monkeypatch.setattr(module, "TASK_CACHE", tmp_path / "tasks.json")

    task = module.Task(id="task-2", prompt="Buy", url="http://demo", web_project_id="wp1", use_case={"name": "BUY"})
    monkeypatch.setattr(module, "load_tasks", lambda **kwargs: [task])

    class FakeEvaluator:
        async def reset(self):
            return SimpleNamespace(
                snapshot=SimpleNamespace(html="<html></html>", url="http://demo?seed=5", screenshot=None),
                score=SimpleNamespace(raw_score=0.0, success=False),
                action_result=None,
            )

        async def step(self, action):
            return SimpleNamespace(
                snapshot=SimpleNamespace(html="<html></html>", url="http://demo?seed=5", screenshot=None),
                score=SimpleNamespace(raw_score=0.0, success=False),
                action_result=None,
            )

        async def close(self):
            return None

        def __init__(self, task, web_agent_id):
            pass

    monkeypatch.setattr(module, "AsyncStatefulEvaluator", FakeEvaluator)
    monkeypatch.setenv("START_AGENT_SERVER", "0")
    monkeypatch.setenv("AGENT_BASE_URL", "http://agent.test")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:9000/openai/v1")

    class FakeResponse:
        async def json(self):
            return {
                "actions": [{"type": "ClickAction"}],
                "metrics": {
                    "llm": {
                        "model": "different-model",
                        "llm_calls": 1,
                        "llm_usages": [{"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}],
                    }
                },
            }

        def raise_for_status(self):
            return None

    class FakePostContext:
        async def __aenter__(self):
            return FakeResponse()

        async def __aexit__(self, exc_type, exc, tb):
            return None

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        def post(self, url, json):
            return FakePostContext()

    sys.modules["aiohttp"] = types.SimpleNamespace(ClientSession=FakeSession)

    result = asyncio.run(
        module.run_evaluation(
            provider="openai",
            model="gpt-5-mini",
            num_tasks=1,
            max_steps=1,
            opts={"task_cache": str(tmp_path / "tasks.json"), "strict_model": True},
        )
    )

    assert result["errors"] == 1
    assert result["model_mismatch_errors"] == 1


def test_run_evaluation_start_server_failure_and_traces(monkeypatch, tmp_path: Path) -> None:
    """Run evaluation start server failure and traces."""
    module = _load_eval_module("eval_failure_module")
    monkeypatch.setattr(module, "SCRIPT_DIR", tmp_path)
    monkeypatch.setattr(module, "OPERATOR_ROOT", tmp_path)
    monkeypatch.setattr(module, "TASK_CACHE", tmp_path / "tasks.json")

    tasks = [
        module.Task(id="task-1", prompt="Task one", url="http://demo", web_project_id="wp1", use_case={"name": "LOGIN"}),
        module.Task(id="task-2", prompt="Task two", url="http://demo", web_project_id="wp1", use_case={"name": "BUY"}),
    ]
    monkeypatch.setattr(module, "load_tasks", lambda **kwargs: tasks)

    class FakeLog:
        def flush(self):
            return None

        def close(self):
            return None

    class FakeProc:
        def terminate(self):
            return None

        def wait(self, timeout):
            return None

    async def fake_to_thread(fn, *args):
        if fn.__name__ == "_start_agent_server_sync":
            return FakeProc(), FakeLog()
        return fn(*args)

    monkeypatch.setattr(module.asyncio, "to_thread", fake_to_thread)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(module.asyncio, "sleep", lambda _: real_sleep(0))
    monkeypatch.setenv("START_AGENT_SERVER", "1")
    monkeypatch.setenv("AGENT_PORT", "5000")
    monkeypatch.setenv("EVAL_SAVE_TRACES", "1")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:9000/openai/v1")

    class FakeEvaluator:
        def __init__(self, task, web_agent_id):
            self.task = task
            self.step_calls = 0

        async def reset(self):
            return SimpleNamespace(
                snapshot=SimpleNamespace(html="<html><body>initial</body></html>", url="http://demo?seed=1", screenshot=None),
                score=SimpleNamespace(raw_score=0.0, success=False),
                action_result=None,
            )

        async def step(self, action):
            self.step_calls += 1
            return SimpleNamespace(
                snapshot=SimpleNamespace(html="<html><body>still here</body></html>", url="http://demo?seed=1", screenshot=None),
                score=SimpleNamespace(raw_score=0.2, success=False),
                action_result=SimpleNamespace(successfully_executed=False, error="failed"),
            )

        async def close(self):
            return None

    monkeypatch.setattr(module, "AsyncStatefulEvaluator", FakeEvaluator)

    class FakeResponse:
        async def json(self):
            return {"actions": [], "metrics": {"decision": "wait"}}

        def raise_for_status(self):
            return None

    class FakePostContext:
        async def __aenter__(self):
            return FakeResponse()

        async def __aexit__(self, exc_type, exc, tb):
            return None

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        def post(self, url, json):
            return FakePostContext()

    sys.modules["aiohttp"] = types.SimpleNamespace(ClientSession=FakeSession)

    result = asyncio.run(
        module.run_evaluation(
            provider="openai",
            model="gpt-5-mini",
            num_tasks=2,
            max_steps=1,
            opts={"distinct_use_cases": True, "task_cache": str(tmp_path / "tasks.json")},
        )
    )

    assert result["num_tasks"] == 2
    assert result["failures"] == 2
    assert (tmp_path / "data" / "traces").exists()
    assert (tmp_path / "data" / "failures").exists()


def test_eval_main_passes_cli_args_to_run_evaluation(monkeypatch) -> None:
    """Eval main passes cli args to run evaluation."""
    module = _load_eval_module("eval_main_module")

    captured = {}
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["eval.py", "--provider", "anthropic", "--model", "claude-sonnet-4", "--num-tasks", "2", "--max-steps", "3", "--seed", "7", "--repeat", "2", "--temperature", "0.4", "--distinct-use-cases"],
    )
    monkeypatch.setattr(module.asyncio, "run", lambda arg: captured.setdefault("asyncio_arg", arg))

    def fake_run_evaluation(provider, model, num_tasks, max_steps, opts):
        captured["provider"] = provider
        captured["model"] = model
        captured["num_tasks"] = num_tasks
        captured["max_steps"] = max_steps
        captured["opts"] = opts
        return "sentinel"

    monkeypatch.setattr(module, "run_evaluation", fake_run_evaluation)

    module.main()

    assert captured["provider"] == "anthropic"
    assert captured["model"] == "claude-sonnet-4"
    assert captured["num_tasks"] == 2
    assert captured["max_steps"] == 3
    assert captured["opts"]["seed"] == 7
    assert captured["opts"]["repeat"] == 2
    assert captured["opts"]["temperature"] == 0.4
    assert captured["opts"]["distinct_use_cases"] is True
    assert captured["asyncio_arg"] == "sentinel"
