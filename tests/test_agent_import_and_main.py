import importlib.util
import logging
import runpy
import sys
import types
from pathlib import Path


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_agent_import_success_path_and_logging(monkeypatch):
    """Agent import success path and logging."""
    path = Path(__file__).resolve().parents[1] / "agent.py"

    root = types.ModuleType("autoppia_iwa")
    src = types.ModuleType("autoppia_iwa.src")
    execution = types.ModuleType("autoppia_iwa.src.execution")
    actions_pkg = types.ModuleType("autoppia_iwa.src.execution.actions")
    actions_actions = types.ModuleType("autoppia_iwa.src.execution.actions.actions")
    actions_base = types.ModuleType("autoppia_iwa.src.execution.actions.base")
    data_generation = types.ModuleType("autoppia_iwa.src.data_generation")
    tasks_pkg = types.ModuleType("autoppia_iwa.src.data_generation.tasks")
    tasks_classes = types.ModuleType("autoppia_iwa.src.data_generation.tasks.classes")
    web_agents_pkg = types.ModuleType("autoppia_iwa.src.web_agents")
    web_agents_classes = types.ModuleType("autoppia_iwa.src.web_agents.classes")

    class FakeTask:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeBaseAction:
        __module__ = "fake.base_action"

    class FakeWebAgent:
        pass

    tasks_classes.Task = FakeTask
    actions_base.BaseAction = FakeBaseAction
    web_agents_classes.IWebAgent = FakeWebAgent

    monkeypatch.setitem(sys.modules, "autoppia_iwa", root)
    monkeypatch.setitem(sys.modules, "autoppia_iwa.src", src)
    monkeypatch.setitem(sys.modules, "autoppia_iwa.src.execution", execution)
    monkeypatch.setitem(sys.modules, "autoppia_iwa.src.execution.actions", actions_pkg)
    monkeypatch.setitem(sys.modules, "autoppia_iwa.src.execution.actions.actions", actions_actions)
    monkeypatch.setitem(sys.modules, "autoppia_iwa.src.execution.actions.base", actions_base)
    monkeypatch.setitem(sys.modules, "autoppia_iwa.src.data_generation", data_generation)
    monkeypatch.setitem(sys.modules, "autoppia_iwa.src.data_generation.tasks", tasks_pkg)
    monkeypatch.setitem(sys.modules, "autoppia_iwa.src.data_generation.tasks.classes", tasks_classes)
    monkeypatch.setitem(sys.modules, "autoppia_iwa.src.web_agents", web_agents_pkg)
    monkeypatch.setitem(sys.modules, "autoppia_iwa.src.web_agents.classes", web_agents_classes)

    basic_calls = []
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: basic_calls.append(kwargs))
    monkeypatch.setattr(logging.getLogger(), "handlers", [])

    module = _load_module("agent_import_success_module", path)

    assert module._AUTOPPIA_IWA_IMPORT_OK is True
    assert module.Task is FakeTask
    assert module.BaseAction is FakeBaseAction
    assert basic_calls


def test_agent_run_as_main_calls_uvicorn(monkeypatch):
    """Agent run as main calls uvicorn."""
    path = Path(__file__).resolve().parents[1] / "agent.py"

    captured = {}
    fake_uvicorn = types.ModuleType("uvicorn")
    fake_uvicorn.run = lambda app, host, port, reload: captured.update(
        {"app": app, "host": host, "port": port, "reload": reload}
    )
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)

    runpy.run_path(str(path), run_name="__main__")

    assert captured == {
        "app": "main:app",
        "host": "0.0.0.0",
        "port": 8000,
        "reload": True,
    }
