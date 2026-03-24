import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SAMPLE_HTML = """
<html>
  <head>
    <title>Demo Shop</title>
  </head>
  <body>
    <header>
      <nav>
        <a href="/home?seed=7">Home</a>
        <a href="/catalog?seed=7">Catalog</a>
      </nav>
    </header>
    <main>
      <h1>Featured products</h1>
      <form id="search-form" method="post" action="/search">
        <label for="search-box">Search products</label>
        <input id="search-box" name="query" placeholder="Search catalog" required />
        <select name="category">
          <option value="books">Books</option>
          <option value="games">Games</option>
        </select>
        <button type="submit">Search</button>
      </form>
      <section>
        <article>
          <h2>Camera 3000</h2>
          <p>Price 199</p>
          <a href="/products/camera?seed=7">View camera</a>
          <button aria-label="Add camera">Add to cart</button>
        </article>
        <article>
          <h2>Retro Console</h2>
          <p>Price 89</p>
          <a href="/products/console">View console</a>
          <button>Buy now</button>
        </article>
      </section>
      <input type="hidden" name="token" value="secret" />
    </main>
  </body>
</html>
""".strip()


@pytest.fixture
def sample_html() -> str:
    return SAMPLE_HTML


def load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def install_fake_autoppia_iwa() -> None:
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
