from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

local_src_dir = str((ROOT / "src").resolve())
src_init_file = ROOT / "src" / "__init__.py"
src_spec = importlib.util.spec_from_file_location(
    "src",
    src_init_file,
    submodule_search_locations=[local_src_dir],
)
if src_spec is not None and src_spec.loader is not None:
    src_module = importlib.util.module_from_spec(src_spec)
    sys.modules["src"] = src_module
    src_spec.loader.exec_module(src_module)
