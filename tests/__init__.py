from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Prefer the repository-local `src` package over any third-party `src` namespace.
loaded_src = sys.modules.get("src")
if loaded_src is not None:
    local_src = str((ROOT / "src").resolve())
    loaded_src_file = str(getattr(loaded_src, "__file__", "") or "")
    loaded_src_paths = [str(Path(p).resolve()) for p in list(getattr(loaded_src, "__path__", []))]
    is_local_src = loaded_src_file.startswith(local_src) or any(path.startswith(local_src) for path in loaded_src_paths)
    if not is_local_src:
        sys.modules.pop("src", None)
