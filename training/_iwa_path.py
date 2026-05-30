"""Bootstrap sys.path for training imports.

1. **Operator repo root** (parent of `training/`) so `import src.operator...` works
   (used by e.g. `format_for_sft` without relying on each script to set cwd).
2. **autoppia_iwa** sibling checkout when not installed: `<workspace>/autoppia_iwa/`
   with inner package `autoppia_iwa/autoppia_iwa/`.

Override: set `AUTOPPIA_IWA_ROOT` to the autoppia_iwa *repository* root
(the directory that contains the inner `autoppia_iwa` package).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def ensure_operator_repo_on_path() -> None:
    training_dir = Path(__file__).resolve().parent
    operator_root = training_dir.parent
    s = str(operator_root)
    if s not in sys.path:
        sys.path.insert(0, s)


def _iwa_repo_root() -> Path | None:
    training_dir = Path(__file__).resolve().parent
    operator_root = training_dir.parent
    default_candidate = operator_root.parent / "autoppia_iwa"
    if (default_candidate / "autoppia_iwa" / "__init__.py").is_file():
        return default_candidate
    env = str(os.environ.get("AUTOPPIA_IWA_ROOT", "")).strip()
    if env:
        p = Path(env).expanduser().resolve()
        if (p / "autoppia_iwa" / "__init__.py").is_file():
            return p
    return None


def ensure_autoppia_iwa_on_path() -> None:
    root = _iwa_repo_root()
    if root is None:
        return
    s = str(root)
    if s not in sys.path:
        # Append (not insert(0)) so the operator repo stays first and `import src` resolves
        # to `autoppia_operator/src`, not a sibling IWA `src` tree.
        sys.path.append(s)


ensure_operator_repo_on_path()
ensure_autoppia_iwa_on_path()
