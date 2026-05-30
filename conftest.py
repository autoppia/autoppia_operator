from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)
# Sibling `autoppia_iwa` repo for IWA-backed tests (see training/_iwa_path.py)
_iwa = ROOT.parent / "autoppia_iwa"
if _iwa.is_dir() and str(_iwa) not in sys.path and not os.environ.get("AUTOPPIA_IWA_ROOT"):
    sys.path.append(str(_iwa))
