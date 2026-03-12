from __future__ import annotations

import sys

from src.operator.runtime import completion as _impl

sys.modules[__name__] = _impl
