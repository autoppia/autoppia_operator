from __future__ import annotations

import sys

from infra import llm_gateway as _impl

sys.modules[__name__] = _impl
