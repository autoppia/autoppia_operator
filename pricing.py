from __future__ import annotations

import sys

from infra import pricing as _impl

sys.modules[__name__] = _impl
