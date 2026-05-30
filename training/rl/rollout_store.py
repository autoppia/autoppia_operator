from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RolloutStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def write_episode(self, *, seed: int, episode: dict[str, Any]) -> Path:
        path = self.root / f"seed_{int(seed):04d}.json"
        path.write_text(json.dumps(episode, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def write_summary(self, summary: dict[str, Any]) -> Path:
        path = self.root / "summary.json"
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
