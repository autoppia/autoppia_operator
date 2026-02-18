from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import hashlib
import json
import math
import os
import re
import uuid


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]{2,}", (text or "").lower())


class HashEmbedding:
    """Simple deterministic text embedding without external dependencies."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = int(dim)

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        toks = _tokenize(text)
        if not toks:
            return vec

        for t in toks:
            h = hashlib.sha1(t.encode("utf-8")).hexdigest()
            idx = int(h[:8], 16) % self.dim
            sign = 1.0 if (int(h[8:10], 16) % 2 == 0) else -1.0
            vec[idx] += sign

        nrm = math.sqrt(sum(x * x for x in vec))
        if nrm > 0:
            vec = [x / nrm for x in vec]
        return vec


@dataclass
class MemoryHit:
    score: float
    outcome: str
    task: str
    summary: str
    intents: list[str]
    reward: float
    created_at: str


class TrajectoryMemory:
    """Stores successful and failed trajectories and retrieves similar past behavior."""

    def __init__(self, path: str | None = None, dim: int = 256) -> None:
        self.path = Path(path or os.getenv("TRAJECTORY_MEMORY_PATH", "data/memory/trajectories.jsonl"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = HashEmbedding(dim=dim)

    def save_trajectory(
        self,
        *,
        task_id: str,
        task: str,
        outcome: str,
        reward: float,
        steps: list[dict[str, Any]],
        summary: str,
        intents: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        outcome_norm = "success" if str(outcome).lower() in {"success", "ok", "pass", "true", "1"} else "error"
        row = {
            "id": str(uuid.uuid4()),
            "task_id": str(task_id or ""),
            "task": _norm_ws(task),
            "task_embedding": self.embedder.embed(task),
            "outcome": outcome_norm,
            "reward": float(reward),
            "summary": _norm_ws(summary)[:600],
            "intents": intents,
            "steps": steps,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
        return row

    def _iter_rows(self, max_rows: int = 5000) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []

        rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                    if isinstance(obj, dict):
                        rows.append(obj)
                except Exception:
                    continue
        if len(rows) > max_rows:
            rows = rows[-max_rows:]
        return rows

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        return float(sum(x * y for x, y in zip(a, b)))

    def retrieve_similar(
        self,
        *,
        task: str,
        k_success: int = 3,
        k_error: int = 2,
        min_score: float = 0.15,
    ) -> dict[str, list[MemoryHit]]:
        q = self.embedder.embed(task)
        scored: list[tuple[float, dict[str, Any]]] = []
        for r in self._iter_rows():
            emb = r.get("task_embedding")
            if not isinstance(emb, list):
                continue
            try:
                embf = [float(x) for x in emb]
            except Exception:
                continue
            s = self._cosine(q, embf)
            if s >= float(min_score):
                scored.append((s, r))

        successes: list[MemoryHit] = []
        errors: list[MemoryHit] = []
        scored.sort(key=lambda t: t[0], reverse=True)
        for score, r in scored:
            hit = MemoryHit(
                score=float(score),
                outcome=str(r.get("outcome") or "error"),
                task=str(r.get("task") or ""),
                summary=str(r.get("summary") or ""),
                intents=[str(x) for x in (r.get("intents") or []) if str(x)],
                reward=float(r.get("reward") or 0.0),
                created_at=str(r.get("created_at") or ""),
            )
            if hit.outcome == "success" and len(successes) < int(k_success):
                successes.append(hit)
            elif hit.outcome != "success" and len(errors) < int(k_error):
                errors.append(hit)
            if len(successes) >= int(k_success) and len(errors) >= int(k_error):
                break

        return {"success": successes, "error": errors}


def infer_high_level_intents(task: str) -> list[str]:
    t = (task or "").lower()
    intents: list[str] = []

    rules = [
        ("LOGIN", ["login", "log in", "sign in", "signin"]),
        ("SIGNUP", ["sign up", "signup", "register", "create account"]),
        ("SEARCH", ["search", "find", "look for"]),
        ("OPEN_MOVIE", ["movie", "film", "title", "details", "showtime"]),
        ("BOOKING", ["book", "reserve", "ticket", "seat", "checkout", "payment"]),
        ("NAVIGATION", ["go to", "open", "navigate", "visit"]),
        ("FORM_FILL", ["fill", "form", "enter", "type"]),
        ("FILTER_SORT", ["filter", "sort", "price", "rating", "date"]),
    ]

    for name, keys in rules:
        if any(k in t for k in keys):
            intents.append(name)

    if not intents:
        intents.append("GENERAL_WEB_TASK")
    return intents


def format_memory_context(mem: dict[str, list[MemoryHit]]) -> str:
    lines: list[str] = []
    succ = mem.get("success") or []
    err = mem.get("error") or []

    if succ:
        lines.append("SIMILAR SUCCESSFUL TRAJECTORIES:")
        for h in succ:
            intents = ",".join(h.intents[:4])
            lines.append(f"- score={h.score:.2f} intents=[{intents}] summary={_norm_ws(h.summary)[:240]}")
    if err:
        lines.append("SIMILAR FAILED TRAJECTORIES:")
        for h in err:
            intents = ",".join(h.intents[:4])
            lines.append(f"- score={h.score:.2f} intents=[{intents}] summary={_norm_ws(h.summary)[:240]}")

    if not lines:
        return ""
    return "\n".join(lines)
