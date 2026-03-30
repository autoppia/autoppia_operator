from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _recommended_login_action(url: str, state_out: dict[str, Any] | None) -> dict[str, Any]:
    state_out = state_out if isinstance(state_out, dict) else {}
    current_url = str(url or "")
    if "/login" not in current_url:
        return {
            "name": "browser.navigate",
            "arguments": {"url": "/login"},
            "reason": "login_nav",
        }
    if not bool(state_out.get("typed_username")):
        return {
            "name": "browser.input",
            "arguments": {"field_hint": "username", "text": "user1"},
            "reason": "type_username",
        }
    if not bool(state_out.get("typed_password")):
        return {
            "name": "browser.input",
            "arguments": {"field_hint": "password", "text": "Passw0rd!"},
            "reason": "type_password",
        }
    return {
        "name": "browser.click",
        "arguments": {"control_hint": "sign in"},
        "reason": "submit_login",
    }


def build_login_rule_corrections(*, summary_path: Path, output_path: Path) -> dict[str, Any]:
    payload = _load_json(summary_path)
    results = payload.get("results")
    failed = [row for row in results if isinstance(row, dict) and not bool(row.get("success"))] if isinstance(results, list) else []
    rows: list[dict[str, Any]] = []
    for row in failed:
        seed = int(row.get("seed") or 0)
        current_url = str(row.get("final_url") or "")
        correction = _recommended_login_action(current_url, row.get("state_out"))
        rows.append(
            {
                "seed": seed,
                "failure_category": str(row.get("failure_category") or ""),
                "recommended_action": correction,
            }
        )
    _write_jsonl(output_path, rows)
    return {
        "failed_seed_count": len(failed),
        "correction_rows": len(rows),
        "output_path": str(output_path),
    }
