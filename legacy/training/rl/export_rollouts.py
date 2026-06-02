from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    out = [0.0] * len(rewards)
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running = float(rewards[idx]) + float(gamma) * running
        out[idx] = running
    return out


def _assistant_content_from_policy_output(policy_output: dict[str, Any]) -> str:
    return json.dumps(
        {
            "actions": policy_output.get("actions") if isinstance(policy_output.get("actions"), list) else [],
            "done": bool(policy_output.get("done")),
            "content": policy_output.get("content"),
            "reasoning": policy_output.get("reasoning"),
        },
        ensure_ascii=False,
    )


def export_rollouts_to_weighted_sft(
    *,
    rollout_dir: str | Path,
    out_path: str | Path,
    gamma: float = 0.99,
    min_return: float = 0.0,
) -> dict[str, Any]:
    rollout_root = Path(rollout_dir).expanduser().resolve()
    output_path = Path(out_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    episode_files = sorted(p for p in rollout_root.glob("seed_*.json") if p.is_file())
    for path in episode_files:
        episode = json.loads(path.read_text(encoding="utf-8"))
        steps = episode.get("steps") if isinstance(episode.get("steps"), list) else []
        rewards = [float(((step.get("reward") if isinstance(step, dict) else {}) or {}).get("total") or 0.0) for step in steps]
        returns = _discounted_returns(rewards, gamma=float(gamma))
        for step, step_return in zip(steps, returns, strict=False):
            if not isinstance(step, dict):
                continue
            if float(step_return) < float(min_return):
                continue
            user_content = str(step.get("policy_input_text") or "")
            policy_output = step.get("policy_output") if isinstance(step.get("policy_output"), dict) else {}
            assistant_content = _assistant_content_from_policy_output(policy_output)
            weight = max(0.0, float(step_return))
            rows.append(
                {
                    "messages": [
                        {"role": "system", "content": ""},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content},
                    ],
                    "metadata": {
                        "seed": int(episode.get("seed") or 0),
                        "episode_score": float(episode.get("score") or 0.0),
                        "episode_success": bool(episode.get("success")),
                        "step_index": int(step.get("episode_step") or 0),
                        "reward": float(((step.get("reward") if isinstance(step, dict) else {}) or {}).get("total") or 0.0),
                        "return": float(step_return),
                        "weight": float(weight),
                    },
                    "weight": float(weight),
                }
            )
    with output_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    returns = [float(((row.get("metadata") if isinstance(row.get("metadata"), dict) else {}) or {}).get("return") or 0.0) for row in rows]
    summary = {
        "rollout_dir": str(rollout_root),
        "out_path": str(output_path),
        "rows": len(rows),
        "gamma": float(gamma),
        "min_return": float(min_return),
        "avg_return": (sum(returns) / len(returns)) if returns else 0.0,
        "max_return": max(returns) if returns else 0.0,
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
