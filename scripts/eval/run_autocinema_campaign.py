#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.focus_pipeline import (
    build_runpod_job_command,
    consolidate_focus_gold,
    export_focus_sft,
    focus_root,
)
from training.use_case_registry import all_use_case_specs


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _default_use_cases() -> list[str]:
    return [spec.name for spec in all_use_case_specs()]


def _parse_use_cases(raw: str) -> list[str]:
    text = str(raw or "").strip()
    if not text or text.lower() == "all":
        return _default_use_cases()
    out: list[str] = []
    for part in text.split(","):
        name = str(part).strip().upper()
        if name and name not in out:
            out.append(name)
    return out


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _copy_contact_holdout_artifact() -> dict[str, Any] | None:
    src = REPO_ROOT / "tmp" / "contact_eval_50_random.json"
    if not src.exists():
        return None
    dst = REPO_ROOT / "data" / "autocinema" / "contact" / "eval" / "contact_holdout_50_random.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    payload = _read_json(dst)
    summary = {
        "use_case": "CONTACT",
        "artifact_path": str(dst),
        "generated_at": _now_utc(),
        "successes": int(payload.get("successes") or 0),
        "total": int(payload.get("total") or len(payload.get("rows") or [])),
        "success_rate": float(payload.get("success_rate") or 0.0),
        "avg_score": float(payload.get("avg_score") or 0.0),
    }
    _write_json(dst.with_name("contact_holdout_50_summary.json"), summary)
    return summary


def _prepare_use_case_exports(*, use_cases: list[str], sft_dir_name: str, split_seed: int, val_ratio: float) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for use_case in use_cases:
        output_root = focus_root(use_case=use_case)
        episodes_path, summary_path, summary = consolidate_focus_gold(output_root=output_root, use_case=use_case)
        manifest = export_focus_sft(
            episodes_path=episodes_path,
            summary_path=summary_path,
            output_dir=output_root / sft_dir_name,
            split_seed=split_seed,
            val_ratio=val_ratio,
        )
        results.append(
            {
                "use_case": use_case,
                "output_root": str(output_root),
                "sft_dir": str((output_root / sft_dir_name).resolve()),
                "gold_summary": summary,
                "sft_manifest": manifest,
            }
        )
    return results


def _merge_sft_manifests(*, run_name: str, prepared: list[dict[str, Any]], base_model: str) -> dict[str, Any]:
    out_dir = REPO_ROOT / "data" / "autocinema_multi" / run_name / "sft"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    source_manifests: list[str] = []
    use_cases: list[str] = []
    per_use_case: list[dict[str, Any]] = []
    system_prompt = ""
    runtime_aligned = True
    source_mode = "trace_only"
    for item in prepared:
        use_case = str(item["use_case"])
        manifest = dict(item["sft_manifest"])
        sft_dir = Path(str(item["sft_dir"])).resolve()
        manifest_path = Path(str(manifest.get("manifest_path") or (sft_dir / "manifest.json"))).resolve()
        train_path = Path(str(manifest.get("train_path") or (sft_dir / "train.jsonl"))).resolve()
        val_path = Path(str(manifest.get("val_path") or (sft_dir / "val.jsonl"))).resolve()
        source_manifests.append(str(manifest_path))
        use_cases.append(use_case)
        train_payloads = _read_jsonl(train_path)
        val_payloads = _read_jsonl(val_path)
        train_rows.extend(train_payloads)
        val_rows.extend(val_payloads)
        if not system_prompt:
            system_prompt = str(manifest.get("system_prompt") or "")
        runtime_aligned = runtime_aligned and bool(manifest.get("runtime_aligned", True))
        source_mode = str(manifest.get("source_mode") or source_mode)
        per_use_case.append(
            {
                "use_case": use_case,
                "manifest_path": str(manifest_path),
                "train_examples": len(train_payloads),
                "val_examples": len(val_payloads),
                "episodes_total": int(manifest.get("episodes_total") or 0),
            }
        )
    _write_jsonl(out_dir / "train.jsonl", train_rows)
    _write_jsonl(out_dir / "val.jsonl", val_rows)
    merged_manifest = {
        "format_version": "autocinema.browser_use.sft.v3",
        "base_model": base_model,
        "training_method": "lora_qlora",
        "system_prompt": system_prompt,
        "source_manifests": source_manifests,
        "use_cases": use_cases,
        "episodes_total": sum(int(item["episodes_total"]) for item in per_use_case),
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "runtime_aligned": runtime_aligned,
        "source_mode": source_mode,
        "run_name": run_name,
        "generated_at": _now_utc(),
        "per_use_case": per_use_case,
        "train_path": str(out_dir / "train.jsonl"),
        "val_path": str(out_dir / "val.jsonl"),
        "manifest_path": str(out_dir / "manifest.json"),
    }
    _write_json(out_dir / "manifest.json", merged_manifest)
    return merged_manifest


def _write_campaign_plan(
    *,
    run_name: str,
    merged_manifest: dict[str, Any],
    use_cases: list[str],
    existing_pod_id: str,
    epochs: int,
    lora_rank: int,
    all_use_cases_eval_tasks: int,
) -> dict[str, Any]:
    run_root = REPO_ROOT / "data" / "autocinema_multi" / run_name
    train_cmd = build_runpod_job_command(
        sft_dir=run_root / "sft",
        output_dir=REPO_ROOT / "models" / run_name,
        existing_pod_id=existing_pod_id,
        epochs=epochs,
        lora_rank=lora_rank,
        include_val_data=True,
    )
    eval_cmd = [
        sys.executable,
        "scripts/sn36_ops.py",
        "eval",
        "--project-id",
        "autocinema",
        "--all-use-cases",
        "--tasks-per-use-case",
        str(int(all_use_cases_eval_tasks)),
        "--distinct-use-cases",
    ]
    payload = {
        "run_name": run_name,
        "generated_at": _now_utc(),
        "use_cases": use_cases,
        "merged_manifest_path": str(run_root / "sft" / "manifest.json"),
        "train_command": train_cmd,
        "eval_command": eval_cmd,
    }
    _write_json(run_root / "campaign_plan.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare a continuous Autocinema all-use-case campaign")
    parser.add_argument("--run-name", default=f"all_use_cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--use-cases", default="all")
    parser.add_argument("--sft-dir-name", default="sft_campaign")
    parser.add_argument("--split-seed", type=int, default=36)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--base-model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--existing-pod-id", default="")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--all-use-cases-eval-tasks", type=int, default=3)
    args = parser.parse_args(argv)

    use_cases = _parse_use_cases(args.use_cases)
    contact_summary = _copy_contact_holdout_artifact()
    prepared = _prepare_use_case_exports(
        use_cases=use_cases,
        sft_dir_name=str(args.sft_dir_name),
        split_seed=int(args.split_seed),
        val_ratio=float(args.val_ratio),
    )
    merged_manifest = _merge_sft_manifests(
        run_name=str(args.run_name),
        prepared=prepared,
        base_model=str(args.base_model),
    )
    plan = _write_campaign_plan(
        run_name=str(args.run_name),
        merged_manifest=merged_manifest,
        use_cases=use_cases,
        existing_pod_id=str(args.existing_pod_id),
        epochs=int(args.epochs),
        lora_rank=int(args.lora_rank),
        all_use_cases_eval_tasks=int(args.all_use_cases_eval_tasks),
    )
    payload = {
        "contact_resolution": contact_summary,
        "prepared_use_cases": prepared,
        "merged_manifest": merged_manifest,
        "campaign_plan": plan,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
