#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


def _wait_for_job(client: OpenAI, job_id: str, poll_s: float = 20.0, timeout_s: float = 60 * 60 * 8) -> dict[str, Any]:
    t0 = time.time()
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = str(getattr(job, "status", "unknown"))
        print(json.dumps({"job_id": job_id, "status": status}, ensure_ascii=True))
        if status in {"succeeded", "failed", "cancelled"}:
            return job.model_dump() if hasattr(job, "model_dump") else dict(job)
        if (time.time() - t0) > timeout_s:
            raise TimeoutError(f"Timed out waiting for fine-tune job {job_id}")
        time.sleep(float(poll_s))


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    ap = argparse.ArgumentParser(description="Upload training file and start OpenAI fine-tune job")
    ap.add_argument("--train-file", required=True, help="Path to OpenAI chat-format JSONL")
    ap.add_argument("--base-model", default="gpt-4o-mini-2024-07-18")
    ap.add_argument("--suffix", default="autoppia-autocinema-v2")
    ap.add_argument("--n-epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--learning-rate-multiplier", type=float, default=1.0)
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--out", default=None, help="Output JSON with job/model ids")
    args = ap.parse_args()

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY is missing.")

    client = OpenAI(api_key=key)
    train_path = Path(args.train_file).resolve()
    if not train_path.exists():
        raise SystemExit(f"Train file not found: {train_path}")

    with train_path.open("rb") as f:
        up = client.files.create(file=f, purpose="fine-tune")

    hp = {
        "n_epochs": int(args.n_epochs),
        "batch_size": int(args.batch_size),
        "learning_rate_multiplier": float(args.learning_rate_multiplier),
    }
    job = client.fine_tuning.jobs.create(
        model=str(args.base_model),
        training_file=up.id,
        suffix=str(args.suffix),
        hyperparameters=hp,
    )

    payload: dict[str, Any] = {
        "uploaded_file_id": up.id,
        "job_id": job.id,
        "job_status": job.status,
        "base_model": str(args.base_model),
        "train_file": str(train_path),
    }
    print(json.dumps(payload, ensure_ascii=True))

    if args.wait:
        final = _wait_for_job(client, job.id)
        payload["final_job"] = final
        try:
            payload["fine_tuned_model"] = final.get("fine_tuned_model")
        except Exception:
            payload["fine_tuned_model"] = None

    if args.out:
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
