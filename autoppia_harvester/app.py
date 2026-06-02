from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException

from autoppia_harvester.claude_code import ClaudeCodeHarvester
from autoppia_harvester.models import FindTrayectoryRequest, FindTrayectoryResponse
from autoppia_harvester.openai_provider import OpenAIHarvester


app = FastAPI(title="Autoppia Harvester", version="0.1.0")


def build_harvester():
    provider = str(os.getenv("AUTOPPIA_HARVESTER_PROVIDER", "claude_code")).strip().lower()
    if provider in {"openai", "gpt5", "gpt-5"}:
        return OpenAIHarvester()
    if provider in {"claude", "claude_code", "claude-code"}:
        return ClaudeCodeHarvester()
    raise RuntimeError(f"Unsupported harvester provider: {provider}")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/find_trayectory", response_model=FindTrayectoryResponse)
async def find_trayectory(request: FindTrayectoryRequest) -> FindTrayectoryResponse:
    if not request.prompt:
        raise HTTPException(status_code=400, detail="prompt is required")
    if not request.url:
        raise HTTPException(status_code=400, detail="url is required")
    try:
        return await build_harvester().find_trayectory(request)
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/harvest", response_model=FindTrayectoryResponse, deprecated=True)
async def harvest(request: FindTrayectoryRequest) -> FindTrayectoryResponse:
    return await find_trayectory(request)
