from __future__ import annotations

from fastapi import FastAPI, HTTPException

from autoppia_harvester.claude_code import ClaudeCodeHarvester
from autoppia_harvester.models import HarvestRequest, HarvestResponse


app = FastAPI(title="Autoppia Harvester", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/harvest", response_model=HarvestResponse)
async def harvest(request: HarvestRequest) -> HarvestResponse:
    if not request.prompt:
        raise HTTPException(status_code=400, detail="prompt is required")
    if not request.url:
        raise HTTPException(status_code=400, detail="url is required")
    try:
        return await ClaudeCodeHarvester().harvest(request)
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
