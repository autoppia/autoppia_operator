from __future__ import annotations

from fastapi import FastAPI, HTTPException

from autoppia_harvester.claude_code import ClaudeCodeHarvester
from autoppia_harvester.models import FindTrayectoryRequest, FindTrayectoryResponse


app = FastAPI(title="Autoppia Harvester", version="0.1.0")


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
        return await ClaudeCodeHarvester().find_trayectory(request)
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/harvest", response_model=FindTrayectoryResponse, deprecated=True)
async def harvest(request: FindTrayectoryRequest) -> FindTrayectoryResponse:
    return await find_trayectory(request)
