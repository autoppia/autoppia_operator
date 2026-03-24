from __future__ import annotations

import inspect
from importlib import import_module
from typing import Annotated, Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Autoppia Web Agent API")

_cors_kwargs = {
    "allow_origins": [
        "http://127.0.0.1",
        "http://localhost",
        "http://127.0.0.1:5060",
        "http://localhost:5060",
    ],
    "allow_origin_regex": r"chrome-extension://.*",
    "allow_credentials": False,
    "allow_methods": ["*"],
    "allow_headers": ["*"],
}
if "allow_private_network" in inspect.signature(CORSMiddleware.__init__).parameters:
    _cors_kwargs["allow_private_network"] = True
app.add_middleware(CORSMiddleware, **_cors_kwargs)


def _operator():
    return import_module("src.operator.entrypoint").OPERATOR


@app.get("/health", summary="Health check")
async def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/capabilities", summary="Operator capabilities and protocol metadata")
async def capabilities() -> dict[str, Any]:
    return _operator().capabilities_payload()


@app.post("/act", summary="Decide next agent actions")
async def act(payload: Annotated[dict[str, Any], Body(...)]) -> dict[str, Any]:
    try:
        return await _operator().respond_from_payload(payload)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"act_failed:{type(exc).__name__}") from exc


@app.post("/step", summary="Alias for /act")
async def step(payload: Annotated[dict[str, Any], Body(...)]) -> dict[str, Any]:
    try:
        return await _operator().respond_from_payload(payload)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"step_failed:{type(exc).__name__}") from exc
