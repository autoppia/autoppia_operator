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


async def _respond(payload: dict[str, Any], *, endpoint_name: str) -> dict[str, Any]:
    try:
        return await _operator().respond_from_payload(payload)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{endpoint_name}_failed:{type(exc).__name__}") from exc


@app.get("/health", summary="Health check")
async def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/capabilities", summary="Operator capabilities and protocol metadata")
async def capabilities() -> dict[str, Any]:
    return _operator().capabilities_payload()


@app.post("/step", summary="Decide next agent actions")
async def step(payload: Annotated[dict[str, Any], Body(...)]) -> dict[str, Any]:
    return await _respond(payload, endpoint_name="step")


@app.post("/act", summary="Compatibility alias for /step")
async def act(payload: Annotated[dict[str, Any], Body(...)]) -> dict[str, Any]:
    return await _respond(payload, endpoint_name="act")
