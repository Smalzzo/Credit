"""FastAPI application entrypoint."""

from __future__ import annotations

import json
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.deps import init_state, state
from api.routes import router
from credit_scoring.config import LOG_PATH, MODEL_VERSION
from credit_scoring.logging_utils import append_jsonl, hash_payload


app = FastAPI(title="Credit Scoring API", version="0.1.0")
app.include_router(router)


@app.middleware("http")
async def structured_logging_middleware(request: Request, call_next):
    started = perf_counter()
    raw_body = await request.body()
    payload_hash = None
    payload_keys: list[str] | None = None

    if raw_body:
        try:
            parsed_body = json.loads(raw_body.decode("utf-8"))
            if isinstance(parsed_body, dict):
                payload_hash = hash_payload(parsed_body)
                payload_keys = sorted(parsed_body.keys())
            else:
                payload_hash = hash_payload({"payload_type": type(parsed_body).__name__})
        except Exception:
            payload_hash = hash_payload({"raw_payload": raw_body.decode("utf-8", errors="ignore")[:2048]})

    async def receive() -> dict:
        return {"type": "http.request", "body": raw_body, "more_body": False}

    request_with_body = Request(request.scope, receive)
    response_status = 500

    try:
        response = await call_next(request_with_body)
        response_status = response.status_code
        return response
    except Exception as exc:
        request_with_body.state.error_message = str(exc)
        response_status = 500
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})
    finally:
        latency_ms = (perf_counter() - started) * 1000
        inference_ms = float(getattr(request_with_body.state, "inference_ms", 0.0))
        state.metrics.record(latency_ms, inference_ms, is_error=response_status >= 400)

        error_message = getattr(request_with_body.state, "error_message", None)
        if error_message is None and response_status == 422:
            error_message = "validation_error"

        event = {
            "endpoint": request.url.path,
            "method": request.method,
            "status_code": response_status,
            "latency_ms": latency_ms,
            "inference_ms": inference_ms,
            "payload_hash": payload_hash,
            "payload_keys": payload_keys,
            "score": getattr(request_with_body.state, "score", None),
            "decision": getattr(request_with_body.state, "decision", None),
            "model_version": getattr(request_with_body.state, "model_version", MODEL_VERSION),
            "error": error_message,
        }
        try:
            append_jsonl(LOG_PATH, event)
        except Exception:
            pass


@app.on_event("startup")
def startup_event() -> None:
    init_state()
