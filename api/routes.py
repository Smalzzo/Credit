"""FastAPI routes for prediction and service health."""

from __future__ import annotations

from time import perf_counter

from fastapi import APIRouter, HTTPException

from api.deps import state
from credit_scoring.config import LOG_PATH, MODEL_VERSION
from credit_scoring.inference import run_inference
from credit_scoring.logging_utils import append_jsonl, hash_payload
from credit_scoring.schema import ClientFeatures, HealthResponse, PredictionResponse


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=MODEL_VERSION,
        model_loaded=state.model is not None,
    )


@router.get("/metrics")
def metrics() -> dict:
    return state.metrics.summary()


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: ClientFeatures) -> PredictionResponse:
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    started = perf_counter()
    try:
        result = run_inference(state.model, payload)
        latency_ms = (perf_counter() - started) * 1000
        state.metrics.record(latency_ms, result.inference_ms, is_error=False)
        append_jsonl(
            LOG_PATH,
            {
                "endpoint": "/predict",
                "status_code": 200,
                "latency_ms": latency_ms,
                "inference_ms": result.inference_ms,
                "payload_hash": hash_payload(payload.model_dump()),
                "score": result.score,
                "model_version": result.model_version,
            },
        )
        return PredictionResponse(
            score=result.score,
            decision=result.decision,
            model_version=result.model_version,
            latency_ms=latency_ms,
        )
    except HTTPException:
        raise
    except Exception as exc:
        latency_ms = (perf_counter() - started) * 1000
        state.metrics.record(latency_ms, 0.0, is_error=True)
        append_jsonl(
            LOG_PATH,
            {
                "endpoint": "/predict",
                "status_code": 500,
                "latency_ms": latency_ms,
                "inference_ms": 0.0,
                "payload_hash": hash_payload(payload.model_dump()),
                "error": str(exc),
                "model_version": getattr(state.model, "model_version", MODEL_VERSION),
            },
        )
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
