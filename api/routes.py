"""FastAPI routes for prediction and service health."""

from __future__ import annotations

from time import perf_counter

from fastapi import APIRouter, HTTPException, Request

from api.deps import state
from credit_scoring.config import MODEL_VERSION
from credit_scoring.inference import run_inference
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
def predict(payload: ClientFeatures, request: Request) -> PredictionResponse:
    if state.model is None:
        request.state.error_message = "model_not_loaded"
        raise HTTPException(status_code=503, detail="Model is not loaded")
    started = perf_counter()
    try:
        result = run_inference(state.model, payload)
        latency_ms = (perf_counter() - started) * 1000
        request.state.inference_ms = result.inference_ms
        request.state.score = result.score
        request.state.decision = result.decision
        request.state.model_version = result.model_version
        return PredictionResponse(
            score=result.score,
            decision=result.decision,
            model_version=result.model_version,
            latency_ms=latency_ms,
        )
    except HTTPException:
        raise
    except Exception as exc:
        request.state.error_message = str(exc)
        request.state.model_version = getattr(state.model, "model_version", MODEL_VERSION)
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
