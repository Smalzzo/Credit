"""Inference orchestration for scoring requests."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import pandas as pd

from credit_scoring.config import DECISION_THRESHOLD, MODEL_VERSION
from credit_scoring.model import predict_score
from credit_scoring.preprocessing import to_feature_frame
from credit_scoring.schema import ClientFeatures


@dataclass
class InferenceResult:
    score: float
    decision: str
    inference_ms: float
    model_version: str


def run_inference(model, payload: ClientFeatures) -> InferenceResult:
    start = perf_counter()
    frame = to_feature_frame(payload)
    score = predict_score(model, frame)
    elapsed = (perf_counter() - start) * 1000
    decision = "ACCEPT" if score >= DECISION_THRESHOLD else "REJECT"
    version = getattr(model, "model_version", MODEL_VERSION)
    return InferenceResult(
        score=score,
        decision=decision,
        inference_ms=elapsed,
        model_version=version,
    )


def run_inference_from_feature_dict(model, features: dict[str, float | int | None]) -> InferenceResult:
    start = perf_counter()
    frame = pd.DataFrame([features])
    score = predict_score(model, frame)
    elapsed = (perf_counter() - start) * 1000
    decision = "ACCEPT" if score >= DECISION_THRESHOLD else "REJECT"
    version = getattr(model, "model_version", MODEL_VERSION)
    return InferenceResult(
        score=score,
        decision=decision,
        inference_ms=elapsed,
        model_version=version,
    )
