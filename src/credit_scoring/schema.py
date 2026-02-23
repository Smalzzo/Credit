"""Pydantic schemas used by the FastAPI service."""

from pydantic import BaseModel, Field


class ClientFeatures(BaseModel):
    age: int = Field(ge=0, le=120)
    income: float = Field(gt=0)
    credit_amount: float = Field(gt=0)
    annuity: float = Field(gt=0)
    employment_years: float = Field(ge=0)
    family_members: float = Field(gt=0)


class PredictionResponse(BaseModel):
    score: float
    decision: str
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
