"""Preprocessing utilities extracted from notebook feature logic."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from credit_scoring.schema import ClientFeatures


RAW_FEATURE_ORDER = [
    "age",
    "income",
    "credit_amount",
    "annuity",
    "employment_years",
    "family_members",
]

MODEL_FEATURE_ORDER = [
    "age",
    "income",
    "credit_amount",
    "annuity",
    "employment_years",
    "family_members",
    "employment_age_perc",
    "income_credit_perc",
    "income_per_person",
    "annuity_income_perc",
    "payment_rate",
]


def sanitize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", str(col)).strip("_") for col in frame.columns]
    return frame


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    den = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    num = pd.to_numeric(numerator, errors="coerce")
    return num / den


def build_model_features(raw_frame: pd.DataFrame) -> pd.DataFrame:
    frame = raw_frame.copy()
    frame["employment_age_perc"] = _safe_div(frame["employment_years"], frame["age"])
    frame["income_credit_perc"] = _safe_div(frame["income"], frame["credit_amount"])
    frame["income_per_person"] = _safe_div(frame["income"], frame["family_members"])
    frame["annuity_income_perc"] = _safe_div(frame["annuity"], frame["income"])
    frame["payment_rate"] = _safe_div(frame["annuity"], frame["credit_amount"])

    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.fillna(frame.median(numeric_only=True))
    frame = sanitize_columns(frame)
    return frame[MODEL_FEATURE_ORDER]


def to_feature_frame(payload: ClientFeatures) -> pd.DataFrame:
    values = payload.model_dump()
    raw = pd.DataFrame([{key: values[key] for key in RAW_FEATURE_ORDER}])
    return build_model_features(raw)
