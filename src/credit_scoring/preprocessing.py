"""Preprocessing utilities for model inference."""

import pandas as pd

from credit_scoring.schema import ClientFeatures


FEATURE_ORDER = [
    "age",
    "income",
    "credit_amount",
    "annuity",
    "employment_years",
]


def to_feature_frame(payload: ClientFeatures) -> pd.DataFrame:
    values = payload.model_dump()
    return pd.DataFrame([{key: values[key] for key in FEATURE_ORDER}])
