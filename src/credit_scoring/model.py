"""Model loading and prediction entry points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


@dataclass
class DemoModel:
    model_version: str = "demo-0.1.0"

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        raw = (
            0.002 * frame["age"]
            + 0.000001 * frame["income"]
            - 0.000001 * frame["credit_amount"]
            - 0.000005 * frame["annuity"]
            + 0.01 * frame["employment_years"]
        )
        score = 1.0 / (1.0 + np.exp(-raw.to_numpy(dtype=float)))
        return np.column_stack([1 - score, score])


def load_model(model_path: Path):
    if model_path.exists():
        return joblib.load(model_path)
    return DemoModel()


def predict_score(model, frame: pd.DataFrame) -> float:
    proba = model.predict_proba(frame)
    return float(proba[0, 1])
