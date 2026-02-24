"""Model loading and prediction entry points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


@dataclass
class DemoModel:
    model_version: str = "demo-0.1.0"

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        raw = (
            0.015 * frame["employment_age_perc"]
            + 0.10 * frame["income_credit_perc"]
            - 2.5 * frame["annuity_income_perc"]
            - 1.8 * frame["payment_rate"]
            + 0.000001 * frame["income"]
        )
        score = 1.0 / (1.0 + np.exp(-raw.to_numpy(dtype=float)))
        return np.column_stack([1 - score, score])


@dataclass
class NotebookModelBundle:
    model: Any
    feature_names: list[str]
    medians: dict[str, float]
    model_version: str = "notebook-smail-6-31122025"


def _to_bundle(artifact: dict) -> NotebookModelBundle:
    feature_names = list(artifact.get("feature_names", []))
    medians = {str(k): float(v) for k, v in artifact.get("medians", {}).items()}
    version = str(artifact.get("model_version", "notebook-smail-6-31122025"))
    return NotebookModelBundle(
        model=artifact["model"],
        feature_names=feature_names,
        medians=medians,
        model_version=version,
    )


def load_model(model_path: Path):
    if model_path.exists():
        artifact = joblib.load(model_path)
        if isinstance(artifact, dict) and {"model", "feature_names", "medians"}.issubset(artifact.keys()):
            return _to_bundle(artifact)
        return artifact
    return DemoModel()


def predict_score(model, frame: pd.DataFrame) -> float:
    if isinstance(model, NotebookModelBundle):
        aligned = frame.reindex(columns=model.feature_names)
        fill_values = {feature: model.medians.get(feature, 0.0) for feature in model.feature_names}
        aligned = aligned.fillna(fill_values)
        proba = model.model.predict_proba(aligned)
        return float(proba[0, 1])

    proba = model.predict_proba(frame)
    return float(proba[0, 1])
