"""Train and export a reproducible demo credit scoring pipeline."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_demo_dataset(n_rows: int = 500, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    rng = pd.Series(range(n_rows)).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    features = pd.DataFrame(
        {
            "age": 21 + (rng % 50),
            "income": 18000 + (rng * 130),
            "credit_amount": 3000 + (rng * 40),
            "annuity": 300 + (rng * 3),
            "employment_years": rng % 20,
        }
    )
    target = (
        0.003 * features["age"]
        + 0.000002 * features["income"]
        - 0.000001 * features["credit_amount"]
        - 0.000004 * features["annuity"]
        + 0.015 * features["employment_years"]
        > 0.2
    ).astype(int)
    return features, target


def main() -> None:
    features, target = build_demo_dataset()
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )
    pipeline.fit(features, target)
    output_dir = Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_dir / "pipeline.joblib")


if __name__ == "__main__":
    main()
