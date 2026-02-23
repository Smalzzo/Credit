"""Train and export a reproducible demo credit scoring pipeline."""

from __future__ import annotations

from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from credit_scoring.preprocessing import build_model_features


def build_demo_dataset(n_rows: int = 500, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    rng = pd.Series(range(n_rows)).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    raw_features = pd.DataFrame(
        {
            "age": 21 + (rng % 50),
            "income": 18000 + (rng * 130),
            "credit_amount": 3000 + (rng * 40),
            "annuity": 300 + (rng * 3),
            "employment_years": rng % 20,
            "family_members": 1 + (rng % 5),
        }
    )
    features = build_model_features(raw_features)
    risk_signal = (
        0.03 * features["employment_age_perc"]
        + 0.08 * features["income_credit_perc"]
        - 2.0 * features["annuity_income_perc"]
        - 1.2 * features["payment_rate"]
        + 0.000002 * features["income"]
    )
    threshold = float(risk_signal.median())
    target = (risk_signal >= threshold).astype(int)
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
