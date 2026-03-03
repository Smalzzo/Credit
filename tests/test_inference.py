from credit_scoring.inference import run_inference
from credit_scoring.model import DemoModel
from credit_scoring.schema import ClientFeatures
from credit_scoring.config import DECISION_THRESHOLD


def test_inference_returns_score_in_range():
    model = DemoModel()
    payload = ClientFeatures(
        age=42,
        income=65000,
        credit_amount=15000,
        annuity=1700,
        employment_years=8,
        family_members=3,
    )
    result = run_inference(model, payload)
    assert 0 <= result.score <= 1
    assert result.decision in {"ACCEPT", "REJECT"}


class _FixedRiskModel:
    def __init__(self, risk_proba: float):
        self.risk_proba = risk_proba
        self.model_version = "test-fixed-risk"

    def predict_proba(self, frame):
        import numpy as np

        p = float(self.risk_proba)
        return np.array([[1 - p, p]])


def test_high_risk_score_is_rejected():
    payload = ClientFeatures(
        age=35,
        income=55000,
        credit_amount=12000,
        annuity=1200,
        employment_years=7,
        family_members=3,
    )
    model = _FixedRiskModel(risk_proba=DECISION_THRESHOLD + 0.1)
    result = run_inference(model, payload)

    assert result.score >= DECISION_THRESHOLD
    assert result.decision == "REJECT"
