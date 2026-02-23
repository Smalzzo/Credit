from credit_scoring.inference import run_inference
from credit_scoring.model import DemoModel
from credit_scoring.schema import ClientFeatures


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
