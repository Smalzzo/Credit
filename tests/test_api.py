from fastapi.testclient import TestClient

from api.main import app
from api.deps import init_state


client = TestClient(app)


def test_predict_ok():
    init_state()
    payload = {
        "age": 35,
        "income": 50000,
        "credit_amount": 12000,
        "annuity": 1200,
        "employment_years": 6,
        "family_members": 3,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0 <= data["score"] <= 1


def test_predict_validation_error():
    payload = {
        "age": -5,
        "income": 50000,
        "credit_amount": 12000,
        "annuity": 1200,
        "employment_years": 6,
        "family_members": 3,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
