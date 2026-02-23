import pytest
from pydantic import ValidationError

from credit_scoring.schema import ClientFeatures


def test_invalid_age_raises_validation_error():
    with pytest.raises(ValidationError):
        ClientFeatures(
            age=-1,
            income=1000,
            credit_amount=500,
            annuity=50,
            employment_years=2,
            family_members=2,
        )


def test_invalid_income_raises_validation_error():
    with pytest.raises(ValidationError):
        ClientFeatures(
            age=30,
            income=0,
            credit_amount=500,
            annuity=50,
            employment_years=2,
            family_members=2,
        )
