"""Send synthetic requests to local API to generate production-like logs."""

from __future__ import annotations

import random

import requests


def build_payload(valid: bool = True) -> dict:
    if not valid:
        return {
            "age": -10,
            "income": 0,
            "credit_amount": 12000,
            "annuity": 1200,
            "employment_years": 5,
            "family_members": 2,
        }
    return {
        "age": random.randint(20, 70),
        "income": random.randint(20000, 120000),
        "credit_amount": random.randint(1500, 35000),
        "annuity": random.randint(200, 3000),
        "employment_years": random.randint(0, 35),
        "family_members": random.randint(1, 6),
    }


def main(n_valid: int = 50, n_invalid: int = 5, api_url: str = "http://127.0.0.1:8000") -> None:
    endpoint = f"{api_url}/predict"
    for _ in range(n_valid):
        requests.post(endpoint, json=build_payload(valid=True), timeout=10)
    for _ in range(n_invalid):
        requests.post(endpoint, json=build_payload(valid=False), timeout=10)


if __name__ == "__main__":
    main()
