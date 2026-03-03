"""Génère un payload de test compatible avec notebook_model.joblib.

Usage:
    python -m scripts.generate_notebook_payload
    python -m scripts.generate_notebook_payload --output data/payload_notebook_sample.json
    python -m scripts.generate_notebook_payload --call-api
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib


def build_payload(model_path: Path) -> dict:
    artifact = joblib.load(model_path)

    if not isinstance(artifact, dict) or "feature_names" not in artifact or "medians" not in artifact:
        raise ValueError(
            "Le fichier modèle n'a pas le format bundle attendu: "
            "{'model', 'feature_names', 'medians', ...}."
        )

    feature_names = list(artifact["feature_names"])
    medians = dict(artifact["medians"])

    payload = {
        "features": {name: float(medians.get(name, 0.0)) for name in feature_names}
    }
    return payload


def save_payload(payload: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def call_api(payload: dict, api_url: str) -> tuple[int, dict | str]:
    import requests

    response = requests.post(f"{api_url.rstrip('/')}/predict", json=payload, timeout=120)
    try:
        content = response.json()
    except Exception:
        content = response.text
    return response.status_code, content


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Génère un payload features à partir du modèle notebook.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/notebook_model.joblib",
        help="Chemin du modèle notebook exporté",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/payload_notebook_sample.json",
        help="Chemin de sortie du JSON payload",
    )
    parser.add_argument(
        "--call-api",
        action="store_true",
        help="Appeler automatiquement l'endpoint /predict après génération.",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="URL de base de l'API",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable: {model_path}")

    payload = build_payload(model_path)
    save_payload(payload, Path(args.output))

    print(f"✅ Payload généré: {args.output}")
    print(f"✅ Nombre de features: {len(payload['features'])}")

    if args.call_api:
        status_code, content = call_api(payload, args.api_url)
        print(f"📡 Réponse API: HTTP {status_code}")
        print(content)


if __name__ == "__main__":
    main()
