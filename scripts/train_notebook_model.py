"""Entraîne et exporte un modèle compatible avec le notebook Home Credit.

Ce script produit un artefact `models/notebook_model.joblib` contenant:
- le modèle LightGBM
- la liste ordonnée des features
- les médianes d'imputation
- la version du modèle
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import pandas as pd
from lightgbm import LGBMClassifier

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from generate_submission import build_home_credit_dataset
from scripts.train import build_demo_dataset


def _to_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    converted = frame.copy()
    for col in converted.columns:
        if converted[col].dtype == "object":
            converted[col] = pd.to_numeric(converted[col], errors="coerce")
    return converted


def _prepare_training_set(input_dir: str, debug: bool, demo: bool) -> tuple[pd.DataFrame, pd.Series, str]:
    input_path = Path(input_dir)
    train_file = input_path / "application_train.csv"
    test_file = input_path / "application_test.csv"

    if train_file.exists() and test_file.exists():
        df_all = build_home_credit_dataset(input_dir=input_dir, debug=debug)

        train_df = df_all[df_all["TARGET"].notnull()].copy()
        feats = [
            c
            for c in train_df.columns
            if c not in ["TARGET", "SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV", "index"]
        ]
        X = _to_numeric_frame(train_df[feats])
        y = train_df["TARGET"].astype(int)
        return X, y, "notebook-smail-6-31122025"

    if demo:
        X, y = build_demo_dataset(n_rows=1200, seed=2026)
        return X, y.astype(int), "notebook-smail-6-31122025-demo"

    raise FileNotFoundError(
        "CSV Home Credit introuvables dans data/raw. "
        "Ajoute application_train.csv et application_test.csv, "
        "ou lance le script avec --demo."
    )


def main(input_dir: str = "data/raw", debug: bool = True, demo: bool = False) -> Path:
    X, y, model_version = _prepare_training_set(input_dir=input_dir, debug=debug, demo=demo)

    feats = list(X.columns)

    medians = X.median(numeric_only=True)
    X = X.fillna(medians)

    pos_count = int((y == 1).sum())
    neg_count = int((y == 0).sum())
    scale_pos_weight = neg_count / max(pos_count, 1)

    model = LGBMClassifier(
        n_estimators=2500,
        learning_rate=0.03,
        num_leaves=34,
        subsample=0.85,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=1001,
        n_jobs=-1,
        verbose=-1,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X, y)

    artifact = {
        "model": model,
        "feature_names": feats,
        "medians": medians.to_dict(),
        "model_version": model_version,
        "source_notebook": "notebooks/smail_nasr-allah_6_31122025.ipynb",
    }

    output_path = ROOT_DIR / "models" / "notebook_model.joblib"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)

    print(f"✓ Modèle notebook exporté: {output_path}")
    print(f"✓ Nombre de features: {len(feats)}")
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entraîner et exporter un modèle notebook pour l'API.")
    parser.add_argument("--input-dir", type=str, default="data/raw", help="Dossier des CSV Home Credit")
    parser.add_argument("--full", action="store_true", help="Désactive le mode debug (plus lent)")
    parser.add_argument("--demo", action="store_true", help="Mode synthétique si les CSV ne sont pas disponibles")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(input_dir=args.input_dir, debug=not args.full, demo=args.demo)
