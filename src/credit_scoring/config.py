"""Application configuration for credit scoring runtime."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"


def resolve_model_path() -> Path:
	env_model_path = os.getenv("MODEL_PATH")
	if env_model_path:
		return Path(env_model_path)

	notebook_model_path = MODEL_DIR / "notebook_model.joblib"
	if notebook_model_path.exists():
		return notebook_model_path

	return MODEL_DIR / "pipeline.joblib"


MODEL_PATH = resolve_model_path()
DATA_DIR = PROJECT_ROOT / "data"
LOG_PATH = DATA_DIR / "production_logs.jsonl"
REFERENCE_DIR = DATA_DIR / "reference"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_VERSION = "0.1.0"
DECISION_THRESHOLD = 0.4889
DATABASE_URL = os.getenv("DATABASE_URL", "")
