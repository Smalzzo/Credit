"""Application configuration for credit scoring runtime."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "pipeline.joblib"
DATA_DIR = PROJECT_ROOT / "data"
LOG_PATH = DATA_DIR / "production_logs.jsonl"
REFERENCE_DIR = DATA_DIR / "reference"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_VERSION = "0.1.0"
DECISION_THRESHOLD = 0.5
