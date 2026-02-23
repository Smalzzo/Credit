"""Application dependencies and singleton state."""

from __future__ import annotations

from dataclasses import dataclass

from credit_scoring.config import MODEL_PATH
from credit_scoring.model import load_model
from credit_scoring.monitoring import RuntimeMetrics


@dataclass
class AppState:
    model: object | None = None
    metrics: RuntimeMetrics = RuntimeMetrics()


state = AppState()


def init_state() -> None:
    if state.model is None:
        state.model = load_model(MODEL_PATH)
