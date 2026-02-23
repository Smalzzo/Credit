"""Helpers for data drift report generation."""

from pathlib import Path

import pandas as pd


def load_reference(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_production(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)
