"""Structured JSON logging helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def hash_payload(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def append_jsonl(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def extract_model_inputs(payload: dict[str, Any], max_features: int = 3000) -> dict[str, float]:
    if not isinstance(payload, dict):
        return {}

    source: dict[str, Any]
    if isinstance(payload.get("features"), dict):
        source = payload["features"]
    else:
        source = payload

    extracted: dict[str, float] = {}
    for key, value in source.items():
        if len(extracted) >= max_features:
            break
        if isinstance(value, bool):
            extracted[str(key)] = float(value)
            continue
        if isinstance(value, (int, float)):
            extracted[str(key)] = float(value)
            continue
    return extracted
