"""PostgreSQL persistence helpers for production API events."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


try:
    import psycopg
except Exception:  # pragma: no cover - optional dependency at runtime
    psycopg = None


@dataclass
class PostgresStorage:
    database_url: str

    def is_available(self) -> bool:
        return bool(self.database_url and psycopg is not None)

    def ensure_schema(self) -> None:
        if not self.is_available():
            return
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS api_calls (
                        id BIGSERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        endpoint TEXT NOT NULL,
                        method TEXT NOT NULL,
                        status_code INTEGER NOT NULL,
                        latency_ms DOUBLE PRECISION NOT NULL,
                        inference_ms DOUBLE PRECISION NOT NULL,
                        payload_hash TEXT,
                        payload_keys JSONB,
                        input_features JSONB,
                        score DOUBLE PRECISION,
                        decision TEXT,
                        model_version TEXT,
                        error TEXT
                    )
                    """
                )
                cursor.execute(
                    """
                    ALTER TABLE api_calls
                    ALTER COLUMN decision TYPE TEXT
                    USING decision::text
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS monitoring_runs (
                        id BIGSERIAL PRIMARY KEY,
                        run_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        operational JSONB NOT NULL,
                        drift JSONB NOT NULL
                    )
                    """
                )
            connection.commit()

    def insert_event(self, event: dict[str, Any]) -> None:
        if not self.is_available():
            return
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO api_calls (
                        timestamp,
                        endpoint,
                        method,
                        status_code,
                        latency_ms,
                        inference_ms,
                        payload_hash,
                        payload_keys,
                        input_features,
                        score,
                        decision,
                        model_version,
                        error
                    )
                    VALUES (
                        COALESCE(%(timestamp)s::timestamptz, NOW()),
                        %(endpoint)s,
                        %(method)s,
                        %(status_code)s,
                        %(latency_ms)s,
                        %(inference_ms)s,
                        %(payload_hash)s,
                        %(payload_keys)s::jsonb,
                        %(input_features)s::jsonb,
                        %(score)s,
                        %(decision)s,
                        %(model_version)s,
                        %(error)s
                    )
                    """,
                    {
                        "timestamp": event.get("timestamp"),
                        "endpoint": event.get("endpoint"),
                        "method": event.get("method"),
                        "status_code": event.get("status_code"),
                        "latency_ms": event.get("latency_ms"),
                        "inference_ms": event.get("inference_ms"),
                        "payload_hash": event.get("payload_hash"),
                        "payload_keys": json.dumps(event.get("payload_keys") or []),
                        "input_features": json.dumps(event.get("input_features") or {}),
                        "score": event.get("score"),
                        "decision": event.get("decision"),
                        "model_version": event.get("model_version"),
                        "error": event.get("error"),
                    },
                )
            connection.commit()

    def fetch_recent_events(self, limit: int = 5000) -> list[dict[str, Any]]:
        if not self.is_available():
            return []
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        timestamp,
                        endpoint,
                        method,
                        status_code,
                        latency_ms,
                        inference_ms,
                        payload_hash,
                        payload_keys,
                        input_features,
                        score,
                        decision,
                        model_version,
                        error
                    FROM api_calls
                    ORDER BY timestamp DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cursor.fetchall()

        events: list[dict[str, Any]] = []
        for row in rows:
            events.append(
                {
                    "timestamp": row[0].isoformat() if row[0] is not None else None,
                    "endpoint": row[1],
                    "method": row[2],
                    "status_code": row[3],
                    "latency_ms": row[4],
                    "inference_ms": row[5],
                    "payload_hash": row[6],
                    "payload_keys": row[7],
                    "input_features": row[8],
                    "score": row[9],
                    "decision": row[10],
                    "model_version": row[11],
                    "error": row[12],
                }
            )
        return events

    def insert_monitoring_run(self, summary: dict[str, Any]) -> None:
        if not self.is_available():
            return
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO monitoring_runs (
                        operational,
                        drift
                    )
                    VALUES (
                        %(operational)s::jsonb,
                        %(drift)s::jsonb
                    )
                    """,
                    {
                        "operational": json.dumps(summary.get("operational") or {}),
                        "drift": json.dumps(summary.get("drift") or {}),
                    },
                )
            connection.commit()

    def fetch_latest_monitoring_run(self) -> dict[str, Any] | None:
        if not self.is_available():
            return None
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT run_at, operational, drift
                    FROM monitoring_runs
                    ORDER BY run_at DESC
                    LIMIT 1
                    """
                )
                row = cursor.fetchone()
        if row is None:
            return None
        return {
            "run_at": row[0].isoformat() if row[0] is not None else None,
            "operational": row[1] or {},
            "drift": row[2] or {},
        }

    def fetch_recent_monitoring_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.is_available():
            return []
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        run_at,
                        operational,
                        drift,
                        (operational ->> 'error_rate')::double precision AS error_rate,
                        (operational ->> 'latency_p95_ms')::double precision AS latency_p95_ms,
                        COALESCE((drift ->> 'enabled')::boolean, false) AS drift_enabled
                    FROM monitoring_runs
                    ORDER BY run_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cursor.fetchall()

        history: list[dict[str, Any]] = []
        for row in rows:
            history.append(
                {
                    "run_at": row[0].isoformat() if row[0] is not None else None,
                    "operational": row[1] or {},
                    "drift": row[2] or {},
                    "error_rate": row[3],
                    "latency_p95_ms": row[4],
                    "drift_enabled": row[5],
                }
            )
        return history

    def clear_api_calls(self) -> int:
        if not self.is_available():
            return 0
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM api_calls")
                count = int(cursor.fetchone()[0])
                cursor.execute("TRUNCATE TABLE api_calls RESTART IDENTITY")
            connection.commit()
        return count

    def clear_monitoring_runs(self) -> int:
        if not self.is_available():
            return 0
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM monitoring_runs")
                count = int(cursor.fetchone()[0])
                cursor.execute("TRUNCATE TABLE monitoring_runs RESTART IDENTITY")
            connection.commit()
        return count

    def clear_all_monitoring_data(self) -> dict[str, int]:
        return {
            "api_calls": self.clear_api_calls(),
            "monitoring_runs": self.clear_monitoring_runs(),
        }
