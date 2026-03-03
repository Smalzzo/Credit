"""Automated monitoring analysis: drift + operational anomalies."""

from __future__ import annotations

import csv
import io
import json
import warnings
from pathlib import Path
from typing import Any
import sys

import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from credit_scoring.config import DATABASE_URL
from credit_scoring.storage import PostgresStorage


def _build_reference_row_from_raw_json(raw_reference_path: Path, model_bundle_path: Path) -> dict[str, float] | None:
    if not raw_reference_path.exists() or not model_bundle_path.exists():
        return None

    try:
        raw_json = json.loads(raw_reference_path.read_text(encoding="utf-8"))
        columns_csv = str(raw_json.get("columns_csv", "")).strip()
        row_csv = str(raw_json.get("row_csv", "")).strip()
        if not columns_csv or not row_csv:
            return None

        columns = next(csv.reader(io.StringIO(columns_csv)))
        values = next(csv.reader(io.StringIO(row_csv)))
        row_dict = dict(zip(columns, values))

        artifact = joblib.load(model_bundle_path)
        if not isinstance(artifact, dict):
            return None

        feature_names = list(artifact.get("feature_names") or [])
        medians = dict(artifact.get("medians") or {})
        if not feature_names:
            return None

        base_features = {name: float(medians.get(name, 0.0)) for name in feature_names}
        for key in feature_names:
            if key not in row_dict:
                continue
            value = row_dict.get(key)
            if value in (None, ""):
                continue
            try:
                base_features[key] = float(value)
            except (TypeError, ValueError):
                continue
        return base_features
    except Exception:
        return None


def ensure_reference_csv(reference_path: Path) -> bool:
    if reference_path.exists():
        return True

    raw_reference_path = Path("data/reference/home_credit_reference_raw.json")
    model_bundle_path = Path("models/notebook_model.joblib")
    reference_row = _build_reference_row_from_raw_json(raw_reference_path, model_bundle_path)
    if not reference_row:
        return False

    reference_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([reference_row]).to_csv(reference_path, index=False)
    return True


def load_production_logs(limit: int = 15000) -> pd.DataFrame:
    storage = PostgresStorage(DATABASE_URL)
    events: list[dict[str, Any]] = []

    if storage.is_available():
        try:
            events = storage.fetch_recent_events(limit=limit)
        except Exception:
            events = []

    if events:
        return pd.DataFrame(events)

    fallback_path = Path("data/production_logs.jsonl")
    if fallback_path.exists():
        return pd.read_json(fallback_path, lines=True)

    return pd.DataFrame()


def build_features_frame(logs: pd.DataFrame) -> pd.DataFrame:
    if logs.empty or "input_features" not in logs.columns:
        return pd.DataFrame()

    feature_rows: list[dict[str, float]] = []
    for payload in logs["input_features"].tolist():
        if isinstance(payload, dict) and payload:
            row: dict[str, float] = {}
            for key, value in payload.items():
                if isinstance(value, (int, float)):
                    row[str(key)] = float(value)
            if row:
                feature_rows.append(row)
    if not feature_rows:
        return pd.DataFrame()
    return pd.DataFrame(feature_rows)


def build_drift_windows(logs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if logs.empty:
        return pd.DataFrame(), pd.DataFrame(), {"reason": "Aucun log disponible."}

    ordered = logs.copy()
    if "timestamp" in ordered.columns:
        ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], errors="coerce")
        ordered = ordered.dropna(subset=["timestamp"]).sort_values("timestamp")

    if "status_code" in ordered.columns:
        status = pd.to_numeric(ordered["status_code"], errors="coerce")
        ordered = ordered[status < 400]

    if "input_features" in ordered.columns:
        ordered = ordered[ordered["input_features"].apply(lambda payload: isinstance(payload, dict) and bool(payload))]

    n = len(ordered)
    if n < 40:
        return pd.DataFrame(), pd.DataFrame(), {
            "reason": "Pas assez de logs valides pour baseline/current.",
            "n_valid_logs": int(n),
        }

    default_current = 50
    default_baseline = 200
    if n >= (default_current + default_baseline):
        current_n = default_current
        baseline_n = default_baseline
    else:
        current_n = max(20, min(50, n // 5))
        baseline_n = max(20, n - current_n)

    baseline_logs = ordered.iloc[-(baseline_n + current_n):-current_n]
    current_logs = ordered.iloc[-current_n:]

    baseline_features = build_features_frame(baseline_logs)
    current_features = build_features_frame(current_logs)

    return baseline_features, current_features, {
        "n_valid_logs": int(n),
        "baseline_window": int(len(baseline_logs)),
        "current_window": int(len(current_logs)),
    }


def compute_operational_summary(logs: pd.DataFrame) -> dict[str, Any]:
    if logs.empty:
        return {
            "n_requests": 0,
            "error_rate": None,
            "latency_p95_ms": None,
            "alerts": ["Aucun log de production disponible."],
        }

    status_series = pd.to_numeric(logs.get("status_code"), errors="coerce")
    latency_series = pd.to_numeric(logs.get("latency_ms"), errors="coerce")

    error_mask = status_series >= 400
    error_rate = float(error_mask.mean()) if len(error_mask) else 0.0
    latency_p95 = float(latency_series.dropna().quantile(0.95)) if latency_series.notna().any() else 0.0

    alerts: list[str] = []
    if error_rate > 0.05:
        alerts.append(f"Taux d'erreur élevé: {error_rate:.2%} (> 5%).")

    recent_window = min(len(logs), 300)
    if recent_window > 30:
        recent_latency = latency_series.head(recent_window).dropna()
        baseline_latency = latency_series.dropna()
        if not recent_latency.empty and not baseline_latency.empty:
            recent_p95 = float(recent_latency.quantile(0.95))
            baseline_p95 = float(baseline_latency.quantile(0.95))
            if baseline_p95 > 0 and recent_p95 > (baseline_p95 * 1.5):
                alerts.append(
                    f"Latence anormale: p95 récent={recent_p95:.2f} ms vs baseline={baseline_p95:.2f} ms."
                )

    return {
        "n_requests": int(len(logs)),
        "error_rate": error_rate,
        "latency_p95_ms": latency_p95,
        "alerts": alerts,
    }


def run_drift(reference: pd.DataFrame, production_features: pd.DataFrame, report_path: Path) -> dict[str, Any]:
    if reference.empty or production_features.empty:
        return {
            "enabled": False,
            "reason": "Jeu de référence ou features de production indisponibles.",
        }

    common = [
        col
        for col in reference.columns
        if col in production_features.columns and pd.api.types.is_numeric_dtype(reference[col])
    ]
    if not common:
        return {
            "enabled": False,
            "reason": "Aucune feature numérique commune entre référence et production.",
        }

    ref = reference[common].copy()
    cur = production_features[common].copy()
    ref = ref.fillna(ref.median(numeric_only=True))
    cur = cur.fillna(ref.median(numeric_only=True))

    valid_columns: list[str] = []
    for col in common:
        ref_col = pd.to_numeric(ref[col], errors="coerce")
        cur_col = pd.to_numeric(cur[col], errors="coerce")

        if ref_col.notna().sum() < 2 or cur_col.notna().sum() < 2:
            continue
        if ref_col.nunique(dropna=True) <= 1 or cur_col.nunique(dropna=True) <= 1:
            continue
        valid_columns.append(col)

    if not valid_columns:
        return {
            "enabled": False,
            "reason": "Features communes non informatives (constantes ou insuffisantes) pour le drift.",
        }

    ref = ref[valid_columns]
    cur = cur[valid_columns]

    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            snapshot = report.run(reference_data=ref, current_data=cur)
        if hasattr(snapshot, "save_html"):
            snapshot.save_html(str(report_path))
        elif hasattr(report, "save_html"):
            report.save_html(str(report_path))
        elif hasattr(snapshot, "get_html_str"):
            report_path.write_text(snapshot.get_html_str(), encoding="utf-8")
        else:
            raise RuntimeError("Export HTML Evidently non supporté par cette version")
        return {
            "enabled": True,
            "n_features_compared": len(valid_columns),
            "report": str(report_path),
        }
    except Exception as exc:
        return {
            "enabled": False,
            "reason": f"Evidently indisponible ou erreur: {exc}",
        }


def main() -> None:
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "monitoring_summary.json"
    drift_report_path = reports_dir / "drift_report.html"
    reference_path = Path("data/reference/reference.csv")

    ensure_reference_csv(reference_path)

    logs = load_production_logs()
    summary = compute_operational_summary(logs)

    reference = pd.read_csv(reference_path) if reference_path.exists() else pd.DataFrame()
    baseline_features, current_features, window_meta = build_drift_windows(logs)

    if not baseline_features.empty and not current_features.empty:
        drift_summary = run_drift(baseline_features, current_features, drift_report_path)
        drift_summary["reference_source"] = "logs_baseline_vs_current"
        drift_summary["windowing"] = window_meta
    else:
        production_features = build_features_frame(logs)
        drift_summary = run_drift(reference, production_features, drift_report_path)
        drift_summary["reference_source"] = "reference_csv_vs_all_logs"
        drift_summary["windowing"] = window_meta

    final_summary = {
        "operational": summary,
        "drift": drift_summary,
    }

    storage = PostgresStorage(DATABASE_URL)
    if storage.is_available():
        try:
            storage.ensure_schema()
            storage.insert_monitoring_run(final_summary)
        except Exception:
            pass

    summary_path.write_text(json.dumps(final_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Monitoring summary generated at: {summary_path}")
    if drift_summary.get("enabled"):
        print(f"Drift report generated at: {drift_report_path}")


if __name__ == "__main__":
    main()
