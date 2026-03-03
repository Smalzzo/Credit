"""Monitoring dashboard (PostgreSQL first, JSONL fallback)."""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
import random
import sys

import joblib
import pandas as pd
import requests
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from credit_scoring.config import DATABASE_URL
from credit_scoring.storage import PostgresStorage


LOG_PATH = Path("data/production_logs.jsonl")
SUMMARY_PATH = Path("reports/monitoring_summary.json")
NOTEBOOK_PAYLOAD_PATH = Path("data/payload_notebook_sample.json")
RAW_REFERENCE_PAYLOAD_PATH = Path("data/reference/home_credit_reference_raw.json")
MODEL_BUNDLE_PATH = Path("models/notebook_model.joblib")
DRIFT_REPORT_PATH = Path("reports/drift_report.html")


def load_logs(limit: int = 5000) -> pd.DataFrame:
    storage = PostgresStorage(DATABASE_URL)
    if storage.is_available():
        try:
            rows = storage.fetch_recent_events(limit=limit)
            if rows:
                return pd.DataFrame(rows)
        except Exception:
            pass

    if LOG_PATH.exists():
        return pd.read_json(LOG_PATH, lines=True)
    return pd.DataFrame()


def load_monitoring_summary() -> dict:
    storage = PostgresStorage(DATABASE_URL)
    if storage.is_available():
        try:
            latest = storage.fetch_latest_monitoring_run()
            if latest:
                return {
                    "operational": latest.get("operational", {}),
                    "drift": latest.get("drift", {}),
                }
        except Exception:
            pass

    if SUMMARY_PATH.exists():
        try:
            return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def load_monitoring_history(limit: int = 30) -> pd.DataFrame:
    storage = PostgresStorage(DATABASE_URL)
    if storage.is_available():
        try:
            rows = storage.fetch_recent_monitoring_runs(limit=limit)
            if rows:
                frame = pd.DataFrame(rows)
                frame["run_at"] = pd.to_datetime(frame["run_at"], errors="coerce")
                return frame.dropna(subset=["run_at"]).sort_values("run_at")
        except Exception:
            pass
    return pd.DataFrame()


def build_auto_notebook_payload() -> tuple[dict, str]:
    if RAW_REFERENCE_PAYLOAD_PATH.exists() and MODEL_BUNDLE_PATH.exists():
        try:
            raw_json = json.loads(RAW_REFERENCE_PAYLOAD_PATH.read_text(encoding="utf-8"))
            columns_csv = str(raw_json.get("columns_csv", "")).strip()
            row_csv = str(raw_json.get("row_csv", "")).strip()
            if columns_csv and row_csv:
                columns = next(csv.reader(io.StringIO(columns_csv)))
                values = next(csv.reader(io.StringIO(row_csv)))
                row_dict = dict(zip(columns, values))

                artifact = joblib.load(MODEL_BUNDLE_PATH)
                if isinstance(artifact, dict):
                    feature_names = list(artifact.get("feature_names") or [])
                    medians = dict(artifact.get("medians") or {})
                    if feature_names:
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
                        return {"features": base_features}, (
                            f"Payload généré depuis {MODEL_BUNDLE_PATH} + référence {RAW_REFERENCE_PAYLOAD_PATH}"
                        )
        except Exception:
            pass

    if NOTEBOOK_PAYLOAD_PATH.exists():
        try:
            payload = json.loads(NOTEBOOK_PAYLOAD_PATH.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("features"), dict) and payload["features"]:
                return payload, f"Payload chargé depuis {NOTEBOOK_PAYLOAD_PATH}"
        except Exception:
            pass

    if MODEL_BUNDLE_PATH.exists():
        try:
            artifact = joblib.load(MODEL_BUNDLE_PATH)
            if isinstance(artifact, dict):
                feature_names = list(artifact.get("feature_names") or [])
                medians = dict(artifact.get("medians") or {})
                if feature_names:
                    payload = {
                        "features": {name: float(medians.get(name, 0.0)) for name in feature_names}
                    }
                    return payload, f"Payload généré depuis {MODEL_BUNDLE_PATH}"
        except Exception:
            pass

    return {}, "Aucun payload notebook auto disponible (fichier sample ou modèle bundle manquant)."


def apply_notebook_manual_inputs(payload: dict, manual_values: dict, notebook_overrides: dict | None = None) -> dict:
    if not payload or not isinstance(payload.get("features"), dict):
        return payload

    features = dict(payload["features"])

    age = float(manual_values["age"])
    income = float(manual_values["income"])
    credit_amount = float(manual_values["credit_amount"])
    annuity = float(manual_values["annuity"])
    employment_years = float(manual_values["employment_years"])
    family_members = float(manual_values["family_members"])

    features["AMT_INCOME_TOTAL"] = income
    features["AMT_CREDIT"] = credit_amount
    features["AMT_ANNUITY"] = annuity
    features["AMT_GOODS_PRICE"] = credit_amount
    features["CNT_FAM_MEMBERS"] = family_members
    features["DAYS_BIRTH"] = -age * 365.0
    features["DAYS_EMPLOYED"] = -employment_years * 365.0

    if family_members > 0:
        features["INCOME_PER_PERSON"] = income / family_members
    if credit_amount > 0:
        features["PAYMENT_RATE"] = annuity / credit_amount
        features["INCOME_CREDIT_PERC"] = income / credit_amount
    if income > 0:
        features["ANNUITY_INCOME_PERC"] = annuity / income
    if age > 0:
        features["DAYS_EMPLOYED_PERC"] = employment_years / age

    if notebook_overrides:
        for feature_name, feature_value in notebook_overrides.items():
            if feature_value is None:
                continue
            try:
                features[feature_name] = float(feature_value)
            except (TypeError, ValueError):
                continue

    return {"features": features}


def apply_percentage_variation(payload: dict, variation_pct: float) -> dict:
    if variation_pct <= 0 or not payload or not isinstance(payload.get("features"), dict):
        return payload

    delta = float(variation_pct) / 100.0
    varied_features: dict[str, float] = {}
    for key, value in payload["features"].items():
        if not isinstance(value, (int, float)):
            continue

        numeric_value = float(value)
        if numeric_value in (0.0, 1.0):
            varied_features[key] = numeric_value
            continue

        factor = random.uniform(1.0 - delta, 1.0 + delta)
        varied_features[key] = numeric_value * factor

    return {"features": varied_features}

st.title("Credit Scoring Monitoring")

st.subheader("Tester l'API")
st.caption("Le score retourné par l'API est une probabilité de défaut (TARGET=1) : plus il est élevé, plus le risque est élevé.")
default_api_url = os.getenv("API_BASE_URL")
if not default_api_url:
    default_api_url = "http://api:8000" if Path("/.dockerenv").exists() else "http://localhost:8000"

api_base_url = st.text_input("URL API", value=default_api_url)
mode = st.radio("Mode de payload", ["notebook", "compact"], horizontal=True)

col1, col2, col3 = st.columns(3)
age = col1.number_input("age", min_value=18, max_value=100, value=35)
income = col2.number_input("income", min_value=0.0, value=55000.0, step=1000.0)
credit_amount = col3.number_input("credit_amount", min_value=0.0, value=12000.0, step=500.0)

col4, col5, col6 = st.columns(3)
annuity = col4.number_input("annuity (manuel)", min_value=1.0, value=1200.0, step=100.0)
employment_years = col5.number_input("employment_years", min_value=0.0, max_value=60.0, value=7.0, step=1.0)
family_members = col6.number_input("family_members", min_value=1.0, max_value=20.0, value=3.0, step=1.0)

manual_values = {
    "age": float(age),
    "income": float(income),
    "credit_amount": float(credit_amount),
    "annuity": float(annuity),
    "employment_years": float(employment_years),
    "family_members": float(family_members),
}

notebook_overrides: dict[str, float] = {}
variation_percent = 0.0
batch_predictions = 1

if mode == "notebook":
    st.markdown("Variables importantes (mode notebook)")
    ext_col1, ext_col2, ext_col3 = st.columns(3)
    ext_source_1 = ext_col1.number_input("EXT_SOURCE_1", min_value=0.0, max_value=1.0, value=0.50, step=0.01)
    ext_source_2 = ext_col2.number_input("EXT_SOURCE_2", min_value=0.0, max_value=1.0, value=0.50, step=0.01)
    ext_source_3 = ext_col3.number_input("EXT_SOURCE_3", min_value=0.0, max_value=1.0, value=0.50, step=0.01)

    ratio_col1, ratio_col2 = st.columns(2)
    default_payment_rate = float(annuity) / float(credit_amount) if float(credit_amount) > 0 else 0.0
    payment_rate = ratio_col1.number_input(
        "PAYMENT_RATE",
        min_value=0.0,
        value=float(default_payment_rate),
        step=0.001,
        format="%.6f",
    )
    days_birth = ratio_col2.number_input(
        "DAYS_BIRTH",
        min_value=-50000.0,
        max_value=-3000.0,
        value=float(-age * 365.0),
        step=365.0,
    )

    notebook_overrides = {
        "EXT_SOURCE_1": float(ext_source_1),
        "EXT_SOURCE_2": float(ext_source_2),
        "EXT_SOURCE_3": float(ext_source_3),
        "PAYMENT_RATE": float(payment_rate),
        "DAYS_BIRTH": float(days_birth),
        "AMT_ANNUITY": float(annuity),
    }

    gen_col1, gen_col2 = st.columns(2)
    variation_percent = float(
        gen_col1.number_input(
            "Variation aléatoire (%)",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.1,
        )
    )
    batch_predictions = int(
        gen_col2.number_input(
            "Prédictions par clic",
            min_value=1,
            max_value=500,
            value=1,
            step=1,
        )
    )
    st.caption("À chaque requête notebook, les features numériques peuvent varier de ±x% autour de la valeur actuelle.")

request_payload: dict = {}
endpoint = "/predict-compact"
payload_message = ""

if mode == "compact":
    request_payload = {
        "age": int(manual_values["age"]),
        "income": float(manual_values["income"]),
        "credit_amount": float(manual_values["credit_amount"]),
        "annuity": float(manual_values["annuity"]),
        "employment_years": float(manual_values["employment_years"]),
        "family_members": float(manual_values["family_members"]),
    }
else:
    endpoint = "/predict-notebook"
    request_payload, payload_message = build_auto_notebook_payload()
    request_payload = apply_notebook_manual_inputs(
        payload=request_payload,
        manual_values=manual_values,
        notebook_overrides=notebook_overrides,
    )
    if request_payload:
        st.success(f"{payload_message} ({len(request_payload.get('features', {}))} features)")
    else:
        st.error(payload_message)

if st.button("Predict", type="primary"):
    if not request_payload:
        st.error("Payload vide ou invalide.")
    else:
        target_url = f"{api_base_url.rstrip('/')}{endpoint}"
        try:
            n_calls = batch_predictions if mode == "notebook" else 1
            status_codes: list[int] = []
            scores: list[float] = []
            response_body: dict | str | None = None

            for _ in range(n_calls):
                payload_to_send = request_payload
                if mode == "notebook" and variation_percent > 0:
                    payload_to_send = apply_percentage_variation(request_payload, variation_percent)

                response = requests.post(target_url, json=payload_to_send, timeout=30)
                status_codes.append(response.status_code)

                try:
                    response_body = response.json()
                except Exception:
                    response_body = response.text

                if isinstance(response_body, dict) and "score" in response_body:
                    try:
                        scores.append(float(response_body["score"]))
                    except (TypeError, ValueError):
                        pass

            if n_calls == 1:
                st.write(f"HTTP {status_codes[0]}")
            else:
                ok_calls = sum(1 for code in status_codes if 200 <= code < 300)
                st.write(f"Batch envoyé: {n_calls} requêtes (OK: {ok_calls}, erreurs: {n_calls - ok_calls})")

            if isinstance(response_body, dict):
                st.json(response_body)
                if scores:
                    risk_score = float(scores[-1])
                    st.info(f"Score de risque de défaut : {risk_score:.4f} (plus proche de 1 = plus risqué)")
                if len(scores) > 1:
                    score_mean = sum(scores) / len(scores)
                    st.write(
                        {
                            "score_min": min(scores),
                            "score_mean": score_mean,
                            "score_max": max(scores),
                        }
                    )
            elif response_body is not None:
                st.text(str(response_body))
        except Exception as exc:
            st.error(f"Erreur appel API: {exc}")

st.divider()

st.subheader("Administration")
confirm_reset = st.checkbox("Confirmer la remise à zéro des données monitoring", value=False)
if st.button("Reset monitoring", type="secondary"):
    if not confirm_reset:
        st.warning("Cochez la confirmation avant la remise à zéro.")
    else:
        deleted_api_calls = 0
        deleted_runs = 0
        storage = PostgresStorage(DATABASE_URL)
        if storage.is_available():
            try:
                storage.ensure_schema()
                deleted = storage.clear_all_monitoring_data()
                deleted_api_calls = int(deleted.get("api_calls", 0))
                deleted_runs = int(deleted.get("monitoring_runs", 0))
            except Exception as exc:
                st.error(f"Erreur reset PostgreSQL: {exc}")

        removed_files = []
        for path in [LOG_PATH, SUMMARY_PATH, DRIFT_REPORT_PATH]:
            try:
                if path.exists():
                    path.unlink()
                    removed_files.append(str(path))
            except Exception:
                pass

        st.success(
            f"Reset terminé: {deleted_api_calls} requêtes supprimées, {deleted_runs} runs supprimés, "
            f"{len(removed_files)} fichiers locaux supprimés."
        )
        st.rerun()

logs = load_logs()
if logs.empty:
    st.info("Aucun log disponible. Lancez l'API puis générez du trafic de test.")

if not logs.empty:
    st.metric("Requests", int(len(logs)))
    st.metric("Error Rate", float((logs["status_code"] >= 400).mean()))
    st.metric("Latency p95 (ms)", float(pd.to_numeric(logs["latency_ms"], errors="coerce").quantile(0.95)))

    if "timestamp" in logs.columns:
        ts = pd.to_datetime(logs["timestamp"], errors="coerce")
        latency = pd.to_numeric(logs["latency_ms"], errors="coerce")
        latency_df = pd.DataFrame({"timestamp": ts, "latency_ms": latency}).dropna().sort_values("timestamp")
        if not latency_df.empty:
            st.subheader("Latency over time")
            st.line_chart(latency_df.set_index("timestamp")["latency_ms"])

    if "score" in logs.columns:
        st.subheader("Score distribution")
        st.bar_chart(logs["score"].dropna())

    if "latency_ms" in logs.columns:
        st.subheader("Latency summary")
        st.write({"p50": logs["latency_ms"].quantile(0.5), "p95": logs["latency_ms"].quantile(0.95)})

    summary = load_monitoring_summary()
    if summary:
        st.subheader("Automated alerts")
        for alert in summary.get("operational", {}).get("alerts", []):
            st.warning(alert)

        drift = summary.get("drift", {})
        if drift.get("enabled"):
            st.success(f"Drift report available: {drift.get('report')}")
        else:
            st.info(f"Drift non exécuté: {drift.get('reason', 'raison non renseignée')}")

    history = load_monitoring_history(limit=50)
    if not history.empty:
        st.subheader("Monitoring runs history")
        trend = history[["run_at", "error_rate", "latency_p95_ms"]].set_index("run_at")
        st.line_chart(trend)

        drift_counts = (
            history["drift_enabled"].fillna(False).map({True: "enabled", False: "disabled"}).value_counts()
        )
        st.write(
            {
                "runs_count": int(len(history)),
                "drift_enabled_runs": int(drift_counts.get("enabled", 0)),
                "drift_disabled_runs": int(drift_counts.get("disabled", 0)),
            }
        )
