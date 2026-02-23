"""Simple monitoring dashboard from production JSONL logs."""

from pathlib import Path

import pandas as pd
import streamlit as st


LOG_PATH = Path("data/production_logs.jsonl")

st.title("Credit Scoring Monitoring")

if not LOG_PATH.exists():
    st.info("No production log found. Run scripts/simulate_production.py first.")
    st.stop()

logs = pd.read_json(LOG_PATH, lines=True)
st.metric("Requests", int(len(logs)))
st.metric("Error Rate", float((logs["status_code"] >= 400).mean()))

if "score" in logs.columns:
    st.subheader("Score distribution")
    st.bar_chart(logs["score"].dropna())

if "latency_ms" in logs.columns:
    st.subheader("Latency")
    st.write({"p50": logs["latency_ms"].quantile(0.5), "p95": logs["latency_ms"].quantile(0.95)})
