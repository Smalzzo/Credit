# Historique des versions

## v1.3.0 - 2026-03-03
- Ajout d'un notebook dédié d'analyse de drift: `notebooks/04_data_drift_analysis.ipynb`.
- Analyse drift en mode baseline vs current (fenêtrage logs) dans `drift/run_drift.py`.
- Compatibilité Evidently (export HTML via snapshot/report selon version).
- Filtrage des colonnes non informatives avant calcul de drift.

## v1.2.0 - 2026-02-27
- Dashboard Streamlit enrichi: variables métier (`EXT_SOURCE_1/2/3`, `PAYMENT_RATE`, `DAYS_BIRTH`).
- Génération batch de prédictions et variation aléatoire (%) pour simulation.
- Clarification UI: score = probabilité de défaut (TARGET=1).

## v1.1.0 - 2026-02-26
- Correction logique décision API: score élevé (risque) => `REJECT`.
- Ajout test de non-régression sur décision dans `tests/test_inference.py`.
- Support exécution drift dans Docker (copie du dossier `drift` dans l'image API).

## v1.0.0 - Initial
- API FastAPI de scoring crédit.
- Monitoring Streamlit + stockage PostgreSQL.
- Docker Compose (`postgres`, `api`, `streamlit`).
- Pipeline CI de base.
