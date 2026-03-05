# Credit Scoring MLOps (FastAPI + Monitoring + Drift)

Projet de scoring de crédit industrialisé à partir d’un notebook existant, avec API FastAPI, tests automatisés, conteneur Docker, pipeline CI/CD GitHub Actions, logging JSON structuré, stockage PostgreSQL, dashboard de monitoring et rapport de data drift.

## Objectifs

- Exposer un modèle de scoring via API REST (`/predict`, `/health`, `/metrics`)
- Charger le modèle une seule fois au démarrage de l’API
- Garantir la reproductibilité locale (entraînement demo, tests, conteneur)
- Tracer les requêtes en JSON structuré (PostgreSQL + fallback JSONL local)
- Surveiller la performance opérationnelle (latence, erreurs, distribution des scores)
- Détecter le drift entre un jeu de référence et des données de production loggées

## Structure du dépôt

```text
.
├── api/
│   ├── deps.py
│   ├── main.py
│   └── routes.py
├── data/
│   └── reference/
├── docs/
│   └── screenshots/
├── drift/
│   └── run_drift.py
├── monitoring/
│   └── streamlit_app.py
├── models/
├── reports/
├── scripts/
│   ├── export_model.py
│   ├── simulate_production.py
│   └── train.py
├── src/
│   └── credit_scoring/
│       ├── config.py
│       ├── inference.py
│       ├── logging_utils.py
│       ├── model.py
│       ├── monitoring.py
│       ├── preprocessing.py
│       └── schema.py
├── tests/
├── .github/workflows/ci.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Données et sécurité

- Ne pas commiter de données brutes ni de secrets.
- Le dossier `data/` est ignoré par Git, sauf placeholders techniques.
- Les logs API (`data/production_logs.jsonl`) contiennent un hash du payload et des métadonnées, pas de PII brute.
- Les variables sensibles éventuelles doivent être passées via variables d’environnement, jamais en dur.

## Installation locale (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configuration PostgreSQL (PoC local)

Variable d’environnement utilisée par l’API, le script d’analyse et Streamlit:

```powershell
$env:DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/credit_monitoring"
```

Si `DATABASE_URL` est absent, le système continue en mode local via `data/production_logs.jsonl`.

Pour forcer une stratégie PostgreSQL-first en local, exportez toujours `DATABASE_URL` avant de lancer API / Streamlit / drift.

### Backfill des logs JSONL vers PostgreSQL

Si vous avez déjà des logs dans `data/production_logs.jsonl`, vous pouvez les charger en base:

```powershell
$env:DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/credit_monitoring"
python -m scripts.backfill_postgres_logs --input data/production_logs.jsonl
```

## Entraîner / exporter le modèle

Mode demo reproductible (dataset synthétique):

```powershell
$env:PYTHONPATH = "src"
python -m scripts.train
```

Artefact généré:

- `models/pipeline.joblib`

## Lancer l’API

```powershell
$env:PYTHONPATH = "src"
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health` : état service, version, modèle chargé
- `POST /predict` : score, décision, version modèle, latence
- `GET /metrics` : métriques runtime en mémoire

### Exemple `curl` pour `/predict`

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
	-H "Content-Type: application/json" \
	-d '{
		"age": 35,
		"income": 55000,
		"credit_amount": 12000,
		"annuity": 1200,
		"employment_years": 7,
		"family_members": 3
	}'
```

## Générer des logs de production simulés

1. Démarrer l’API.
2. Dans un second terminal:

```powershell
python -m scripts.simulate_production
```

Fichier de logs attendu:

- `data/production_logs.jsonl`

Table PostgreSQL générée automatiquement au démarrage API (si `DATABASE_URL` défini):

- `api_calls` (timestamp, endpoint, status_code, latence, input_features, score, décision, erreurs, etc.)

## Monitoring Streamlit

```powershell
streamlit run monitoring/streamlit_app.py
```

Le dashboard affiche:

- Volume de requêtes
- Taux d’erreur
- Distribution des scores
- Statistiques de latence (p50/p95)
- Série temporelle de latence
- Alertes automatiques (si générées par l’analyse)

## Data Drift (Evidently)

Préparer un jeu de référence non sensible dans:

- `data/reference/reference.csv`

Alternative automatique déjà branchée:

- si `data/reference/reference.csv` est absent, le script reconstruit un CSV de référence depuis `data/reference/home_credit_reference_raw.json` + `models/notebook_model.joblib`.

Puis lancer:

```powershell
python -m drift.run_drift
```

Sorties générées:

- `reports/drift_report.html`
- `reports/monitoring_summary.json`

Le script calcule automatiquement:

- dérive des données (Evidently, sur features numériques communes référence/production),
- taux d’erreur global,
- anomalie de latence (comparaison p95 fenêtre récente vs baseline),
- alertes de vigilance opérationnelle.

## Optimisation ONNX (démo)

Le projet inclut une démo d'export ONNX et benchmark de latence:

```powershell
python -m scripts.onnx_optimization_demo --runs 300 --batch-size 1
```

Sorties:

- Modèle ONNX exporté: `models/notebook_model.onnx`
- Comparaison latence moyenne `sklearn` vs `onnxruntime`
- Écart max de probabilité (`Max |proba diff|`)

## Tests

```powershell
python -m pytest -q tests
```

Tests inclus:

- Validation des entrées Pydantic
- Inference (score dans le domaine attendu)
- Intégration API (`/predict` succès + erreur 422)

## Docker

### Build

```powershell
docker build -t credit-scoring:local .
```

### Run

```powershell
docker run --rm -p 8000:8000 credit-scoring:local
```

### Docker Compose (PostgreSQL + API + Streamlit)

PoC local complet en un lancement:

```powershell
docker compose up -d --build
```

Services disponibles:

- API: `http://localhost:8000` (Swagger: `/docs`)
- Streamlit: `http://localhost:8501`
- PostgreSQL: `localhost:5432` (`postgres/postgres`, DB `credit_monitoring`)

Arrêt et nettoyage:

```powershell
docker compose down
```

Pour supprimer aussi le volume PostgreSQL:

```powershell
docker compose down -v
```

## Déploiement API sur Hugging Face Spaces

Objectif: déployer uniquement l'API FastAPI dans un Space Docker.

### 1) Créer un Space

- Type: **Docker**
- Visibilité: selon besoin (public/private)
- Nom suggéré: `credit-scoring-api`

### 2) Préparer le repository du Space

Le Space doit contenir au minimum:

- `Dockerfile`
- `requirements.txt`
- `api/`
- `src/`
- `models/`
- `drift/` (optionnel, utile si vous lancez l'analyse drift dans le container)

Ajouter un `README.md` avec front-matter Hugging Face:

```yaml
---
title: Credit Scoring API
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---
```

### 3) Variables d'environnement (Space Settings)

- `MODEL_PATH` (optionnel): chemin du modèle à charger, ex. `/app/models/notebook_model.joblib`
- `DATABASE_URL` (optionnel): laisser vide si vous ne connectez pas Postgres sur le Space

### 4) Push vers le Space

Exemple Git:

```powershell
git remote add hf https://huggingface.co/spaces/<user>/<space-name>
git push hf master
```

### 5) Vérification

Une fois le build terminé, endpoints attendus:

- `/health`
- `/docs`
- `/predict` (ou `/predict-compact`, `/predict-notebook`)

Note: l'image supporte le port dynamique du Space via `PORT`.

## CI/CD (GitHub Actions)

Workflow: `.github/workflows/ci.yml`

Jobs:

1. `lint-test` : installation dépendances + `pytest`
2. `build-docker` : build image Docker si tests OK
3. `deploy-simulated` : simulation de déploiement (`echo "deploy ok"`)

## Interprétation monitoring et drift

- **Latence p95 en hausse**: possible saturation CPU, modèle trop lourd, I/O disque.
- **Taux d’erreur > 1-2%**: vérifier validation input, dépendances externes, santé du modèle.
- **Distribution score qui se décale**: possible changement population ou qualité des entrées.
- **Drift features significatif**: réentraîner le modèle ou recalibrer les seuils de décision.

## Captures d’écran attendues (manuel)

Voir le guide dans `docs/screenshots/README.md`.

## Limites actuelles

- Pipeline de drift dépend de la qualité et du schéma des logs de production.
- Le mode demo utilise un modèle synthétique pour garantir la reproductibilité sans dataset sensible.

## Licence

Usage éducatif et démonstration MLOps.
