# Fraud Detection MLOps

> Production-grade MLOps pipeline for real-time fraud scoring on 590k IEEE-CIS transactions — XGBoost + SMOTE, MLflow experiment tracking, FastAPI serving, Evidently monitoring, and a GitHub Actions CI/CD gate that only deploys when AUC-PR improves.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-FF6600?style=flat-square&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![MLflow](https://img.shields.io/badge/MLflow-2.10.2-0194E2?style=flat-square&logo=mlflow&logoColor=white)](https://mlflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![DVC](https://img.shields.io/badge/DVC-3.41-945DD6?style=flat-square&logo=dvc&logoColor=white)](https://dvc.org)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![CI/CD](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF?style=flat-square&logo=githubactions&logoColor=white)](https://github.com/features/actions)
[![Tests](https://img.shields.io/badge/Tests-47%2F47_passing-brightgreen?style=flat-square&logo=pytest&logoColor=white)](tests/)

---

## Results

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.9296** |
| AUC-PR (Average Precision) | **0.6414** |
| Training set | 590,540 transactions |
| Fraud rate | 3.5% (severe imbalance — handled with SMOTE) |
| Features | 433 raw + 21 engineered = 454 total |
| Test suite | **47 / 47 passing** |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                             DATA LAYER                                       │
│                                                                              │
│  data/raw/                    data_preprocessing.py          data/processed/ │
│  ├─ train_transaction.csv ──► merge on TransactionID   ──►  train.parquet   │
│  ├─ train_identity.csv        impute (median/constant)       test.parquet    │
│  ├─ test_transaction.csv      label-encode categoricals                      │
│  └─ test_identity.csv         save as Parquet                                │
│                    (DVC-tracked — never committed to Git)                    │
└─────────────────────────────────────┬────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          FEATURE ENGINEERING                                 │
│                                                                              │
│  feature_engineering.py                                                      │
│  ├─ Time features      tx_hour, tx_dayofweek, tx_is_weekend, tx_is_night    │
│  ├─ Amount features    log1p(TransactionAmt), cents, is_round               │
│  ├─ Email features     domain_match, is_top_domain, is_anonymous            │
│  ├─ Card features      card1/card2 ratio, amt/card ratio                    │
│  └─ Count features     per-card1/addr1/addr2 frequency encoding             │
└─────────────────────────────────────┬────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING (train.py)                                │
│                                                                              │
│  Train split (80%)  ──►  SMOTE (strategy=0.10)  ──►  XGBClassifier         │
│                          472k → 501k rows              n_estimators=300      │
│                          45k synthetic fraud           max_depth=6           │
│                          samples generated             learning_rate=0.05    │
│                                                        early_stopping=30     │
│  Val split (20%)    ──►  evaluate ──► ROC-AUC 0.9296, AUC-PR 0.6414        │
└──────────────┬───────────────────────────────────────────────┬───────────────┘
               │                                               │
               ▼                                               ▼
┌──────────────────────────┐              ┌────────────────────────────────────┐
│     MLFLOW TRACKING      │              │        DVC PIPELINE                │
│                          │              │                                    │
│  Experiment: fraud_det.  │              │  dvc repro                         │
│  ├─ All hyperparameters  │              │  └─► preprocess                    │
│  ├─ Train/val metrics    │              │       └─► train                    │
│  ├─ Feature importances  │              │            └─► evaluate            │
│  └─ Model artifact       │              │                                    │
│         │                │              │  params.yaml = single source       │
│         ▼                │              │  of truth for all config           │
│  Model Registry          │              └────────────────────────────────────┘
│  fraud_detector v1 ──────┼──────────────────────────┐
└──────────────────────────┘                           │
                                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        CI/CD (GitHub Actions)                                │
│                                                                              │
│  git push ──► lint (ruff) ──► test (pytest 47/47) ──► retrain (DVC)        │
│                                                            │                 │
│                                                     evaluate & compare       │
│                                                     AUC-PR vs baseline       │
│                                                            │                 │
│                                             delta ≥ 0.5%? │                 │
│                                          YES ──────────────┤                 │
│                                   build Docker image       │                 │
│                                   push to GHCR            │                 │
│                                   deploy to prod           │                 │
│                                          NO ───────────────┘                 │
│                                   skip (model did not improve)               │
└──────────────────────────────────────────────┬───────────────────────────────┘
                                               │
                                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        SERVING (FastAPI)                                     │
│                                                                              │
│  POST /predict                         GET /health                           │
│  ├─ Pydantic v2 validation             ├─ status: ok                         │
│  ├─ Feature engineering (serving)      ├─ model_loaded: true                 │
│  ├─ Load model from MLflow Registry    └─ model_version: "1"                 │
│  └─ Return fraud_probability [0,1]                                           │
│       + is_fraud (bool at threshold)                                         │
│       + model_version                                                        │
└──────────────────────────────────────────────┬───────────────────────────────┘
                                               │  prediction log
                                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                       MONITORING (Evidently AI)                              │
│                                                                              │
│  Reference: training distribution                                            │
│  Current:   incoming prediction log                                          │
│  Reports:   Data Drift · Data Quality · Target Drift                        │
│  Output:    monitoring/reports/drift_report.html                             │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/s9b/fraud-detection-mlops.git
cd fraud-detection-mlops

# 2. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Add data (Kaggle IEEE-CIS dataset — not committed, tracked by DVC)
#    Place CSVs in data/raw/ OR run: dvc pull  (after configuring remote)

# 4. Preprocess
python src/data_preprocessing.py
# → data/processed/train.parquet  (590k rows, 433 cols)
# → data/processed/test.parquet   (506k rows, 432 cols)

# 5. Train
python src/train.py
# → XGBoost + SMOTE, ROC-AUC 0.9296, AUC-PR 0.6414
# → Model registered in MLflow as fraud_detector v1

# 6. View experiments
mlflow ui --port 5001
# open http://localhost:5001

# 7. Serve
uvicorn api.main:app --reload
# open http://localhost:8000/docs

# 8. Run tests
pytest tests/ -v
# → 47/47 passing

# 9. Drift report
python monitoring/drift_report.py
# → monitoring/reports/drift_report.html
```

---

## Full Project Structure

```
fraud-detection-mlops/
│
├── data/
│   ├── raw/                        # Original CSVs — DVC-tracked, never committed
│   │   ├── train_transaction.csv   # 590k transactions, 394 columns
│   │   ├── train_identity.csv      # 144k identity rows, 41 columns
│   │   ├── test_transaction.csv    # 506k test transactions
│   │   ├── test_identity.csv       # 141k test identity rows
│   │   └── sample_submission.csv   # Kaggle submission format
│   └── processed/                  # Parquet outputs — DVC-tracked
│       ├── train.parquet           # Merged, imputed, encoded training data
│       ├── test.parquet            # Merged, imputed, encoded test data
│       └── run_meta.yaml           # MLflow run ID + metric snapshot
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py       # Load CSVs → merge → impute → encode → parquet
│   ├── feature_engineering.py      # Time, email, card, amount, frequency features
│   ├── train.py                    # XGBoost + SMOTE + full MLflow logging + registry
│   └── evaluate.py                 # Metrics computation + AUC-PR CI/CD gate logic
│
├── api/
│   ├── __init__.py
│   ├── main.py                     # FastAPI app: /health + /predict, lifespan loader
│   └── schemas.py                  # Pydantic v2: all 394 tx + 41 identity fields
│
├── monitoring/
│   ├── drift_report.py             # Evidently: data drift + quality + target drift
│   └── reports/                    # Generated HTML reports (gitignored)
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py       # 15 tests: merge, drop, impute, encode, split
│   ├── test_api.py                 # 17 tests: /health, /predict, validation, 422/503
│   └── test_model.py               # 15 tests: feature engineering, metrics, XGBoost
│
├── .github/
│   └── workflows/
│       └── ci_cd.yml               # lint → test → retrain → AUC-PR gate → deploy
│
├── dvc.yaml                        # DVC pipeline: preprocess → train → evaluate
├── params.yaml                     # Single source of truth for ALL hyperparameters
├── MLproject                       # MLflow Projects entrypoints for each stage
├── Dockerfile                      # Multi-stage: builder (compile) + slim runtime
├── docker-compose.yml              # API + MLflow server + training job (--profile train)
├── requirements.txt                # Pinned dependencies
└── README.md                       # This file
```

---

## Component Deep-Dive

### params.yaml — Single Source of Truth

Every hyperparameter lives here. Nothing is hardcoded:

```yaml
model:
  n_estimators: 300
  max_depth: 6
  learning_rate: 0.05
  scale_pos_weight: 10      # Handles class imbalance weight
  early_stopping_rounds: 30

smote:
  sampling_strategy: 0.10   # Minority:majority ratio after oversampling
  k_neighbors: 5

evaluation:
  threshold: 0.5
  auc_pr_improvement_threshold: 0.005   # CI/CD deploy gate: must improve by 0.5%
```

### MLflow — Experiment Tracking + Model Registry

Every training run logs:
- All 13 hyperparameters
- Validation metrics: ROC-AUC, AUC-PR, F1, Precision, Recall
- Feature importance CSV (453 features ranked)
- Model artifact with input signature + example
- Automatically registered in Model Registry as `fraud_detector`

```python
# Access latest model anywhere
model = mlflow.xgboost.load_model("models:/fraud_detector/latest")
```

### DVC — Data + Pipeline Versioning

Three reproducible stages, each with tracked inputs/outputs:

```
dvc repro
└─► preprocess   deps: raw CSVs + params.yaml
     └─► train   deps: train.parquet + params.yaml
          └─► evaluate   deps: train.parquet + run_meta.yaml
```

Data files are never committed to Git — only tracked via DVC with a remote (S3, GDrive, etc.).

### FastAPI — Real-Time Serving

```bash
POST http://localhost:8000/predict
Content-Type: application/json

{
  "transaction": {
    "TransactionAmt": 299.00,
    "ProductCD": "W",
    "card4": "visa",
    "P_emaildomain": "gmail.com"
  },
  "identity": {
    "DeviceType": "mobile",
    "id_01": -1.0
  }
}
```

Response:
```json
{
  "fraud_probability": 0.0342,
  "is_fraud": false,
  "threshold": 0.5,
  "model_version": "1"
}
```

All 394 transaction + 41 identity fields are in the Pydantic schema — all `Optional`, so you can send as few fields as available. The serving layer applies identical feature engineering as training.

### Evidently AI — Data Drift Monitoring

Compares the training distribution against incoming prediction logs:

```bash
python monitoring/drift_report.py \
  --current data/processed/predictions_log.parquet \
  --output  monitoring/reports/drift_report.html
```

Generates three reports:
- **Data Drift** — Which features have drifted from training distribution
- **Data Quality** — Missing values, outliers, type mismatches in current data
- **Target Drift** — Shift in fraud rate over time (when labels are available)

### GitHub Actions CI/CD Pipeline

```
on: push to main / PR to main
         │
         ▼
   ┌─── lint ────────────────────────────────┐
   │  ruff check src/ api/ monitoring/ tests/ │
   │  ruff format --check                     │
   └──────────────────┬──────────────────────┘
                      │
                      ▼
   ┌─── test ────────────────────────────────┐
   │  pytest tests/ -v --cov=src --cov=api   │
   │  47/47 must pass                         │
   └──────────────────┬──────────────────────┘
                      │  (main branch only)
                      ▼
   ┌─── retrain ─────────────────────────────┐
   │  python src/data_preprocessing.py        │
   │  python src/train.py                     │
   │  python src/evaluate.py → metrics.json   │
   └──────────────────┬──────────────────────┘
                      │
                      ▼
   ┌─── compare AUC-PR ──────────────────────┐
   │  current AUC-PR vs previous baseline     │
   │  delta ≥ 0.005 (0.5%)? → DEPLOY         │
   │  delta < 0.005?         → SKIP           │
   └──────────────────┬──────────────────────┘
                      │ if deploy
                      ▼
   ┌─── build + push ────────────────────────┐
   │  docker build → push to GHCR            │
   │  tagged: sha-<commit>, latest           │
   └──────────────────┬──────────────────────┘
                      │
                      ▼
   ┌─── deploy ──────────────────────────────┐
   │  Pluggable: kubectl / ECS / fly.io       │
   │  environment: production (manual gate)   │
   └─────────────────────────────────────────┘
```

### Docker

```bash
# Build and run full stack
docker compose up -d
# API   → http://localhost:8000/docs
# MLflow → http://localhost:5001

# One-shot training job
docker compose --profile train up train
```

Multi-stage Dockerfile: `builder` stage compiles all native extensions (XGBoost, numpy), `runtime` stage is a minimal Python 3.11-slim image. Non-root user. HEALTHCHECK built in.

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info + links |
| `/health` | GET | Liveness check + model status |
| `/predict` | POST | Score a transaction for fraud |
| `/docs` | GET | Swagger UI (interactive) |
| `/redoc` | GET | ReDoc documentation |

### Validation

The Pydantic schema enforces:
- `TransactionAmt >= 0` (no negative amounts)
- No extra top-level fields (`extra = "forbid"`)
- All feature fields `Optional` — send only what you have
- Automatic coercion: string categoricals → hash-encoded int → float64 for XGBoost

---

## Screenshots

> **MLflow Experiment Tracking** — Runs table showing ROC-AUC, AUC-PR, F1 across experiments
![MLflow Experiment Tracking](docs/screenshots/mlflow_experiments.png)

> **MLflow Run Detail** — Hyperparameters panel + metrics panel for the best run
![MLflow Run Detail](docs/screenshots/mlflow_run_detail.png)

> **MLflow Model Registry** — fraud_detector model with version history and stage
![MLflow Model Registry](docs/screenshots/mlflow_registry.png)

> **FastAPI Swagger UI** — Interactive /predict endpoint with full schema
![FastAPI Swagger UI](docs/screenshots/fastapi_swagger.png)

> **FastAPI Live Prediction** — Executed /predict returning fraud_probability
![FastAPI Prediction](docs/screenshots/fastapi_predict_response.png)

> **Evidently Drift Report** — Feature drift heatmap comparing training vs current
![Evidently Drift Report](docs/screenshots/evidently_drift.png)

---

## Why This Architecture?

| Decision | Rationale |
|----------|-----------|
| **SMOTE not class_weight** | SMOTE generates synthetic minority samples before the train/val split, giving the model richer signal; `scale_pos_weight` is set as a secondary guard |
| **AUC-PR over AUC-ROC for gate** | At 3.5% fraud rate, ROC-AUC is misleading (a model that flags everything as legit still scores 0.96). AUC-PR directly measures precision/recall tradeoff on the minority class |
| **Parquet over CSV** | 10-15x faster read, typed schema, columnar compression — critical for 590k rows with 430+ columns |
| **Pydantic v2 with all fields Optional** | Serving data is always partial — a transaction may have no identity row. The model handles missing values via the same imputation as training |
| **MLflow registry over file-based model** | Enables zero-downtime model rollout, version comparison, stage promotion (Staging → Production), and audit trail |
| **DVC for data, Git for code** | Data files are gigabytes; Git is for logic. DVC locks data versions to code versions via `dvc.lock`, ensuring full reproducibility |

---

## Reproducing Results

```bash
# Full DVC pipeline (preprocess + train + evaluate)
dvc repro

# Check metrics
cat metrics.json

# Compare runs in MLflow
mlflow ui --port 5001

# Hyperparameter sweep via MLflow Projects
mlflow run . -e train \
  -P n_estimators=500 \
  -P learning_rate=0.03 \
  -P smote_sampling_strategy=0.15
```

---

## License

MIT
