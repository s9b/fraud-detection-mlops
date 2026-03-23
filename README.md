# Fraud Detection MLOps

Production-grade MLOps pipeline for the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset.

**Tech stack:** XGBoost · SMOTE · MLflow · DVC · FastAPI · Evidently AI · GitHub Actions · Docker

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Data & Training Layer                            │
│                                                                         │
│  data/raw/  ──────►  data_preprocessing.py  ──────►  data/processed/   │
│   (DVC)          (merge, impute, encode)           (parquet, DVC)      │
│                              │                                          │
│                              ▼                                          │
│               feature_engineering.py                                   │
│          (time, email, card, amount features)                          │
│                              │                                          │
│                              ▼                                          │
│          train.py  ──► SMOTE ──► XGBoostClassifier                    │
│               │         (imbalanced-learn)                             │
│               │                                                         │
│               ▼                                                         │
│          MLflow Tracking  ──►  Model Registry                          │
│      (params, metrics, artefacts, model versions)                      │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Serving Layer                                    │
│                                                                         │
│   Client  ──►  FastAPI /predict  ──►  MLflow load_model  ──►  Score   │
│               (Pydantic validation)    (latest registry version)       │
│                     │                                                   │
│                     ▼                                                   │
│              /health endpoint (liveness + model status)                │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Monitoring Layer                                   │
│                                                                         │
│   Prediction log  ──►  Evidently AI  ──►  drift_report.html           │
│  (data/processed/      (data drift,        (feature drift,            │
│  predictions_log)       data quality)        target drift)             │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       CI/CD Pipeline (GitHub Actions)                  │
│                                                                         │
│  push/PR  ──►  lint (ruff)  ──►  test (pytest)  ──►  retrain (DVC)   │
│                                                          │              │
│                                              evaluate & compare AUC-PR │
│                                                          │              │
│                                              AUC-PR improves ≥ 0.5%?  │
│                                             YES ──► build + push image │
│                                                   ──► deploy to prod   │
│                                              NO  ──► skip deploy       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Preprocess data

```bash
python src/data_preprocessing.py
# or via DVC:
dvc repro preprocess
```

### 3. Train model

```bash
python src/train.py
# or via DVC (full pipeline):
dvc repro
```

### 4. Launch MLflow UI

```bash
mlflow ui
# open http://localhost:5000
```

### 5. Serve the API

```bash
uvicorn api.main:app --reload
# open http://localhost:8000/docs
```

### 6. Run tests

```bash
pytest tests/ -v --cov=src --cov=api
```

### 7. Generate drift report

```bash
python monitoring/drift_report.py
# report saved to monitoring/reports/drift_report.html
```

---

## Docker

```bash
# Start the full stack (API + MLflow)
docker compose up -d

# Run the training job
docker compose --profile train up train

# API docs
open http://localhost:8000/docs

# MLflow UI
open http://localhost:5000
```

---

## API Reference

### `POST /predict`

Score a transaction for fraud.

**Request body** (JSON):

```json
{
  "transaction": {
    "TransactionDT": 86400,
    "TransactionAmt": 150.00,
    "ProductCD": "W",
    "card1": 9500,
    "card4": "visa",
    "card6": "debit",
    "P_emaildomain": "gmail.com"
  },
  "identity": {
    "DeviceType": "mobile",
    "DeviceInfo": "Samsung Galaxy S21",
    "id_01": -1.0
  }
}
```

**Response**:

```json
{
  "fraud_probability": 0.0342,
  "is_fraud": false,
  "threshold": 0.5,
  "model_version": "3"
}
```

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "3",
  "uptime_seconds": 120.45
}
```

---

## Project Structure

```
fraud-detection-mlops/
├── data/
│   ├── raw/                        # Original CSVs (DVC-tracked)
│   │   ├── train_transaction.csv
│   │   ├── train_identity.csv
│   │   ├── test_transaction.csv
│   │   ├── test_identity.csv
│   │   └── sample_submission.csv
│   └── processed/                  # Parquet outputs (DVC-tracked)
│       ├── train.parquet
│       └── test.parquet
├── src/
│   ├── data_preprocessing.py       # Merge, impute, encode, save parquet
│   ├── feature_engineering.py      # Time, email, card, amount features
│   ├── train.py                    # XGBoost + SMOTE + MLflow logging
│   └── evaluate.py                 # Metrics + AUC-PR gating logic
├── api/
│   ├── main.py                     # FastAPI app (lifespan, /health, /predict)
│   └── schemas.py                  # Pydantic v2 request/response models
├── monitoring/
│   └── drift_report.py             # Evidently AI drift reports
├── tests/
│   ├── test_preprocessing.py       # 15 unit tests for preprocessing
│   ├── test_api.py                 # 15 unit tests for FastAPI endpoints
│   └── test_model.py               # 15 unit tests for features & model
├── .github/workflows/ci_cd.yml     # lint → test → retrain → compare → deploy
├── dvc.yaml                        # preprocess → train → evaluate stages
├── params.yaml                     # All hyperparameters (single source of truth)
├── MLproject                       # MLflow Projects entrypoints
├── Dockerfile                      # Multi-stage build (builder + runtime)
├── docker-compose.yml              # API + MLflow + training job
└── requirements.txt
```

---

## Configuration

All model and pipeline hyperparameters live in **`params.yaml`** — never hardcoded:

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `model` | `n_estimators` | 300 | XGBoost trees |
| `model` | `max_depth` | 6 | Tree depth |
| `model` | `learning_rate` | 0.05 | Shrinkage |
| `model` | `scale_pos_weight` | 10 | Class imbalance weight |
| `smote` | `sampling_strategy` | 0.1 | Minority/majority ratio after SMOTE |
| `smote` | `k_neighbors` | 5 | SMOTE neighbourhood size |
| `evaluation` | `threshold` | 0.5 | Fraud decision threshold |
| `evaluation` | `auc_pr_improvement_threshold` | 0.005 | Min AUC-PR delta to trigger deploy |

---

## Dataset

IEEE-CIS Fraud Detection (Kaggle):
- 590,540 transactions · 394 transaction features · 41 identity features
- Merged on `TransactionID` (left join)
- Target: `isFraud` (3.5% positive class → severe imbalance → SMOTE)

---

## CI/CD Gate Logic

```
Retrain → evaluate → compare AUC-PR (current vs previous run)
   if delta ≥ 0.5%  → build Docker image → push to GHCR → deploy
   else              → skip deploy (model did not improve)
```

Override with `workflow_dispatch` → `force_deploy: true`.

---

## Monitoring

The Evidently drift report compares:
- **Reference**: training data distribution
- **Current**: incoming prediction request log

Run manually:
```bash
python monitoring/drift_report.py \
  --current data/processed/predictions_log.parquet \
  --output  monitoring/reports/drift_report.html
```

Reports generated: data drift · data quality · target drift (when labels available).
