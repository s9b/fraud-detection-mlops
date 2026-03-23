# Fraud Detection MLOps Project

## Project Overview
Production MLOps pipeline for IEEE-CIS fraud detection.
590k transactions, XGBoost classifier, severe class imbalance (3.5% fraud).

## Tech Stack
- Model: XGBoost + SMOTE (imbalanced-learn)
- Tracking: MLflow
- Versioning: DVC
- Serving: FastAPI + Pydantic
- Monitoring: Evidently AI
- CI/CD: GitHub Actions
- Container: Docker + docker-compose
- Tests: pytest

## Commands
- Train model: `python src/train.py`
- Run API: `uvicorn api.main:app --reload`
- Run tests: `pytest tests/ -v`
- DVC pipeline: `dvc repro`
- MLflow UI: `mlflow ui`
- Evidently report: `python monitoring/drift_report.py`

## Data
- Raw files in `data/raw/`
- train_transaction.csv + train_identity.csv merge on TransactionID
- Target column: isFraud
- Class imbalance: ~3.5% positive class

## Conventions
- All hyperparameters in params.yaml, never hardcoded
- All experiments logged to MLflow
- Feature engineering in src/feature_engineering.py
- Pydantic schemas in api/schemas.py
- Never commit data files (handled by DVC)
