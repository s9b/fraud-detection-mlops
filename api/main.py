"""
FastAPI serving layer for the fraud detection model.
Endpoints:
  GET  /health   — liveness + model status
  POST /predict  — return fraud probability for a single transaction
"""

import logging
import time
from contextlib import asynccontextmanager

import mlflow
import mlflow.xgboost
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import HealthResponse, PredictRequest, PredictResponse
from src.feature_engineering import run_feature_engineering

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Global state ──────────────────────────────────────────────────────────────
_model = None
_model_version: str = "unknown"
_feature_names: list[str] = []
_start_time: float = time.time()
_params: dict = {}
_threshold: float = 0.5


def _load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def _load_model(params: dict) -> tuple:
    """Load the latest production/staging/none model from MLflow registry."""
    registry_name = params["mlflow"]["model_registry_name"]
    client = mlflow.tracking.MlflowClient()

    try:
        versions = client.get_latest_versions(
            registry_name, stages=["Production", "Staging", "None"]
        )
        if not versions:
            raise RuntimeError(f"No model versions found for '{registry_name}'")
        latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
        model_uri = f"models:/{registry_name}/{latest.version}"
        logger.info(
            "Loading model %s version %s from MLflow", registry_name, latest.version
        )
        model = mlflow.xgboost.load_model(model_uri)
        return model, str(latest.version)
    except Exception as exc:
        logger.warning("Could not load model from MLflow registry: %s", exc)
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    global _model, _model_version, _params, _threshold, _feature_names

    logger.info("Starting fraud detection API …")
    _params = _load_params()
    _threshold = _params["evaluation"]["threshold"]

    try:
        _model, _model_version = _load_model(_params)
        logger.info("Model v%s loaded successfully", _model_version)
    except Exception as exc:
        logger.error("Failed to load model: %s — /predict will be unavailable", exc)

    yield

    logger.info("Shutting down fraud detection API")


app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud scoring for IEEE-CIS transactions using XGBoost",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_feature_row(request: PredictRequest, params: dict) -> pd.DataFrame:
    """Convert a PredictRequest into a single-row DataFrame ready for the model."""
    tx = request.transaction.model_dump()
    identity = request.identity.model_dump() if request.identity else {}

    # Merge into one dict
    row = {**tx, **identity}

    # Encode string/categorical fields the same way preprocessing does (label encoding → int)
    str_cols = [k for k, v in row.items() if isinstance(v, str)]
    for col in str_cols:
        # Simple hash-based encoding for serving (consistent with LabelEncoder range)
        row[col] = abs(hash(row[col])) % 100_000

    df = pd.DataFrame([row])

    # Coerce every column to float64 — None values become NaN, then fill with 0.
    # This is necessary because a single-row DataFrame built from a dict of
    # mostly-None values leaves those columns as object dtype.
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Apply feature engineering
    df = run_feature_engineering(df, params)

    # Final dtype enforcement: ensure no object columns remain before XGBoost
    df = df.astype(float)

    return df


def _align_to_model(df: pd.DataFrame, model) -> pd.DataFrame:
    """Ensure the dataframe has exactly the features the model was trained on."""
    if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
        model_features = list(model.feature_names_in_)
        # Add missing columns as 0
        for col in model_features:
            if col not in df.columns:
                df[col] = 0
        df = df[model_features]
    return df


@app.get("/health", response_model=HealthResponse, tags=["infra"])
async def health() -> HealthResponse:
    """Liveness check — always returns 200 if the server is up."""
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        model_version=_model_version,
        uptime_seconds=round(time.time() - _start_time, 2),
    )


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
async def predict(request: PredictRequest) -> PredictResponse:
    """
    Score a single transaction for fraud.

    Returns fraud probability, binary label, and the threshold used.
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please ensure the model has been trained and registered.",
        )

    try:
        df = _build_feature_row(request, _params)
        df = _align_to_model(df, _model)
        proba = float(_model.predict_proba(df)[0, 1])
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    is_fraud = proba >= _threshold

    logger.info(
        "TransactionAmt=%.2f  fraud_prob=%.4f  is_fraud=%s",
        request.transaction.TransactionAmt or 0.0,
        proba,
        is_fraud,
    )

    return PredictResponse(
        fraud_probability=proba,
        is_fraud=is_fraud,
        threshold=_threshold,
        model_version=_model_version,
    )


@app.get("/", tags=["infra"])
async def root():
    return {"message": "Fraud Detection API", "docs": "/docs", "health": "/health"}
