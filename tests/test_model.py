"""
Tests for model training logic, feature engineering, and prediction shape/behaviour.
Uses small synthetic datasets — no real CSV loading needed.
"""

import numpy as np
import pandas as pd
import pytest
import yaml
from pathlib import Path
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from src.feature_engineering import (
    add_card_features,
    add_email_domain_features,
    add_time_features,
    add_transaction_amount_features,
    add_count_features,
    run_feature_engineering,
)
from src.train import compute_metrics


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "TransactionDT": rng.integers(86400, 86400 * 365, size=n).astype(float),
            "TransactionAmt": rng.uniform(1, 5000, size=n),
            "ProductCD": rng.choice(["W", "H", "C", "S", "R"], size=n),
            "card1": rng.integers(1000, 20000, size=n).astype(float),
            "card2": rng.integers(100, 600, size=n).astype(float),
            "card4": rng.choice(["visa", "mastercard", "discover"], size=n),
            "card5": rng.integers(100, 600, size=n).astype(float),
            "card6": rng.choice(["debit", "credit"], size=n),
            "addr1": rng.integers(100, 600, size=n).astype(float),
            "addr2": rng.integers(1, 100, size=n).astype(float),
            "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", "anonymous.com"], size=n),
            "R_emaildomain": rng.choice(["gmail.com", "hotmail.com", "comcast.net"], size=n),
            "C1": rng.uniform(0, 10, size=n),
            "D1": rng.uniform(0, 500, size=n),
            "isFraud": rng.choice([0, 1], size=n, p=[0.965, 0.035]),
        }
    )
    return df


def _make_params() -> dict:
    return {
        "feature_engineering": {
            "time_features": True,
            "email_domain_features": True,
            "card_features": True,
        }
    }


def _train_small_model(X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=20,
        max_depth=3,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X, y)
    return model


# ── Feature engineering tests ─────────────────────────────────────────────────

def test_add_time_features_creates_expected_columns():
    df = _make_df(50)
    result = add_time_features(df)
    for col in ["tx_hour", "tx_dayofweek", "tx_day", "tx_month", "tx_is_weekend", "tx_is_night"]:
        assert col in result.columns, f"Missing column: {col}"


def test_add_time_features_hour_in_range():
    df = _make_df(100)
    result = add_time_features(df)
    assert result["tx_hour"].between(0, 23).all()


def test_add_email_domain_features_creates_match_column():
    df = _make_df(50)
    result = add_email_domain_features(df)
    assert "email_domain_match" in result.columns
    assert result["email_domain_match"].isin([0, 1]).all()


def test_add_email_domain_features_is_top_binary():
    df = _make_df(50)
    result = add_email_domain_features(df)
    assert "p_email_is_top" in result.columns
    assert result["p_email_is_top"].isin([0, 1]).all()


def test_add_card_features_no_nan_after():
    df = _make_df(50)
    result = add_card_features(df)
    card_feat_cols = [c for c in result.columns if "card1_card2" in c or "amt_card" in c]
    assert len(card_feat_cols) > 0
    assert result[card_feat_cols].isna().sum().sum() == 0


def test_add_transaction_amount_features_log_positive():
    df = _make_df(100)
    result = add_transaction_amount_features(df)
    assert "tx_amt_log" in result.columns
    assert (result["tx_amt_log"] >= 0).all()


def test_add_count_features_non_negative():
    df = _make_df(100)
    result = add_count_features(df)
    assert "card1_freq" in result.columns
    assert (result["card1_freq"] >= 0).all()


def test_run_feature_engineering_increases_column_count():
    df = _make_df(100)
    params = _make_params()
    original_cols = df.shape[1]
    result = run_feature_engineering(df.copy(), params)
    assert result.shape[1] > original_cols


def test_run_feature_engineering_no_nans():
    df = _make_df(100)
    params = _make_params()
    result = run_feature_engineering(df.copy(), params)
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    assert result[numeric_cols].isna().sum().sum() == 0


# ── compute_metrics tests ─────────────────────────────────────────────────────

def test_compute_metrics_returns_all_keys():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    metrics = compute_metrics(y_true, y_proba, threshold=0.5)
    for key in ["roc_auc", "average_precision", "f1", "precision", "recall"]:
        assert key in metrics


def test_compute_metrics_perfect_model():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.01, 0.02, 0.03, 0.97, 0.98, 0.99])
    metrics = compute_metrics(y_true, y_proba, threshold=0.5)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)


def test_compute_metrics_values_in_unit_interval():
    rng = np.random.default_rng(1)
    y_true = rng.choice([0, 1], size=100, p=[0.9, 0.1])
    y_proba = rng.uniform(0, 1, size=100)
    metrics = compute_metrics(y_true, y_proba, threshold=0.5)
    for k, v in metrics.items():
        assert 0.0 <= v <= 1.0, f"Metric {k}={v} out of range"


# ── Model fit / predict shape tests ──────────────────────────────────────────

def test_model_predict_proba_shape():
    df = _make_df(200)
    params = _make_params()
    df_fe = run_feature_engineering(df.copy(), params)
    X = df_fe.drop(columns=["isFraud"])
    y = df_fe["isFraud"]

    model = _train_small_model(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)


def test_model_predict_proba_sums_to_one():
    df = _make_df(200)
    params = _make_params()
    df_fe = run_feature_engineering(df.copy(), params)
    X = df_fe.drop(columns=["isFraud"])
    y = df_fe["isFraud"]

    model = _train_small_model(X, y)
    proba = model.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-5)


def test_model_fraud_proba_column_index():
    """Ensure class index 1 corresponds to fraud class."""
    df = _make_df(200)
    params = _make_params()
    df_fe = run_feature_engineering(df.copy(), params)
    X = df_fe.drop(columns=["isFraud"])
    y = df_fe["isFraud"]

    model = _train_small_model(X, y)
    assert model.classes_[1] == 1  # fraud label


def test_model_feature_importances_length():
    df = _make_df(200)
    params = _make_params()
    df_fe = run_feature_engineering(df.copy(), params)
    X = df_fe.drop(columns=["isFraud"])
    y = df_fe["isFraud"]

    model = _train_small_model(X, y)
    assert len(model.feature_importances_) == X.shape[1]


def test_model_average_precision_above_random():
    """A trained model should beat a random baseline on AUC-PR."""
    df = _make_df(500, seed=7)
    params = _make_params()
    df_fe = run_feature_engineering(df.copy(), params)
    X = df_fe.drop(columns=["isFraud"])
    y = df_fe["isFraud"]

    model = _train_small_model(X, y)
    proba = model.predict_proba(X)[:, 1]
    ap = average_precision_score(y, proba)
    # Random baseline AUC-PR ≈ fraud prevalence (~3.5%)
    assert ap > y.mean(), "Model should outperform random baseline"
