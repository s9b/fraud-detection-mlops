"""
Tests for the FastAPI application.
Covers: health endpoint, predict validation, schema enforcement, error handling.
Uses httpx AsyncClient with the ASGI transport (no live server required).
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import numpy as np


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """
    Create a TestClient with the model loading mocked out so tests
    don't require a trained MLflow model on disk.
    """
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.85, 0.15]])
    mock_model.feature_names_in_ = None  # skip alignment

    with patch("api.main._load_model", return_value=(mock_model, "test-v1")):
        with patch("api.main._load_params", return_value=_mock_params()):
            from api.main import app
            with TestClient(app) as c:
                yield c


def _mock_params() -> dict:
    return {
        "mlflow": {
            "experiment_name": "fraud_detection",
            "model_registry_name": "fraud_detector",
            "artifact_path": "model",
        },
        "evaluation": {"threshold": 0.5},
        "feature_engineering": {
            "time_features": True,
            "email_domain_features": True,
            "card_features": True,
        },
    }


def _minimal_transaction() -> dict:
    return {
        "TransactionDT": 86400,
        "TransactionAmt": 150.0,
        "ProductCD": "W",
        "card1": 9500,
        "card4": "visa",
        "card6": "debit",
    }


def _full_predict_payload() -> dict:
    return {
        "transaction": _minimal_transaction(),
        "identity": {
            "DeviceType": "mobile",
            "DeviceInfo": "Samsung",
            "id_01": -1.0,
        },
    }


# ── Health endpoint ───────────────────────────────────────────────────────────

def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_schema(client):
    response = client.get("/health")
    body = response.json()
    assert "status" in body
    assert "model_loaded" in body
    assert "model_version" in body
    assert "uptime_seconds" in body


def test_health_status_ok(client):
    response = client.get("/health")
    assert response.json()["status"] == "ok"


def test_health_model_loaded_true(client):
    response = client.get("/health")
    assert response.json()["model_loaded"] is True


# ── Root endpoint ─────────────────────────────────────────────────────────────

def test_root_returns_200(client):
    response = client.get("/")
    assert response.status_code == 200


# ── Predict endpoint — valid requests ─────────────────────────────────────────

def test_predict_returns_200_with_valid_payload(client):
    response = client.post("/predict", json=_full_predict_payload())
    assert response.status_code == 200


def test_predict_response_has_fraud_probability(client):
    response = client.post("/predict", json=_full_predict_payload())
    body = response.json()
    assert "fraud_probability" in body
    assert 0.0 <= body["fraud_probability"] <= 1.0


def test_predict_response_has_is_fraud_bool(client):
    response = client.post("/predict", json=_full_predict_payload())
    body = response.json()
    assert "is_fraud" in body
    assert isinstance(body["is_fraud"], bool)


def test_predict_response_has_threshold_and_version(client):
    response = client.post("/predict", json=_full_predict_payload())
    body = response.json()
    assert "threshold" in body
    assert "model_version" in body


def test_predict_transaction_only_no_identity(client):
    """Identity is optional — request without it should still succeed."""
    payload = {"transaction": _minimal_transaction()}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


def test_predict_all_null_features_accepted(client):
    """All-null transaction features should be accepted and not crash."""
    payload = {
        "transaction": {
            "TransactionDT": None,
            "TransactionAmt": None,
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


# ── Predict endpoint — invalid requests ───────────────────────────────────────

def test_predict_rejects_negative_transaction_amount(client):
    payload = {
        "transaction": {
            "TransactionAmt": -500.0,
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_extra_top_level_fields(client):
    """Schema has extra='forbid' at the top level."""
    payload = {
        "transaction": _minimal_transaction(),
        "unexpected_field": "should fail",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_empty_body_returns_422(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_missing_transaction_key_returns_422(client):
    payload = {"identity": {"DeviceType": "mobile"}}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


# ── Model unavailable ─────────────────────────────────────────────────────────

def test_predict_returns_503_when_model_not_loaded():
    """When model loading fails, /predict should return 503."""
    with patch("api.main._load_model", side_effect=RuntimeError("no model")):
        with patch("api.main._load_params", return_value=_mock_params()):
            from importlib import reload
            import api.main as main_module

            # Directly set global to None to simulate failed load
            original_model = main_module._model
            main_module._model = None

            with TestClient(main_module.app) as c:
                response = c.post("/predict", json=_full_predict_payload())
                assert response.status_code == 503

            main_module._model = original_model
