"""
Evaluation script: loads best registered model from MLflow,
runs it on the validation split, prints and logs metrics.
Used by CI/CD to gate deployment based on AUC-PR improvement.
"""

import json
import logging
import sys
from pathlib import Path

# Allow running as `python src/evaluate.py` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.feature_engineering import run_feature_engineering

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def load_latest_model(registry_name: str, mlflow_cfg: dict):
    """Load the latest version of the registered model."""
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(registry_name, stages=["None", "Staging", "Production"])
    if not versions:
        raise RuntimeError(f"No versions found for model '{registry_name}'")
    latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
    model_uri = f"models:/{registry_name}/{latest.version}"
    logger.info("Loading model: %s  (version %s)", registry_name, latest.version)
    model = mlflow.xgboost.load_model(model_uri)
    return model, latest.version


def evaluate(params_path: str = "params.yaml", output_json: str = "metrics.json") -> dict:
    params = load_params(params_path)
    data_cfg = params["data"]
    eval_cfg = params["evaluation"]
    mlflow_cfg = params["mlflow"]

    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    # Load processed training data (re-derive validation split)
    processed_train = Path(data_cfg["processed_train"])
    if not processed_train.exists():
        raise FileNotFoundError(
            f"Processed training data not found at {processed_train}. Run preprocessing first."
        )

    df = pd.read_parquet(processed_train)
    df = run_feature_engineering(df, params)

    target_col = data_cfg["target_column"]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    _, X_val, _, y_val = train_test_split(
        X,
        y,
        test_size=data_cfg["test_size"],
        random_state=data_cfg["random_state"],
        stratify=y,
    )

    # Load model
    model, model_version = load_latest_model(mlflow_cfg["model_registry_name"], mlflow_cfg)

    # Predict
    threshold = eval_cfg["threshold"]
    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # Compute metrics
    metrics = {
        "roc_auc": float(roc_auc_score(y_val, y_proba)),
        "average_precision": float(average_precision_score(y_val, y_proba)),
        "f1": float(f1_score(y_val, y_pred, zero_division=0)),
        "precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "recall": float(recall_score(y_val, y_pred, zero_division=0)),
        "model_version": model_version,
        "threshold": threshold,
        "n_val_samples": len(y_val),
        "n_fraud_val": int(y_val.sum()),
    }

    logger.info("Evaluation metrics (model v%s):", model_version)
    for k, v in metrics.items():
        logger.info("  %-25s %s", k, v)

    # Print classification report
    logger.info("\n%s", classification_report(y_val, y_pred, target_names=["legit", "fraud"]))

    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    logger.info("Confusion matrix:\n%s", cm)

    # Save to JSON for CI/CD pipeline comparison
    with open(output_json, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", output_json)

    # Log to MLflow as a standalone run
    with mlflow.start_run(run_name=f"evaluate_v{model_version}"):
        mlflow.log_params({"model_version": model_version, "threshold": threshold})
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)
        mlflow.log_artifact(output_json)

    return metrics


def compare_metrics(
    current_json: str,
    previous_json: str,
    improvement_threshold: float,
) -> bool:
    """Return True if current AUC-PR improves over previous by at least improvement_threshold."""
    with open(current_json) as f:
        current = json.load(f)
    with open(previous_json) as f:
        previous = json.load(f)

    current_ap = current["average_precision"]
    previous_ap = previous["average_precision"]
    delta = current_ap - previous_ap

    logger.info(
        "AUC-PR comparison: current=%.4f  previous=%.4f  delta=%.4f  threshold=%.4f",
        current_ap,
        previous_ap,
        delta,
        improvement_threshold,
    )
    return delta >= improvement_threshold


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate fraud detection model")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--output", default="metrics.json")
    parser.add_argument("--previous", default=None, help="Path to previous metrics.json for comparison")
    args = parser.parse_args()

    metrics = evaluate(params_path=args.params, output_json=args.output)

    if args.previous:
        params = load_params(args.params)
        threshold = params["evaluation"]["auc_pr_improvement_threshold"]
        should_deploy = compare_metrics(args.output, args.previous, threshold)
        logger.info("Should deploy: %s", should_deploy)
        sys.exit(0 if should_deploy else 1)
