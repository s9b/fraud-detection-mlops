"""
Training pipeline: load processed data, apply feature engineering,
train XGBoost with SMOTE, log everything to MLflow, register best model.
"""

import logging
import os
import sys
from pathlib import Path

# Allow running as `python src/train.py` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.data_preprocessing import preprocess, split_features_target
from src.feature_engineering import run_feature_engineering

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    y_pred = (y_pred_proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
        "average_precision": average_precision_score(y_true, y_pred_proba),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def train(params_path: str = "params.yaml") -> None:
    params = load_params(params_path)
    data_cfg = params["data"]
    model_cfg = params["model"]
    smote_cfg = params["smote"]
    eval_cfg = params["evaluation"]
    mlflow_cfg = params["mlflow"]

    # ── Ensure processed data exists ──────────────────────────────────────────
    processed_train = Path(data_cfg["processed_train"])
    if not processed_train.exists():
        logger.info("Processed data not found — running preprocessing first")
        preprocess(params_path)

    # ── Load processed training data ──────────────────────────────────────────
    logger.info("Loading processed training data from %s", processed_train)
    df = pd.read_parquet(processed_train)
    logger.info("Loaded shape: %s", df.shape)

    # ── Feature engineering ───────────────────────────────────────────────────
    df = run_feature_engineering(df, params)

    # ── Split features / target ───────────────────────────────────────────────
    X, y = split_features_target(df, data_cfg["target_column"])
    logger.info("Class distribution:\n%s", y.value_counts())

    # ── Train / validation split ──────────────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=data_cfg["test_size"],
        random_state=data_cfg["random_state"],
        stratify=y,
    )
    logger.info("Train shape: %s  Val shape: %s", X_train.shape, X_val.shape)

    # ── SMOTE oversampling on training set ────────────────────────────────────
    logger.info(
        "Applying SMOTE (sampling_strategy=%.2f, k=%d)",
        smote_cfg["sampling_strategy"],
        smote_cfg["k_neighbors"],
    )
    sm = SMOTE(
        sampling_strategy=smote_cfg["sampling_strategy"],
        k_neighbors=smote_cfg["k_neighbors"],
        random_state=smote_cfg["random_state"],
    )
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    logger.info(
        "After SMOTE — train shape: %s  class distribution:\n%s",
        X_train_res.shape,
        pd.Series(y_train_res).value_counts(),
    )

    # ── MLflow setup ──────────────────────────────────────────────────────────
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info("MLflow run_id: %s", run_id)

        # Log all params
        mlflow.log_params(
            {
                "n_estimators": model_cfg["n_estimators"],
                "max_depth": model_cfg["max_depth"],
                "learning_rate": model_cfg["learning_rate"],
                "subsample": model_cfg["subsample"],
                "colsample_bytree": model_cfg["colsample_bytree"],
                "min_child_weight": model_cfg["min_child_weight"],
                "scale_pos_weight": model_cfg["scale_pos_weight"],
                "reg_alpha": model_cfg["reg_alpha"],
                "reg_lambda": model_cfg["reg_lambda"],
                "random_state": model_cfg["random_state"],
                "smote_sampling_strategy": smote_cfg["sampling_strategy"],
                "smote_k_neighbors": smote_cfg["k_neighbors"],
                "train_size": len(X_train_res),
                "val_size": len(X_val),
                "n_features": X_train.shape[1],
            }
        )

        # ── Build & train model ────────────────────────────────────────────────
        model = XGBClassifier(
            n_estimators=model_cfg["n_estimators"],
            max_depth=model_cfg["max_depth"],
            learning_rate=model_cfg["learning_rate"],
            subsample=model_cfg["subsample"],
            colsample_bytree=model_cfg["colsample_bytree"],
            min_child_weight=model_cfg["min_child_weight"],
            scale_pos_weight=model_cfg["scale_pos_weight"],
            reg_alpha=model_cfg["reg_alpha"],
            reg_lambda=model_cfg["reg_lambda"],
            random_state=model_cfg["random_state"],
            n_jobs=model_cfg["n_jobs"],
            eval_metric=model_cfg["eval_metric"],
            early_stopping_rounds=model_cfg["early_stopping_rounds"],
            use_label_encoder=False,
        )

        logger.info("Training XGBoost model …")
        model.fit(
            X_train_res,
            y_train_res,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )

        # ── Evaluate on validation set ─────────────────────────────────────────
        y_val_proba = model.predict_proba(X_val)[:, 1]
        metrics = compute_metrics(y_val, y_val_proba, threshold=eval_cfg["threshold"])

        logger.info("Validation metrics:")
        for k, v in metrics.items():
            logger.info("  %-22s %.4f", k, v)
            mlflow.log_metric(k, v)

        # ── Log feature importances ────────────────────────────────────────────
        importance_df = pd.DataFrame(
            {
                "feature": X_train.columns.tolist(),
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        importance_path = "feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)
        os.remove(importance_path)

        # ── Log model ──────────────────────────────────────────────────────────
        feature_names = X_train.columns.tolist()
        signature = mlflow.models.infer_signature(
            X_val[:5], model.predict_proba(X_val[:5])[:, 1]
        )
        mlflow.xgboost.log_model(
            model,
            artifact_path=mlflow_cfg["artifact_path"],
            signature=signature,
            registered_model_name=mlflow_cfg["model_registry_name"],
            input_example=X_val[:3],
        )

        # ── Persist run metadata for evaluate.py ──────────────────────────────
        meta = {
            "run_id": run_id,
            "average_precision": float(metrics["average_precision"]),
            "roc_auc": float(metrics["roc_auc"]),
            "feature_names": feature_names,
        }
        meta_path = Path(data_cfg["processed_dir"]) / "run_meta.yaml"
        with open(meta_path, "w") as f:
            yaml.dump(meta, f)
        mlflow.log_artifact(str(meta_path))
        logger.info("Run metadata saved to %s", meta_path)

    logger.info("Training complete. Best model registered as '%s'.", mlflow_cfg["model_registry_name"])


if __name__ == "__main__":
    train()
