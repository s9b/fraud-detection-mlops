"""
Data preprocessing pipeline for IEEE-CIS Fraud Detection dataset.
Merges transaction + identity, imputes missing values, encodes categoricals.
"""

import logging
import sys
from pathlib import Path

# Allow running as `python src/data_preprocessing.py` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def load_raw_data(
    transaction_path: str,
    identity_path: str,
) -> pd.DataFrame:
    logger.info("Loading transaction data from %s", transaction_path)
    transactions = pd.read_csv(transaction_path)
    logger.info("Loaded %d transactions with %d columns", *transactions.shape)

    logger.info("Loading identity data from %s", identity_path)
    identity = pd.read_csv(identity_path)
    logger.info("Loaded %d identity rows with %d columns", *identity.shape)

    logger.info("Merging on TransactionID (left join)")
    df = transactions.merge(identity, on="TransactionID", how="left")
    logger.info("Merged shape: %s", df.shape)
    return df


def drop_columns(df: pd.DataFrame, drop_cols: list) -> pd.DataFrame:
    existing = [c for c in drop_cols if c in df.columns]
    logger.info("Dropping columns: %s", existing)
    return df.drop(columns=existing, errors="ignore")


def impute_missing(
    df: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_strategy: str = "constant",
    categorical_fill_value: str = "unknown",
) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    logger.info(
        "Imputing %d numeric cols with '%s' strategy", len(numeric_cols), numeric_strategy
    )
    if numeric_strategy == "median":
        for col in numeric_cols:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    elif numeric_strategy == "mean":
        for col in numeric_cols:
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)
    elif numeric_strategy == "zero":
        df[numeric_cols] = df[numeric_cols].fillna(0)

    logger.info(
        "Imputing %d categorical cols with '%s' strategy",
        len(categorical_cols),
        categorical_strategy,
    )
    if categorical_strategy == "constant":
        df[categorical_cols] = df[categorical_cols].fillna(categorical_fill_value)
    elif categorical_strategy == "mode":
        for col in categorical_cols:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else categorical_fill_value
            df[col] = df[col].fillna(mode_val)

    return df


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Label-encode all object/category columns. Returns (df, encoder_map)."""
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    encoder_map = {}

    logger.info("Label encoding %d categorical columns", len(categorical_cols))
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoder_map[col] = le

    return df, encoder_map


def split_features_target(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def preprocess(params_path: str = "params.yaml") -> None:
    params = load_params(params_path)
    data_cfg = params["data"]
    prep_cfg = params["preprocessing"]

    Path(data_cfg["processed_dir"]).mkdir(parents=True, exist_ok=True)

    # --- Training data ---
    train_df = load_raw_data(
        transaction_path=data_cfg["train_transaction"],
        identity_path=data_cfg["train_identity"],
    )
    train_df = drop_columns(train_df, prep_cfg["drop_cols"])
    train_df = impute_missing(
        train_df,
        numeric_strategy=prep_cfg["numeric_impute_strategy"],
        categorical_strategy=prep_cfg["categorical_impute_strategy"],
        categorical_fill_value=prep_cfg["categorical_fill_value"],
    )
    train_df, _ = encode_categoricals(train_df)

    train_out = data_cfg["processed_train"]
    train_df.to_parquet(train_out, index=False)
    logger.info("Saved processed training data to %s  shape=%s", train_out, train_df.shape)

    # --- Test data (no isFraud column) ---
    test_df = load_raw_data(
        transaction_path=data_cfg["test_transaction"],
        identity_path=data_cfg["test_identity"],
    )
    # Remove TransactionID only (no isFraud in test)
    test_df = drop_columns(test_df, prep_cfg["drop_cols"])
    test_df = impute_missing(
        test_df,
        numeric_strategy=prep_cfg["numeric_impute_strategy"],
        categorical_strategy=prep_cfg["categorical_impute_strategy"],
        categorical_fill_value=prep_cfg["categorical_fill_value"],
    )
    test_df, _ = encode_categoricals(test_df)

    test_out = data_cfg["processed_test"]
    test_df.to_parquet(test_out, index=False)
    logger.info("Saved processed test data to %s  shape=%s", test_out, test_df.shape)


if __name__ == "__main__":
    preprocess()
