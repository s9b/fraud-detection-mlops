"""
Tests for src/data_preprocessing.py
Covers: loading, merging, imputation, encoding, splitting, column dropping.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from src.data_preprocessing import (
    drop_columns,
    encode_categoricals,
    impute_missing,
    load_raw_data,
    split_features_target,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_transaction_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5],
            "isFraud": [0, 1, 0, 0, 1],
            "TransactionDT": [86400, 172800, 259200, 345600, 432000],
            "TransactionAmt": [100.0, np.nan, 300.0, 400.0, np.nan],
            "ProductCD": ["W", "H", np.nan, "C", "S"],
            "card1": [1234, 5678, 9012, np.nan, 3456],
            "card4": ["visa", "mastercard", "visa", np.nan, "discover"],
        }
    )


@pytest.fixture
def sample_identity_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [1, 3, 5],
            "id_01": [-1.0, -5.0, 0.0],
            "DeviceType": ["mobile", "desktop", "mobile"],
        }
    )


@pytest.fixture
def clean_numeric_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
            "isFraud": [0, 1, 0],
        }
    )


# ── Test: load_raw_data ───────────────────────────────────────────────────────


def test_load_raw_data_merges_on_transaction_id(
    tmp_path, sample_transaction_df, sample_identity_df
):
    tx_path = tmp_path / "train_transaction.csv"
    id_path = tmp_path / "train_identity.csv"
    sample_transaction_df.to_csv(tx_path, index=False)
    sample_identity_df.to_csv(id_path, index=False)

    merged = load_raw_data(str(tx_path), str(id_path))

    assert "TransactionID" in merged.columns
    assert "id_01" in merged.columns
    assert "DeviceType" in merged.columns
    assert len(merged) == 5  # left join preserves all transactions


def test_load_raw_data_left_join_preserves_all_transactions(
    tmp_path, sample_transaction_df, sample_identity_df
):
    tx_path = tmp_path / "train_transaction.csv"
    id_path = tmp_path / "train_identity.csv"
    sample_transaction_df.to_csv(tx_path, index=False)
    sample_identity_df.to_csv(id_path, index=False)

    merged = load_raw_data(str(tx_path), str(id_path))

    # Transactions 2 and 4 have no identity row → NaN in id_01
    tx2_row = merged[merged["TransactionID"] == 2]
    assert pd.isna(tx2_row["id_01"].values[0])


# ── Test: drop_columns ────────────────────────────────────────────────────────


def test_drop_columns_removes_specified(sample_transaction_df):
    result = drop_columns(sample_transaction_df, ["TransactionID"])
    assert "TransactionID" not in result.columns


def test_drop_columns_ignores_missing_cols(sample_transaction_df):
    result = drop_columns(sample_transaction_df, ["NonExistentCol"])
    assert result.shape == sample_transaction_df.shape


def test_drop_columns_multiple(sample_transaction_df):
    result = drop_columns(sample_transaction_df, ["TransactionID", "card1"])
    assert "TransactionID" not in result.columns
    assert "card1" not in result.columns


# ── Test: impute_missing ──────────────────────────────────────────────────────


def test_impute_missing_numeric_median(sample_transaction_df):
    result = impute_missing(sample_transaction_df.copy(), numeric_strategy="median")
    assert result["TransactionAmt"].isna().sum() == 0
    assert result["card1"].isna().sum() == 0


def test_impute_missing_categorical_constant(sample_transaction_df):
    result = impute_missing(
        sample_transaction_df.copy(),
        categorical_strategy="constant",
        categorical_fill_value="unknown",
    )
    assert result["ProductCD"].isna().sum() == 0
    assert result["card4"].isna().sum() == 0
    assert "unknown" in result["ProductCD"].values


def test_impute_missing_no_nans_after_imputation(sample_transaction_df):
    result = impute_missing(
        sample_transaction_df.copy(),
        numeric_strategy="median",
        categorical_strategy="constant",
        categorical_fill_value="unknown",
    )
    assert result.isna().sum().sum() == 0


# ── Test: encode_categoricals ─────────────────────────────────────────────────


def test_encode_categoricals_converts_strings_to_int(sample_transaction_df):
    df = impute_missing(sample_transaction_df.copy())
    encoded, encoder_map = encode_categoricals(df)

    # card4 was object → should now be numeric
    assert pd.api.types.is_numeric_dtype(encoded["card4"])


def test_encode_categoricals_returns_encoder_map(sample_transaction_df):
    df = impute_missing(sample_transaction_df.copy())
    _, encoder_map = encode_categoricals(df)

    assert isinstance(encoder_map, dict)
    assert len(encoder_map) > 0
    for col, enc in encoder_map.items():
        assert isinstance(enc, LabelEncoder)


def test_encode_categoricals_no_object_dtypes_remaining(sample_transaction_df):
    df = impute_missing(sample_transaction_df.copy())
    encoded, _ = encode_categoricals(df)
    obj_cols = encoded.select_dtypes(include=["object"]).columns.tolist()
    assert len(obj_cols) == 0


# ── Test: split_features_target ───────────────────────────────────────────────


def test_split_features_target_separates_correctly(clean_numeric_df):
    X, y = split_features_target(clean_numeric_df, "isFraud")
    assert "isFraud" not in X.columns
    assert y.name == "isFraud"
    assert len(X) == len(y)


def test_split_features_target_raises_on_missing_target(clean_numeric_df):
    with pytest.raises(ValueError, match="Target column"):
        split_features_target(clean_numeric_df, "nonexistent_col")


def test_split_features_target_feature_count(clean_numeric_df):
    X, y = split_features_target(clean_numeric_df, "isFraud")
    assert X.shape[1] == clean_numeric_df.shape[1] - 1
