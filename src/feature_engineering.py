"""
Feature engineering for IEEE-CIS Fraud Detection.
Generates time features, email domain features, and card interaction features.
All logic is controlled by params.yaml flags.
"""

import logging

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Reference start date for TransactionDT (seconds since this date)
_REFERENCE_DATE = pd.Timestamp("2017-12-01")

# Top email domains by fraud prevalence (used for domain grouping)
_TOP_DOMAINS = {
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "outlook.com",
    "live.com",
    "icloud.com",
    "anonymous.com",
    "protonmail.com",
}


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Decompose TransactionDT (seconds offset) into calendar features."""
    if "TransactionDT" not in df.columns:
        logger.warning("TransactionDT not found, skipping time features")
        return df

    logger.info("Adding time-based features from TransactionDT")
    dt = _REFERENCE_DATE + pd.to_timedelta(df["TransactionDT"], unit="s")
    df["tx_hour"] = dt.dt.hour
    df["tx_dayofweek"] = dt.dt.dayofweek
    df["tx_day"] = dt.dt.day
    df["tx_month"] = dt.dt.month
    df["tx_is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    df["tx_is_night"] = ((dt.dt.hour >= 22) | (dt.dt.hour < 6)).astype(int)
    return df


def add_email_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and encode email domain features from P_emaildomain / R_emaildomain."""
    for col in ["P_emaildomain", "R_emaildomain"]:
        if col not in df.columns:
            continue
        prefix = "p_email" if col.startswith("P") else "r_email"
        # Is domain a known top domain?
        df[f"{prefix}_is_top"] = df[col].apply(
            lambda x: 1 if str(x).lower() in _TOP_DOMAINS else 0
        )
        # Is domain anonymous?
        df[f"{prefix}_is_anonymous"] = (
            df[col].astype(str).str.lower().str.contains("anonymous").astype(int)
        )
        # Do both purchaser and recipient share the same domain?
    if "P_emaildomain" in df.columns and "R_emaildomain" in df.columns:
        df["email_domain_match"] = (
            df["P_emaildomain"].astype(str) == df["R_emaildomain"].astype(str)
        ).astype(int)
    logger.info("Added email domain features")
    return df


def add_card_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create card interaction features."""
    logger.info("Adding card interaction features")
    if "card1" in df.columns and "card2" in df.columns:
        df["card1_card2_ratio"] = df["card1"] / (df["card2"].replace(0, np.nan))
        df["card1_card2_sum"] = df["card1"] + df["card2"]

    if "TransactionAmt" in df.columns:
        if "card1" in df.columns:
            df["amt_card1_ratio"] = df["TransactionAmt"] / (df["card1"].replace(0, np.nan))
        if "card5" in df.columns:
            df["amt_card5_ratio"] = df["TransactionAmt"] / (df["card5"].replace(0, np.nan))

    # Fill NaNs introduced by division
    new_cols = [c for c in df.columns if c in [
        "card1_card2_ratio", "card1_card2_sum", "amt_card1_ratio", "amt_card5_ratio"
    ]]
    df[new_cols] = df[new_cols].fillna(0)
    return df


def add_transaction_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Log-transform and bucket TransactionAmt."""
    if "TransactionAmt" not in df.columns:
        return df
    logger.info("Adding transaction amount features")
    df["tx_amt_log"] = np.log1p(df["TransactionAmt"])
    df["tx_amt_cents"] = (df["TransactionAmt"] % 1 * 100).round(0)
    df["tx_amt_is_round"] = (df["TransactionAmt"] % 1 == 0).astype(int)
    return df


def add_count_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate count features: how often each card/addr combo appears."""
    logger.info("Adding count-based aggregation features")
    for col in ["card1", "addr1", "addr2"]:
        if col in df.columns:
            freq = df[col].map(df[col].value_counts())
            df[f"{col}_freq"] = freq.fillna(0)
    return df


def run_feature_engineering(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Apply all enabled feature engineering steps."""
    fe_cfg = params.get("feature_engineering", {})

    if fe_cfg.get("time_features", True):
        df = add_time_features(df)

    df = add_transaction_amount_features(df)
    df = add_count_features(df)

    if fe_cfg.get("email_domain_features", True):
        df = add_email_domain_features(df)

    if fe_cfg.get("card_features", True):
        df = add_card_features(df)

    # Fill any new NaNs from feature engineering
    numeric_new = df.select_dtypes(include=[np.number]).columns
    df[numeric_new] = df[numeric_new].fillna(0)

    logger.info("Feature engineering complete. Total columns: %d", df.shape[1])
    return df


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)
