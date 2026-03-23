"""
Evidently AI drift report: compares training data distribution
against incoming prediction requests (reference vs current).

Usage:
    python monitoring/drift_report.py \
        --reference data/processed/train.parquet \
        --current   data/processed/predictions_log.parquet \
        --output    monitoring/reports/drift_report.html
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from evidently import ColumnMapping
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)
from evidently.report import Report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def load_reference(params: dict) -> pd.DataFrame:
    """Load processed training data as the reference distribution."""
    path = params["data"]["processed_train"]
    logger.info("Loading reference data from %s", path)
    df = pd.read_parquet(path)
    logger.info("Reference shape: %s", df.shape)
    return df


def load_current(current_path: str) -> pd.DataFrame:
    """Load current (prediction-time) data."""
    logger.info("Loading current data from %s", current_path)
    if current_path.endswith(".parquet"):
        df = pd.read_parquet(current_path)
    elif current_path.endswith(".csv"):
        df = pd.read_csv(current_path)
    else:
        raise ValueError(f"Unsupported file format: {current_path}")
    logger.info("Current shape: %s", df.shape)
    return df


_PRIORITY_COLS = [
    # Key transaction features
    "TransactionAmt",
    "TransactionDT",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "dist1",
    "dist2",
    "P_emaildomain",
    "R_emaildomain",
    # Count / timedelta features
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C13",
    "C14",
    "D1",
    "D2",
    "D3",
    "D4",
    "D10",
    "D15",
    # Top Vesta features by typical importance
    "V258",
    "V257",
    "V201",
    "V169",
    "V87",
    "V82",
    "V83",
    "V53",
    "V54",
    "V75",
    "V76",
    "V61",
    "V62",
    # Engineered
    "tx_amt_log",
    "tx_hour",
    "tx_dayofweek",
    "tx_is_weekend",
    "email_domain_match",
    "card1_freq",
    # Target
    "isFraud",
]


def _select_common_columns(
    ref: pd.DataFrame,
    cur: pd.DataFrame,
    max_cols: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keep only columns present in both datasets, limited to max_cols.
    Priority columns are selected first so the report focuses on the most
    informative features rather than timing out on 433 columns.
    """
    common_all = set(ref.columns) & set(cur.columns)
    # Select priority cols first, then fill up to max_cols with remaining
    priority = [c for c in _PRIORITY_COLS if c in common_all]
    remaining = [c for c in sorted(common_all) if c not in priority]
    selected = (priority + remaining)[:max_cols]
    logger.info(
        "Columns for drift analysis: %d selected from %d common (max_cols=%d)",
        len(selected),
        len(common_all),
        max_cols,
    )
    return ref[selected], cur[selected]


def _get_column_mapping(
    df: pd.DataFrame,
    target_col: str,
    prediction_col: str | None = None,
) -> ColumnMapping:
    """Build Evidently column mapping from the dataframe schema."""
    str_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove target / prediction from feature lists
    for col in [target_col, prediction_col]:
        if col in str_cols:
            str_cols.remove(col)
        if col in num_cols:
            num_cols.remove(col)

    mapping = ColumnMapping(
        target=target_col if target_col in df.columns else None,
        prediction=prediction_col
        if prediction_col and prediction_col in df.columns
        else None,
        numerical_features=num_cols,
        categorical_features=str_cols,
    )
    return mapping


def generate_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    target_col: str,
    prediction_col: str | None,
    output_path: str,
) -> None:
    """Generate Evidently HTML drift report."""
    reference, current = _select_common_columns(reference, current)

    column_mapping = _get_column_mapping(reference, target_col, prediction_col)

    report = Report(
        metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ]
    )

    logger.info("Running Evidently report …")
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    report.save_html(output_path)
    logger.info("Drift report saved to %s", output_path)


def generate_target_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    target_col: str,
    prediction_col: str | None,
    output_path: str,
) -> None:
    """Generate target drift report (if labels available in current)."""
    if target_col not in current.columns:
        logger.warning(
            "Target column '%s' not in current data — skipping target drift", target_col
        )
        return

    reference, current = _select_common_columns(reference, current)
    column_mapping = _get_column_mapping(reference, target_col, prediction_col)

    report = Report(metrics=[TargetDriftPreset()])
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping,
    )
    report.save_html(output_path)
    logger.info("Target drift report saved to %s", output_path)


def simulate_current_from_reference(
    reference: pd.DataFrame, n: int = 5000, seed: int = 42
) -> pd.DataFrame:
    """
    If no real prediction log exists, create a synthetic current dataset
    by sampling + adding small noise — useful for testing the pipeline.
    """
    rng = np.random.default_rng(seed)
    sample = reference.sample(n=min(n, len(reference)), random_state=seed).copy()
    num_cols = sample.select_dtypes(include=[np.number]).columns

    # Add gaussian noise to numeric cols
    for col in num_cols:
        std = sample[col].std()
        if std > 0:
            sample[col] = sample[col] + rng.normal(0, std * 0.05, size=len(sample))

    logger.info("Created synthetic current dataset (n=%d) from reference", len(sample))
    return sample


def run(
    params_path: str = "params.yaml",
    current_path: str | None = None,
    output_path: str = "monitoring/reports/drift_report.html",
    prediction_col: str | None = "fraud_probability",
) -> None:
    params = load_params(params_path)
    target_col = params["data"]["target_column"]

    reference = load_reference(params)

    # Sample reference to 10k rows for Evidently performance (590k is very slow)
    if len(reference) > 10_000:
        reference = reference.sample(n=10_000, random_state=42)
        logger.info("Sampled reference to 10k rows for report performance")

    if current_path and Path(current_path).exists():
        current = load_current(current_path)
    else:
        logger.warning(
            "No current data path provided or file not found — using synthetic sample"
        )
        current = simulate_current_from_reference(reference)

    generate_drift_report(reference, current, target_col, prediction_col, output_path)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_output = output_path.replace(".html", f"_target_{ts}.html")
    generate_target_drift_report(
        reference, current, target_col, prediction_col, target_output
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Evidently drift report")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument(
        "--reference", default=None, help="Override reference data path"
    )
    parser.add_argument(
        "--current", default=None, help="Path to current/prediction-log data"
    )
    parser.add_argument("--output", default="monitoring/reports/drift_report.html")
    parser.add_argument("--prediction-col", default="fraud_probability")
    args = parser.parse_args()

    run(
        params_path=args.params,
        current_path=args.current,
        output_path=args.output,
        prediction_col=args.prediction_col,
    )
