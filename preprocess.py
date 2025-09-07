"""Data preprocessing pipeline for churn dataset.

Steps:
    * Coerce TotalCharges and drop tenure==0 rows
    * Engineer spend / service count features
    * Binarize select Yes/No columns
    * Enforce concise dtypes (numeric + category)
    * Drop raw / low-signal columns and de-duplicate
    * Persist cleaned dataset to processed path
"""

import os
from typing import List, Tuple
import pandas as pd
import numpy as np


RAW_PATH_DEFAULT = "./data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUT_PATH_DEFAULT = "./data/processed/telco_clean.csv"


def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def drop_columns_if_exist(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    present = [c for c in cols if c in df.columns]
    return df.drop(columns=present)


def coerce_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    if "TotalCharges" in df.columns:
        df = df.copy()
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    return df


def drop_not_yet_billable(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Drop rows where tenure == 0 (the known 11 'new' customers).
    Handles the case even if additional bad rows exist after coercion.
    """
    if "tenure" not in df.columns:
        return df, 0
    mask = (df["tenure"] == 0)
    dropped = int(mask.sum())
    return df.loc[~mask].copy(), dropped


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # avg_monthly_spend (safe because tenure==0 rows are already dropped)
    if {"TotalCharges", "tenure"}.issubset(df.columns):
        df["avg_monthly_spend"] = (df["TotalCharges"] / df["tenure"]).astype(np.float32)

    # streaming_count: from StreamingTV, StreamingMovies
    for col in ["StreamingTV", "StreamingMovies"]:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")
    df["streaming_count"] = (
        (df["StreamingTV"].astype(str) == "Yes").astype(np.int8)
        + (df["StreamingMovies"].astype(str) == "Yes").astype(np.int8)
    ).astype(np.int8)

    # security_support_count: Yes-count over OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport
    sec_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
    for col in sec_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")
    df["security_support_count"] = (
        (df["OnlineSecurity"].astype(str) == "Yes").astype(np.int8)
        + (df["OnlineBackup"].astype(str) == "Yes").astype(np.int8)
        + (df["DeviceProtection"].astype(str) == "Yes").astype(np.int8)
        + (df["TechSupport"].astype(str) == "Yes").astype(np.int8)
    ).astype(np.int8)

    return df


def binarize_yes_no(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    mapping = {"Yes": 1, "No": 0}
    for col in cols:
        if col in df.columns:
            df[col] = df[col].map(mapping).astype("Int8")
    return df


def enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Numerics
    if "tenure" in df.columns:
        df["tenure"] = df["tenure"].astype("int32")
    if "MonthlyCharges" in df.columns:
        df["MonthlyCharges"] = df["MonthlyCharges"].astype("float32")
    if "avg_monthly_spend" in df.columns:
        df["avg_monthly_spend"] = df["avg_monthly_spend"].astype("float32")

    # SeniorCitizen is already 0/1; make it compact int8
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype("int8")

    # Multi-category as category dtype (models will one-hot later)
    multi_cat = [
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"
    ]
    for col in multi_cat:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop raw TotalCharges after using it
    df = drop_columns_if_exist(df, ["TotalCharges"])

    # Drop agreed low-signal / identifier columns
    df = drop_columns_if_exist(df, ["customerID", "gender", "PhoneService", "MultipleLines"])

    # De-duplicate full rows across remaining features
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    deduped = before - len(df)

    # Sanity: no NaNs should remain
    if df.isna().sum().sum() != 0:
        raise ValueError("NaNs remain after preprocessing; please investigate.")

    return df


def preprocess(path_in: str = RAW_PATH_DEFAULT, path_out: str = OUT_PATH_DEFAULT) -> None:
    ensure_dir(path_out)

    df_raw = load_raw(path_in)
    n_raw = len(df_raw)

    # Coerce and drop new customers (tenure==0)
    df = coerce_total_charges(df_raw)
    df, dropped_new = drop_not_yet_billable(df)

    # Engineer features
    df = add_engineered_features(df)

    # Binary mappings
    df = binarize_yes_no(df, ["Churn", "Partner", "Dependents", "PaperlessBilling"])

    # Enforce dtypes
    df = enforce_dtypes(df)

    # Finalize (drop raw cols, low-signal cols, dedup, validate)
    df_final = finalize(df)
    n_final = len(df_final)

    # Basic class balance info (post-drop)
    churn_col = "Churn" if "Churn" in df_final.columns else None
    churn_stats = ""
    if churn_col:
        vc = df_final[churn_col].value_counts(dropna=False)
        total = vc.sum()
        pct = (vc / total * 100).round(2)
        churn_stats = f"Churn distribution (0/1): {vc.to_dict()} | %: {pct.to_dict()}"

    # Persist
    df_final.to_csv(path_out, index=False)

    # Summary
    print("=== Preprocessing Summary ===")
    print(f"Input rows: {n_raw}")
    print(f"Dropped tenure==0 rows: {dropped_new}")
    print(f"Output rows: {n_final}")
    print(f"Columns: {list(df_final.columns)}")
    if churn_stats:
        print(churn_stats)
    # Quick assertion aligned with our assumptions
    assert "avg_monthly_spend" in df_final.columns, "avg_monthly_spend missing"
    assert all(c in df_final.columns for c in ["streaming_count", "security_support_count"]), "engineered counts missing"
    assert "TotalCharges" not in df_final.columns, "TotalCharges should have been dropped"
    assert "customerID" not in df_final.columns, "customerID should have been dropped"
    assert "gender" not in df_final.columns, "gender should have been dropped"
    assert "PhoneService" not in df_final.columns, "PhoneService should have been dropped"
    assert "MultipleLines" not in df_final.columns, "MultipleLines should have been dropped"


if __name__ == "__main__":
    preprocess(path_in=RAW_PATH_DEFAULT, path_out=OUT_PATH_DEFAULT)
