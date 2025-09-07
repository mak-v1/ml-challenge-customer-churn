"""FastAPI inference service for churn model.

Loads a serialized sklearn pipeline (or compatible object with predict_proba),
reconstructs engineered features applied during training, and serves a
probability + thresholded label. Threshold priority: env THRESHOLD > metrics
file > default 0.5.
"""

import os
import json
from typing import List, Dict, Any
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

# Default to logistic regression artifacts directory
DEFAULT_MODEL_DIR = os.getenv("MODEL_DIR", "./reports/models/lr")
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "model.joblib")
METRICS_PATH = os.path.join(DEFAULT_MODEL_DIR, "final_test_metrics.json")

# Threshold: env overrides metrics; fallback to 0.5
def load_threshold() -> float:
    """Resolve decision threshold from env > metrics JSON > default 0.5."""
    if "THRESHOLD" in os.environ:
        try:
            return float(os.environ["THRESHOLD"])
        except ValueError:
            pass
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return float(data.get("threshold") or data.get("threshold_info", {}).get("best_threshold") or 0.5)
    except Exception:
        return 0.5

THRESHOLD = load_threshold()
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)

# Raw-like fields required (subset of original raw; engineer rest inside)
RAW_SCHEMA: List[str] = [
    "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PaperlessBilling", "PaymentMethod", "Contract",
    "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "MonthlyCharges", "TotalCharges"
]

app = FastAPI(title="Churn Inference API", version="1.0.0")


def preprocess_payload(record: Dict[str, Any]) -> pd.DataFrame:
    """Validate and transform raw-like JSON record into model feature frame."""
    missing = [c for c in RAW_SCHEMA if c not in record]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing fields: {missing}")

    df = pd.DataFrame([record]).copy()
    # Coerce numerics
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").astype("Int64")
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").astype("float32")

    if df["tenure"].isna().any() or (df["tenure"] <= 0).any():
        raise HTTPException(status_code=400, detail="tenure must be >= 1")

    # Engineered features (align with training pipeline expectations)
    df["avg_monthly_spend"] = (df["TotalCharges"] / df["tenure"]).astype("float32")
    df["streaming_count"] = (
        (df["StreamingTV"].astype(str) == "Yes").astype("int8") +
        (df["StreamingMovies"].astype(str) == "Yes").astype("int8")
    ).astype("int8")
    df["security_support_count"] = (
        (df["OnlineSecurity"].astype(str) == "Yes").astype("int8") +
        (df["OnlineBackup"].astype(str) == "Yes").astype("int8") +
        (df["DeviceProtection"].astype(str) == "Yes").astype("int8") +
        (df["TechSupport"].astype(str) == "Yes").astype("int8")
    ).astype("int8")

    # Drop TotalCharges as in training
    df = df.drop(columns=["TotalCharges"], errors="ignore")
    df["tenure"] = df["tenure"].astype("int32")
    return df


# Load model once
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found at {MODEL_PATH}. Train model first.")
pipe = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "model_path": MODEL_PATH, "threshold": THRESHOLD}


@app.post("/predict")
def predict(record: Dict[str, Any]):
    X = preprocess_payload(record)
    try:
        proba = float(pipe.predict_proba(X)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {e}")
    label = int(proba >= THRESHOLD)
    return {
        "churn_probability": proba,
        "churn_label": label,
        "threshold": THRESHOLD,
        "model_version": os.path.basename(MODEL_PATH)
    }
