"""Shared utilities for churn model training.

This module centralizes reusable operations:
    * Data loading / splitting helpers
    * Column inference (numeric vs categorical with light numeric-like coercion)
    * Metric / curve utilities (ROC, PR, threshold tuning for F1)
    * Feature importance extraction (coefficients, impurity, permutation)
    * Generic persistence helpers (JSON / text reports)

All plotting uses a non‑interactive backend so the code is safe in headless
environments (CI, containers).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance


DATA_PATH_DEFAULT = "./data/processed/telco_clean.csv"
REPORTS_DIR = "./reports/models"


def ensure_dir(path: str) -> None:
    """Create directory (and parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    """Load processed dataset and sanity‑check the presence of the target."""
    df = pd.read_csv(path)
    if "Churn" not in df.columns:
        raise ValueError("Processed CSV must contain 'Churn' column as target")
    return df


def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix X and integer target y."""
    y = df["Churn"].astype(int)
    X = df.drop(columns=["Churn"])  # keep engineered features
    return X, y


def infer_columns(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Infer numeric vs categorical columns with light numeric-like promotion.

    Object columns that parse to numeric (>=95% non‑NA when coerced) are
    reclassified as numeric to ensure appropriate scaling / encoding.
    """
    num_cols = X.select_dtypes(include=["int8", "int16", "int32", "int64", "float16", "float32", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cat_cols[:]:
        parsed = pd.to_numeric(X[c], errors="coerce")
        if parsed.notna().mean() >= 0.95:
            X[c] = parsed
            cat_cols.remove(c)
            num_cols.append(c)
    return num_cols, cat_cols


def pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Average precision (area under precision‑recall curve)."""
    return float(average_precision_score(y_true, y_prob))


def tune_threshold_f1(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """Return threshold maximizing F1 plus supporting precision / recall stats."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = (2 * prec * rec) / np.clip(prec + rec, 1e-12, None)
    thr = np.concatenate(([0.0], thr))  # align lengths with precision/recall arrays
    best_idx = int(np.nanargmax(f1))
    return float(thr[best_idx]), {
        "best_threshold": float(thr[best_idx]),
        "best_f1": float(f1[best_idx]),
        "precision_at_best": float(prec[best_idx]),
        "recall_at_best": float(rec[best_idx]),
    }


def plot_curves(y_true: np.ndarray, y_prob: np.ndarray, out_dir: str) -> None:
    """Persist ROC and PR curve PNG files to ``out_dir``."""
    ensure_dir(out_dir)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.plot(fpr, tpr, label="ROC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc.png"), dpi=150)
    plt.close()

    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    baseline = (y_true == 1).mean()
    plt.figure(figsize=(5, 4))
    plt.plot([0, 1], [baseline, baseline], linestyle="--", color="grey", label="Baseline")
    plt.plot(rec, prec, label="PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pr.png"), dpi=150)
    plt.close()


def get_feature_names(pre: ColumnTransformer) -> List[str]:
    """Best‑effort extraction of transformed feature names from a ColumnTransformer."""
    try:
        return pre.get_feature_names_out().tolist()
    except Exception:
        names: List[str] = []
        for name, trans, cols in pre.transformers_:
            if name == "remainder" and trans == "drop":
                continue
            if trans == "passthrough":
                names.extend(cols)  # type: ignore[arg-type]
            else:
                try:
                    fn = trans.get_feature_names_out(cols).tolist()  # type: ignore
                except Exception:
                    if isinstance(cols, (list, tuple)):
                        fn = [f"{name}__{c}" for c in cols]
                    else:
                        fn = [f"{name}__{cols}"]
                names.extend(fn)
        return names


def compute_and_save_feature_importance(pipe: Pipeline, Xv: pd.DataFrame, yv: pd.Series, out_dir: str) -> None:
    """Compute coefficients / impurity / permutation importances and save CSV."""
    ensure_dir(out_dir)
    pre: ColumnTransformer = pipe.named_steps["pre"]  # type: ignore
    names = get_feature_names(pre)
    est = pipe.named_steps["est"]  # type: ignore

    rows: List[Dict[str, Any]] = []
    if hasattr(est, "coef_"):
        for n, w in zip(names, est.coef_.ravel()):  # type: ignore
            rows.append({"feature": n, "type": "coefficient", "value": float(w)})
    if hasattr(est, "feature_importances_"):
        for n, w in zip(names, est.feature_importances_):  # type: ignore
            rows.append({"feature": n, "type": "impurity_importance", "value": float(w)})
    try:
        perm = permutation_importance(pipe, Xv, yv, scoring="average_precision", n_repeats=8, random_state=42)
        for n, w in zip(names, perm.importances_mean):
            rows.append({"feature": n, "type": "permutation_AP", "value": float(w)})
    except Exception:
        pass

    if rows:
        df_imp = pd.DataFrame(rows)
        df_imp.sort_values(by=["type", "value"], ascending=[True, False], inplace=True)
        df_imp.to_csv(os.path.join(out_dir, "feature_importance.csv"), index=False)


def save_text_report(path: str, header: str, report: str, cm: np.ndarray | None = None) -> None:
    """Write a plain‑text report (optionally with confusion matrix)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(header.rstrip() + "\n\n")
        f.write(report)
        if cm is not None:
            f.write("\nConfusion matrix:\n")
            f.write(np.array2string(cm))


def save_json(path: str, data: Dict[str, Any]) -> None:
    """Persist a JSON dictionary with stable key ordering."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def evaluate_pipeline(pipe: Pipeline, Xv: pd.DataFrame, yv: pd.Series) -> Dict[str, Any]:
    """Evaluate fitted pipeline on validation/test set returning metrics + probs."""
    y_prob = pipe.predict_proba(Xv)[:, 1]
    roc = roc_auc_score(yv, y_prob)
    ap = pr_auc(yv.values, y_prob)
    thr, thr_info = tune_threshold_f1(yv.values, y_prob)
    y_pred = (y_prob >= thr).astype(int)
    rep = classification_report(yv, y_pred, digits=3)
    cm = confusion_matrix(yv, y_pred)
    return {
        "roc_auc": roc,
        "pr_auc": ap,
        "threshold": thr,
        "threshold_info": thr_info,
        "classification_report": rep,
        "confusion_matrix": cm.tolist(),
        "y_prob": y_prob.tolist(),
    }


__all__ = [
    "DATA_PATH_DEFAULT",
    "REPORTS_DIR",
    "ensure_dir",
    "load_data",
    "split_xy",
    "infer_columns",
    "evaluate_pipeline",
    "plot_curves",
    "compute_and_save_feature_importance",
    "save_text_report",
    "save_json",
]
