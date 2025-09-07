"""Train a CatBoost churn model emphasizing PR‑AUC and calibrated threshold.

Artifacts saved under reports/models/cb:
    * baseline_val_metrics / final_test_metrics (JSON + text)
    * feature_importance.csv (PredictionValuesChange)
    * model.cbm
"""

import argparse
import os
import json
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score

from model_common import (
    DATA_PATH_DEFAULT,
    REPORTS_DIR,
    load_data,
    split_xy,
    ensure_dir,
    tune_threshold_f1,  # type: ignore
    pr_auc,             # type: ignore
)

try:
    from catboost import CatBoostClassifier
except ImportError as e:
    raise SystemExit("CatBoost not installed. Install with: pip install catboost") from e


def identify_cat_features(df: pd.DataFrame) -> List[str]:
    """Return list of multi-category columns retained as categorical."""
    candidates = [
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"
    ]
    return [c for c in candidates if c in df.columns]


def evaluate(model: CatBoostClassifier, Xv: pd.DataFrame, yv: pd.Series) -> Dict[str, Any]:
    """Compute metrics + threshold tuning for a fitted CatBoost model."""
    y_prob = model.predict_proba(Xv)[:, 1]
    roc = roc_auc_score(yv, y_prob)
    ap = average_precision_score(yv, y_prob)
    thr, thr_info = tune_threshold_f1(yv.values, y_prob)  # type: ignore
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


def save_json(path: str, data: Dict[str, Any]):
    """Persist JSON with indentation."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_text_report(path: str, header: str, report: str, cm: List[List[int]] | None = None):
    """Write plain-text classification report and optional confusion matrix."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(header.rstrip() + "\n\n")
        f.write(report)
        if cm is not None:
            f.write("\nConfusion matrix:\n")
            f.write(np.array2string(np.array(cm)))


def feature_importance(model: CatBoostClassifier, feature_names: List[str]) -> pd.DataFrame:
    """Return sorted DataFrame of feature importance values."""
    try:
        imp_vals = model.get_feature_importance(type="PredictionValuesChange")
    except Exception:
        imp_vals = model.get_feature_importance()
    df_imp = pd.DataFrame({"feature": feature_names, "importance": imp_vals})
    df_imp.sort_values("importance", ascending=False, inplace=True)
    return df_imp


def main():
    """Entrypoint: split, train with early stopping, refit, evaluate, persist."""
    parser = argparse.ArgumentParser(description="Train CatBoost churn model")
    parser.add_argument("--data", default=DATA_PATH_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--l2_leaf_reg", type=float, default=3.0)
    parser.add_argument("--early_stopping_rounds", type=int, default=100)
    args = parser.parse_args()

    df = load_data(args.data)
    X_all, y_all = split_xy(df)

    # 60/20/20 split stratified
    X_temp, Xt, y_temp, yt = train_test_split(
        X_all, y_all, test_size=0.2, random_state=args.seed, stratify=y_all
    )
    X, Xv, y, yv = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=args.seed, stratify=y_temp
    )

    cat_features = identify_cat_features(X)
    feature_order = X.columns.tolist()  # preserve order for importance mapping

    # CatBoost expects categorical feature indices
    cat_idx = [feature_order.index(c) for c in cat_features]

    # Class weights: inverse frequency
    pos_ratio = (y == 1).mean()
    class_weights = [1.0, (1 - pos_ratio) / pos_ratio] if pos_ratio > 0 else [1.0, 1.0]

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="PRAUC",  # optimize for PR-AUC
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        random_seed=args.seed,
        class_weights=class_weights,
        verbose=False,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    model.fit(
        X,
        y,
        eval_set=(Xv, yv),
        cat_features=cat_idx,
        use_best_model=True,
    )

    out_dir = os.path.join(REPORTS_DIR, "cb")
    ensure_dir(out_dir)

    # Validation metrics
    val_metrics = evaluate(model, Xv, yv)
    save_json(os.path.join(out_dir, "baseline_val_metrics.json"), val_metrics)
    save_text_report(
        os.path.join(out_dir, "baseline_val_report.txt"),
        header=(
            f"CatBoost Validation Metrics\nROC-AUC: {val_metrics['roc_auc']:.6f}\n"
            f"PR-AUC: {val_metrics['pr_auc']:.6f}\nBest threshold: {val_metrics['threshold']:.4f}"
        ),
        report=val_metrics["classification_report"],
        cm=val_metrics["confusion_matrix"],
    )

    # Final fit (train+val) with early stopping on held-out val? Simpler: refit with best params using full train+val, no eval set.
    Xtv = pd.concat([X, Xv], axis=0)
    ytv = pd.concat([y, yv], axis=0)
    final_model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="PRAUC",
        iterations=model.tree_count_,  # use best iteration found
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        random_seed=args.seed,
        class_weights=class_weights,
        verbose=False,
    )
    final_model.fit(Xtv, ytv, cat_features=cat_idx)

    test_metrics = evaluate(final_model, Xt, yt)
    save_json(os.path.join(out_dir, "final_test_metrics.json"), test_metrics)
    save_text_report(
        os.path.join(out_dir, "final_test_report.txt"),
        header=(
            f"CatBoost FINAL Test Metrics\nROC-AUC: {test_metrics['roc_auc']:.6f}\n"
            f"PR-AUC: {test_metrics['pr_auc']:.6f}\nChosen threshold: {test_metrics['threshold']:.4f}"
        ),
        report=test_metrics["classification_report"],
        cm=test_metrics["confusion_matrix"],
    )

    # Feature importance
    df_imp = feature_importance(final_model, feature_order)
    df_imp.to_csv(os.path.join(out_dir, "feature_importance.csv"), index=False)

    # Save model
    final_model.save_model(os.path.join(out_dir, "model.cbm"))
    print(f"Saved CatBoost model + artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
