"""Train a Random Forest churn model and persist standardized artifacts."""

import argparse
import os
from joblib import dump

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from model_common import (
    DATA_PATH_DEFAULT,
    REPORTS_DIR,
    ensure_dir,
    load_data,
    split_xy,
    infer_columns,
    evaluate_pipeline,
    plot_curves,
    compute_and_save_feature_importance,
    save_text_report,
    save_json,
)


def build_pipeline(num_cols, cat_cols, seed: int) -> Pipeline:
    """Return RF pipeline with passthrough numerics + full one‑hot encoding."""
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )
    est = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced",
        random_state=seed,
    )
    return Pipeline([("pre", pre), ("est", est)])


def main():
    """Entrypoint: split, train, evaluate (val + test), save artifacts."""
    parser = argparse.ArgumentParser(description="Train Random Forest churn model")
    parser.add_argument("--data", default=DATA_PATH_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_data(args.data)
    X_all, y_all = split_xy(df)

    X_temp, Xt, y_temp, yt = train_test_split(
        X_all, y_all, test_size=0.2, random_state=args.seed, stratify=y_all
    )
    X, Xv, y, yv = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=args.seed, stratify=y_temp
    )

    num_cols, cat_cols = infer_columns(X)
    pipe = build_pipeline(num_cols, cat_cols, args.seed)
    pipe.fit(X, y)
    out_dir = os.path.join(REPORTS_DIR, "rf")
    ensure_dir(out_dir)

    val_metrics = evaluate_pipeline(pipe, Xv, yv)
    plot_curves(yv.values, np.array(val_metrics["y_prob"]), out_dir)
    compute_and_save_feature_importance(pipe, Xv, yv, out_dir)
    save_json(os.path.join(out_dir, "baseline_val_metrics.json"), val_metrics)
    save_text_report(
        os.path.join(out_dir, "baseline_val_report.txt"),
        header=(
            f"Random Forest Validation Metrics\nROC-AUC: {val_metrics['roc_auc']:.6f}\n"
            f"PR-AUC: {val_metrics['pr_auc']:.6f}\nBest threshold: {val_metrics['threshold']:.4f}"
        ),
        report=val_metrics["classification_report"],
    )

    # Final fit on train+val
    Xtv = pd.concat([X, Xv], axis=0)
    ytv = pd.concat([y, yv], axis=0)
    pipe.fit(Xtv, ytv)
    test_metrics = evaluate_pipeline(pipe, Xt, yt)
    plot_curves(yt.values, np.array(test_metrics["y_prob"]), out_dir)
    save_json(os.path.join(out_dir, "final_test_metrics.json"), test_metrics)
    save_text_report(
        os.path.join(out_dir, "final_test_report.txt"),
        header=(
            f"Random Forest FINAL Test Metrics\nROC-AUC: {test_metrics['roc_auc']:.6f}\n"
            f"PR-AUC: {test_metrics['pr_auc']:.6f}\nChosen threshold: {test_metrics['threshold']:.4f}"
        ),
        report=test_metrics["classification_report"],
    )

    dump(pipe, os.path.join(out_dir, "model.joblib"))
    print(f"Saved RF model + artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
