"""Generate a comparison PDF for churn models (LR / RF / SVM / CatBoost).

Inputs (per model directory under reports/models/<key>):
    * baseline_val_metrics.json
    * final_test_metrics.json

Outputs: reports/model_comparison.pdf with summary text, validation + test
heatmaps, confusion matrices, and bar charts for ROC/PR/F1 (test).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


MODEL_KEYS = ["lr", "rf", "svm", "cb"]


def load_json(path: str) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def derive_basic_metrics(cm: List[List[int]]) -> Dict[str, float]:
    if not cm or len(cm) != 2 or len(cm[0]) != 2 or len(cm[1]) != 2:
        return {k: float("nan") for k in ["accuracy", "precision", "recall", "f1"]}
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    total = tn + fp + fn + tp
    accuracy = (tn + tp) / total if total else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else float("nan")
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def collect_metrics(reports_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict]]:
    val_rows = []
    test_rows = []
    raw: Dict[str, Dict] = {}
    for key in MODEL_KEYS:
        model_dir = os.path.join(reports_dir, key)
        val_path = os.path.join(model_dir, "baseline_val_metrics.json")
        test_path = os.path.join(model_dir, "final_test_metrics.json")
        val = load_json(val_path)
        test = load_json(test_path)
        raw[key] = {"val": val, "test": test}

        if val:
            cm_val = val.get("confusion_matrix")
            derived_val = derive_basic_metrics(cm_val)
            val_rows.append({
                "model": key,
                "roc_auc": val.get("roc_auc", float("nan")),
                "pr_auc": val.get("pr_auc", float("nan")),
                "threshold": val.get("threshold", float("nan")),
                **derived_val,
            })
        if test:
            cm_test = test.get("confusion_matrix")
            derived_test = derive_basic_metrics(cm_test)
            test_rows.append({
                "model": key,
                "roc_auc": test.get("roc_auc", float("nan")),
                "pr_auc": test.get("pr_auc", float("nan")),
                "threshold": test.get("threshold", float("nan")),
                **derived_test,
            })

    val_df = pd.DataFrame(val_rows).set_index("model") if val_rows else pd.DataFrame()
    test_df = pd.DataFrame(test_rows).set_index("model") if test_rows else pd.DataFrame()
    return val_df, test_df, raw


def plot_heatmap(ax, data: pd.DataFrame, title: str, cmap: str = "Blues") -> None:
    if data.empty:
        ax.set_axis_off()
        ax.set_title(f"{title}\n<no data>")
        return
    metrics = data.columns.tolist()
    models = data.index.tolist()
    mat = data.to_numpy()
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    # annotations
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isnan(val):
                txt = "-"
            else:
                txt = f"{val:.3f}" if val <= 0.9995 else f"{val:.3f}"  # keep 3 decimals
            ax.text(j, i, txt, ha="center", va="center", color="black", fontsize=8)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Score", rotation=270, labelpad=12)


def plot_confusion(ax, cm: List[List[int]], title: str) -> None:
    if not cm:
        ax.set_axis_off()
        ax.set_title(f"{title}\n<missing>")
        return
    arr = np.array(cm)
    im = ax.imshow(arr, cmap="Purples")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, str(arr[i, j]), ha="center", va="center", color="black", fontsize=10)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_metric_bars(ax, test_df: pd.DataFrame, metrics: List[str], title: str) -> None:
    if test_df.empty:
        ax.set_axis_off()
        ax.set_title(f"{title}\n<no data>")
        return
    x = np.arange(len(test_df.index))
    width = 0.8 / len(metrics)
    for idx, m in enumerate(metrics):
        vals = test_df[m].values
        ax.bar(x + idx * width, vals, width=width, label=m)
        for xi, v in zip(x + idx * width, vals):
            if not np.isnan(v):
                ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(x + width * (len(metrics)-1) / 2)
    ax.set_xticklabels(test_df.index.tolist())
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(fontsize=7)


def add_text_page(pdf: PdfPages, title: str, lines: List[str]):
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape size
    fig.suptitle(title, fontsize=16, y=0.95)
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.88])
    ax.axis("off")
    y = 0.95
    for ln in lines:
        ax.text(0.01, y, ln, ha="left", va="top", family="monospace", fontsize=10)
        y -= 0.035
        if y < 0.02:
            break
    pdf.savefig(fig)
    plt.close(fig)


def build_summary_lines(val_df: pd.DataFrame, test_df: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    def fmt_row(df: pd.DataFrame, model: str) -> str:
        r = df.loc[model]
        return (
            f"{model.upper()}: Acc={r['accuracy']:.3f} Pre={r['precision']:.3f} Rec={r['recall']:.3f} "
            f"F1={r['f1']:.3f} ROC={r['roc_auc']:.3f} PR={r['pr_auc']:.3f} Thr={r['threshold']:.3f}"
        )
    if not val_df.empty:
        lines.append("Validation Metrics:")
        for m in val_df.index:
            lines.append("  " + fmt_row(val_df, m))
        lines.append("")
    if not test_df.empty:
        lines.append("Test Metrics:")
        for m in test_df.index:
            lines.append("  " + fmt_row(test_df, m))
    return lines


def main():
    parser = argparse.ArgumentParser(description="Compare churn models and generate PDF with visualizations")
    parser.add_argument("--reports_dir", default="./reports/models")
    parser.add_argument("--out_pdf", default="./reports/model_comparison.pdf")
    args = parser.parse_args()

    val_df, test_df, raw = collect_metrics(args.reports_dir)

    # Order columns for heatmaps
    desired_cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    val_hm = val_df.reindex(columns=desired_cols)
    test_hm = test_df.reindex(columns=desired_cols)

    # PDF generation
    os.makedirs(os.path.dirname(args.out_pdf), exist_ok=True)
    with PdfPages(args.out_pdf) as pdf:
        # Page 1: summary text
        lines = build_summary_lines(val_df, test_df)
        add_text_page(pdf, "Model Performance Summary", lines)

        # Page 2: validation heatmap
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        plot_heatmap(ax, val_hm, "Validation Metrics Heatmap")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: test heatmap
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        plot_heatmap(ax, test_hm, "Test Metrics Heatmap")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Confusion matrices (test)
        for key in MODEL_KEYS:
            cm = raw.get(key, {}).get("test", {}).get("confusion_matrix")
            fig, ax = plt.subplots(figsize=(5, 4))
            plot_confusion(ax, cm, f"{key.upper()} Test Confusion Matrix")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # Bar charts for selected test metrics
        if not test_df.empty:
            fig, ax = plt.subplots(figsize=(11.69, 5))
            plot_metric_bars(ax, test_df, ["roc_auc", "pr_auc", "f1"], "Test Metrics Comparison (ROC/PR/F1)")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Model comparison PDF saved to: {args.out_pdf}")


if __name__ == "__main__":
    main()
