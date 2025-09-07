"""Exploratory data analysis (EDA) PDF generator for the churn dataset.

Produces a multi-page PDF with schema overview, duplication analysis, missing
value variants, categorical distributions, churn rate by category, numeric
summaries, histograms, and edge-case diagnostics.
"""

import os
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        if p:
            os.makedirs(p, exist_ok=True)


def load_dataset(file_path: str) -> pd.DataFrame:
    # Audit raw; do not set na_values here
    return pd.read_csv(file_path)


def normalize_customer_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def duplicate_report(df: pd.DataFrame, id_col: str = "customerID") -> Dict:
    report: Dict = {}
    if id_col not in df.columns:
        raise ValueError(f"ID column '{id_col}' not found.")

    df_tmp = df.copy()
    df_tmp["_ID_NORM"] = normalize_customer_id(df_tmp[id_col])

    # Duplicate IDs
    dup_id_mask = df_tmp.duplicated(subset=["_ID_NORM"], keep=False)
    dup_id_df = df_tmp.loc[dup_id_mask].copy()

    exact, conflicting = [], []
    if not dup_id_df.empty:
        for norm_id, g in dup_id_df.groupby("_ID_NORM"):
            g_noid = g.drop(columns=[id_col, "_ID_NORM"])
            is_exact = (g_noid.nunique().sum() == 0)
            rows = g.index.tolist()
            ids = g[id_col].tolist()
            if is_exact:
                exact.append({"norm_id": norm_id, "rows": rows, "orig_ids": ids})
            else:
                conflicting.append({"norm_id": norm_id, "rows": rows, "orig_ids": ids})

    report["duplicate_id_groups_total"] = len(exact) + len(conflicting)
    report["duplicate_id_groups_exact"] = len(exact)
    report["duplicate_id_groups_conflicting"] = len(conflicting)
    report["df_dup_ids"] = dup_id_df.drop(columns=["_ID_NORM"]) if not dup_id_df.empty else pd.DataFrame()

    # Full-row duplicates INCLUDING ID (for awareness)
    full_dups_incl = df_tmp.duplicated(keep=False)
    report["full_row_dupe_count_including_id"] = int(full_dups_incl.sum())
    report["df_full_dupes_incl"] = df_tmp.loc[full_dups_incl].drop(columns=["_ID_NORM"]) if full_dups_incl.any() else pd.DataFrame()

    # Full-row duplicates EXCLUDING ID
    exclude_cols = [c for c in [id_col, "_ID_NORM"] if c in df_tmp.columns]
    subset_cols = [c for c in df_tmp.columns if c not in exclude_cols]
    full_dups_excl = df_tmp.duplicated(subset=subset_cols, keep=False)
    report["full_row_dupe_count_excluding_id"] = int(full_dups_excl.sum())
    report["df_full_dupes_excl"] = df_tmp.loc[full_dups_excl].drop(columns=["_ID_NORM"]) if full_dups_excl.any() else pd.DataFrame()

    return report


def robust_missing_counts(s: pd.Series) -> Dict[str, int]:
    na_count = int(s.isna().sum())
    s_str = s.astype(str)
    empty_str_count = int((s_str == "").sum())
    whitespace_only_count = int(((s_str.str.strip() == "") & (s_str != "")).sum())
    return {
        "na_count": na_count,
        "empty_str_count": empty_str_count,
        "whitespace_only_count": whitespace_only_count,
    }


def schema_overview(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        dtype = df[col].dtype
        non_null = int(df[col].notna().sum())
        unique = int(df[col].nunique(dropna=True))
        miss = robust_missing_counts(df[col])
        rows.append({
            "column": col,
            "dtype": str(dtype),
            "non_null": non_null,
            "unique": unique,
            "na_count": miss["na_count"],
            "empty_str_count": miss["empty_str_count"],
            "whitespace_only_count": miss["whitespace_only_count"],
        })
    return pd.DataFrame(rows).sort_values("column").reset_index(drop=True)


def split_categoricals_numerics(df: pd.DataFrame, id_col: str = "customerID") -> Tuple[List[str], List[str]]:
    cat_cols, num_cols = [], []
    for col in df.columns:
        if col == id_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            num_cols.append(col)
        else:
            cat_cols.append(col)
    return cat_cols, num_cols


def categorical_distributions(df: pd.DataFrame, cat_cols: List[str], top_k: int = 10) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for col in cat_cols:
        vc = df[col].value_counts(dropna=False)
        total = vc.sum()
        dist = pd.DataFrame({
            "value": vc.index.astype(str),
            "count": vc.values,
            "percent": (vc.values / total) * 100.0
        })
        if len(dist) > top_k:
            dist = dist.head(top_k)
        out[col] = dist
    return out


def numeric_summary(df: pd.DataFrame, num_cols: List[str]) -> pd.DataFrame:
    if not num_cols:
        return pd.DataFrame()
    desc = df[num_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    desc = desc.rename(columns={"50%": "median", "25%": "p25", "75%": "p75"})
    return desc.reset_index().rename(columns={"index": "column"})


def numeric_coercion_preview(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            coerced = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce")
            rows.append({
                "column": col,
                "dtype_now": str(df[col].dtype),
                "non_numeric_if_coerced": int(coerced.isna().sum()),
                "total_rows": int(len(df[col])),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("non_numeric_if_coerced", ascending=False).reset_index(drop=True)
    return out


def special_placeholder_counts(df: pd.DataFrame) -> pd.DataFrame:
    placeholders = {
        "No internet service": [
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"
        ],
        "No phone service": ["MultipleLines"],
    }
    rows = []
    for token, cols in placeholders.items():
        for col in cols:
            if col in df.columns:
                count = int((df[col].astype(str) == token).sum())
                rows.append({"column": col, "placeholder": token, "count": count})
    return pd.DataFrame(rows).sort_values(["placeholder", "column"]).reset_index(drop=True)


def edge_case_summary_lines(df: pd.DataFrame) -> List[str]:
    lines = []
    n = len(df)

    has_tenure = "tenure" in df.columns
    has_mc = "MonthlyCharges" in df.columns
    has_tc = "TotalCharges" in df.columns

    tenure0 = (df["tenure"] == 0) if has_tenure else pd.Series(False, index=df.index)
    mc = pd.to_numeric(df["MonthlyCharges"], errors="coerce") if has_mc else pd.Series(np.nan, index=df.index)
    mc_zero = (mc == 0) if has_mc else pd.Series(False, index=df.index)
    s_tc = df["TotalCharges"].astype(str) if has_tc else pd.Series("", index=df.index)
    tc_blank = s_tc.str.strip() == "" if has_tc else pd.Series(False, index=df.index)
    tc_missing_like = (df["TotalCharges"].isna() | tc_blank) if has_tc else pd.Series(False, index=df.index)

    c_tenure0 = int(tenure0.sum())
    c_mc0 = int(mc_zero.sum())
    c_tc_missing = int(tc_missing_like.sum())

    lines.append(f"Rows: {n}")
    lines.append(f"tenure == 0: {c_tenure0}")
    lines.append(f"MonthlyCharges == 0: {c_mc0}")
    lines.append(f"TotalCharges blank/empty: {c_tc_missing}")

    # Subset checks
    tc_blank_subset_tenure0 = (tc_missing_like & ~tenure0).sum() == 0
    tenure0_subset_tc_blank = (tenure0 & ~tc_missing_like).sum() == 0
    lines.append(f"All TotalCharges blank ⇒ tenure==0: {tc_blank_subset_tenure0}")
    lines.append(f"All tenure==0 ⇒ TotalCharges blank: {tenure0_subset_tc_blank}")

    if c_mc0 > 0:
        mc0_subset_tenure0 = (mc_zero & ~tenure0).sum() == 0
        tenure0_subset_mc0 = (tenure0 & ~mc_zero).sum() == 0
        lines.append(f"All MonthlyCharges==0 ⇒ tenure==0: {mc0_subset_tenure0}")
        lines.append(f"All tenure==0 ⇒ MonthlyCharges==0: {tenure0_subset_mc0}")

    return lines


def infer_churn_categorical_cols(
    df: pd.DataFrame,
    id_col: str = "customerID",
    target_col: str = "Churn",
    max_card: int = 50,
    numeric_like_threshold: float = 0.9,
) -> List[str]:
    """
    Auto-pick categorical columns for churn-rate bar charts:
      - exclude id & target
      - exclude numeric dtypes
      - exclude object columns that are 'numeric-like' (>= numeric_like_threshold parsable)
      - exclude very high-cardinality columns via max_card
    """
    cols: List[str] = []
    for col in df.columns:
        if col in (id_col, target_col):
            continue
        # native numeric → skip
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        # object/string that is actually numeric-like → skip
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            s = df[col].astype(str).str.strip()
            parsed = pd.to_numeric(s, errors="coerce")
            frac_numeric = parsed.notna().mean()
            if frac_numeric >= numeric_like_threshold:
                continue
        # cardinality guard
        if df[col].nunique(dropna=False) <= max_card:
            cols.append(col)
    return cols


def category_order_for_plot(col: str) -> List[str] | None:
    """Return preset ordering for known tri-level service flags if applicable."""
    if col == "MultipleLines":
        return ["Yes", "No", "No phone service"]
    tri = {"OnlineSecurity", "OnlineBackup", "DeviceProtection",
           "TechSupport", "StreamingTV", "StreamingMovies"}
    if col in tri:
        return ["Yes", "No", "No internet service"]
    return None


def churn_rate_by_category(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if "Churn" not in df.columns:
        return pd.DataFrame()
    g = df.groupby(col)["Churn"].value_counts(dropna=False).unstack(fill_value=0)
    if "Yes" not in g.columns and "No" not in g.columns:
        return pd.DataFrame()
    yes = g.get("Yes", pd.Series(0, index=g.index))
    no = g.get("No", pd.Series(0, index=g.index))
    total = yes + no
    rate = (yes / total.replace(0, np.nan)) * 100.0
    return pd.DataFrame({
        col: total.index.astype(str),
        "count": total.values,
        "churn_rate_percent": rate.values
    }).sort_values("churn_rate_percent", ascending=False).reset_index(drop=True)


def add_text_page(pdf: PdfPages, title: str, lines: List[str]) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
    fig.suptitle(title, fontsize=16, y=0.95)
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.85])
    ax.axis("off")
    y = 0.95
    for line in lines:
        ax.text(0.01, y, line, va="top", ha="left", fontsize=11, family="monospace")
        y -= 0.05
    pdf.savefig(fig)
    plt.close(fig)


def add_table_page(pdf: PdfPages, title: str, df: pd.DataFrame, max_rows: int = 25) -> None:
    if df.empty:
        add_text_page(pdf, title, ["<empty>"])
        return
    chunks = [df.iloc[i:i+max_rows] for i in range(0, len(df), max_rows)]
    for idx, chunk in enumerate(chunks, 1):
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.suptitle(f"{title} (page {idx}/{len(chunks)})", fontsize=16, y=0.97)
        ax = fig.add_axes([0.02, 0.05, 0.96, 0.86])
        ax.axis("off")
        tbl = ax.table(cellText=chunk.values,
                       colLabels=chunk.columns,
                       loc="center",
                       cellLoc="left")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.3)
        pdf.savefig(fig)
        plt.close(fig)


def plot_bar(pdf: PdfPages, title: str, data: pd.Series, xlabel: str, ylabel: str, rotate: int = 0):
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.bar(data.index.astype(str), data.values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if rotate:
        plt.setp(ax.get_xticklabels(), rotation=rotate, ha="right")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_hist(pdf: PdfPages, title: str, data: pd.Series, bins: int = 30, xlabel: str = "", ylabel: str = "Count"):
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.hist(data.dropna().values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel or data.name)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def run_eda(
    file_path: str = "./data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    id_col: str = "customerID",
    pdf_path: str = "./reports/EDA_Report.pdf",
    top_k_cat: int = 10
) -> None:
    ensure_dirs(os.path.dirname(pdf_path))

    df = load_dataset(file_path)
    dup = duplicate_report(df, id_col=id_col)
    schema = schema_overview(df)
    cat_cols, num_cols = split_categoricals_numerics(df, id_col=id_col)
    cat_dists = categorical_distributions(df, cat_cols, top_k=top_k_cat)
    num_desc = numeric_summary(df, num_cols)
    coercion_preview = numeric_coercion_preview(df)
    placeholders = special_placeholder_counts(df)
    edge_lines = edge_case_summary_lines(df)

    churn_summary = None
    if "Churn" in df.columns:
        churn_counts = df["Churn"].value_counts(dropna=False)
        churn_pct = (churn_counts / churn_counts.sum()) * 100.0
        churn_summary = pd.DataFrame({"Churn": churn_counts.index.astype(str),
                                      "count": churn_counts.values,
                                      "percent": churn_pct.values})

    # Auto-pick all categorical columns (excluding ID/Churn) for churn-rate plots
    key_cats = infer_churn_categorical_cols(df, id_col=id_col, target_col="Churn", max_card=50)
    churn_rate_tables = {c: churn_rate_by_category(df, c) for c in key_cats}

    # Prepare a short listing to show exactly what got selected
    selected_lines = ["Selected columns:"]
    if key_cats:
        for i in range(0, len(key_cats), 6):
            selected_lines.append(", ".join(key_cats[i:i+6]))
    else:
        selected_lines.append("<none>")

    with PdfPages(pdf_path) as pdf:
        # Summary
        lines = [
            f"Source file: {file_path}",
            f"Rows: {len(df):,}",
            f"Columns: {len(df.columns)}",
            "",
            "Duplicate Summary:",
            f"- Duplicate ID groups: total={dup['duplicate_id_groups_total']}, exact={dup['duplicate_id_groups_exact']}, conflicting={dup['duplicate_id_groups_conflicting']}",
            f"- Full-row duplicates (including ID): {dup['full_row_dupe_count_including_id']}",
            f"- Full-row duplicates (excluding ID): {dup['full_row_dupe_count_excluding_id']}",
            "",
            "Notes:",
            "- No data has been mutated.",
            "- Hidden missingness and numeric coercion preview provided to support decisions.",
            "- Edge-case summary captures tenure==0 / blank TotalCharges / zero MonthlyCharges relationships.",
        ]
        add_text_page(pdf, "EDA Summary", lines)

        # Schema + hidden missingness
        add_table_page(pdf, "Schema Overview (dtype, unique, missing variants)", schema, max_rows=24)
        missing_focus = schema[(schema["na_count"] > 0) |
                               (schema["empty_str_count"] > 0) |
                               (schema["whitespace_only_count"] > 0)]
        add_table_page(pdf, "Hidden Missingness Focus (NaN / '' / whitespace-only)", missing_focus)

        # Numeric coercion preview (object -> numeric)
        add_table_page(pdf, "Numeric Coercion Preview (object columns -> numeric)", coercion_preview, max_rows=30)

        # Edge-case summary (concise text)
        add_text_page(pdf, "Edge-case Summary (tenure / charges)", edge_lines)

        # Churn distribution
        if churn_summary is not None:
            plot_bar(pdf,
                     "Churn Distribution",
                     churn_summary.set_index("Churn")["percent"],
                     xlabel="Churn",
                     ylabel="Percent")
            add_table_page(pdf, "Churn Distribution (Counts & Percentages)", churn_summary)

        # Placeholder tokens
        if placeholders is not None and not placeholders.empty:
            add_table_page(pdf, "Special Placeholder Tokens (counts)", placeholders, max_rows=30)

        # Categorical distributions (top-k)
        for col, dist in cat_dists.items():
            plot_bar(pdf,
                     f"Categorical Distribution: {col} (Top {top_k_cat})",
                     dist.set_index("value")["count"],
                     xlabel=col,
                     ylabel="Count",
                     rotate=45)
            add_table_page(pdf, f"Top-{top_k_cat} Values: {col}", dist)

        # Show which columns were auto-selected for churn-rate plots
        add_text_page(pdf, "Churn-Rate Columns (auto-selected)", selected_lines)

        # Churn rate by category (auto-selected)
        for col, tbl in churn_rate_tables.items():
            if tbl is None or tbl.empty:
                continue
            order = category_order_for_plot(col)
            if order is not None:
                existing = [o for o in order if o in set(tbl[col].tolist())]
                if existing:
                    tbl = tbl.set_index(col).reindex(existing).reset_index()
            plot_bar(pdf,
                     f"Churn Rate by {col}",
                     tbl.set_index(col)["churn_rate_percent"],
                     xlabel=col,
                     ylabel="Churn Rate (%)",
                     rotate=45)
            add_table_page(pdf, f"Churn Rate by {col} (counts + %)", tbl)

        # Numeric distributions
        if num_cols:
            for col in num_cols:
                plot_hist(pdf, f"Histogram: {col}", df[col], bins=30, xlabel=col)
        if num_desc is not None and not num_desc.empty:
            add_table_page(pdf, "Numeric Summary (describe + percentiles)", num_desc)

    print(f"EDA PDF report saved to: {pdf_path}")


if __name__ == "__main__":
    run_eda(
        file_path="./data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        id_col="customerID",
        pdf_path="./reports/EDA_Report.pdf",
        top_k_cat=10
    )
