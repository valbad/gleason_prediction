"""
Build final summary tables for communication and project tracking.

Reads analysis outputs from reports/ and produces cleaned, presentation-ready
CSV and Markdown tables.  All input files are optional — a missing file
produces a warning and a placeholder section rather than a crash.

Inputs
------
  reports/shareable_tabular_results_binary_label_int.csv
  reports/shareable_tabular_results_binary_label_gg3plus_int.csv
  reports/risk_stratification.csv
  reports/univariate_geometry_effects.csv
  reports/patient_level_performance.csv
  reports/core_counts_by_patient_audit.csv

Outputs
-------
  reports/final_main_performance_table.csv / .md
  reports/final_univariate_effects_table.csv / .md
  reports/final_risk_stratification_table.csv / .md
  reports/final_patient_level_table.csv / .md
  reports/final_core_count_audit_summary.csv / .md

Usage
-----
    python src/build_final_summary_tables.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT   = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"

LABEL_COLS = ["binary_label_int", "binary_label_gg3plus_int"]
LABEL_SHORT = {
    "binary_label_int":         "GG2+ / csPCa",
    "binary_label_gg3plus_int": "GG3+ / high-grade",
}

TARGET_EXPERIMENTS = [
    "target_geometry_no_availability",
    "all_geometry_no_availability_plus_clinical",
]

UNIVARIATE_FEATURES = [
    "distance_midpoint_to_target_surface_mm",
    "distance_midpoint_to_target_centroid_mm",
    "trajectory_intersects_target",
    "approximate_fraction_of_centerline_inside_target",
    "psa_density",
    "log_psa_ng_ml",
]


# ── Shared utilities ──────────────────────────────────────────────────────────

def _f(v, dec: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{dec}f}"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return str(v)


def _pct(v, dec: int = 1) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{float(v) * 100:.{dec}f}%"


def _pval(p: float) -> str:
    if np.isnan(p):
        return "n/a"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def _ci(val, lo, hi, dec: int = 3) -> str:
    """Format 'value [lo – hi]', falling back gracefully on NaN."""
    v_str  = _f(val, dec)
    lo_str = _f(lo,  dec)
    hi_str = _f(hi,  dec)
    if lo_str == "n/a" or hi_str == "n/a":
        return v_str
    return f"{v_str} [{lo_str} – {hi_str}]"


def _sign(v: float) -> str:
    if np.isnan(v):
        return "—"
    return "▲" if v >= 0 else "▼"


def _load(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        warnings.warn(f"[{label}] not found: {path.name} — section will be empty.")
        return None
    df = pd.read_csv(path)
    if df.empty:
        warnings.warn(f"[{label}] file is empty: {path.name}")
        return None
    return df


def _save(df: pd.DataFrame, path: Path, label: str) -> None:
    df.to_csv(path, index=False)
    print(f"  Saved {label} -> {path.name}")


def _write_md(lines: list[str], path: Path, label: str) -> None:
    path.write_text("\n".join(lines) + "\n")
    print(f"  Saved {label} -> {path.name}")


def _md_table(df: pd.DataFrame, col_map: dict[str, str]) -> list[str]:
    """Render a DataFrame as a markdown table using col_map = {df_col: header}."""
    headers = list(col_map.values())
    cols    = list(col_map.keys())
    lines   = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for _, r in df.iterrows():
        cells = [str(r[c]) if c in r.index else "n/a" for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _missing_section(title: str, filename: str) -> list[str]:
    return [
        f"## {title}",
        "",
        f"_Source file `{filename}` not found. Run the corresponding analysis script first._",
        "",
    ]


# ── 1. Main performance table ─────────────────────────────────────────────────

def build_main_performance(out_csv: Path, out_md: Path) -> None:
    print("\n[1] Main performance table")

    frames = []
    for lc in LABEL_COLS:
        path = REPORTS_DIR / f"shareable_tabular_results_{lc}.csv"
        df   = _load(path, lc)
        if df is not None:
            # Normalise label column name (the script stores it as 'label_column')
            if "label_column" in df.columns:
                df = df.rename(columns={"label_column": "label_col"})
            elif "label_col" not in df.columns:
                df["label_col"] = lc
            frames.append(df)

    if not frames:
        _write_md(_missing_section("Main Model Performance", "shareable_tabular_results_*.csv"),
                  out_md, "main_performance.md (empty)")
        return

    raw = pd.concat(frames, ignore_index=True)

    # Filter to full_geometry_dataset mode and target experiments
    raw = raw[
        (raw["mode"] == "full_geometry_dataset")
        & raw["experiment"].isin(TARGET_EXPERIMENTS)
    ].copy()

    if raw.empty:
        warnings.warn("No matching rows for target experiments in full_geometry_dataset.")
        _write_md(_missing_section("Main Model Performance", "—"), out_md, "main_performance.md (empty)")
        return

    # Best model per (label_col, experiment)
    rows_out = []
    for (lc, exp), grp in raw.groupby(["label_col", "experiment"]):
        best = grp.loc[grp["test_roc_auc"].idxmax()]

        # CI columns (bootstrap; absent when --no-bootstrap was used)
        roc_ci = _ci(best.get("test_roc_auc"),
                     best.get("roc_auc_ci_low"),
                     best.get("roc_auc_ci_high"))
        pr_ci  = _ci(best.get("test_pr_auc"),
                     best.get("pr_auc_ci_low"),
                     best.get("pr_auc_ci_high"))

        prev = best.get("test_prevalence", np.nan)
        pr_auc = best.get("test_pr_auc", np.nan)
        pr_lift = pr_auc / prev if (not np.isnan(pr_auc) and not np.isnan(prev) and prev > 0) else np.nan

        rows_out.append({
            "label_col":          lc,
            "endpoint":           LABEL_SHORT.get(lc, lc),
            "test_prevalence":    _pct(prev),
            "experiment":         exp,
            "model":              best.get("model", "n/a"),
            "roc_auc_ci":         roc_ci,
            "pr_auc_ci":          pr_ci,
            "pr_lift":            _f(pr_lift, 2),
            "sensitivity":        _f(best.get("test_sensitivity")),
            "specificity":        _f(best.get("test_specificity")),
            "precision":          _f(best.get("test_precision")),
            "f1":                 _f(best.get("test_f1")),
            # Raw values for CSV
            "_test_roc_auc":      best.get("test_roc_auc", np.nan),
            "_test_pr_auc":       best.get("test_pr_auc", np.nan),
            "_test_prevalence":   prev,
            "_pr_lift_raw":       pr_lift,
        })

    display_cols = [
        ("endpoint",        "Endpoint"),
        ("test_prevalence", "Prevalence"),
        ("experiment",      "Experiment"),
        ("model",           "Best model"),
        ("roc_auc_ci",      "ROC-AUC [95% CI]"),
        ("pr_auc_ci",       "PR-AUC [95% CI]"),
        ("pr_lift",         "PR-lift"),
        ("sensitivity",     "Sensitivity"),
        ("specificity",     "Specificity"),
        ("precision",       "Precision"),
        ("f1",              "F1"),
    ]
    col_map = dict(display_cols)

    df_out = pd.DataFrame(rows_out)
    # Sort: GG2+ first, then by experiment order
    exp_order = {e: i for i, e in enumerate(TARGET_EXPERIMENTS)}
    lc_order  = {lc: i for i, lc in enumerate(LABEL_COLS)}
    df_out["_exp_ord"] = df_out["experiment"].map(exp_order)
    df_out["_lc_ord"]  = df_out["label_col"].map(lc_order)
    df_out = df_out.sort_values(["_exp_ord", "_lc_ord"]).drop(columns=["_exp_ord", "_lc_ord"])

    _save(df_out, out_csv, "main_performance.csv")

    md = [
        "# Main Model Performance",
        "",
        "Mode: `full_geometry_dataset`. Best model selected by test ROC-AUC within each "
        "(endpoint, experiment) group.  ",
        "95% CI from patient-level bootstrap (omitted if experiments were run with `--no-bootstrap`).",
        "",
        *_md_table(df_out, col_map),
        "",
        "> **PR-lift** = PR-AUC / test prevalence. Values > 1 indicate the model beats "
        "the no-skill baseline.",
        "",
    ]
    _write_md(md, out_md, "main_performance.md")


# ── 2. Univariate effects table ───────────────────────────────────────────────

def build_univariate_effects(out_csv: Path, out_md: Path) -> None:
    print("\n[2] Univariate effects table")

    df = _load(REPORTS_DIR / "univariate_geometry_effects.csv", "univariate")
    if df is None:
        _write_md(_missing_section("Univariate Feature Effects",
                                   "univariate_geometry_effects.csv"), out_md, "univariate.md (empty)")
        return

    df = df[df["feature"].isin(UNIVARIATE_FEATURES)].copy()

    rows_out = []
    for lc in LABEL_COLS:
        sub = df[df["label_col"] == lc]
        for feat in UNIVARIATE_FEATURES:
            r = sub[sub["feature"] == feat]
            if r.empty:
                continue
            r = r.iloc[0]
            prev = r.get("test_prevalence", np.nan)
            pr   = r.get("test_pr_auc", np.nan)
            lift = pr / prev if (not np.isnan(pr) and not np.isnan(prev) and prev > 0) else np.nan
            rows_out.append({
                "label_col":    lc,
                "endpoint":     LABEL_SHORT.get(lc, lc),
                "feature":      feat,
                "is_boolean":   bool(r.get("is_boolean", False)),
                "lr_coef":      _f(r.get("lr_coef"), 4),
                "direction":    _sign(r.get("lr_coef", np.nan)),
                "roc_auc":      _f(r.get("test_roc_auc")),
                "pr_auc":       _f(r.get("test_pr_auc")),
                "pr_lift":      _f(lift, 2),
                "spearman_rho": _f(r.get("spearman_r"), 3),
                "spearman_p":   _pval(r.get("spearman_p", np.nan)),
                # Raw for CSV
                "_lr_coef_raw":  r.get("lr_coef", np.nan),
                "_roc_auc_raw":  r.get("test_roc_auc", np.nan),
            })

    col_map = {
        "endpoint":     "Endpoint",
        "feature":      "Feature",
        "is_boolean":   "Boolean?",
        "direction":    "Dir",
        "lr_coef":      "LR coef (std)",
        "roc_auc":      "ROC-AUC",
        "pr_auc":       "PR-AUC",
        "pr_lift":      "PR-lift",
        "spearman_rho": "Spearman ρ",
        "spearman_p":   "p-value",
    }

    df_out = pd.DataFrame(rows_out)
    _save(df_out, out_csv, "univariate_effects.csv")

    md_sections = [
        "# Univariate Feature Effects",
        "",
        "Univariate logistic regression (LR) fit on the train split, evaluated on the test split.  ",
        "**LR coef** is standardised for numeric features (1 unit = 1 σ) and natural 0→1 for booleans.  ",
        "**Dir** ▲ = higher feature value → higher predicted probability of label = 1.  ",
        "**Spearman ρ** computed on all filtered rows (train + val + test).",
        "",
    ]
    for lc in LABEL_COLS:
        sub = df_out[df_out["label_col"] == lc]
        # Sort by |lr_coef_raw| descending
        sub = sub.assign(_abs=sub["_lr_coef_raw"].abs()).sort_values("_abs", ascending=False).drop(columns="_abs")
        md_sections += [
            f"## {LABEL_SHORT.get(lc, lc)} (`{lc}`)",
            "",
            *_md_table(sub, col_map),
            "",
        ]

    _write_md(md_sections, out_md, "univariate_effects.md")


# ── 3. Risk stratification table ─────────────────────────────────────────────

def build_risk_stratification(out_csv: Path, out_md: Path) -> None:
    print("\n[3] Risk stratification table")

    df = _load(REPORTS_DIR / "risk_stratification.csv", "risk_stratification")
    if df is None:
        _write_md(_missing_section("Risk Stratification",
                                   "risk_stratification.csv"), out_md, "risk_strat.md (empty)")
        return

    # Baseline prevalence per (label_col, model) = n_test_positives / n_test
    base = df[["label_col", "model", "test_roc_auc", "test_pr_auc",
               "n_test", "n_test_positives"]].drop_duplicates(subset=["label_col", "model"])
    base = base.copy()
    base["baseline_prevalence"] = base["n_test_positives"] / base["n_test"]

    rows_out = []
    for lc in LABEL_COLS:
        lc_base = base[base["label_col"] == lc]
        if lc_base.empty:
            continue
        # Best model by test_roc_auc
        best_model = lc_base.loc[lc_base["test_roc_auc"].idxmax(), "model"]
        brow = lc_base[lc_base["model"] == best_model].iloc[0]
        prev = brow["baseline_prevalence"]

        cap_df = df[(df["label_col"] == lc) & (df["model"] == best_model)
                    & (df["table"] == "capture")]
        dec_df = df[(df["label_col"] == lc) & (df["model"] == best_model)
                    & (df["table"] == "decile")]

        def _cap(pct):
            r = cap_df[np.isclose(cap_df["top_pct"].astype(float), pct, atol=1e-6)]
            return float(r["capture_rate"].iloc[0]) if not r.empty else np.nan

        hi_dec = dec_df[dec_df["decile"] == 10]
        hi_prev = float(hi_dec["observed_prevalence"].iloc[0]) if not hi_dec.empty else np.nan

        rows_out.append({
            "label_col":          lc,
            "endpoint":           LABEL_SHORT.get(lc, lc),
            "model":              best_model,
            "test_roc_auc":       _f(brow["test_roc_auc"]),
            "test_pr_auc":        _f(brow["test_pr_auc"]),
            "baseline_prevalence": _pct(prev),
            "decile10_prevalence": _pct(hi_prev),
            "capture_top5pct":    _pct(_cap(0.05)),
            "capture_top10pct":   _pct(_cap(0.10)),
            "capture_top20pct":   _pct(_cap(0.20)),
        })

    col_map = {
        "endpoint":            "Endpoint",
        "model":               "Model",
        "test_roc_auc":        "ROC-AUC",
        "test_pr_auc":         "PR-AUC",
        "baseline_prevalence": "Baseline prevalence",
        "decile10_prevalence": "Decile 10 prevalence",
        "capture_top5pct":     "Capture top 5%",
        "capture_top10pct":    "Capture top 10%",
        "capture_top20pct":    "Capture top 20%",
    }

    df_out = pd.DataFrame(rows_out)
    _save(df_out, out_csv, "risk_stratification.csv")

    md = [
        "# Risk Stratification Summary",
        "",
        "Best model per endpoint (by test ROC-AUC) from `analyze_risk_stratification.py`.  ",
        "Feature set: `all_geometry_no_availability_plus_clinical` (target_mesh_available excluded).  ",
        "**Decile 10** = highest-risk 10% of test cores.  ",
        "**Capture top N%** = fraction of all positive cores found in the top N% by predicted risk.",
        "",
        *_md_table(df_out, col_map),
        "",
    ]
    _write_md(md, out_md, "risk_stratification.md")


# ── 4. Patient-level table ────────────────────────────────────────────────────

def build_patient_level(out_csv: Path, out_md: Path) -> None:
    print("\n[4] Patient-level table")

    df = _load(REPORTS_DIR / "patient_level_performance.csv", "patient_level")
    if df is None:
        _write_md(_missing_section("Patient-level Performance",
                                   "patient_level_performance.csv"), out_md, "patient_level.md (empty)")
        return

    metrics = df[df["table"] == "metrics"].copy()
    if metrics.empty:
        warnings.warn("No 'metrics' rows found in patient_level_performance.csv")
        return

    AGG_LABEL = {
        "max_prob":             "Max prob",
        "top3_mean_prob":       "Top-3 mean",
        "mean_prob":            "Mean prob",
        "prop_above_threshold": "Prop > thr",
    }

    rows_out = []
    for (lc, model), grp in metrics.groupby(["label_col", "model"]):
        # Best aggregation by roc_auc
        best_idx = grp["roc_auc"].idxmax()
        best_agg = grp.loc[best_idx]

        rows_out.append({
            "label_col":          lc,
            "endpoint":           LABEL_SHORT.get(lc, lc),
            "model":              model,
            "best_aggregation":   AGG_LABEL.get(best_agg["aggregation"], best_agg["aggregation"]),
            "patient_prevalence": _pct(best_agg.get("patient_prevalence")),
            "roc_auc":            _f(best_agg.get("roc_auc")),
            "pr_auc":             _f(best_agg.get("pr_auc")),
            "sensitivity":        _f(best_agg.get("sensitivity")),
            "specificity":        _f(best_agg.get("specificity")),
            "f1":                 _f(best_agg.get("f1")),
            "capture_top10pct":   _pct(best_agg.get("capture_top10pct")),
            "capture_top20pct":   _pct(best_agg.get("capture_top20pct")),
        })

    col_map = {
        "endpoint":           "Endpoint",
        "model":              "Model",
        "best_aggregation":   "Best aggregation",
        "patient_prevalence": "Prevalence",
        "roc_auc":            "ROC-AUC",
        "pr_auc":             "PR-AUC",
        "sensitivity":        "Sensitivity",
        "specificity":        "Specificity",
        "f1":                 "F1",
        "capture_top10pct":   "Capture @10%",
        "capture_top20pct":   "Capture @20%",
    }

    df_out = pd.DataFrame(rows_out)
    _save(df_out, out_csv, "patient_level.csv")

    # Also show all aggregation methods per (endpoint, model) in collapsible detail
    md = [
        "# Patient-level Performance",
        "",
        "Core-level predicted probabilities aggregated to patient level.  ",
        "A patient is **positive** if at least one core is positive for the endpoint.  ",
        "**Best aggregation** selected by patient-level ROC-AUC on the test split.",
        "",
        "## Best aggregation per endpoint and model",
        "",
        *_md_table(df_out, col_map),
        "",
        "## All aggregation methods",
        "",
    ]

    all_agg_cols = {
        "endpoint":           "Endpoint",
        "model":              "Model",
        "best_aggregation":   "Aggregation",
        "patient_prevalence": "Prevalence",
        "roc_auc":            "ROC-AUC",
        "pr_auc":             "PR-AUC",
        "sensitivity":        "Sensitivity",
        "specificity":        "Specificity",
        "f1":                 "F1",
        "capture_top10pct":   "Capture @10%",
        "capture_top20pct":   "Capture @20%",
    }

    for (lc, model), grp in metrics.groupby(["label_col", "model"]):
        grp = grp.sort_values("roc_auc", ascending=False).copy()
        grp["endpoint"]     = LABEL_SHORT.get(lc, lc)
        grp["best_aggregation"] = grp["aggregation"].map(
            lambda a: AGG_LABEL.get(a, a))

        def _fmt_row(row):
            return {
                "endpoint":           LABEL_SHORT.get(lc, lc),
                "model":              model,
                "best_aggregation":   AGG_LABEL.get(row["aggregation"], row["aggregation"]),
                "patient_prevalence": _pct(row.get("patient_prevalence")),
                "roc_auc":            _f(row.get("roc_auc")),
                "pr_auc":             _f(row.get("pr_auc")),
                "sensitivity":        _f(row.get("sensitivity")),
                "specificity":        _f(row.get("specificity")),
                "f1":                 _f(row.get("f1")),
                "capture_top10pct":   _pct(row.get("capture_top10pct")),
                "capture_top20pct":   _pct(row.get("capture_top20pct")),
            }

        formatted = pd.DataFrame([_fmt_row(r) for _, r in grp.iterrows()])
        md += [
            f"### {LABEL_SHORT.get(lc, lc)} — {model}",
            "",
            *_md_table(formatted, all_agg_cols),
            "",
        ]

    _write_md(md, out_md, "patient_level.md")


# ── 5. Core-count audit summary ───────────────────────────────────────────────

def build_core_count_summary(out_csv: Path, out_md: Path) -> None:
    print("\n[5] Core-count audit summary")

    df = _load(REPORTS_DIR / "core_counts_by_patient_audit.csv", "core_count_audit")
    if df is None:
        _write_md(_missing_section("Core Count Audit Summary",
                                   "core_counts_by_patient_audit.csv"), out_md, "core_count.md (empty)")
        return

    rows_out = []
    for lc in LABEL_COLS:

        # Overall distribution (split == "all")
        dist = df[(df["section"] == "distribution") & (df["label_col"] == lc)
                  & (df["split"] == "all")]
        overall_median = float(dist["median"].iloc[0]) if not dist.empty else np.nan
        overall_mean   = float(dist["mean"].iloc[0])   if not dist.empty else np.nan

        # Positive vs negative (split == "all")
        pn = df[(df["section"] == "pos_vs_neg") & (df["label_col"] == lc)
                & (df["split"] == "all")]
        pos_row = pn[pn["group"] == "positive"]
        neg_row = pn[pn["group"] == "negative"]
        mean_pos = float(pos_row["mean_n_cores"].iloc[0]) if not pos_row.empty else np.nan
        mean_neg = float(neg_row["mean_n_cores"].iloc[0]) if not neg_row.empty else np.nan

        # Spearman: n_cores vs patient_label
        corr = df[(df["section"] == "correlation") & (df["label_col"] == lc)
                  & (df["target"] == "patient_label")]
        rho = float(corr["spearman_r"].iloc[0]) if not corr.empty else np.nan
        pv  = float(corr["spearman_p"].iloc[0]) if not corr.empty else np.nan

        # Interpretation
        if not np.isnan(rho) and abs(rho) >= 0.10 and not np.isnan(pv) and pv < 0.05:
            interp = "Weak but significant: positive patients tend to have slightly more cores"
        elif not np.isnan(rho) and not np.isnan(pv) and pv < 0.05:
            interp = "Very weak but significant association"
        else:
            interp = "No significant association"

        rows_out.append({
            "label_col":          lc,
            "endpoint":           LABEL_SHORT.get(lc, lc),
            "overall_median_n_cores":   _f(overall_median, 1),
            "overall_mean_n_cores":     _f(overall_mean, 1),
            "mean_n_cores_positive_pat": _f(mean_pos, 1),
            "mean_n_cores_negative_pat": _f(mean_neg, 1),
            "spearman_rho_vs_label":    _f(rho, 3),
            "spearman_p":               _pval(pv),
            "interpretation":           interp,
        })

    col_map = {
        "endpoint":                   "Endpoint",
        "overall_median_n_cores":     "Median n_cores",
        "overall_mean_n_cores":       "Mean n_cores",
        "mean_n_cores_positive_pat":  "Mean (positive pat.)",
        "mean_n_cores_negative_pat":  "Mean (negative pat.)",
        "spearman_rho_vs_label":      "Spearman ρ (vs label)",
        "spearman_p":                 "p-value",
        "interpretation":             "Interpretation",
    }

    df_out = pd.DataFrame(rows_out)
    _save(df_out, out_csv, "core_count_summary.csv")

    md = [
        "# Core Count Audit Summary",
        "",
        "Per-patient biopsy core counts, stratified by endpoint and label group.  ",
        "**Spearman ρ** measures the association between number of cores and the patient label  ",
        "(positive = at least one core positive). A significant positive correlation would  ",
        "mean positive patients systematically receive more cores, which could mildly  ",
        "advantage core-count-sensitive aggregation methods (max-prob, top-3 mean).",
        "",
        *_md_table(df_out, col_map),
        "",
    ]
    _write_md(md, out_md, "core_count_summary.md")


# ── 6. Calibration table ─────────────────────────────────────────────────────

def _calib_interp(
    bss_before: float,
    bss_after: float,
    ece_before: float,
    ece_after: float,
    low_bin_frac_after: float,
) -> str:
    """Build a short interpretation string from calibration metrics."""
    parts: list[str] = []

    # ECE improvement
    ece_delta = ece_before - ece_after
    if not np.isnan(ece_delta) and ece_delta > 0.01:
        parts.append("Recalibration improves calibration.")

    # Brier skill score sign flip
    if (not np.isnan(bss_before) and not np.isnan(bss_after)
            and bss_before < 0 <= bss_after):
        parts.append(
            "Raw probabilities poor; recalibrated probabilities usable with caution."
        )

    # Low-bin concentration (only when detectable from the bins data)
    if not np.isnan(low_bin_frac_after) and low_bin_frac_after > 0.90:
        parts.append(
            "After recalibration, predictions concentrate in the lowest probability "
            "bin; absolute risk estimates remain conservative."
        )

    # Mandatory caution
    parts.append(
        "Discrimination / ranking should be interpreted separately from "
        "absolute risk calibration."
    )

    return " ".join(parts)


def build_calibration_table(out_csv: Path, out_md: Path) -> None:
    print("\n[6] Calibration table")

    df = _load(REPORTS_DIR / "calibration_analysis.csv", "calibration")
    if df is None:
        _write_md(
            _missing_section("Calibration Analysis", "calibration_analysis.csv"),
            out_md, "calibration.md (empty)",
        )
        return

    metrics = df[df["table"] == "metrics"].copy()
    if metrics.empty:
        warnings.warn("No 'metrics' rows in calibration_analysis.csv")
        return

    # Split before/after and index on (label_col, model) for easy pairing
    before = metrics[metrics["calibration"] == "before"].set_index(["label_col", "model"])
    after  = metrics[metrics["calibration"] == "after"].set_index(["label_col", "model"])

    # Low-bin concentration: fraction of test samples in bin_index == 1 after recalibration
    bins = df[(df["table"] == "bins") & (df["calibration"] == "after")].copy()
    low_bin_fracs: dict[tuple, float] = {}
    if not bins.empty and "bin_index" in bins.columns:
        for (lc, model), grp in bins.groupby(["label_col", "model"]):
            total = grp["n_cores"].sum()
            bin1  = grp[grp["bin_index"] == 1]["n_cores"].sum()
            low_bin_fracs[(lc, model)] = float(bin1 / total) if total > 0 else np.nan

    rows_out: list[dict] = []
    for (lc, model) in before.index:
        b = before.loc[(lc, model)]
        a = after.loc[(lc, model)] if (lc, model) in after.index else pd.Series(dtype=float)

        def _get(src, col):
            try:
                v = src[col]
                return float(v) if not (isinstance(v, float) and np.isnan(v)) else np.nan
            except (KeyError, TypeError):
                return np.nan

        brier_b = _get(b, "brier_score");         brier_a = _get(a, "brier_score")
        bss_b   = _get(b, "brier_skill_score");   bss_a   = _get(a, "brier_skill_score")
        ece_b   = _get(b, "ece");                 ece_a   = _get(a, "ece")
        slope_b = _get(b, "cal_slope");           slope_a = _get(a, "cal_slope")
        int_b   = _get(b, "cal_intercept");       int_a   = _get(a, "cal_intercept")

        low_frac = low_bin_fracs.get((lc, model), np.nan)
        interp   = _calib_interp(bss_b, bss_a, ece_b, ece_a, low_frac)

        rows_out.append({
            "label_col":                lc,
            "endpoint":                 LABEL_SHORT.get(lc, lc),
            "model":                    model,
            "brier_before":             _f(brier_b),
            "brier_after":              _f(brier_a),
            "brier_skill_score_before": _f(bss_b),
            "brier_skill_score_after":  _f(bss_a),
            "ece_before":               _f(ece_b),
            "ece_after":                _f(ece_a),
            "cal_slope_before":         _f(slope_b),
            "cal_slope_after":          _f(slope_a),
            "cal_intercept_before":     _f(int_b),
            "cal_intercept_after":      _f(int_a),
            "interpretation":           interp,
        })

    col_map = {
        "endpoint":                   "Endpoint",
        "model":                      "Model",
        "brier_before":               "Brier (before)",
        "brier_after":                "Brier (after)",
        "brier_skill_score_before":   "BSS (before)",
        "brier_skill_score_after":    "BSS (after)",
        "ece_before":                 "ECE (before)",
        "ece_after":                  "ECE (after)",
        "cal_slope_before":           "Slope (before)",
        "cal_slope_after":            "Slope (after)",
        "cal_intercept_before":       "Intercept (before)",
        "cal_intercept_after":        "Intercept (after)",
        "interpretation":             "Interpretation",
    }

    df_out = pd.DataFrame(rows_out)
    _save(df_out, out_csv, "calibration.csv")

    md = [
        "# Calibration Analysis Summary",
        "",
        "Calibration metrics before and after logistic recalibration (Platt scaling  ",
        "on val-split logit(predicted probabilities)).  ",
        "**BSS** = Brier skill score (1 − Brier / Brier_null); higher is better; < 0 = worse than naive.  ",
        "**ECE** = expected calibration error (10 equal-width bins); lower is better.  ",
        "**Slope** b ≈ 1 → well-spread; b < 1 → over-confident; b > 1 → under-confident.",
        "",
        *_md_table(df_out, col_map),
        "",
    ]
    _write_md(md, out_md, "calibration.md")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== Building final summary tables ===")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    build_main_performance(
        REPORTS_DIR / "final_main_performance_table.csv",
        REPORTS_DIR / "final_main_performance_table.md",
    )
    build_univariate_effects(
        REPORTS_DIR / "final_univariate_effects_table.csv",
        REPORTS_DIR / "final_univariate_effects_table.md",
    )
    build_risk_stratification(
        REPORTS_DIR / "final_risk_stratification_table.csv",
        REPORTS_DIR / "final_risk_stratification_table.md",
    )
    build_patient_level(
        REPORTS_DIR / "final_patient_level_table.csv",
        REPORTS_DIR / "final_patient_level_table.md",
    )
    build_core_count_summary(
        REPORTS_DIR / "final_core_count_audit_summary.csv",
        REPORTS_DIR / "final_core_count_audit_summary.md",
    )
    build_calibration_table(
        REPORTS_DIR / "final_calibration_table.csv",
        REPORTS_DIR / "final_calibration_table.md",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
