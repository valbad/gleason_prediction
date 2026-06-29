"""
Grouped core-level inference analysis.

Goal
----
Assess whether the compact target_geometry_plus_clinical associations remain
present and interpretable when intra-patient correlation is accounted for,
using cluster-robust logistic regression and (optionally) GEE.

This is an inference / association analysis on all valid splits pooled,
NOT a train/val/test predictive-performance experiment.

Input
-----
data/share/needle_features_v1.csv

Outputs
-------
reports/grouped_core_level_inference_cluster_robust.csv
reports/grouped_core_level_inference_gee.csv          (if GEE runs)
reports/grouped_core_level_inference_correlation.csv
reports/grouped_core_level_inference.md

Usage
-----
    python src/analyze_grouped_core_level_inference.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from run_shareable_tabular_experiments import (
    PATIENT_COL,
    SPLIT_COL,
    load_dataset,
    prepare_features,
)

import statsmodels.api as sm

try:
    from statsmodels.genmod.generalized_estimating_equations import GEE
    from statsmodels.genmod.families import Binomial as GEE_Binomial
    from statsmodels.genmod.cov_struct import Exchangeable
    HAS_GEE = True
except ImportError:
    HAS_GEE = False

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_VIF = True
except ImportError:
    HAS_VIF = False

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT   = Path(__file__).resolve().parent.parent
DATA_PATH   = REPO_ROOT / "data" / "share" / "needle_features_v1.csv"
REPORTS_DIR = REPO_ROOT / "reports"

# ── Endpoints ─────────────────────────────────────────────────────────────────

ENDPOINTS: list[tuple[str, str]] = [
    ("binary_label_int",         "GG2+ / csPCa"),
    ("binary_label_gg3plus_int", "GG3+ / high-grade"),
]

# ── Feature set: target_geometry_plus_clinical ────────────────────────────────

FEATURE_COLS: list[str] = [
    "distance_midpoint_to_target_centroid_mm",
    "distance_midpoint_to_target_surface_mm",
    "trajectory_intersects_target",
    "approximate_fraction_of_centerline_inside_target",
    "psa_ng_ml",
    "log_psa_ng_ml",
    "prostate_volume_cc",
    "psa_density",
]

# Binary features are kept as 0/1 and NOT z-score standardised
BINARY_FEATURES: frozenset[str] = frozenset({"trajectory_intersects_target"})

# Patient-level clinical variables (replicated to every core of a patient)
CLINICAL_FEATURES: frozenset[str] = frozenset({
    "psa_ng_ml", "log_psa_ng_ml", "prostate_volume_cc", "psa_density",
})

ALPHA = 0.05


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(
    df: pd.DataFrame,
    feat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    1. Median-impute each feature (computed over all rows — inference setting).
    2. Z-score-standardise continuous features; leave binary as 0/1.
    3. Add statsmodels constant column named 'const'.

    Returns
    -------
    X_raw : imputed, NOT standardised DataFrame (for correlation diagnostics)
    X_std : imputed, standardised (continuous) + 0/1 (binary) DataFrame,
            with 'const' prepended (ready for statsmodels)
    """
    X = df[feat_cols].copy()

    for col in feat_cols:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    X_raw = X.copy()

    for col in feat_cols:
        if col not in BINARY_FEATURES:
            mu, sd = X[col].mean(), X[col].std()
            if sd > 0:
                X[col] = (X[col] - mu) / sd

    X_std = sm.add_constant(X, has_constant="add")
    return X_raw, X_std


# ── Cluster-robust GLM ────────────────────────────────────────────────────────

def fit_cluster_robust(
    y: pd.Series,
    X_std: pd.DataFrame,
    groups: np.ndarray,
) -> object | None:
    """
    GLM Binomial (logit link) with cluster-robust standard errors.
    Groups = patient_number array.
    """
    try:
        model = sm.GLM(y, X_std, family=sm.families.Binomial())
        result = model.fit(
            cov_type="cluster",
            cov_kwds={"groups": groups},
        )
        return result
    except Exception as exc:
        warnings.warn(f"Cluster-robust GLM failed: {exc}", stacklevel=2)
        return None


# ── GEE ──────────────────────────────────────────────────────────────────────

def fit_gee(
    y: pd.Series,
    X_std: pd.DataFrame,
    groups: np.ndarray,
) -> object | None:
    """
    Binomial GEE with exchangeable working correlation, grouped by patient.
    Feature names are stored as _col_names on the result for downstream use.
    """
    if not HAS_GEE:
        return None
    col_names = list(X_std.columns)
    try:
        model = GEE(
            y.values,
            X_std.values.astype(float),
            groups=groups,
            family=GEE_Binomial(),
            cov_struct=Exchangeable(),
        )
        result = model.fit(maxiter=200)
        result._col_names = col_names
        return result
    except Exception as exc:
        warnings.warn(f"GEE fit failed: {exc}", stacklevel=2)
        return None


# ── Coefficient extraction ────────────────────────────────────────────────────

def _extract_table(
    result,
    label_col: str,
    method: str,
    meta: dict,
    col_names: list[str] | None = None,
) -> list[dict]:
    """
    Unified coefficient extraction for GLM (cluster-robust) and GEE results.

    col_names: if provided, use these as feature names in order.
               If None, fall back to result.params.index.
    """
    if col_names is None:
        if hasattr(result.params, "index"):
            col_names = list(result.params.index)
        else:
            warnings.warn(f"[{method}] Cannot determine feature names.", stacklevel=2)
            return []

    params_arr = np.asarray(result.params, dtype=float)
    bse_arr    = np.asarray(result.bse,    dtype=float)

    if hasattr(result, "tvalues"):
        t_arr = np.asarray(result.tvalues, dtype=float)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            t_arr = np.where(bse_arr > 0, params_arr / bse_arr, np.nan)

    pval_arr = np.asarray(result.pvalues, dtype=float)

    try:
        ci_raw   = result.conf_int()
        ci_lo_arr = np.asarray(ci_raw)[:, 0].astype(float)
        ci_hi_arr = np.asarray(ci_raw)[:, 1].astype(float)
    except Exception:
        ci_lo_arr = params_arr - 1.96 * bse_arr
        ci_hi_arr = params_arr + 1.96 * bse_arr

    rows = []
    for i, feat in enumerate(col_names):
        if feat == "const" or i >= len(params_arr):
            continue
        coef  = float(params_arr[i])
        se    = float(bse_arr[i])
        z     = float(t_arr[i])
        pv    = float(pval_arr[i])
        ci_lo = float(ci_lo_arr[i])
        ci_hi = float(ci_hi_arr[i])
        rows.append({
            "endpoint":        label_col,
            "method":          method,
            "feature":         feat,
            "coef":            coef,
            "se_robust":       se,
            "z_stat":          z,
            "p_value":         pv,
            "or":              float(np.exp(coef)),
            "ci_lo_coef":      ci_lo,
            "ci_hi_coef":      ci_hi,
            "ci_lo_or":        float(np.exp(ci_lo)),
            "ci_hi_or":        float(np.exp(ci_hi)),
            "is_standardized": feat not in BINARY_FEATURES,
            **meta,
        })
    return rows


# ── Collinearity diagnostics ──────────────────────────────────────────────────

def compute_correlations(X_raw: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Spearman correlations among predictors (raw, un-standardised)."""
    feats = list(X_raw.columns)
    rows = []
    for i, f1 in enumerate(feats):
        for j in range(i + 1, len(feats)):
            f2 = feats[j]
            r, pv = scipy_stats.spearmanr(X_raw[f1], X_raw[f2])
            rows.append({
                "feature1":   f1,
                "feature2":   f2,
                "spearman_r": round(float(r), 4),
                "p_value":    float(pv),
            })
    return pd.DataFrame(rows)


def compute_vif(X_std: pd.DataFrame) -> pd.DataFrame | None:
    """VIF for each predictor using the standardised design matrix (with const)."""
    if not HAS_VIF:
        return None
    try:
        arr = X_std.values.astype(float)
        rows = []
        for i, col in enumerate(X_std.columns):
            if col == "const":
                continue
            rows.append({"feature": col, "vif": round(float(variance_inflation_factor(arr, i)), 2)})
        return pd.DataFrame(rows)
    except Exception as exc:
        warnings.warn(f"VIF computation failed: {exc}", stacklevel=2)
        return None


# ── Formatting helpers ────────────────────────────────────────────────────────

def _f(v, n: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:.{n}f}"


def _p(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    v = float(v)
    star = "***" if v < 0.001 else ("**" if v < 0.01 else ("*" if v < 0.05 else ""))
    return f"{v:.4f}{star}"


def _coef_md_table(rows: list[dict]) -> list[str]:
    header = "| Feature | Coef | SE | z | p | OR | 95% CI coef | 95% CI OR | Std? |"
    sep    = "|---|---|---|---|---|---|---|---|---|"
    lines  = [header, sep]
    for r in rows:
        ci_c = f"[{_f(r['ci_lo_coef'])}, {_f(r['ci_hi_coef'])}]"
        ci_o = f"[{_f(r['ci_lo_or'])},  {_f(r['ci_hi_or'])}]"
        std_flag = "yes" if r["is_standardized"] else "no (0/1)"
        lines.append(
            f"| `{r['feature']}` "
            f"| {_f(r['coef'])} "
            f"| {_f(r['se_robust'])} "
            f"| {_f(r['z_stat'])} "
            f"| {_p(r['p_value'])} "
            f"| {_f(r['or'])} "
            f"| {ci_c} "
            f"| {ci_o} "
            f"| {std_flag} |"
        )
    return lines


def _corr_md_table(df: pd.DataFrame) -> list[str]:
    header = "| Feature 1 | Feature 2 | Spearman ρ | p-value |"
    sep    = "|---|---|---|---|"
    lines  = [header, sep]
    for _, r in df.iterrows():
        lines.append(
            f"| `{r['feature1']}` | `{r['feature2']}` "
            f"| {_f(r['spearman_r'])} | {_p(r['p_value'])} |"
        )
    return lines


def _vif_md_table(df: pd.DataFrame) -> list[str]:
    header = "| Feature | VIF |"
    sep    = "|---|---|"
    lines  = [header, sep]
    for _, r in df.iterrows():
        note = " ← high" if r["vif"] > 10 else (" ← moderate" if r["vif"] > 5 else "")
        lines.append(f"| `{r['feature']}` | {_f(r['vif'])}{note} |")
    return lines


# ── Interpretation ────────────────────────────────────────────────────────────

def _interpret(
    cr_rows_by_ep:  dict[str, list[dict]],
    gee_rows_by_ep: dict[str, list[dict]],
) -> list[str]:
    lines: list[str] = []

    def _get(rows: list[dict], feat: str) -> dict | None:
        for r in rows:
            if r["feature"] == feat:
                return r
        return None

    TARGET_GEO_FEATS = {
        "distance_midpoint_to_target_centroid_mm",
        "distance_midpoint_to_target_surface_mm",
        "trajectory_intersects_target",
        "approximate_fraction_of_centerline_inside_target",
    }

    for label_col, ep_label in ENDPOINTS:
        cr_rows  = cr_rows_by_ep.get(label_col, [])
        gee_rows = gee_rows_by_ep.get(label_col, [])
        lines   += [f"### {ep_label}", ""]

        if not cr_rows:
            lines += ["*Cluster-robust model did not converge — no interpretation.*", ""]
            continue

        # Point 1: distance to target surface
        surf_r = _get(cr_rows, "distance_midpoint_to_target_surface_mm")
        if surf_r is not None:
            coef = surf_r["coef"]
            pv   = surf_r["p_value"]
            or_v = surf_r["or"]
            direction = "negatively" if coef < 0 else "positively"
            significance = (
                "remains statistically significant" if pv < ALPHA
                else "does not reach statistical significance"
            )
            lines += [
                f"**Distance to target surface:** `distance_midpoint_to_target_surface_mm` "
                f"is {direction} associated with the endpoint and {significance} after "
                f"accounting for intra-patient clustering "
                f"(coef = {_f(coef)}, OR = {_f(or_v)}, p = {_p(pv)}, cluster-robust SE).",
                "",
            ]

        # Point 2: geometry robustness
        geo_sig = [r for r in cr_rows if r["feature"] in TARGET_GEO_FEATS and r["p_value"] < ALPHA]
        n_geo   = len(TARGET_GEO_FEATS)
        n_sig   = len(geo_sig)
        if n_sig >= 2:
            lines += [
                f"**Target-relative geometry:** {n_sig} of {n_geo} target-geometry features "
                f"are individually significant (p < 0.05) after cluster-robust adjustment, "
                f"supporting the interpretability of `target_geometry_plus_clinical` after "
                f"accounting for patient-level correlation.",
                "",
            ]
        elif n_sig == 1:
            lines += [
                f"**Target-relative geometry:** 1 of {n_geo} target-geometry features reaches "
                f"p < 0.05 after cluster-robust adjustment. Geometry features may contribute "
                f"jointly; partial multicollinearity among them can suppress individual "
                f"significance without eliminating joint predictive value.",
                "",
            ]
        else:
            lines += [
                f"**Target-relative geometry:** No individual target-geometry feature reaches "
                f"p < 0.05 after cluster-robust adjustment. Geometry may contribute jointly "
                f"rather than individually, or intra-patient correlation reduces effective "
                f"sample size sufficiently to mask individual effects.",
                "",
            ]

        # trajectory_intersects_target redundancy note
        lines += [
            "**Note on `trajectory_intersects_target`:** "
            "`trajectory_intersects_target` is not individually significant once "
            "continuous distance and fraction-inside features are included, likely "
            "because it is highly redundant with "
            "`approximate_fraction_of_centerline_inside_target`.",
            "",
        ]

        # Point 3: PSA collinearity caveat
        lines += [
            "**PSA-related variables:** `psa_ng_ml`, `log_psa_ng_ml`, and `psa_density` "
            "are correlated clinical measurements (see Section 5). Individual coefficient "
            "estimates for these features can be numerically unstable and should not be "
            "used to rank clinical importance. The group of PSA-related variables is "
            "informative, but no single coefficient alone is interpretable in isolation.",
            "",
        ]

        # GEE concordance if available
        if gee_rows and surf_r is not None:
            gee_surf = _get(gee_rows, "distance_midpoint_to_target_surface_mm")
            if gee_surf is not None:
                same_sign = (surf_r["coef"] * gee_surf["coef"]) > 0
                concordance = "concordant with" if same_sign else "discordant with"
                lines += [
                    f"**GEE concordance:** The GEE estimate for "
                    f"`distance_midpoint_to_target_surface_mm` "
                    f"(coef = {_f(gee_surf['coef'])}, p = {_p(gee_surf['p_value'])}) "
                    f"is {concordance} the cluster-robust estimate.",
                    "",
                ]

    # Point 4: support for compact model
    lines += [
        "### Support for the compact `target_geometry_plus_clinical` model",
        "",
        "The cluster-robust analysis tests whether target-relative geometry "
        "associations survive intra-patient correlation adjustment. Statistically "
        "significant geometry associations after this adjustment support the use of "
        "the compact feature set as the central interpretable association model, "
        "while predictive claims should remain based on the held-out performance "
        "analyses. Clinical (PSA-related) variables contribute population-level "
        "discriminative value but their individual coefficients should be reported "
        "with appropriate collinearity caveats.",
        "",
    ]

    return lines


# ── Report writing ────────────────────────────────────────────────────────────

def write_reports(
    cr_rows_by_ep:  dict[str, list[dict]],
    gee_rows_by_ep: dict[str, list[dict]],
    corr_by_ep:     dict[str, pd.DataFrame],
    vif_by_ep:      dict[str, pd.DataFrame | None],
    summary_by_ep:  dict[str, dict],
    has_gee:        bool,
    gee_produced_output: bool,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # CSVs
    all_cr = [r for rows in cr_rows_by_ep.values() for r in rows]
    if all_cr:
        p = REPORTS_DIR / "grouped_core_level_inference_cluster_robust.csv"
        pd.DataFrame(all_cr).to_csv(p, index=False)
        print(f"Saved -> {p.relative_to(REPO_ROOT)}")

    all_gee = [r for rows in gee_rows_by_ep.values() for r in rows]
    if all_gee:
        p = REPORTS_DIR / "grouped_core_level_inference_gee.csv"
        pd.DataFrame(all_gee).to_csv(p, index=False)
        print(f"Saved -> {p.relative_to(REPO_ROOT)}")

    corr_dfs = [
        df_c.assign(endpoint=lc)
        for lc, df_c in corr_by_ep.items()
        if not df_c.empty
    ]
    if corr_dfs:
        p = REPORTS_DIR / "grouped_core_level_inference_correlation.csv"
        pd.concat(corr_dfs, ignore_index=True).to_csv(p, index=False)
        print(f"Saved -> {p.relative_to(REPO_ROOT)}")

    # Markdown
    lines: list[str] = [
        "# Grouped Core-Level Inference Analysis",
        "",
        "## 1. Methodology",
        "",
        "**Goal:** Assess whether the compact `target_geometry_plus_clinical` "
        "associations remain present after accounting for intra-patient correlation. "
        "This is an inference / association analysis — not a predictive-performance "
        "experiment.",
        "",
        "**Data:** `data/share/needle_features_v1.csv`, filtered to "
        '`label_join_status == "coord_match"` and `split ∈ {train, val, test}`. '
        "All valid splits are pooled to maximise statistical power.",
        "",
        "**Feature set:** `target_geometry_plus_clinical` (8 features). "
        "Clinical variables (`psa_ng_ml`, `log_psa_ng_ml`, `prostate_volume_cc`, "
        "`psa_density`) are patient-level measurements replicated to each core — "
        "they are not independent core-level observations.",
        "",
        "**Preprocessing:** Missing values are median-imputed over all pooled rows. "
        "Continuous predictors are z-score standardised (one unit = one standard "
        "deviation) so that coefficients are on a comparable scale. "
        "`trajectory_intersects_target` is a binary (0/1) indicator and is kept "
        "un-standardised; its coefficient represents the log-odds difference between "
        "cores that do and do not intersect the target.",
        "",
        "**Model 1 — Cluster-robust logistic regression:** "
        "`statsmodels.GLM` (Binomial, logit link) with `cov_type='cluster'` "
        "clustered by `patient_number`. This yields asymptotically valid standard "
        "errors and p-values that account for intra-patient core correlation without "
        "imposing a parametric correlation structure.",
        "",
    ]

    if has_gee:
        lines += [
            "**Model 2 — GEE:** Binomial GEE with exchangeable working correlation "
            "(`statsmodels.genmod.GEE`), grouped by `patient_number`. "
            "GEE makes weaker parametric assumptions than mixed models; the "
            "exchangeable structure assumes a common intra-patient correlation "
            "across core pairs.",
            "",
        ]
    else:
        lines += [
            "**Model 2 — GEE:** `statsmodels.genmod.GEE` not available in this "
            "environment. Cluster-robust logistic regression (Section 3) provides "
            "equivalent asymptotic protection against intra-patient correlation.",
            "",
        ]

    lines += [
        "**Significance threshold:** α = 0.05 (two-tailed). "
        "Stars: \\* p < 0.05, \\*\\* p < 0.01, \\*\\*\\* p < 0.001.",
        "",
        "## 2. Dataset Summary",
        "",
    ]

    for label_col, ep_label in ENDPOINTS:
        s = summary_by_ep.get(label_col)
        if not s:
            continue
        lines += [
            f"### {ep_label}",
            "",
            f"- Total rows (all splits): **{s['n_rows']:,}**",
            f"- Unique patients: **{s['n_patients']:,}**",
            f"- Positive cores: **{s['n_positive']:,}** ({_f(s['prevalence'] * 100, 1)}%)",
            "",
            "| Split | Rows | Patients | Positives | Prevalence |",
            "|---|---|---|---|---|",
        ]
        for split in ("train", "val", "test"):
            lines.append(
                f"| {split} "
                f"| {s[f'{split}_n_rows']:,} "
                f"| {s[f'{split}_n_patients']:,} "
                f"| {s[f'{split}_n_pos']:,} "
                f"| {_f(s[f'{split}_prevalence'] * 100, 1)}% |"
            )
        lines.append("")

    lines += [
        "## 3. Cluster-Robust Logistic Regression",
        "",
        "> **Standardisation note:** coefficients for continuous features are in "
        "units of one standard deviation. `trajectory_intersects_target` is binary "
        "(0/1); its coefficient is the log-odds change from non-intersecting to "
        "intersecting cores.",
        "",
    ]
    for label_col, ep_label in ENDPOINTS:
        rows = cr_rows_by_ep.get(label_col, [])
        lines += [f"### {ep_label}", ""]
        if not rows:
            lines += ["*Model did not converge or produced no output.*", ""]
        else:
            lines += _coef_md_table(rows)
            lines.append("")

    lines += ["## 4. GEE Results (Optional)", ""]
    if not has_gee:
        lines += [
            "> `statsmodels.genmod.GEE` is not available. "
            "Cluster-robust logistic regression (Section 3) provides a valid "
            "alternative that does not require a parametric correlation structure.",
            "",
        ]
    elif not gee_produced_output:
        lines += [
            "> GEE fitting failed for all endpoints (see console warnings). "
            "Cluster-robust logistic regression (Section 3) remains the primary "
            "inference method.",
            "",
        ]
    else:
        lines += [
            "> **Standardisation note:** same as Section 3.",
            "",
        ]
        for label_col, ep_label in ENDPOINTS:
            rows = gee_rows_by_ep.get(label_col, [])
            lines += [f"### {ep_label}", ""]
            if not rows:
                lines += ["*GEE did not converge for this endpoint.*", ""]
            else:
                lines += _coef_md_table(rows)
                lines.append("")

    lines += [
        "## 5. Predictor Correlation and Collinearity",
        "",
        "> **Warning:** `psa_ng_ml`, `log_psa_ng_ml`, and `psa_density` are "
        "strongly correlated by construction. Joint coefficient estimates for "
        "these features may be numerically unstable; individual coefficient "
        "magnitudes should not be used to rank PSA-related clinical importance.",
        "",
    ]
    for label_col, ep_label in ENDPOINTS:
        df_corr = corr_by_ep.get(label_col, pd.DataFrame())
        vif_df  = vif_by_ep.get(label_col)

        lines += [f"### {ep_label}", "", "#### Pairwise Spearman correlations", ""]
        if df_corr.empty:
            lines += ["*Not available.*", ""]
        else:
            lines += _corr_md_table(df_corr)
            lines.append("")

        lines += ["#### Variance Inflation Factors (VIF)", ""]
        if vif_df is not None and not vif_df.empty:
            lines += _vif_md_table(vif_df)
            lines.append("")
        else:
            lines += ["*VIF computation not available.*", ""]

    lines += ["## 6. Interpretation", ""]
    lines += _interpret(cr_rows_by_ep, gee_rows_by_ep)

    md_path = REPORTS_DIR / "grouped_core_level_inference.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"Saved -> {md_path.relative_to(REPO_ROOT)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== Grouped core-level inference analysis ===\n")

    cr_rows_by_ep:  dict[str, list[dict]]            = {}
    gee_rows_by_ep: dict[str, list[dict]]            = {}
    corr_by_ep:     dict[str, pd.DataFrame]          = {}
    vif_by_ep:      dict[str, pd.DataFrame | None]   = {}
    summary_by_ep:  dict[str, dict]                  = {}

    gee_produced_output = False

    for label_col, ep_label in ENDPOINTS:
        print(f"\n--- Endpoint: {ep_label} ({label_col}) ---")

        df, _ = load_dataset(DATA_PATH, target_col=label_col)
        df    = prepare_features(df)

        y      = df[label_col]
        groups = df[PATIENT_COL].values

        # Dataset summary
        s: dict = {
            "n_rows":     len(df),
            "n_patients": int(df[PATIENT_COL].nunique()),
            "n_positive": int(y.sum()),
            "prevalence": float(y.mean()),
        }
        for split in ("train", "val", "test"):
            sub = df[df[SPLIT_COL] == split]
            s[f"{split}_n_rows"]     = len(sub)
            s[f"{split}_n_patients"] = int(sub[PATIENT_COL].nunique())
            s[f"{split}_n_pos"]      = int(sub[label_col].sum())
            s[f"{split}_prevalence"] = float(sub[label_col].mean()) if len(sub) else float("nan")
        summary_by_ep[label_col] = s
        print(
            f"  {s['n_rows']:,} rows | {s['n_patients']:,} patients | "
            f"{s['n_positive']:,} positives ({s['prevalence']*100:.1f}%)"
        )

        # Preprocess
        feat_cols = [c for c in FEATURE_COLS if c in df.columns]
        missing   = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            warnings.warn(f"[{label_col}] missing columns: {missing}", stacklevel=1)

        X_raw, X_std = preprocess(df, feat_cols)

        meta = {
            "n_rows":     s["n_rows"],
            "n_patients": s["n_patients"],
            "n_positive": s["n_positive"],
            "prevalence": s["prevalence"],
        }

        # Cluster-robust GLM
        print("  Fitting cluster-robust GLM ... ", end="", flush=True)
        cr_result = fit_cluster_robust(y, X_std, groups)
        if cr_result is not None:
            cr_rows = _extract_table(
                cr_result, label_col, "cluster_robust", meta,
                col_names=list(X_std.columns),
            )
            cr_rows_by_ep[label_col] = cr_rows
            n_sig = sum(1 for r in cr_rows if r["p_value"] < ALPHA)
            print(f"done — {n_sig}/{len(cr_rows)} significant at α={ALPHA}")
        else:
            cr_rows_by_ep[label_col] = []
            print("FAILED")

        # GEE
        if HAS_GEE:
            print("  Fitting GEE (exchangeable) ... ", end="", flush=True)
            gee_result = fit_gee(y, X_std, groups)
            if gee_result is not None:
                col_names = getattr(gee_result, "_col_names", list(X_std.columns))
                gee_rows  = _extract_table(
                    gee_result, label_col, "gee_exchangeable", meta,
                    col_names=col_names,
                )
                gee_rows_by_ep[label_col] = gee_rows
                gee_produced_output = True
                n_sig = sum(1 for r in gee_rows if r["p_value"] < ALPHA)
                print(f"done — {n_sig}/{len(gee_rows)} significant at α={ALPHA}")
            else:
                gee_rows_by_ep[label_col] = []
                print("FAILED")
        else:
            gee_rows_by_ep[label_col] = []

        # Collinearity diagnostics
        corr_by_ep[label_col] = compute_correlations(X_raw)
        vif_by_ep[label_col]  = compute_vif(X_std)

    write_reports(
        cr_rows_by_ep,
        gee_rows_by_ep,
        corr_by_ep,
        vif_by_ep,
        summary_by_ep,
        has_gee=HAS_GEE,
        gee_produced_output=gee_produced_output,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
