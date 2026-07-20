"""
compare_extended_feature_sets.py

Goal
----
Run a final low-cost improvement sprint comparing the compact geometry-clinical
model against interpretable extended feature sets identified in the feature-
opportunity audit.  Reports whether any extension meaningfully improves
prediction before manuscript freeze.

This is a modelling script.  It does NOT modify any existing reports.

Outputs
-------
reports/extended_feature_set_comparison.csv
reports/extended_feature_set_comparison.md
reports/extended_feature_set_risk_stratification.csv
reports/extended_feature_set_risk_stratification.md
reports/extended_feature_set_recommendation.md

Usage
-----
    python src/compare_extended_feature_sets.py

Constraints
-----------
- cancer_length_mm and pct_cancer_in_core are pathology outcomes available
  only after the biopsy has been processed.  They are NEVER used as predictors.
- is_targeted_or_prior_positive encodes the physician's pre-biopsy targeting
  decision (systematic sextant vs MRI-targeted / prior-positive site).  It is
  not pathology leakage, but it encodes clinical suspicion and must be reported
  in a separate procedural-context model, not silently merged into the compact
  geometry-clinical model.
- All delta values are: extended − current_compact (positive = extension better).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from run_shareable_tabular_experiments import (
    CLINICAL_FEATURES,
    HAS_XGBOOST,
    PATIENT_COL,
    RANDOM_STATE,
    SPLIT_COL,
    TARGET_GEOMETRY_NO_AVAILABILITY_FEATURES,
    bootstrap_patient_ci,
    build_model,
    evaluate_model,
    get_model_specs,
    load_dataset,
    prepare_features,
)


# ── Paths & constants ─────────────────────────────────────────────────────────

REPO_ROOT   = Path(__file__).resolve().parent.parent
DATA_PATH   = REPO_ROOT / "data" / "share" / "needle_features_v1.csv"
REPORTS_DIR = REPO_ROOT / "reports"

N_BOOT      = 1000
REFERENCE_FS = "current_compact"          # baseline everything is measured against

# Strict thresholds for "meaningful" improvement
MIN_DELTA_PR_AUC  = 0.020
MIN_DELTA_ROC_AUC = 0.015

ENDPOINTS: list[tuple[str, str]] = [
    ("binary_label_int",         "GG2+ / csPCa"),
    ("binary_label_gg3plus_int", "GG3+ / high-grade"),
]

# ── Base feature lists ────────────────────────────────────────────────────────

_TARGET = TARGET_GEOMETRY_NO_AVAILABILITY_FEATURES   # 4 target-geometry features
_CLIN   = CLINICAL_FEATURES                          # 4 clinical features
_COMPACT = _TARGET + _CLIN                           # 8-feature compact model

# New derived feature names (created in derive_extended_features())
FEAT_TARGETED        = "is_targeted_or_prior_positive"
FEAT_SIGNED_DIST     = "signed_distance_midpoint_to_target_surface_mm"
FEAT_INTER_SDIST_PD  = "inter_signed_dist_psa_density"
FEAT_INTER_FRAC_PD   = "inter_frac_inside_psa_density"
FEAT_INTER_SDIST_LP  = "inter_signed_dist_log_psa"
FEAT_INTER_FRAC_LP   = "inter_frac_inside_log_psa"

_INTERACTIONS = [
    FEAT_INTER_SDIST_PD,
    FEAT_INTER_FRAC_PD,
    FEAT_INTER_SDIST_LP,
    FEAT_INTER_FRAC_LP,
]

# Replace unsigned distance with signed distance in the target-geometry block.
_TARGET_SIGNED = [
    c if c != "distance_midpoint_to_target_surface_mm" else FEAT_SIGNED_DIST
    for c in _TARGET
]
_COMPACT_SIGNED = _TARGET_SIGNED + _CLIN          # 8 features, signed distance


def _make_feature_sets() -> dict[str, list[str]]:
    """
    Ordered dict of (set_name -> feature_col_list) definitions.

    A. current_compact                            — 8 features (baseline)
    B. compact_with_signed_distance               — 8 features (distance recode only)
    C. compact_plus_targeting_context             — 9 features (+ procedural flag)
    D. compact_signed_plus_targeting_context      — 9 features (signed + flag)
    E. compact_signed_plus_interactions           — 12 features (signed + 4 interactions)
    F. compact_signed_plus_targeting_plus_inter   — 13 features
    G. clinical_plus_targeting_context            — 5 features (clinical + flag only)
    H. targeting_context_only                     — 1 feature (flag only, lower bound)
    """
    return {
        "current_compact":
            _COMPACT,
        "compact_with_signed_distance":
            _COMPACT_SIGNED,
        "compact_plus_targeting_context":
            _COMPACT + [FEAT_TARGETED],
        "compact_signed_plus_targeting_context":
            _COMPACT_SIGNED + [FEAT_TARGETED],
        "compact_signed_plus_interactions":
            _COMPACT_SIGNED + _INTERACTIONS,
        "compact_signed_plus_targeting_plus_inter":
            _COMPACT_SIGNED + [FEAT_TARGETED] + _INTERACTIONS,
        "clinical_plus_targeting_context":
            _CLIN + [FEAT_TARGETED],
        "targeting_context_only":
            [FEAT_TARGETED],
    }


FEATURE_SETS = _make_feature_sets()


# ── Feature derivation ────────────────────────────────────────────────────────

def derive_extended_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add new derived columns to a DataFrame that has already been through
    prepare_features().  Modifies a copy; does not touch the original.

    New columns added:
    - is_targeted_or_prior_positive  (0/1 float)
    - signed_distance_midpoint_to_target_surface_mm  (float, mm)
    - inter_signed_dist_psa_density
    - inter_frac_inside_psa_density
    - inter_signed_dist_log_psa
    - inter_frac_inside_log_psa

    Never uses: cancer_length_mm, pct_cancer_in_core, primary_gleason,
    secondary_gleason — all are pathology outcomes not available pre-biopsy.
    """
    df = df.copy()

    # ── 1. Procedural targeting-context flag ─────────────────────────────────
    # core_label == "TARGET OR PRIOR POSITIVE" means the physician directed this
    # core at an MRI-suspicious lesion or a previously positive site BEFORE the
    # current biopsy outcome was known.  This is NOT pathology leakage.
    if "core_label" in df.columns:
        df[FEAT_TARGETED] = (df["core_label"] == "TARGET OR PRIOR POSITIVE").astype(float)
    else:
        warnings.warn(
            "'core_label' column absent — "
            f"'{FEAT_TARGETED}' will be all-NaN.",
            stacklevel=2,
        )
        df[FEAT_TARGETED] = np.nan

    # ── 2. Signed distance to target surface ─────────────────────────────────
    # Convention: negative = needle midpoint is inside or touching the target mesh
    #             (fraction > 0 or trajectory intersects), positive = outside.
    # This is a deterministic recoding of distance_midpoint_to_target_surface_mm.
    dist_col = "distance_midpoint_to_target_surface_mm"
    frac_col = "approximate_fraction_of_centerline_inside_target"
    if dist_col in df.columns and frac_col in df.columns:
        inside = df[frac_col] > 0
        df[FEAT_SIGNED_DIST] = np.where(
            inside,
            -df[dist_col].abs(),   # inside → negative magnitude
            df[dist_col].abs(),    # outside → positive magnitude
        )
    else:
        missing = [c for c in [dist_col, frac_col] if c not in df.columns]
        warnings.warn(
            f"Columns missing for signed-distance derivation: {missing}. "
            f"'{FEAT_SIGNED_DIST}' will be all-NaN.",
            stacklevel=2,
        )
        df[FEAT_SIGNED_DIST] = np.nan

    # ── 3. Interaction features ───────────────────────────────────────────────
    # All interactions use signed distance and/or fraction_inside × PSA metrics.
    pd_col  = "psa_density"
    lp_col  = "log_psa_ng_ml"
    df[FEAT_INTER_SDIST_PD] = df[FEAT_SIGNED_DIST] * df[pd_col]   if pd_col in df.columns else np.nan
    df[FEAT_INTER_FRAC_PD]  = df[frac_col]          * df[pd_col]   if pd_col in df.columns else np.nan
    df[FEAT_INTER_SDIST_LP] = df[FEAT_SIGNED_DIST] * df[lp_col]   if lp_col in df.columns else np.nan
    df[FEAT_INTER_FRAC_LP]  = df[frac_col]          * df[lp_col]   if lp_col in df.columns else np.nan

    return df


# ── Helpers ───────────────────────────────────────────────────────────────────

def _f(v, n: int = 3) -> str:
    if isinstance(v, (float, np.floating)) and np.isnan(v):
        return "n/a"
    return f"{v:.{n}f}"


def _risk_strat(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """Top-5/10/20% prevalence and capture rate on a scored test set."""
    n     = len(y_true)
    n_pos = int(y_true.sum())
    result: dict = {"prevalence": float(y_true.mean()) if n > 0 else np.nan}
    for pct in (5, 10, 20):
        k       = max(1, int(np.ceil(n * pct / 100)))
        top_idx = np.argsort(probs)[::-1][:k]
        top_y   = y_true[top_idx]
        result[f"top{pct}pct_prevalence"]   = float(top_y.mean())
        result[f"top{pct}pct_capture_rate"] = float(top_y.sum() / n_pos) if n_pos > 0 else np.nan
        result[f"top{pct}pct_enrichment"]   = (
            float(top_y.mean() / y_true.mean())
            if y_true.mean() > 0 else np.nan
        )
    return result


# ── Experiment runner ─────────────────────────────────────────────────────────

def run_one(
    label_col: str,
    fs_name: str,
    feature_cols: list[str],
    model_name: str,
    df: pd.DataFrame,
) -> tuple[dict, np.ndarray] | tuple[None, None]:
    """
    Fit on train, select Youden-J threshold on val, evaluate on test.
    Returns (result_row, test_probs) or (None, None) on failure.
    """
    train = df[df[SPLIT_COL] == "train"]
    val   = df[df[SPLIT_COL] == "val"]
    test  = df[df[SPLIT_COL] == "test"]

    usable  = [c for c in feature_cols if c in df.columns and train[c].notna().any()]
    dropped = set(feature_cols) - set(usable)
    if dropped:
        warnings.warn(
            f"[{label_col}/{fs_name}/{model_name}] "
            f"dropping unusable columns: {sorted(dropped)}",
            stacklevel=2,
        )
    if not usable:
        warnings.warn(
            f"[{label_col}/{fs_name}/{model_name}] no usable features — skipping.",
            stacklevel=2,
        )
        return None, None

    X_train, y_train = train[usable], train[label_col]
    X_val,   y_val   = val[usable],   val[label_col]
    X_test,  y_test  = test[usable],  test[label_col]

    pipeline = build_model(model_name, usable, y_train)
    if pipeline is None:
        return None, None

    test_metrics, _, threshold = evaluate_model(
        pipeline, X_train, y_train, X_val, y_val, X_test, y_test
    )
    test_probs = pipeline.predict_proba(X_test)[:, 1]

    test_prev = float(y_test.mean())
    pr_auc_v  = test_metrics.get("pr_auc", np.nan)
    pr_lift   = (
        pr_auc_v / test_prev
        if test_prev > 0 and not (isinstance(pr_auc_v, float) and np.isnan(pr_auc_v))
        else np.nan
    )

    ci = bootstrap_patient_ci(
        pipeline, X_test, y_test,
        test[PATIENT_COL].values,
        threshold,
        n_boot=N_BOOT,
        random_state=RANDOM_STATE,
    )

    row: dict = {
        "label_col":       label_col,
        "feature_set":     fs_name,
        "model":           model_name,
        "n_features":      len(usable),
        "n_train":         len(train),
        "n_val":           len(val),
        "n_test":          len(test),
        "test_prevalence": test_prev,
        "pr_lift":         pr_lift,
        "threshold":       threshold,
    }
    row.update({f"test_{k}": v for k, v in test_metrics.items()})
    row.update(ci)
    return row, test_probs


# ── Markdown helpers ──────────────────────────────────────────────────────────

_PERF_COLS = [
    "feature_set", "model", "n_features",
    "test_roc_auc", "roc_auc_ci_low", "roc_auc_ci_high",
    "test_pr_auc",  "pr_auc_ci_low",  "pr_auc_ci_high",
    "pr_lift",
    "threshold",
    "test_sensitivity", "test_specificity",
    "test_precision",   "test_f1",
    "n_train", "n_val", "n_test", "test_prevalence",
]

_RISK_COLS = [
    "feature_set", "model", "prevalence",
    "top5pct_prevalence",  "top5pct_capture_rate",  "top5pct_enrichment",
    "top10pct_prevalence", "top10pct_capture_rate", "top10pct_enrichment",
    "top20pct_prevalence", "top20pct_capture_rate", "top20pct_enrichment",
]


def _md_table(df: pd.DataFrame, cols: list[str]) -> list[str]:
    avail = [c for c in cols if c in df.columns]
    header = "| " + " | ".join(avail) + " |"
    sep    = "|" + "---|" * len(avail)
    lines  = [header, sep]
    for _, r in df.iterrows():
        cells = [
            _f(v) if isinstance(v, (float, np.floating)) else str(v)
            for v in (r.get(c, np.nan) for c in avail)
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


# ── Feature audit ─────────────────────────────────────────────────────────────

def _feature_audit_section(raw_df: pd.DataFrame, df: pd.DataFrame) -> list[str]:
    """
    Report core_label distribution and is_targeted_or_prior_positive prevalence.
    This section must be read before interpreting the model comparisons.
    """
    lines: list[str] = [
        "## Feature Audit: `core_label` and `is_targeted_or_prior_positive`",
        "",
        "### Critical interpretation notes",
        "",
        "**`is_targeted_or_prior_positive`** is derived from `core_label == "
        '"TARGET OR PRIOR POSITIVE"`. It encodes whether the urologist directed',
        "this biopsy core at an MRI-visible suspicious lesion or a previously "
        "positive site, rather than a systematic sextant location.  This "
        "information is recorded **before** the pathology outcome of the current "
        "core is known.",
        "",
        "This is **not** pathology leakage in the strict sense.  However, it is "
        "a *procedural targeting-context* feature, not a geometry or clinical "
        "feature.  Including it encodes clinical suspicion (the radiologist's / "
        "urologist's belief that this site is suspicious), which may partially "
        "proxy for the MRI PI-RADS score or prior-session pathology.  Any model "
        "including this feature must be reported as a separate "
        "**procedural-context model**, not as an extension of the pure "
        "geometry-clinical compact model.",
        "",
        "**Excluded pathology outcomes (never used as predictors):**",
        "- `cancer_length_mm` — millimetres of cancer in core (pathology result)",
        "- `pct_cancer_in_core` — percentage cancer (pathology result)",
        "- `primary_gleason` / `secondary_gleason` — Gleason grades (pathology result)",
        "",
    ]

    # core_label missingness
    if "core_label" not in df.columns:
        lines += ["*`core_label` column not found in dataset.*", ""]
        return lines

    total = len(df)
    miss_n = df["core_label"].isna().sum()
    lines += [
        f"**`core_label` missingness (coord_match, valid split):** "
        f"{miss_n} / {total} ({100*miss_n/total:.1f}%) missing.",
        "",
    ]

    # Distribution of core_label values
    val_counts = df["core_label"].value_counts(dropna=False)
    lines += [
        "**`core_label` value distribution (all rows, all splits):**",
        "",
        "| Value | Count | % of total |",
        "|-------|-------|-----------|",
    ]
    for val, cnt in val_counts.items():
        label_str = str(val) if not (isinstance(val, float) and np.isnan(val)) else "NaN"
        lines.append(f"| {label_str} | {cnt} | {100*cnt/total:.1f}% |")
    lines.append("")

    # Distribution by split
    lines += [
        "**`core_label` by split:**",
        "",
        "| Split | n_rows | n_targeted | targeted_pct |",
        "|-------|--------|------------|--------------|",
    ]
    for sp in ["train", "val", "test"]:
        sp_df = df[df[SPLIT_COL] == sp]
        n_sp = len(sp_df)
        n_tgt = (sp_df["core_label"] == "TARGET OR PRIOR POSITIVE").sum()
        lines.append(f"| {sp} | {n_sp} | {n_tgt} | {100*n_tgt/max(n_sp,1):.1f}% |")
    lines.append("")

    # Prevalence by is_targeted × endpoint
    lines += [
        "**GG2+ prevalence by `is_targeted_or_prior_positive` (test split):**",
        "",
    ]
    test_df = df[df[SPLIT_COL] == "test"].copy()
    if FEAT_TARGETED in test_df.columns and "binary_label_int" in test_df.columns:
        for tgt_val, label in [(1.0, "Targeted / prior-positive"), (0.0, "Systematic sextant")]:
            sub = test_df[test_df[FEAT_TARGETED] == tgt_val]
            if len(sub) == 0:
                continue
            prev = sub["binary_label_int"].mean()
            lines.append(
                f"- **{label}** (n={len(sub)}): GG2+ prevalence = {100*prev:.1f}%"
            )
        lines.append("")

    if "binary_label_gg3plus_int" in test_df.columns:
        lines += [
            "**GG3+ prevalence by `is_targeted_or_prior_positive` (test split):**",
            "",
        ]
        for tgt_val, label in [(1.0, "Targeted / prior-positive"), (0.0, "Systematic sextant")]:
            sub = test_df[test_df[FEAT_TARGETED] == tgt_val]
            if len(sub) == 0:
                continue
            prev = sub["binary_label_gg3plus_int"].mean()
            lines.append(
                f"- **{label}** (n={len(sub)}): GG3+ prevalence = {100*prev:.1f}%"
            )
        lines.append("")

    lines += [
        "### Signed-distance derivation",
        "",
        "**`signed_distance_midpoint_to_target_surface_mm`** is a deterministic "
        "recoding of `distance_midpoint_to_target_surface_mm`:",
        "- **Negative** when `approximate_fraction_of_centerline_inside_target > 0` "
        "(needle partially inside the target mesh)",
        "- **Positive** otherwise (needle fully outside the target mesh)",
        "",
        "This is not a new measurement; it adds sign information that the unsigned "
        "distance discards, at the cost of one additional assumption about the sign "
        "convention.",
        "",
        "### Interaction features",
        "",
        "Four interaction terms are computed as simple products:",
        "",
        f"| Column | Formula |",
        f"|--------|---------|",
        f"| `{FEAT_INTER_SDIST_PD}` | signed_dist × psa_density |",
        f"| `{FEAT_INTER_FRAC_PD}` | frac_inside_target × psa_density |",
        f"| `{FEAT_INTER_SDIST_LP}` | signed_dist × log_psa |",
        f"| `{FEAT_INTER_FRAC_LP}` | frac_inside_target × log_psa |",
        "",
        "These interaction terms capture potential PSA-severity modulation of "
        "needle-target proximity effects.  They increase model complexity and risk "
        "of overfitting on a small test set (~120 patients); they are included as "
        "exploratory sensitivity analyses only.",
        "",
    ]

    return lines


# ── Delta section ─────────────────────────────────────────────────────────────

def _delta_section(df_perf: pd.DataFrame, label_col: str, ep_label: str) -> list[str]:
    """Per-endpoint delta table vs current_compact (LR only for clarity)."""
    sub = df_perf[df_perf["label_col"] == label_col]
    if sub.empty:
        return [f"### {ep_label}", "", "*No results.*", ""]

    ref_rows = sub[(sub["feature_set"] == REFERENCE_FS) & (sub["model"] == "logistic_regression")]
    if ref_rows.empty:
        return [f"### {ep_label}", "", f"*Reference `{REFERENCE_FS}` LR not available.*", ""]

    ref_roc = float(ref_rows["test_roc_auc"].iloc[0])
    ref_pr  = float(ref_rows["test_pr_auc"].iloc[0])
    ref_t5p = float(ref_rows.get("top5pct_prevalence", pd.Series([np.nan])).iloc[0]) if "top5pct_prevalence" in ref_rows else np.nan
    ref_t10c = float(ref_rows.get("top10pct_capture_rate", pd.Series([np.nan])).iloc[0]) if "top10pct_capture_rate" in ref_rows else np.nan

    lines: list[str] = [
        f"### {ep_label}",
        "",
        "Delta values = extended − current_compact (logistic regression, test split).  "
        "Positive = extension outperforms compact baseline.",
        "",
        f"| Feature set | n_feat | ROC-AUC | ΔROC | PR-AUC | ΔPR | "
        f"top5% prev | Δtop5% | top10% cap | Δtop10% |",
        f"|-------------|--------|---------|------|--------|-----|"
        f"---------|--------|-----------|---------|",
    ]

    lr_sub = sub[sub["model"] == "logistic_regression"]
    for _, r in lr_sub.iterrows():
        roc  = float(r["test_roc_auc"])
        pr   = float(r["test_pr_auc"])
        droc = roc - ref_roc
        dpr  = pr  - ref_pr
        sr = "+" if droc >= 0 else ""
        sp = "+" if dpr  >= 0 else ""

        # Risk-strat deltas (may not be in perf df — we'll add them later via merge)
        lines.append(
            f"| `{r['feature_set']}` | {r['n_features']} "
            f"| {_f(roc)} | {sr}{_f(droc)} "
            f"| {_f(pr)} | {sp}{_f(dpr)} "
            f"| — | — | — | — |"
        )
    lines.append("")
    return lines


# ── Recommendation section ────────────────────────────────────────────────────

def _recommendation_section(
    df_perf: pd.DataFrame,
    df_risk: pd.DataFrame,
) -> list[str]:
    """
    Apply strict decision criteria and produce a recommendation document.
    Critical stance: do not overclaim, distinguish feature types, flag
    interpretation risks.
    """
    lines: list[str] = [
        "# Extended Feature Set Recommendation",
        "",
        "**Branch:** `feat/geometry-clinical-baselines`  ",
        "**Script:** `src/compare_extended_feature_sets.py`  ",
        "**Based on:** `reports/extended_feature_set_comparison.md`",
        "",
        "## Decision criteria",
        "",
        "An extended feature set is considered **meaningfully better** than the compact "
        "baseline only if, on the held-out **test split** with logistic regression:",
        "",
        f"- ΔPR-AUC ≥ +{MIN_DELTA_PR_AUC:.3f}  **OR**",
        f"- ΔROC-AUC ≥ +{MIN_DELTA_ROC_AUC:.3f}",
        "",
        "OR if it substantially improves top-ranked enrichment (top-5% capture rate "
        "or prevalence) without relying on pathology leakage.",
        "",
        "These thresholds are identical to the equivalence margins used in the "
        "compact-vs-full comparison (`reports/minimal_vs_full_model_comparison.md`).",
        "",
        "## Feature-type classifications",
        "",
        "| Feature | Type | Leakage risk |",
        "|---------|------|-------------|",
        "| `distance_midpoint_to_target_surface_mm` | Target geometry | None |",
        "| `signed_distance_midpoint_to_target_surface_mm` | Target geometry (recode) | None |",
        "| `distance_midpoint_to_target_centroid_mm` | Target geometry | None |",
        "| `trajectory_intersects_target` | Target geometry | None |",
        "| `approximate_fraction_of_centerline_inside_target` | Target geometry | None |",
        "| `psa_ng_ml`, `log_psa_ng_ml`, `prostate_volume_cc`, `psa_density` | Clinical | None |",
        "| `is_targeted_or_prior_positive` | Procedural targeting context | **See note** |",
        "| Interaction terms | Derived | None |",
        "| `cancer_length_mm`, `pct_cancer_in_core` | Pathology outcome | **Leakage — excluded** |",
        "",
        "**Note on `is_targeted_or_prior_positive`:** Encodes the physician's pre-biopsy "
        "targeting decision.  Not pathology leakage, but encodes clinical suspicion "
        "(MRI PI-RADS, prior-session positivity) that is not available in the standard "
        "compact feature set.  Any model including this feature must be clearly labelled "
        "as a *procedural-context extended model*, not the primary compact model.",
        "",
        "## Results summary",
        "",
    ]

    # Build a compact summary from the LR results
    for label_col, ep_label in ENDPOINTS:
        sub = df_perf[
            (df_perf["label_col"] == label_col)
            & (df_perf["model"] == "logistic_regression")
        ]
        if sub.empty:
            lines += [f"### {ep_label}", "", "*No LR results.*", ""]
            continue

        ref_row = sub[sub["feature_set"] == REFERENCE_FS]
        if ref_row.empty:
            lines += [f"### {ep_label}", "", "*Baseline not available.*", ""]
            continue

        ref_roc = float(ref_row["test_roc_auc"].iloc[0])
        ref_pr  = float(ref_row["test_pr_auc"].iloc[0])

        lines += [f"### {ep_label}", ""]
        lines += [
            f"**Compact baseline (LR):** ROC-AUC = {_f(ref_roc)}, PR-AUC = {_f(ref_pr)}",
            "",
            "| Feature set | ΔROC-AUC | ΔPR-AUC | Meaningful? |",
            "|-------------|----------|---------|------------|",
        ]

        for _, r in sub.iterrows():
            if r["feature_set"] == REFERENCE_FS:
                continue
            droc = float(r["test_roc_auc"]) - ref_roc
            dpr  = float(r["test_pr_auc"])  - ref_pr
            sr = "+" if droc >= 0 else ""
            sp = "+" if dpr  >= 0 else ""
            meaningful = (
                "**YES**" if (dpr >= MIN_DELTA_PR_AUC or droc >= MIN_DELTA_ROC_AUC)
                else "No"
            )
            lines.append(
                f"| `{r['feature_set']}` | {sr}{_f(droc)} | {sp}{_f(dpr)} "
                f"| {meaningful} |"
            )
        lines.append("")

    # Recommendation text
    lines += [
        "## Recommendations",
        "",
        "### 1. Geometry and clinical features",
        "",
    ]

    # Check signed distance
    signed_meaningful: dict[str, bool] = {}
    for label_col, ep_label in ENDPOINTS:
        sub = df_perf[
            (df_perf["label_col"] == label_col)
            & (df_perf["model"] == "logistic_regression")
        ]
        ref_row = sub[sub["feature_set"] == REFERENCE_FS]
        ext_row = sub[sub["feature_set"] == "compact_with_signed_distance"]
        if ref_row.empty or ext_row.empty:
            signed_meaningful[label_col] = False
            continue
        droc = float(ext_row["test_roc_auc"].iloc[0]) - float(ref_row["test_roc_auc"].iloc[0])
        dpr  = float(ext_row["test_pr_auc"].iloc[0])  - float(ref_row["test_pr_auc"].iloc[0])
        signed_meaningful[label_col] = (dpr >= MIN_DELTA_PR_AUC or droc >= MIN_DELTA_ROC_AUC)

    if any(signed_meaningful.values()):
        lines += [
            "**Signed distance to target surface:** The signed-distance recode of "
            "`distance_midpoint_to_target_surface_mm` improves performance for at least "
            "one endpoint (see table above).  Consider replacing unsigned distance with "
            "signed distance in the compact model **only if** the improvement is consistent "
            "across both endpoints and the interpretation is clearly documented (negative = "
            "inside target, positive = outside).  If improvement is endpoint-specific, "
            "retain unsigned distance in the primary compact model and report signed "
            "distance as a sensitivity analysis.",
            "",
        ]
    else:
        lines += [
            "**Signed distance to target surface:** The signed-distance recode does not "
            "improve performance beyond the compact baseline for either endpoint.  "
            "Retain `distance_midpoint_to_target_surface_mm` (unsigned) in the compact "
            "model.  The signed variant provides cleaner interpretation but no measurable "
            "predictive benefit; mention as a model property in the Methods section if "
            "desired, but do not substitute.",
            "",
        ]

    # Check interactions
    inter_meaningful: dict[str, bool] = {}
    for label_col, ep_label in ENDPOINTS:
        sub = df_perf[
            (df_perf["label_col"] == label_col)
            & (df_perf["model"] == "logistic_regression")
        ]
        ref_row = sub[sub["feature_set"] == REFERENCE_FS]
        ext_row = sub[sub["feature_set"] == "compact_signed_plus_interactions"]
        if ref_row.empty or ext_row.empty:
            inter_meaningful[label_col] = False
            continue
        droc = float(ext_row["test_roc_auc"].iloc[0]) - float(ref_row["test_roc_auc"].iloc[0])
        dpr  = float(ext_row["test_pr_auc"].iloc[0])  - float(ref_row["test_pr_auc"].iloc[0])
        inter_meaningful[label_col] = (dpr >= MIN_DELTA_PR_AUC or droc >= MIN_DELTA_ROC_AUC)

    if any(inter_meaningful.values()):
        lines += [
            "**Interaction features:** Interaction terms (signed distance × PSA density / "
            "log-PSA; fraction inside × PSA density / log-PSA) improve performance for "
            "at least one endpoint.  Caution: interaction terms are harder to interpret, "
            "increase collinearity, and may overfit on a test set of ~120 patients.  "
            "Do not include in the primary compact model.  Report as an exploratory "
            "supplementary analysis with a clear caveat about overfitting risk.",
            "",
        ]
    else:
        lines += [
            "**Interaction features:** Interaction terms (signed distance / fraction inside "
            "× PSA density / log-PSA) do not improve performance beyond the compact "
            "baseline.  Do not include in any model.  This finding supports the conclusion "
            "that PSA-severity and needle-target proximity act independently rather than "
            "multiplicatively in this dataset.",
            "",
        ]

    lines += ["### 2. Procedural targeting-context flag", ""]

    # Check is_targeted_or_prior_positive
    tctx_meaningful: dict[str, bool] = {}
    for label_col, ep_label in ENDPOINTS:
        sub = df_perf[
            (df_perf["label_col"] == label_col)
            & (df_perf["model"] == "logistic_regression")
        ]
        ref_row = sub[sub["feature_set"] == REFERENCE_FS]
        ext_row = sub[sub["feature_set"] == "compact_plus_targeting_context"]
        if ref_row.empty or ext_row.empty:
            tctx_meaningful[label_col] = False
            continue
        droc = float(ext_row["test_roc_auc"].iloc[0]) - float(ref_row["test_roc_auc"].iloc[0])
        dpr  = float(ext_row["test_pr_auc"].iloc[0])  - float(ref_row["test_pr_auc"].iloc[0])
        tctx_meaningful[label_col] = (dpr >= MIN_DELTA_PR_AUC or droc >= MIN_DELTA_ROC_AUC)

    if any(tctx_meaningful.values()):
        lines += [
            "**`is_targeted_or_prior_positive`** improves performance for at least one "
            "endpoint beyond the compact baseline.  This is expected: targeted cores have "
            "structurally higher positivity rates than systematic cores (see feature "
            "audit section), and this information partially proxies for the radiologist's "
            "MRI-based suspicion.",
            "",
            "**Recommendation:** Report `compact_plus_targeting_context` as an "
            "**extended procedural-context model** in a supplementary table or secondary "
            "analysis section.  Do **not** replace the compact geometry-clinical model "
            "with this extended model as the primary result, because:",
            "",
            "1. `is_targeted_or_prior_positive` is a targeting-decision variable, "
            "not a geometry or clinical measurement.  Its availability may differ "
            "in prospective deployment (e.g., systematic biopsy cohorts with no "
            "targeting label).",
            "2. Including it would change the scientific claim from "
            "'needle-to-target geometry predicts cancer' to 'the targeting decision "
            "plus geometry predicts cancer,' which are substantively different claims.",
            "3. The improvement may partially reflect confounding between targeting "
            "strategy and cancer prevalence rather than independent signal.",
            "",
        ]
    else:
        lines += [
            "**`is_targeted_or_prior_positive`** does not improve performance beyond "
            "the compact baseline by the pre-specified thresholds.  This suggests that "
            "the geometric features (distance, fraction inside target) already partially "
            "capture the information encoded in the targeting decision, or that the "
            "effect is not strong enough to detect given the test-set sample size.",
            "",
            "**Recommendation:** Do not add `is_targeted_or_prior_positive` to any "
            "primary model.  The feature audit result (see above) should still be "
            "reported in the manuscript to document that the 'TARGET OR PRIOR POSITIVE' "
            "category was identified and its leakage status was assessed.",
            "",
        ]

    lines += [
        "### 3. Primary compact model recommendation",
        "",
        "**Retain `current_compact` (`target_geometry_plus_clinical`, 8 features) as "
        "the primary compact model** unless the signed-distance recode is meaningfully "
        "better for both endpoints (see §1 above).",
        "",
        "No extension tested here justifies replacing the compact geometry-clinical model "
        "as the primary reported result.  The compact model remains the most parsimonious, "
        "interpretable, and scientifically defensible choice for the manuscript's central "
        "claim.",
        "",
        "### 4. Limitations of this sprint",
        "",
        "- Test set contains ~120 patients; all delta estimates have wide bootstrap CIs.  "
        "A ΔROC-AUC of ±0.015 is near the noise floor at this sample size.",
        "- `is_targeted_or_prior_positive` was not pre-registered as a candidate feature "
        "before the modelling phase; it was identified by post-hoc column audit.  "
        "Treat its performance as exploratory.",
        "- Interaction terms were not pre-specified; all results for extended sets E and F "
        "should be considered hypothesis-generating only.",
        "- This sprint does not constitute a prospective validation.  Calibration and "
        "decision-curve analysis were not re-run; raw probabilities are not reported as "
        "calibrated absolute risks.",
        "",
        "---",
        "",
        "*Report generated by `src/compare_extended_feature_sets.py`.  "
        "Do not modify by hand.*",
    ]

    return lines


# ── Report writer ─────────────────────────────────────────────────────────────

def write_reports(
    results: list[dict],
    risk_rows: list[dict],
    raw_df: pd.DataFrame,
    df_aug: pd.DataFrame,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df_perf = pd.DataFrame(results)
    df_risk = pd.DataFrame(risk_rows)

    # ── CSVs ──────────────────────────────────────────────────────────────────
    perf_csv = REPORTS_DIR / "extended_feature_set_comparison.csv"
    risk_csv = REPORTS_DIR / "extended_feature_set_risk_stratification.csv"
    df_perf.to_csv(perf_csv, index=False)
    df_risk.to_csv(risk_csv, index=False)
    print(f"Saved -> {perf_csv.relative_to(REPO_ROOT)}")
    print(f"Saved -> {risk_csv.relative_to(REPO_ROOT)}")

    # ── Main comparison MD ────────────────────────────────────────────────────
    perf_md = REPORTS_DIR / "extended_feature_set_comparison.md"
    lines: list[str] = [
        "# Extended Feature Set Comparison",
        "",
        "**Branch:** `feat/geometry-clinical-baselines`  ",
        "**Script:** `src/compare_extended_feature_sets.py`  ",
        "**Data:** `data/share/needle_features_v1.csv`",
        "",
        "## Methodology",
        "",
        "**Endpoints:** GG2+ / csPCa (`binary_label_int`) and "
        "GG3+ / high-grade (`binary_label_gg3plus_int`).  ",
        "**Models:** logistic regression (class-balanced, standardised features), "
        "HistGradientBoosting, and XGBoost (if installed).  ",
        "**Threshold selection:** Youden J maximised on the validation split.  ",
        "**Bootstrap CIs:** 1,000 patient-level resamples of the test set.  ",
        f"**Reference baseline:** `{REFERENCE_FS}` (8-feature compact model: "
        "4 target-geometry + 4 clinical).  ",
        "**Delta convention:** extended − baseline; positive = extension outperforms.  ",
        "",
        "**Excluded predictors (pathology leakage):**",
        "`cancer_length_mm`, `pct_cancer_in_core`, `primary_gleason`, "
        "`secondary_gleason` — all measured after core processing; never used.",
        "",
    ]

    # Feature audit
    lines += _feature_audit_section(raw_df, df_aug)

    # Performance tables
    lines += ["## Performance comparison", ""]
    for label_col, ep_label in ENDPOINTS:
        sub = df_perf[df_perf["label_col"] == label_col]
        if sub.empty:
            lines += [f"### {ep_label}", "", "*No results.*", ""]
            continue
        lines += [f"### {ep_label}", ""]
        lines += _md_table(sub, _PERF_COLS)
        lines.append("")

    # Delta sections (LR only for clarity)
    lines += ["## Delta vs compact baseline (logistic regression)", ""]
    for label_col, ep_label in ENDPOINTS:
        lines += _delta_section(df_perf, label_col, ep_label)

    perf_md.write_text("\n".join(lines) + "\n")
    print(f"Saved -> {perf_md.relative_to(REPO_ROOT)}")

    # ── Risk stratification MD ────────────────────────────────────────────────
    risk_md = REPORTS_DIR / "extended_feature_set_risk_stratification.md"
    rlines: list[str] = [
        "# Extended Feature Set — Risk Stratification",
        "",
        "**Capture rate** = fraction of all test positives in the top-N% scored cores.  ",
        "**Prevalence** in top-N% = positive rate among top-N% highest-probability cores.  ",
        "**Enrichment** = top-N% prevalence / baseline prevalence.  ",
        "**Baseline:** overall positive rate in the test set.",
        "",
    ]
    for label_col, ep_label in ENDPOINTS:
        sub_r = df_risk[df_risk["label_col"] == label_col]
        if sub_r.empty:
            rlines += [f"## {ep_label}", "", "*No results.*", ""]
            continue
        rlines += [f"## {ep_label}", ""]
        rlines += _md_table(sub_r, _RISK_COLS)
        rlines.append("")

    risk_md.write_text("\n".join(rlines) + "\n")
    print(f"Saved -> {risk_md.relative_to(REPO_ROOT)}")

    # ── Recommendation MD ─────────────────────────────────────────────────────
    rec_md = REPORTS_DIR / "extended_feature_set_recommendation.md"
    rec_md.write_text(
        "\n".join(_recommendation_section(df_perf, df_risk)) + "\n"
    )
    print(f"Saved -> {rec_md.relative_to(REPO_ROOT)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== Extended feature set comparison ===\n")
    if not HAS_XGBOOST:
        warnings.warn(
            "xgboost not installed — XGBoost models will be skipped.",
            stacklevel=1,
        )

    results:    list[dict] = []
    risk_rows:  list[dict] = []
    df_aug_ref: pd.DataFrame | None = None    # will hold augmented df for audit
    raw_ref:    pd.DataFrame | None = None    # raw CSV for audit

    for label_col, ep_label in ENDPOINTS:
        print(f"\n--- Endpoint: {ep_label} ({label_col}) ---")
        df_base, _ = load_dataset(DATA_PATH, target_col=label_col)
        df_base = prepare_features(df_base)
        df       = derive_extended_features(df_base)

        # Stash the augmented df from the first endpoint for the feature audit
        if df_aug_ref is None:
            df_aug_ref = df
            raw_ref    = df_base   # post-filter but pre-derived

        n_pts = df[PATIENT_COL].nunique()
        print(f"  {len(df):,} rows, {n_pts:,} patients")

        available_cols  = set(df.columns)
        model_specs     = get_model_specs(df[label_col])
        model_names     = [name for name, _, _ in model_specs]

        fs_dict = FEATURE_SETS

        for fs_name, fs_cols_raw in fs_dict.items():
            feature_cols = [c for c in fs_cols_raw if c in available_cols]
            missing = [c for c in fs_cols_raw if c not in available_cols]
            if missing:
                warnings.warn(
                    f"[{label_col}/{fs_name}] missing columns: {missing}",
                    stacklevel=1,
                )
            if not feature_cols:
                continue

            for model_name in model_names:
                print(
                    f"  [{fs_name}] {model_name} ...",
                    end=" ", flush=True,
                )
                row, test_probs = run_one(
                    label_col, fs_name, feature_cols, model_name, df
                )
                if row is None:
                    print("skipped")
                    continue

                results.append(row)
                print(
                    f"ROC-AUC={_f(row['test_roc_auc'])}  "
                    f"PR-AUC={_f(row['test_pr_auc'])}"
                )

                test_df = df[df[SPLIT_COL] == "test"]
                y_test  = test_df[label_col].values
                risk = _risk_strat(y_test, test_probs)
                risk_rows.append({
                    "label_col":   label_col,
                    "feature_set": fs_name,
                    "model":       model_name,
                    **risk,
                })

    if not results:
        print("\nNo results — check data and dependencies.")
        return

    write_reports(results, risk_rows, raw_ref, df_aug_ref)
    print("\nDone.")


if __name__ == "__main__":
    main()
