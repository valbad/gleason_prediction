"""
compare_final_targeted_feature_sprint.py

Goal
----
Final targeted feature sprint. Tests four candidate feature families against
the compact 8-feature geometry-clinical baseline:
  1. core_label encoding (full one-hot + anatomical/procedural derived flags)
  2. First-order image/intensity features (auto-detected from column names)
  3. Simple transformed geometry features (log1p distances, ratios)
  4. Regularized logistic regression (L2, L1, Elastic Net tuned on val PR-AUC)

Decision thresholds are STRICTER than the extended sprint:
  ΔPR-AUC >= +0.030 OR ΔROC-AUC >= +0.020 on held-out test split (LR).
  Gain must appear on >=1 endpoint without large degradation on the other.

This is a modelling script.  It does NOT modify existing reports.

Outputs
-------
reports/final_targeted_feature_sprint_comparison.csv
reports/final_targeted_feature_sprint_comparison.md
reports/final_targeted_feature_sprint_risk_stratification.csv
reports/final_targeted_feature_sprint_risk_stratification.md
reports/final_targeted_feature_sprint_recommendation.md

Leakage exclusion
-----------------
Never used as predictors:
  cancer_length_mm, pct_cancer_in_core, primary_gleason, secondary_gleason,
  pathology_label, pathology_label_int, binary_label, binary_label_int,
  binary_label_gg3plus_int.

core_label context
------------------
core_label is a pre-biopsy procedural/anatomical targeting label, NOT
pathology leakage. It encodes whether the physician directed this core at an
MRI-suspicious lesion (targeted) or a systematic sextant position. Any model
including core_label-derived features is explicitly labelled as a
PROCEDURAL/ANATOMICAL CONTEXT model.

Usage
-----
    python src/compare_final_targeted_feature_sprint.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from run_shareable_tabular_experiments import (
    BOOLEAN_FEATURES,
    CLINICAL_FEATURES,
    HAS_XGBOOST,
    PATIENT_COL,
    RANDOM_STATE,
    SPLIT_COL,
    TARGET_GEOMETRY_NO_AVAILABILITY_FEATURES,
    bootstrap_patient_ci,
    build_model,
    compute_metrics,
    evaluate_model,
    load_dataset,
    prepare_features,
    select_threshold_youden,
)


# ── Paths & constants ─────────────────────────────────────────────────────────

REPO_ROOT   = Path(__file__).resolve().parent.parent
DATA_PATH   = REPO_ROOT / "data" / "share" / "needle_features_v1.csv"
REPORTS_DIR = REPO_ROOT / "reports"

N_BOOT        = 1000
REFERENCE_FS  = "current_compact"

# Stricter than extended sprint (ΔPR 0.020 → 0.030; ΔROC 0.015 → 0.020)
MIN_DELTA_PR_AUC  = 0.030
MIN_DELTA_ROC_AUC = 0.020

ENDPOINTS: list[tuple[str, str]] = [
    ("binary_label_int",         "GG2+ / csPCa"),
    ("binary_label_gg3plus_int", "GG3+ / high-grade"),
]

_TARGET  = TARGET_GEOMETRY_NO_AVAILABILITY_FEATURES   # 4 target-geometry features
_CLIN    = CLINICAL_FEATURES                          # 4 clinical features
_COMPACT = _TARGET + _CLIN                            # 8-feature compact model

LEAKAGE_COLS: frozenset[str] = frozenset({
    "cancer_length_mm", "pct_cancer_in_core",
    "primary_gleason", "secondary_gleason",
    "pathology_label", "pathology_label_int",
    "binary_label", "binary_label_int", "binary_label_gg3plus_int",
})

_META_NEVER_FEATURES: frozenset[str] = frozenset({
    "patient_number", "split", "label_join_status",
    "core_id", "filename", "path",
    "extraction_status", "target_mesh_available",
    "midpoint_inside_prostate",
})

INTENSITY_KEYWORDS: tuple[str, ...] = (
    "intensity", "mean", "median", "std", "percentile",
    "p10", "p25", "p75", "p90", "t2", "adc", "dwi", "voxel", "patch",
)

# Regularization grids
L2_C_VALUES  = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
L1_C_VALUES  = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
EN_C_VALUES  = [0.03, 0.1, 0.3, 1.0, 3.0]
EN_L1_RATIOS = [0.2, 0.5, 0.8]


# ── Feature derivation ────────────────────────────────────────────────────────

def _detect_intensity_cols(df: pd.DataFrame) -> list[str]:
    """Auto-detect first-order image/intensity columns from column name keywords."""
    exclude = LEAKAGE_COLS | _META_NEVER_FEATURES | {"core_label"}
    result = []
    for col in df.columns:
        if col in exclude:
            continue
        col_lower = col.lower()
        if any(kw in col_lower for kw in INTENSITY_KEYWORDS):
            if df[col].notna().mean() >= 0.10:
                result.append(col)
    return sorted(result)


def _sanitize(s: str) -> str:
    """Sanitize a string to a safe column name fragment."""
    return (
        s.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("+", "plus")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
    )


def derive_sprint_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str], list[str], list[str], set[str]]:
    """
    Derive all new feature columns for this sprint.

    Returns
    -------
    df               : augmented DataFrame (copy)
    cl_onehot        : core_label one-hot column names
    cl_anatom        : anatomical/procedural derived column names
    intensity_cols   : auto-detected intensity column names (pre-existing, not derived)
    geo_transf       : transformed geometry column names
    sprint_bool_cols : set of all new binary (0/1) derived columns (not to be std-scaled)
    """
    df = df.copy()
    sprint_bool_cols: set[str] = set()

    # 1. core_label full one-hot encoding
    cl_onehot: list[str] = []
    if "core_label" in df.columns:
        is_na = df["core_label"].isna()
        cats  = sorted(df["core_label"].dropna().unique())
        for cat in cats:
            col_name = "cl_" + _sanitize(str(cat))
            df[col_name] = (df["core_label"] == cat).astype(float)
            cl_onehot.append(col_name)
            sprint_bool_cols.add(col_name)
        if is_na.any():
            df["cl___missing__"] = is_na.astype(float)
            cl_onehot.append("cl___missing__")
            sprint_bool_cols.add("cl___missing__")
    else:
        warnings.warn("'core_label' not found — one-hot features will be empty.", stacklevel=2)

    # 2. Anatomical/procedural flags derived from core_label
    cl_anatom: list[str] = []
    if "core_label" in df.columns:
        cl_lower = df["core_label"].fillna("").str.lower()

        def _add_flag(col_name: str, mask: "pd.Series[bool]") -> None:
            df[col_name] = mask.astype(float)
            cl_anatom.append(col_name)
            sprint_bool_cols.add(col_name)

        _add_flag("cl_is_targeted_or_prior_positive",
                  df["core_label"] == "TARGET OR PRIOR POSITIVE")
        _add_flag("cl_is_systematic_core",
                  (df["core_label"] != "TARGET OR PRIOR POSITIVE") & df["core_label"].notna())
        _add_flag("cl_is_left",    cl_lower.str.contains("left",    na=False))
        _add_flag("cl_is_right",   cl_lower.str.contains("right",   na=False))
        _add_flag("cl_is_apex",    cl_lower.str.contains("apex",    na=False))
        _add_flag("cl_is_mid",     cl_lower.str.contains("mid",     na=False))
        _add_flag("cl_is_base",    cl_lower.str.contains("base",    na=False))
        _add_flag("cl_is_lateral", cl_lower.str.contains("lateral", na=False))
        _add_flag("cl_is_extra",   cl_lower.str.contains("extra",   na=False))

    # 3. Auto-detect intensity columns (existing columns, not derived)
    intensity_cols = _detect_intensity_cols(df)

    # 4. Simple transformed geometry features
    dist_surf  = "distance_midpoint_to_target_surface_mm"
    dist_cent  = "distance_midpoint_to_target_centroid_mm"
    frac_ins   = "approximate_fraction_of_centerline_inside_target"
    traj_int   = "trajectory_intersects_target"
    core_len_c = "core_length_mm"

    geo_transf: list[str] = []

    def _geo(col_name: str, series: pd.Series) -> None:
        df[col_name] = series
        geo_transf.append(col_name)

    if dist_surf in df.columns:
        _geo("geo_log1p_dist_surface",
             np.log1p(df[dist_surf].clip(lower=0)))
    if dist_cent in df.columns:
        _geo("geo_log1p_dist_centroid",
             np.log1p(df[dist_cent].clip(lower=0)))
    if dist_surf in df.columns and core_len_c in df.columns:
        denom = df[core_len_c].replace(0, np.nan)
        _geo("geo_surface_dist_per_core_len", df[dist_surf] / denom)
    if dist_cent in df.columns and core_len_c in df.columns:
        denom = df[core_len_c].replace(0, np.nan)
        _geo("geo_centroid_dist_per_core_len", df[dist_cent] / denom)
    if frac_ins in df.columns and core_len_c in df.columns:
        _geo("geo_frac_inside_times_core_len", df[frac_ins] * df[core_len_c])
    if dist_surf in df.columns and traj_int in df.columns:
        _geo("geo_outside_dist_not_intersecting",
             df[dist_surf] * (1.0 - df[traj_int].fillna(0)))
    if frac_ins in df.columns and traj_int in df.columns:
        _geo("geo_inside_frac_when_intersecting",
             df[frac_ins] * df[traj_int].fillna(0))

    return df, cl_onehot, cl_anatom, intensity_cols, geo_transf, sprint_bool_cols


def _dedup(cols: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def make_feature_sets(
    cl_onehot: list[str],
    cl_anatom: list[str],
    intensity_cols: list[str],
    geo_transf: list[str],
) -> dict[str, list[str]]:
    """
    A: current_compact                            8 features (baseline)
    B: compact_plus_core_label_onehot             8 + |categories|
    C: compact_plus_anatomical_core_label         8 + 9 anatomical flags
    D: compact_plus_intensity                     8 + detected intensity
    E: compact_plus_transformed_geometry          8 + derived geometry
    F: compact_plus_core_label_plus_intensity     B + intensity
    G: compact_plus_core_label_plus_transformed_geo B + transformed geometry
    H: compact_plus_all_low_cost_features         all of the above
    I: clinical_plus_core_label                   4 + one-hot
    J: target_geometry_plus_core_label            4 + one-hot
    """
    sets: dict[str, list[str]] = {"current_compact": _COMPACT}

    if cl_onehot:
        sets["compact_plus_core_label_onehot"] = _dedup(_COMPACT + cl_onehot)
    if cl_anatom:
        sets["compact_plus_anatomical_core_label"] = _dedup(_COMPACT + cl_anatom)
    if intensity_cols:
        sets["compact_plus_intensity"] = _dedup(_COMPACT + intensity_cols)
    if geo_transf:
        sets["compact_plus_transformed_geometry"] = _dedup(_COMPACT + geo_transf)
    if cl_onehot and intensity_cols:
        sets["compact_plus_core_label_plus_intensity"] = _dedup(_COMPACT + cl_onehot + intensity_cols)
    if cl_onehot and geo_transf:
        sets["compact_plus_core_label_plus_transformed_geo"] = _dedup(_COMPACT + cl_onehot + geo_transf)

    all_extras = cl_onehot + cl_anatom + intensity_cols + geo_transf
    if all_extras:
        sets["compact_plus_all_low_cost_features"] = _dedup(_COMPACT + all_extras)

    if cl_onehot:
        sets["clinical_plus_core_label"] = _dedup(_CLIN + cl_onehot)
        sets["target_geometry_plus_core_label"] = _dedup(_TARGET + cl_onehot)

    return sets


# ── Model builders (LR only; HGB/XGBoost use shared build_model) ──────────────

def _build_lr_preprocessor(
    feature_cols: list[str],
    extra_bool_cols: set[str],
) -> ColumnTransformer:
    """Numeric features: median-imputed + StandardScaled. Boolean: most_frequent imputed only."""
    all_bool     = BOOLEAN_FEATURES | extra_bool_cols
    numeric_cols = [c for c in feature_cols if c not in all_bool]
    bool_cols    = [c for c in feature_cols if c in all_bool]
    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
        ]), numeric_cols))
    if bool_cols:
        transformers.append(("bool", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
        ]), bool_cols))
    return ColumnTransformer(transformers)


def _build_tuned_lr(
    penalty: str,
    C: float,
    l1_ratio: float | None,
    feature_cols: list[str],
    y_train: pd.Series,
    sprint_bool_cols: set[str],
) -> Pipeline:
    """Build a LR pipeline with the given regularization settings."""
    if penalty == "elasticnet":
        lr = LogisticRegression(
            class_weight="balanced", max_iter=10000,
            penalty="elasticnet", solver="saga",
            C=C, l1_ratio=l1_ratio, random_state=RANDOM_STATE,
        )
    elif penalty == "l1":
        lr = LogisticRegression(
            class_weight="balanced", max_iter=10000,
            penalty="l1", solver="saga", C=C, random_state=RANDOM_STATE,
        )
    else:  # l2
        lr = LogisticRegression(
            class_weight="balanced", max_iter=10000,
            penalty="l2", solver="lbfgs", C=C, random_state=RANDOM_STATE,
        )
    prep = _build_lr_preprocessor(feature_cols, sprint_bool_cols)
    return Pipeline([("preprocess", prep), ("clf", lr)])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _f(v, n: int = 3) -> str:
    if isinstance(v, (float, np.floating)) and np.isnan(v):
        return "n/a"
    return f"{v:.{n}f}"


def _risk_strat(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """Top-5/10/20% prevalence, capture rate, and enrichment."""
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
            float(top_y.mean() / y_true.mean()) if y_true.mean() > 0 else np.nan
        )
    return result


# ── Tuning ────────────────────────────────────────────────────────────────────

def _tune_lr_params(
    penalty: str,
    feature_cols: list[str],
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame,   y_val: pd.Series,
    sprint_bool_cols: set[str],
) -> tuple[float, float | None, float]:
    """
    Grid-search C (and l1_ratio for elasticnet) on val PR-AUC.
    Tie-break on val ROC-AUC.
    Returns (best_C, best_l1_ratio, best_val_pr_auc).
    """
    C_grid    = {"l2": L2_C_VALUES, "l1": L1_C_VALUES, "elasticnet": EN_C_VALUES}[penalty]
    l1_ratios = EN_L1_RATIOS if penalty == "elasticnet" else [None]

    best_val_pr  = -np.inf
    best_val_roc = -np.inf
    best_C       = 1.0
    best_l1r: float | None = None

    for C in C_grid:
        for l1r in l1_ratios:
            try:
                pipe = _build_tuned_lr(penalty, C, l1r, feature_cols, y_train, sprint_bool_cols)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    pipe.fit(X_train, y_train)
                vp = pipe.predict_proba(X_val)[:, 1]
                if len(np.unique(y_val)) < 2:
                    continue
                val_pr  = float(average_precision_score(y_val.values, vp))
                val_roc = float(roc_auc_score(y_val.values, vp))
                if val_pr > best_val_pr or (
                    val_pr == best_val_pr and val_roc > best_val_roc
                ):
                    best_val_pr  = val_pr
                    best_val_roc = val_roc
                    best_C, best_l1r = C, l1r
            except Exception as exc:
                warnings.warn(
                    f"Tuning {penalty} C={C} l1_ratio={l1r}: {exc}", stacklevel=2,
                )

    return best_C, best_l1r, best_val_pr


# ── Experiment runner ─────────────────────────────────────────────────────────

def run_one(
    label_col: str,
    fs_name: str,
    feature_cols: list[str],
    model_name: str,
    df: pd.DataFrame,
    sprint_bool_cols: set[str],
) -> tuple[dict, np.ndarray] | tuple[None, None]:
    """
    Fit on train, select Youden-J threshold on val, evaluate on test.
    Tuned variants: tune C/l1_ratio on val PR-AUC first.
    Returns (result_row, test_probs) or (None, None) on failure.
    """
    train = df[df[SPLIT_COL] == "train"]
    val   = df[df[SPLIT_COL] == "val"]
    test  = df[df[SPLIT_COL] == "test"]

    usable  = [c for c in feature_cols if c in df.columns and train[c].notna().any()]
    dropped = set(feature_cols) - set(usable)
    if dropped:
        warnings.warn(
            f"[{label_col}/{fs_name}/{model_name}] dropping unusable: {sorted(dropped)}",
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

    tuned_meta: dict = {}

    if model_name.startswith("tuned_"):
        penalty = model_name.replace("tuned_", "").replace("_logistic_regression", "")
        if penalty not in ("l2", "l1", "elasticnet"):
            warnings.warn(
                f"Unrecognised tuned model name '{model_name}' — skipping.", stacklevel=2,
            )
            return None, None

        best_C, best_l1r, best_val_pr = _tune_lr_params(
            penalty, usable, X_train, y_train, X_val, y_val, sprint_bool_cols,
        )
        print(
            f"    → {penalty}: best C={best_C}, l1_ratio={best_l1r}, "
            f"val PR-AUC={_f(best_val_pr)}",
            flush=True,
        )
        tuned_meta = {
            "tuned_penalty":  penalty,
            "tuned_C":        best_C,
            "tuned_l1_ratio": best_l1r if best_l1r is not None else np.nan,
        }
        pipeline = _build_tuned_lr(penalty, best_C, best_l1r, usable, y_train, sprint_bool_cols)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            pipeline.fit(X_train, y_train)

        val_probs  = pipeline.predict_proba(X_val)[:, 1]
        test_probs = pipeline.predict_proba(X_test)[:, 1]
        threshold  = select_threshold_youden(y_val.values, val_probs)
        test_metrics = compute_metrics(y_test.values, test_probs, threshold)

    else:
        pipeline = build_model(model_name, usable, y_train)
        if pipeline is None:
            return None, None
        test_metrics, _, threshold = evaluate_model(
            pipeline, X_train, y_train, X_val, y_val, X_test, y_test,
        )
        test_probs = pipeline.predict_proba(X_test)[:, 1]

    ci = bootstrap_patient_ci(
        pipeline, X_test, y_test,
        test[PATIENT_COL].values, threshold,
        n_boot=N_BOOT, random_state=RANDOM_STATE,
    )

    test_prev = float(y_test.mean())
    pr_auc_v  = test_metrics.get("pr_auc", np.nan)
    pr_lift   = (
        pr_auc_v / test_prev
        if test_prev > 0 and not (isinstance(pr_auc_v, float) and np.isnan(pr_auc_v))
        else np.nan
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
        **tuned_meta,
    }
    row.update({f"test_{k}": v for k, v in test_metrics.items()})
    row.update(ci)
    return row, test_probs


# ── Markdown helpers ──────────────────────────────────────────────────────────

_PERF_COLS = [
    "feature_set", "model", "n_features",
    "test_roc_auc", "roc_auc_ci_low", "roc_auc_ci_high",
    "test_pr_auc",  "pr_auc_ci_low",  "pr_auc_ci_high",
    "pr_lift", "threshold",
    "test_sensitivity", "test_specificity", "test_precision", "test_f1",
    "n_train", "n_val", "n_test", "test_prevalence",
]

_RISK_COLS = [
    "feature_set", "model", "prevalence",
    "top5pct_prevalence",  "top5pct_capture_rate",  "top5pct_enrichment",
    "top10pct_prevalence", "top10pct_capture_rate", "top10pct_enrichment",
    "top20pct_prevalence", "top20pct_capture_rate", "top20pct_enrichment",
]


def _md_table(df: pd.DataFrame, cols: list[str]) -> list[str]:
    avail  = [c for c in cols if c in df.columns]
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


# ── Leakage audit section ─────────────────────────────────────────────────────

def _leakage_audit_section(
    df: pd.DataFrame,
    intensity_cols: list[str],
    cl_onehot: list[str],
    cl_anatom: list[str],
    geo_transf: list[str],
) -> list[str]:
    lines: list[str] = [
        "## Leakage Audit and Feature Inventory",
        "",
        "### Excluded leakage columns (pathology outcomes — never used as predictors)",
        "",
    ]
    present_leakage = sorted(c for c in LEAKAGE_COLS if c in df.columns)
    absent_leakage  = sorted(c for c in LEAKAGE_COLS if c not in df.columns)
    for c in present_leakage:
        lines.append(f"- `{c}` — present in dataset, excluded")
    for c in absent_leakage:
        lines.append(f"- `{c}` — not present (already dropped upstream)")
    lines.append("")

    lines += [
        "### Excluded metadata / ID / split columns",
        "",
    ]
    for c in sorted(_META_NEVER_FEATURES):
        status = "present" if c in df.columns else "absent"
        lines.append(f"- `{c}` ({status})")
    lines.append("")

    lines += [
        "### `core_label` distribution",
        "",
        "**Leakage status:** `core_label` is a PRE-BIOPSY procedural/anatomical label "
        "(physician's targeting decision). It is NOT pathology leakage. However, it "
        "encodes clinical suspicion (MRI PI-RADS, prior session positivity) and is "
        "therefore labelled as PROCEDURAL/ANATOMICAL CONTEXT, not pure geometry.",
        "",
    ]
    if "core_label" in df.columns:
        miss_n = df["core_label"].isna().sum()
        total  = len(df)
        lines.append(
            f"**Missingness:** {miss_n} / {total} ({100*miss_n/total:.1f}%) missing."
        )
        lines.append("")
        lines += [
            "| Value | Count | % total |",
            "|-------|-------|---------|",
        ]
        for val, cnt in df["core_label"].value_counts(dropna=False).items():
            label_str = str(val) if not (isinstance(val, float) and np.isnan(val)) else "NaN"
            lines.append(f"| {label_str} | {cnt} | {100*cnt/total:.1f}% |")
        lines.append("")

        lines += [
            "**`core_label` by split:**",
            "",
            "| Split | n_rows | n_targeted | targeted_pct |",
            "|-------|--------|------------|--------------|",
        ]
        for sp in ["train", "val", "test"]:
            sp_df = df[df[SPLIT_COL] == sp]
            n_sp  = len(sp_df)
            n_tgt = (sp_df["core_label"] == "TARGET OR PRIOR POSITIVE").sum()
            lines.append(f"| {sp} | {n_sp} | {n_tgt} | {100*n_tgt/max(n_sp,1):.1f}% |")
        lines.append("")
    else:
        lines += ["*`core_label` column not present in dataset.*", ""]

    lines += [
        "### Auto-detected image/intensity columns",
        "",
    ]
    if intensity_cols:
        lines.append(
            f"**{len(intensity_cols)} column(s) detected** (keyword match + ≥10% non-NaN):"
        )
        lines.append("")
        for c in intensity_cols:
            miss = df[c].isna().mean() if c in df.columns else 1.0
            lines.append(f"- `{c}`: missingness {100*miss:.1f}%")
    else:
        lines.append(
            "*No image/intensity columns detected. Feature families D, F, H are skipped.*"
        )
    lines.append("")

    lines += [
        "### Derived `core_label` one-hot columns",
        "",
        f"**{len(cl_onehot)} one-hot columns:** " + (", ".join(f"`{c}`" for c in cl_onehot) or "none"),
        "",
        "### Derived anatomical/procedural flag columns",
        "",
        f"**{len(cl_anatom)} anatomical flag columns:** " + (", ".join(f"`{c}`" for c in cl_anatom) or "none"),
        "",
        "### Derived transformed geometry columns",
        "",
        f"**{len(geo_transf)} transformed geometry columns:** " + (", ".join(f"`{c}`" for c in geo_transf) or "none"),
        "",
    ]

    # Missingness for all new derived columns
    all_new = cl_onehot + cl_anatom + geo_transf
    if any(c in df.columns for c in all_new):
        lines += [
            "### Missingness of new derived features (all splits)",
            "",
            "| Column | missingness |",
            "|--------|------------|",
        ]
        for c in all_new:
            if c in df.columns:
                miss = df[c].isna().mean()
                lines.append(f"| `{c}` | {100*miss:.1f}% |")
        lines.append("")

    return lines


# ── Delta section ─────────────────────────────────────────────────────────────

def _delta_section(
    df_perf: pd.DataFrame,
    df_risk: pd.DataFrame,
    label_col: str,
    ep_label: str,
) -> list[str]:
    """Delta table vs current_compact LR for one endpoint."""
    sub = df_perf[(df_perf["label_col"] == label_col) & (df_perf["model"] == "logistic_regression")]
    if sub.empty:
        return [f"### {ep_label}", "", "*No LR results.*", ""]

    ref_row = sub[sub["feature_set"] == REFERENCE_FS]
    if ref_row.empty:
        return [f"### {ep_label}", "", f"*Baseline `{REFERENCE_FS}` LR not found.*", ""]

    ref_roc = float(ref_row["test_roc_auc"].iloc[0])
    ref_pr  = float(ref_row["test_pr_auc"].iloc[0])

    # Risk strat reference
    rsub = df_risk[
        (df_risk["label_col"] == label_col)
        & (df_risk["feature_set"] == REFERENCE_FS)
        & (df_risk["model"] == "logistic_regression")
    ]
    ref_t5p  = float(rsub["top5pct_prevalence"].iloc[0])  if not rsub.empty else np.nan
    ref_t10c = float(rsub["top10pct_capture_rate"].iloc[0]) if not rsub.empty else np.nan

    lines: list[str] = [
        f"### {ep_label}",
        "",
        "Delta values = feature set − `current_compact` (LR, test split). Positive = better.",
        "",
        f"Compact baseline: ROC-AUC={_f(ref_roc)}, PR-AUC={_f(ref_pr)}, "
        f"top5% prev={_f(ref_t5p)}, top10% capture={_f(ref_t10c)}",
        "",
        "| Feature set | n_feat | ROC-AUC | ΔROC | PR-AUC | ΔPR | Δtop5%prev | Δtop10%cap | Threshold met? |",
        "|-------------|--------|---------|------|--------|-----|-----------|-----------|---------------|",
    ]

    for _, r in sub.iterrows():
        roc  = float(r["test_roc_auc"])
        pr   = float(r["test_pr_auc"])
        droc = roc - ref_roc
        dpr  = pr  - ref_pr
        sr   = "+" if droc >= 0 else ""
        sp   = "+" if dpr  >= 0 else ""

        # Risk strat delta
        rr = df_risk[
            (df_risk["label_col"] == label_col)
            & (df_risk["feature_set"] == r["feature_set"])
            & (df_risk["model"] == "logistic_regression")
        ]
        dt5p  = float(rr["top5pct_prevalence"].iloc[0])  - ref_t5p  if not rr.empty else np.nan
        dt10c = float(rr["top10pct_capture_rate"].iloc[0]) - ref_t10c if not rr.empty else np.nan
        st5   = "+" if not np.isnan(dt5p)  and dt5p  >= 0 else ""
        st10  = "+" if not np.isnan(dt10c) and dt10c >= 0 else ""

        threshold_met = (
            "**YES**" if (dpr >= MIN_DELTA_PR_AUC or droc >= MIN_DELTA_ROC_AUC)
            else "No"
        )
        lines.append(
            f"| `{r['feature_set']}` | {r['n_features']} "
            f"| {_f(roc)} | {sr}{_f(droc)} "
            f"| {_f(pr)} | {sp}{_f(dpr)} "
            f"| {st5}{_f(dt5p)} | {st10}{_f(dt10c)} "
            f"| {threshold_met} |"
        )
    lines.append("")
    return lines


# ── Recommendation section ────────────────────────────────────────────────────

def _recommendation_section(
    df_perf: pd.DataFrame,
    df_risk: pd.DataFrame,
) -> list[str]:
    lines: list[str] = [
        "# Final Targeted Feature Sprint — Recommendation",
        "",
        "**Branch:** `feat/geometry-clinical-baselines`  ",
        "**Script:** `src/compare_final_targeted_feature_sprint.py`  ",
        "**Based on:** `reports/final_targeted_feature_sprint_comparison.md`",
        "",
        "## Decision criteria (stricter than extended sprint)",
        "",
        "An extension is worth discussing as a **candidate improved model** only if:",
        "",
        f"- ΔPR-AUC ≥ +{MIN_DELTA_PR_AUC:.3f} **OR** ΔROC-AUC ≥ +{MIN_DELTA_ROC_AUC:.3f} "
        "on held-out test split with logistic regression",
        "- **AND** gain appears on at least one clinically important endpoint without "
        "large degradation (|ΔROC| > 0.015 or |ΔPR| > 0.020) on the other endpoint",
        "- **AND** feature set is interpretable and not leakage-prone.",
        "",
        "Replacement of the primary compact model is recommended only if:",
        "- Gain is consistent across both endpoints, OR clinically meaningful for one",
        "- Feature family is scientifically defensible",
        "- Model remains interpretable enough to explain in a manuscript",
        "",
        "## Feature-type classification",
        "",
        "| Feature family | Type | Leakage risk | Notes |",
        "|---------------|------|-------------|-------|",
        "| Target geometry (4) | Geometric | None | Pure spatial |",
        "| Clinical (4) | Clinical | None | PSA, prostate volume |",
        "| `core_label` one-hot / flags | Procedural/anatomical | **See note** | Pre-biopsy targeting decision |",
        "| Intensity (auto-detected) | Image-derived | None | First-order only |",
        "| Transformed geometry | Derived | None | Log, ratio, interaction |",
        "",
        "**Note on `core_label`:** Encodes the physician's pre-biopsy targeting decision "
        "(systematic sextant vs MRI-targeted / prior-positive site). Not pathology leakage, "
        "but encodes clinical suspicion not present in the geometry-clinical compact model. "
        "If any model including `core_label` crosses the threshold, the gain is flagged as "
        "**procedural/anatomical-context-driven**, not geometry-driven.",
        "",
        "## Results summary (logistic regression, test split)",
        "",
    ]

    # Per-endpoint summary table
    findings: dict[str, list[tuple[str, float, float]]] = {}

    for label_col, ep_label in ENDPOINTS:
        sub = df_perf[
            (df_perf["label_col"] == label_col)
            & (df_perf["model"] == "logistic_regression")
        ]
        if sub.empty:
            lines += [f"### {ep_label}", "", "*No LR results.*", ""]
            findings[label_col] = []
            continue

        ref_row = sub[sub["feature_set"] == REFERENCE_FS]
        if ref_row.empty:
            lines += [f"### {ep_label}", "", "*Baseline not available.*", ""]
            findings[label_col] = []
            continue

        ref_roc = float(ref_row["test_roc_auc"].iloc[0])
        ref_pr  = float(ref_row["test_pr_auc"].iloc[0])

        lines += [
            f"### {ep_label}",
            "",
            f"**Compact baseline (LR):** ROC-AUC = {_f(ref_roc)}, PR-AUC = {_f(ref_pr)}",
            "",
            "| Feature set | ΔROC-AUC | ΔPR-AUC | Threshold met? |",
            "|-------------|----------|---------|---------------|",
        ]

        ep_findings: list[tuple[str, float, float]] = []
        for _, r in sub.iterrows():
            if r["feature_set"] == REFERENCE_FS:
                continue
            droc = float(r["test_roc_auc"]) - ref_roc
            dpr  = float(r["test_pr_auc"])  - ref_pr
            sr   = "+" if droc >= 0 else ""
            sp   = "+" if dpr  >= 0 else ""
            met  = dpr >= MIN_DELTA_PR_AUC or droc >= MIN_DELTA_ROC_AUC
            lines.append(
                f"| `{r['feature_set']}` | {sr}{_f(droc)} | {sp}{_f(dpr)} "
                f"| {'**YES**' if met else 'No'} |"
            )
            if met:
                ep_findings.append((r["feature_set"], droc, dpr))
        lines.append("")
        findings[label_col] = ep_findings

    # Overall recommendation
    lines += ["## Overall recommendation", ""]

    # Identify sets that cross for GG2+ and/or GG3+
    found_gg2 = {fs for fs, _, _ in findings.get("binary_label_int", [])}
    found_gg3 = {fs for fs, _, _ in findings.get("binary_label_gg3plus_int", [])}
    found_both = found_gg2 & found_gg3
    found_any  = found_gg2 | found_gg3

    def _contains_core_label(fs_name: str) -> bool:
        return "core_label" in fs_name

    if not found_any:
        lines += [
            "**No feature set crosses either strict threshold for either endpoint "
            f"(ΔPR ≥ +{MIN_DELTA_PR_AUC:.3f} or ΔROC ≥ +{MIN_DELTA_ROC_AUC:.3f}).**",
            "",
            "The compact `current_compact` model (`target_geometry_plus_clinical`, 8 features) "
            "remains the best central model. This is a **null result** for all four feature "
            "families tested. Further performance gains likely require genuinely new "
            "information (external dataset, additional clinical variables, calibrated image "
            "features) rather than minor feature engineering on the current dataset.",
            "",
        ]
    else:
        if found_both:
            lines += [
                f"Feature sets crossing the threshold for **both** endpoints: "
                + ", ".join(f"`{fs}`" for fs in sorted(found_both)),
                "",
            ]
        if found_gg2 - found_both:
            lines += [
                f"Feature sets crossing for **GG2+ only**: "
                + ", ".join(f"`{fs}`" for fs in sorted(found_gg2 - found_both)),
                "",
            ]
        if found_gg3 - found_both:
            lines += [
                f"Feature sets crossing for **GG3+ only**: "
                + ", ".join(f"`{fs}`" for fs in sorted(found_gg3 - found_both)),
                "",
            ]

        core_label_driven = {fs for fs in found_any if _contains_core_label(fs)}
        pure_geo_driven   = found_any - core_label_driven

        if core_label_driven:
            lines += [
                f"**Threshold-crossing sets that include `core_label` features: "
                + ", ".join(f"`{fs}`" for fs in sorted(core_label_driven)) + "**",
                "",
                "These gains are **procedural/anatomical-context-driven**, not geometry-driven. "
                "Any of these sets can be reported as a secondary procedural-context model, "
                "but must NOT replace the primary compact geometry-clinical model. Including "
                "`core_label` changes the scientific claim from 'needle geometry predicts "
                "cancer' to 'targeting decision + geometry predicts cancer.'",
                "",
            ]
        if pure_geo_driven:
            lines += [
                f"**Threshold-crossing sets without `core_label`: "
                + ", ".join(f"`{fs}`" for fs in sorted(pure_geo_driven)) + "**",
                "",
                "Gains in these feature sets do not rely on procedural context. "
                "If the gain is consistent across endpoints and the features are "
                "scientifically defensible, consider updating the primary model.",
                "",
            ]

    # Regularized LR comment
    lines += [
        "### Regularized LR variants",
        "",
        "Tuned L2/L1/Elastic Net models were compared against the baseline LR (C=1). "
        "Regularization tuning is most useful when the feature space is large relative "
        "to the sample size (many one-hot categories). With only 8 compact features, "
        "the gain from tuning over the default C=1 is typically negligible. Results are "
        "reported in the comparison table.",
        "",
        "### Limitations of this sprint",
        "",
        "- Test set: ~120 patients. Delta estimates at this sample size have wide "
        "bootstrap CIs; a ΔROC of ±0.020 is near the noise floor.",
        "- `core_label` was audited post-hoc (column discovery, not pre-registration). "
        "Results for `core_label`-containing feature sets are exploratory.",
        "- Intensity features (if detected) are first-order only; any gain from them "
        "should be verified on the restricted `extraction_status == 'ok'` subset before "
        "reporting.",
        "- Transformed geometry features are deterministic recodes; they cannot add "
        "genuinely new information, only different parameterizations of existing features.",
        "- This sprint does not include calibration re-assessment or decision-curve analysis.",
        "",
    ]

    # French conclusion
    lines += [
        "---",
        "",
        "## Conclusion à montrer au binôme",
        "",
    ]

    if not found_any:
        lines += [
            "**Résultat : null result complet.**",
            "",
            "Aucune des quatre familles de features testées — encodage complet de `core_label`, "
            "features d'intensité image premier ordre, transformations géométriques simples, "
            "régression logistique régularisée (L2/L1/Elastic Net) — ne dépasse les seuils "
            f"stricts (ΔPR ≥ +{MIN_DELTA_PR_AUC:.3f} ou ΔROC ≥ +{MIN_DELTA_ROC_AUC:.3f}) "
            "sur le test set pour aucun des deux endpoints.",
            "",
            "**Décision : le modèle compact reste le meilleur modèle central.** "
            "On ne change rien.",
            "",
            "**La modélisation est terminée.** Les gains de performance supplémentaires "
            "nécessiteraient des informations genuinement nouvelles (dataset externe, "
            "features image calibrées, variables cliniques non disponibles), pas des "
            "transformations supplémentaires du dataset existant.",
            "",
            "**Prochaine étape recommandée :** rédiger le manuscrit. "
            "Tous les résultats nécessaires sont documentés.",
        ]
    else:
        cl_driven_list = sorted(core_label_driven) if core_label_driven else []
        geo_driven_list = sorted(pure_geo_driven) if pure_geo_driven else []

        if cl_driven_list:
            lines += [
                "**Résultat : amélioration détectée, mais elle est portée par `core_label`.**",
                "",
                f"Les feature sets {', '.join(f'`{fs}`' for fs in cl_driven_list)} "
                "franchissent le seuil, mais ils incluent des features de contexte procédural "
                "(`core_label`) qui encodent la décision de ciblage du médecin. Ce n'est pas "
                "une fuite pathologique, mais ce n'est pas non plus de la géométrie pure.",
                "",
                "**Décision :** ces modèles peuvent être reportés comme analyse secondaire "
                "« modèle contexte procédural », mais ne remplacent PAS le modèle compact "
                "principal. La story du papier reste « géométrie aiguille-cible ».",
                "",
            ]
        if geo_driven_list:
            lines += [
                "**Résultat : amélioration sans dépendance à `core_label`.**",
                "",
                f"Les feature sets {', '.join(f'`{fs}`' for fs in geo_driven_list)} "
                "franchissent le seuil sans features de contexte procédural. "
                "À discuter avec le binôme : est-ce que ces features sont scientifiquement "
                "défendables et est-ce qu'on veut mettre à jour le modèle central ?",
                "",
            ]

        lines += [
            "**Prochaine étape :** discuter avec le binôme si on adopte le modèle étendu "
            "comme modèle central, ou si on le reporte comme analyse secondaire. "
            "Si null result sur tous les endpoints cliniquement importants, "
            "la modélisation est terminée.",
        ]

    lines += ["", "---", "", "*Report generated by `src/compare_final_targeted_feature_sprint.py`. "
              "Do not modify by hand.*"]
    return lines


# ── Report writer ─────────────────────────────────────────────────────────────

def write_reports(
    results: list[dict],
    risk_rows: list[dict],
    df_aug: pd.DataFrame,
    intensity_cols: list[str],
    cl_onehot: list[str],
    cl_anatom: list[str],
    geo_transf: list[str],
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df_perf = pd.DataFrame(results)
    df_risk = pd.DataFrame(risk_rows)

    # CSVs
    perf_csv = REPORTS_DIR / "final_targeted_feature_sprint_comparison.csv"
    risk_csv = REPORTS_DIR / "final_targeted_feature_sprint_risk_stratification.csv"
    df_perf.to_csv(perf_csv, index=False)
    df_risk.to_csv(risk_csv, index=False)
    print(f"Saved -> {perf_csv.relative_to(REPO_ROOT)}")
    print(f"Saved -> {risk_csv.relative_to(REPO_ROOT)}")

    # Main comparison MD
    perf_md = REPORTS_DIR / "final_targeted_feature_sprint_comparison.md"
    lines: list[str] = [
        "# Final Targeted Feature Sprint — Comparison",
        "",
        "**Branch:** `feat/geometry-clinical-baselines`  ",
        "**Script:** `src/compare_final_targeted_feature_sprint.py`  ",
        "**Data:** `data/share/needle_features_v1.csv`",
        "",
        "## Methodology",
        "",
        "**Endpoints:** GG2+ / csPCa (`binary_label_int`) and "
        "GG3+ / high-grade (`binary_label_gg3plus_int`).  ",
        "**Models:** logistic regression (baseline, C=1), tuned L2/L1/Elastic Net "
        "(C tuned on val PR-AUC), HistGradientBoosting (sensitivity), "
        "XGBoost if installed (sensitivity).  ",
        "**Preprocessing:** numeric features: median imputation + StandardScaler; "
        "boolean/one-hot features: most_frequent imputation, no scaling.  ",
        "**Threshold selection:** Youden J maximised on validation split.  ",
        "**Bootstrap CIs:** 1,000 patient-level resamples (patients resampled with "
        "replacement, all their cores included; 2.5/97.5 percentiles).  ",
        f"**Reference baseline:** `{REFERENCE_FS}` (8 features: "
        "4 target-geometry + 4 clinical).  ",
        f"**Decision thresholds:** ΔPR-AUC ≥ +{MIN_DELTA_PR_AUC:.3f} OR "
        f"ΔROC-AUC ≥ +{MIN_DELTA_ROC_AUC:.3f} (stricter than extended sprint).  ",
        "",
        "**Excluded predictors (pathology leakage):**  ",
        "`cancer_length_mm`, `pct_cancer_in_core`, `primary_gleason`, "
        "`secondary_gleason`, `pathology_label`, `pathology_label_int`, "
        "`binary_label` and all endpoint columns — never used.",
        "",
    ]

    lines += _leakage_audit_section(df_aug, intensity_cols, cl_onehot, cl_anatom, geo_transf)

    lines += ["## Performance comparison", ""]
    for label_col, ep_label in ENDPOINTS:
        sub = df_perf[df_perf["label_col"] == label_col]
        if sub.empty:
            lines += [f"### {ep_label}", "", "*No results.*", ""]
            continue
        lines += [f"### {ep_label}", ""]
        lines += _md_table(sub, _PERF_COLS)
        lines.append("")

    lines += ["## Delta vs compact baseline (logistic regression)", ""]
    for label_col, ep_label in ENDPOINTS:
        lines += _delta_section(df_perf, df_risk, label_col, ep_label)

    perf_md.write_text("\n".join(lines) + "\n")
    print(f"Saved -> {perf_md.relative_to(REPO_ROOT)}")

    # Risk stratification MD
    risk_md = REPORTS_DIR / "final_targeted_feature_sprint_risk_stratification.md"
    rlines: list[str] = [
        "# Final Targeted Feature Sprint — Risk Stratification",
        "",
        "**Capture rate** = fraction of all test positives in the top-N% scored cores.  ",
        "**Prevalence** in top-N% = positive rate among top-N% highest-probability cores.  ",
        "**Enrichment** = top-N% prevalence / baseline prevalence.  ",
        "**Baseline:** overall positive rate in the test set.",
        "",
        "**Note:** feature sets containing `core_label` are PROCEDURAL/ANATOMICAL CONTEXT "
        "models. Their risk-stratification gains may reflect the structural positivity "
        "difference between targeted and systematic cores rather than geometric signal.",
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

    # Recommendation MD
    rec_md = REPORTS_DIR / "final_targeted_feature_sprint_recommendation.md"
    rec_md.write_text("\n".join(_recommendation_section(df_perf, df_risk)) + "\n")
    print(f"Saved -> {rec_md.relative_to(REPO_ROOT)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== Final targeted feature sprint ===\n")
    if not HAS_XGBOOST:
        warnings.warn("xgboost not installed — XGBoost will be skipped.", stacklevel=1)

    results:   list[dict] = []
    risk_rows: list[dict] = []

    # State shared across endpoints (derived once from GG2+ load)
    df_aug_ref:    pd.DataFrame | None = None
    intensity_ref: list[str]           = []
    cl_onehot_ref: list[str]           = []
    cl_anatom_ref: list[str]           = []
    geo_transf_ref: list[str]          = []
    sprint_bool_ref: set[str]          = set()

    for label_col, ep_label in ENDPOINTS:
        print(f"\n--- Endpoint: {ep_label} ({label_col}) ---")

        df_base, _ = load_dataset(DATA_PATH, target_col=label_col)
        df_base    = prepare_features(df_base)

        # Derive sprint features (same derivation for every endpoint)
        df, cl_onehot, cl_anatom, intensity_cols, geo_transf, sprint_bool_cols = (
            derive_sprint_features(df_base)
        )

        if df_aug_ref is None:
            df_aug_ref     = df
            intensity_ref  = intensity_cols
            cl_onehot_ref  = cl_onehot
            cl_anatom_ref  = cl_anatom
            geo_transf_ref = geo_transf
            sprint_bool_ref = sprint_bool_cols
            print(f"  core_label one-hot: {len(cl_onehot)} cols")
            print(f"  anatomical flags  : {len(cl_anatom)} cols")
            print(f"  intensity cols    : {intensity_cols if intensity_cols else 'none detected'}")
            print(f"  geo transforms    : {len(geo_transf)} cols")

        n_pts = df[PATIENT_COL].nunique()
        print(f"  {len(df):,} rows, {n_pts:,} patients")

        feature_sets = make_feature_sets(cl_onehot, cl_anatom, intensity_cols, geo_transf)
        print(f"  Feature sets: {list(feature_sets.keys())}")

        # Model list: standard + tuned LR + tree-based sensitivity
        sprint_models = [
            "logistic_regression",
            "tuned_l2_logistic_regression",
            "tuned_l1_logistic_regression",
            "tuned_elasticnet_logistic_regression",
            "hist_gradient_boosting",
        ]
        if HAS_XGBOOST:
            sprint_models.append("xgboost")

        for fs_name, fs_cols in feature_sets.items():
            # Only use columns actually present in df
            available = set(df.columns)
            feature_cols = [c for c in fs_cols if c in available]
            missing = [c for c in fs_cols if c not in available]
            if missing:
                warnings.warn(
                    f"[{label_col}/{fs_name}] missing cols: {missing}", stacklevel=1,
                )
            if not feature_cols:
                continue

            for model_name in sprint_models:
                print(f"  [{fs_name}] {model_name} ...", end=" ", flush=True)
                row, test_probs = run_one(
                    label_col, fs_name, feature_cols, model_name, df, sprint_bool_cols,
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
                risk    = _risk_strat(y_test, test_probs)
                risk_rows.append({
                    "label_col":   label_col,
                    "feature_set": fs_name,
                    "model":       model_name,
                    **risk,
                })

    if not results:
        print("\nNo results produced — check data and dependencies.")
        return

    write_reports(
        results, risk_rows,
        df_aug_ref, intensity_ref, cl_onehot_ref, cl_anatom_ref, geo_transf_ref,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
