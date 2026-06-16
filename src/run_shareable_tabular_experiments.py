"""
Tabular experiments on the shareable processed dataset.

Input
-----
data/share/needle_features_v1.csv
    One row per biopsy core, with geometry/clinical/intensity features
    plus a binary label and a patient-level train/val/test split.

Two experiment modes
---------------------
MODE A — full_geometry_dataset
    All labelled, coordinate-matched rows with a valid split.
    Geometry and clinical features are available for almost every core,
    independent of whether voxel intensities were extracted.

MODE B — extracted_voxel_subset
    Same rows, restricted to extraction_status == "ok" — the subset for
    which centerline intensity features are also available.

Common row filters (both modes)
--------------------------------
    label_join_status == "coord_match"
    binary_label_int is not null
    split in {"train", "val", "test"}

Outputs
-------
reports/shareable_tabular_results.csv
reports/shareable_tabular_results.md
reports/shareable_tabular_dataset_summary.md

Usage
-----
    python src/run_shareable_tabular_experiments.py
    python src/run_shareable_tabular_experiments.py --label-column binary_label_gg3plus_int
    python src/run_shareable_tabular_experiments.py --n-boot 200
    python src/run_shareable_tabular_experiments.py --no-bootstrap
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


# ── Paths & constants ────────────────────────────────────────────────────────

REPO_ROOT   = Path(__file__).resolve().parent.parent
DATA_PATH   = REPO_ROOT / "data" / "share" / "needle_features_v1.csv"
REPORTS_DIR = REPO_ROOT / "reports"

RANDOM_STATE = 42
TARGET_COL   = "binary_label_int"
PATIENT_COL  = "patient_number"
SPLIT_COL    = "split"
VALID_SPLITS = ("train", "val", "test")

# ── Feature groups ───────────────────────────────────────────────────────────

CLINICAL_FEATURES = [
    "psa_ng_ml",
    "log_psa_ng_ml",
    "prostate_volume_cc",
    "psa_density",
]

BIOPSY_GEOMETRY_FEATURES = [
    "core_length_mm",
    "n_centerline_voxels",
    "n_tube_voxels",
    "midpoint_inside_prostate",
    "distance_midpoint_to_prostate_surface_mm",
    "approximate_fraction_of_centerline_inside_prostate",
]

TARGET_GEOMETRY_FEATURES = [
    "target_mesh_available",
    "distance_midpoint_to_target_centroid_mm",
    "distance_midpoint_to_target_surface_mm",
    "trajectory_intersects_target",
    "approximate_fraction_of_centerline_inside_target",
]

# Same four geometric measurements without the availability flag.
# Used to assess whether target_mesh_available itself drives model performance.
TARGET_GEOMETRY_NO_AVAILABILITY_FEATURES = [
    "distance_midpoint_to_target_centroid_mm",
    "distance_midpoint_to_target_surface_mm",
    "trajectory_intersects_target",
    "approximate_fraction_of_centerline_inside_target",
]

CENTERLINE_INTENSITY_FEATURES = [
    "centerline_intensity_mean",
    "centerline_intensity_std",
    "centerline_intensity_p25",
    "centerline_intensity_p75",
]

# Features that are True/False flags rather than continuous measurements —
# imputed by most-frequent value and cast to 0/1, never standard-scaled.
BOOLEAN_FEATURES = {
    "midpoint_inside_prostate",
    "target_mesh_available",
    "trajectory_intersects_target",
}

MODE_A_EXPERIMENTS = {
    "clinical_only":                               ["clinical"],
    "biopsy_geometry_only":                        ["biopsy_geometry"],
    "target_geometry_only":                        ["target_geometry"],
    "target_geometry_no_availability":             ["target_geometry_no_availability"],
    "all_geometry":                                ["biopsy_geometry", "target_geometry"],
    "all_geometry_no_availability":                ["biopsy_geometry", "target_geometry_no_availability"],
    "all_geometry_plus_clinical":                  ["biopsy_geometry", "target_geometry", "clinical"],
    "all_geometry_no_availability_plus_clinical":  ["biopsy_geometry", "target_geometry_no_availability", "clinical"],
}

MODE_B_EXPERIMENTS = {
    "centerline_intensity_only":                       ["centerline_intensity"],
    "clinical_only":                                   ["clinical"],
    "target_geometry_only":                            ["target_geometry"],
    "target_geometry_no_availability":                 ["target_geometry_no_availability"],
    "all_geometry":                                    ["biopsy_geometry", "target_geometry"],
    "all_geometry_no_availability":                    ["biopsy_geometry", "target_geometry_no_availability"],
    "centerline_plus_geometry":                        ["centerline_intensity", "biopsy_geometry", "target_geometry"],
    "centerline_plus_geometry_plus_clinical":          ["centerline_intensity", "biopsy_geometry", "target_geometry", "clinical"],
    "centerline_plus_geometry_no_availability_plus_clinical": ["centerline_intensity", "biopsy_geometry", "target_geometry_no_availability", "clinical"],
}

METRIC_KEYS = ["roc_auc", "pr_auc", "sensitivity", "specificity", "precision", "recall", "f1"]


# ── 1. Loading & feature preparation ────────────────────────────────────────

def load_dataset(
    path: Path = DATA_PATH,
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, int]:
    """
    Load the shareable dataset and apply the common row filters.

    Returns
    -------
    df       : filtered DataFrame (coord_match, labelled, valid split)
    n_total  : number of rows in the raw file, before filtering
    """
    raw = pd.read_csv(path)
    n_total = len(raw)

    if target_col not in raw.columns:
        raise ValueError(
            f"--label-column '{target_col}' not found in {path.name}. "
            f"Available columns: {sorted(raw.columns.tolist())}"
        )

    df = raw[
        (raw["label_join_status"] == "coord_match")
        & raw[target_col].notna()
        & raw[SPLIT_COL].isin(VALID_SPLITS)
    ].copy()

    df[target_col] = df[target_col].astype(int)
    df = df.reset_index(drop=True)
    return df, n_total


def _to_binary(series: pd.Series) -> pd.Series:
    """Map True/False-like values (bool, 'True'/'False', 1/0) to float 0/1, NaN otherwise."""
    truthy  = {True, "True", "TRUE", "true", 1, "1", 1.0}
    falsy   = {False, "False", "FALSE", "false", 0, "0", 0.0}
    out = pd.Series(np.nan, index=series.index, dtype=float)
    out[series.isin(truthy)] = 1.0
    out[series.isin(falsy)] = 0.0
    return out


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features and normalise boolean columns to 0/1 floats."""
    df = df.copy()

    # ── Derived clinical features ────────────────────────────────────────────
    df["log_psa_ng_ml"] = np.log1p(df["psa_ng_ml"])

    valid_volume = df["prostate_volume_cc"].notna() & (df["prostate_volume_cc"] > 0)
    df["psa_density"] = np.where(
        valid_volume & df["psa_ng_ml"].notna(),
        df["psa_ng_ml"] / df["prostate_volume_cc"],
        np.nan,
    )

    # ── Boolean -> 0/1 ────────────────────────────────────────────────────────
    for col in BOOLEAN_FEATURES:
        if col in df.columns:
            df[col] = _to_binary(df[col])

    return df


# ── 2. Feature groups & experiment definitions ──────────────────────────────

def get_feature_groups() -> dict[str, list[str]]:
    return {
        "clinical":                       CLINICAL_FEATURES,
        "biopsy_geometry":                BIOPSY_GEOMETRY_FEATURES,
        "target_geometry":                TARGET_GEOMETRY_FEATURES,
        "target_geometry_no_availability": TARGET_GEOMETRY_NO_AVAILABILITY_FEATURES,
        "centerline_intensity":           CENTERLINE_INTENSITY_FEATURES,
    }


def resolve_feature_list(group_names: list[str], available_cols: set[str]) -> list[str]:
    """Flatten a list of feature-group names into a deduplicated column list,
    dropping any columns absent from the dataset (with a warning)."""
    groups = get_feature_groups()
    cols: list[str] = []
    for g in group_names:
        for c in groups[g]:
            if c not in available_cols:
                warnings.warn(f"Feature '{c}' (group '{g}') not found in dataset — skipping.")
                continue
            if c not in cols:
                cols.append(c)
    return cols


# ── 3. Model & preprocessing ─────────────────────────────────────────────────

def build_preprocessor(feature_cols: list[str], scale_numeric: bool) -> ColumnTransformer:
    """
    Numeric features  -> median imputation (+ standard scaling if scale_numeric)
    Boolean features  -> most-frequent imputation, kept as 0/1 (never scaled)
    """
    numeric_cols = [c for c in feature_cols if c not in BOOLEAN_FEATURES]
    bool_cols    = [c for c in feature_cols if c in BOOLEAN_FEATURES]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)

    bool_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if bool_cols:
        transformers.append(("bool", bool_pipeline, bool_cols))

    return ColumnTransformer(transformers)


def get_model_specs(y_train: pd.Series) -> list[tuple[str, object, bool]]:
    """Return list of (name, estimator, needs_scaling) for available models."""
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    specs = [
        ("logistic_regression",
         LogisticRegression(class_weight="balanced", max_iter=5000, C=1.0),
         True),
        ("hist_gradient_boosting",
         HistGradientBoostingClassifier(random_state=RANDOM_STATE),
         False),
    ]

    if HAS_XGBOOST:
        specs.append((
            "xgboost",
            XGBClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, scale_pos_weight=scale_pos_weight,
                eval_metric="auc", random_state=RANDOM_STATE, verbosity=0,
            ),
            False,
        ))

    return specs


def build_model(model_name: str, feature_cols: list[str], y_train: pd.Series) -> tuple[Pipeline, object] | None:
    """Build a (preprocessor + estimator) Pipeline for the given model name."""
    specs = {name: (est, scale) for name, est, scale in get_model_specs(y_train)}
    if model_name not in specs:
        return None
    estimator, needs_scaling = specs[model_name]
    preprocessor = build_preprocessor(feature_cols, scale_numeric=needs_scaling)
    return Pipeline([("preprocess", preprocessor), ("clf", estimator)])


# ── 4. Evaluation ─────────────────────────────────────────────────────────────

def select_threshold_youden(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Pick the probability threshold maximizing Youden's J = sensitivity + specificity - 1."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    j = tpr - fpr
    thr = thresholds[int(np.argmax(j))]
    return float(np.clip(thr, 0.0, 1.0))


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    preds = (probs >= threshold).astype(int)

    if len(np.unique(y_true)) < 2:
        roc_auc = np.nan
        pr_auc = np.nan
    else:
        roc_auc = roc_auc_score(y_true, probs)
        pr_auc = average_precision_score(y_true, probs)

    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
    }


def evaluate_model(pipeline: Pipeline, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Fit on train, select a threshold on val (Youden J), evaluate once on test.

    Returns (test_metrics, val_metrics, threshold).
    """
    pipeline.fit(X_train, y_train)

    val_probs  = pipeline.predict_proba(X_val)[:, 1]
    test_probs = pipeline.predict_proba(X_test)[:, 1]

    threshold = select_threshold_youden(y_val.values, val_probs)

    val_metrics  = compute_metrics(y_val.values, val_probs, threshold)
    test_metrics = compute_metrics(y_test.values, test_probs, threshold)

    return test_metrics, val_metrics, threshold


# ── 5. Bootstrap (patient-level resampling) ─────────────────────────────────

def bootstrap_patient_ci(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    patient_ids: np.ndarray,
    threshold: float,
    n_boot: int = 500,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    95% percentile CIs for test-set metrics, resampling *patients* (with
    replacement) rather than individual rows so that all cores from a
    patient move together.
    """
    rng = np.random.default_rng(random_state)

    probs = pipeline.predict_proba(X_test)[:, 1]
    y_arr = y_test.values
    patient_ids = np.asarray(patient_ids)

    unique_patients = np.unique(patient_ids)
    idx_by_patient = {p: np.where(patient_ids == p)[0] for p in unique_patients}

    samples = {k: [] for k in METRIC_KEYS}
    for _ in range(n_boot):
        sampled_patients = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        idx = np.concatenate([idx_by_patient[p] for p in sampled_patients])

        y_b = y_arr[idx]
        if len(np.unique(y_b)) < 2:
            continue

        m = compute_metrics(y_b, probs[idx], threshold)
        for k in METRIC_KEYS:
            samples[k].append(m[k])

    ci = {}
    for k, vals in samples.items():
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            ci[f"{k}_ci_low"] = float(np.percentile(vals, 2.5))
            ci[f"{k}_ci_high"] = float(np.percentile(vals, 97.5))
        else:
            ci[f"{k}_ci_low"] = np.nan
            ci[f"{k}_ci_high"] = np.nan
    return ci


# ── 6. Logistic-regression coefficient extraction ────────────────────────────

def _lr_feature_order(feature_cols: list[str]) -> list[str]:
    """
    Return feature names in the order they appear in the ColumnTransformer
    output produced by build_preprocessor: numeric columns first (in their
    original order), then boolean columns (in their original order).
    This mirrors the transformer list built in build_preprocessor exactly.
    """
    numeric_cols = [c for c in feature_cols if c not in BOOLEAN_FEATURES]
    bool_cols    = [c for c in feature_cols if c in BOOLEAN_FEATURES]
    return numeric_cols + bool_cols


def extract_lr_coefficients(
    pipeline: Pipeline,
    feature_cols: list[str],
    mode: str,
    experiment: str,
    label_col: str,
) -> list[dict]:
    """
    Extract standardised logistic-regression coefficients from a fitted pipeline.

    Coefficients for numeric features are in units of one standard deviation
    of each feature (because StandardScaler precedes the classifier).
    Boolean features are unscaled (coefficient = log-odds change for 0→1).

    Returns one dict per feature, sorted by descending coefficient value.
    """
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return []

    coef = clf.coef_[0]
    ordered_names = _lr_feature_order(feature_cols)

    if len(coef) != len(ordered_names):
        warnings.warn(
            f"[{mode}/{experiment}] coefficient count ({len(coef)}) does not "
            f"match expected feature count ({len(ordered_names)}) — skipping "
            f"coefficient export for this experiment."
        )
        return []

    rows = []
    for name, c in zip(ordered_names, coef):
        rows.append(dict(
            mode=mode,
            experiment=experiment,
            label_col=label_col,
            feature=name,
            coefficient=float(c),
            scaled=(name not in BOOLEAN_FEATURES),
        ))
    rows.sort(key=lambda r: r["coefficient"], reverse=True)
    for rank, r in enumerate(rows, 1):
        r["rank_by_value"] = rank
    return rows


# ── 7. Running a single experiment ──────────────────────────────────────────

def split_summary(df: pd.DataFrame, target_col: str = TARGET_COL) -> dict:
    """Per-split row count, patient count, and label prevalence."""
    out = {}
    for split in VALID_SPLITS:
        sub = df[df[SPLIT_COL] == split]
        out[f"{split}_n_rows"] = len(sub)
        out[f"{split}_n_patients"] = int(sub[PATIENT_COL].nunique())
        out[f"{split}_prevalence"] = float(sub[target_col].mean()) if len(sub) else np.nan
    return out


def run_experiment(
    mode_name: str,
    experiment_name: str,
    model_name: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    do_bootstrap: bool,
    n_boot: int,
    target_col: str = TARGET_COL,
    coef_accumulator: list[dict] | None = None,
) -> dict | None:
    """Run one (mode, experiment, model) combination and return a results row.

    If coef_accumulator is provided and the model is logistic_regression,
    extracted coefficient rows are appended to it in-place.
    """
    train = df[df[SPLIT_COL] == "train"]
    val   = df[df[SPLIT_COL] == "val"]
    test  = df[df[SPLIT_COL] == "test"]

    # Drop feature columns that are entirely missing in the train split: a
    # median/most-frequent imputer has nothing to impute from, and sklearn's
    # SimpleImputer silently drops such columns (keep_empty_features=False),
    # which would desync the ColumnTransformer output from the downstream
    # scaler/estimator and crash. Skip the experiment if nothing remains.
    usable_cols = [c for c in feature_cols if train[c].notna().any()]
    dropped_cols = [c for c in feature_cols if c not in usable_cols]
    if dropped_cols:
        warnings.warn(
            f"[{mode_name}/{experiment_name}/{model_name}] feature(s) entirely "
            f"missing in the train split, dropping: {dropped_cols}"
        )
    feature_cols = usable_cols

    if not feature_cols:
        warnings.warn(
            f"[{mode_name}/{experiment_name}/{model_name}] no usable features "
            f"remain after dropping all-missing columns — skipping."
        )
        return None

    X_train, y_train = train[feature_cols], train[target_col]
    X_val,   y_val   = val[feature_cols],   val[target_col]
    X_test,  y_test  = test[feature_cols],  test[target_col]

    pipeline = build_model(model_name, feature_cols, y_train)
    if pipeline is None:
        return None

    test_metrics, val_metrics, threshold = evaluate_model(
        pipeline, X_train, y_train, X_val, y_val, X_test, y_test
    )

    if coef_accumulator is not None and model_name == "logistic_regression":
        coef_rows = extract_lr_coefficients(
            pipeline, feature_cols, mode_name, experiment_name, target_col
        )
        coef_accumulator.extend(coef_rows)

    row = {
        "mode": mode_name,
        "experiment": experiment_name,
        "model": model_name,
        "n_features": len(feature_cols),
        "features": ";".join(feature_cols),
        "threshold_youden_val": threshold,
    }
    row.update(split_summary(df, target_col=target_col))
    row.update({f"val_{k}": v for k, v in val_metrics.items()})
    row.update({f"test_{k}": v for k, v in test_metrics.items()})

    if do_bootstrap and len(test) > 0:
        ci = bootstrap_patient_ci(
            pipeline, X_test, y_test, test[PATIENT_COL].values, threshold, n_boot=n_boot
        )
        row.update(ci)

    return row


# ── 7. Reporting ──────────────────────────────────────────────────────────────

def write_lr_coefficients(
    coef_rows: list[dict],
    csv_path: Path,
    md_path: Path,
    top_n: int = 10,
):
    """
    Write logistic-regression coefficients to CSV and Markdown.

    CSV  — one row per (mode, experiment, feature), suitable for downstream
           analysis or plotting.
    MD   — human-readable summary showing the top positive and top negative
           coefficients for each experiment, sorted by descending |coefficient|
           within each group.

    Coefficients for numeric features are standardised (one unit = one
    standard deviation after StandardScaler).  Boolean feature coefficients
    are in natural 0→1 units (unscaled).  A positive coefficient means higher
    predicted probability of label = 1.
    """
    if not coef_rows:
        print("  [lr coefficients] no logistic-regression results to export — skipping.")
        return

    df = pd.DataFrame(coef_rows)
    df.to_csv(csv_path, index=False)
    print(f"Saved -> {csv_path.relative_to(REPO_ROOT)}")

    def _sign(v: float) -> str:
        return "▲" if v >= 0 else "▼"

    lines = [
        "# Logistic Regression Coefficients",
        "",
        "Coefficients are **standardised** for numeric features (one unit = one σ  ",
        "after `StandardScaler`) and in natural 0→1 units for boolean features.  ",
        "A **positive** coefficient increases the predicted probability of `label = 1`.",
        "",
    ]

    for mode in df["mode"].unique():
        lines += [f"## {mode}", ""]
        mode_df = df[df["mode"] == mode]

        for experiment in mode_df["experiment"].unique():
            exp_df = mode_df[mode_df["experiment"] == experiment].copy()
            exp_df = exp_df.sort_values("coefficient", ascending=False)

            lines += [f"### {experiment}", ""]

            pos = exp_df[exp_df["coefficient"] >= 0].head(top_n)
            neg = exp_df[exp_df["coefficient"] < 0].tail(top_n).iloc[::-1]  # most negative first

            for label, subset in [("Top positive", pos), ("Top negative", neg)]:
                if subset.empty:
                    continue
                lines += [
                    f"**{label} coefficients** (→ higher probability of label = 1  "
                    + ("↑)" if "positive" in label else "↓)"),
                    "",
                    "| Feature | Coefficient | Scaled? |",
                    "|---|---|---|",
                ]
                for _, r in subset.iterrows():
                    scaled_flag = "std-scaled" if r["scaled"] else "0/1 flag"
                    lines.append(
                        f"| `{r['feature']}` "
                        f"| {_sign(r['coefficient'])} {abs(r['coefficient']):.4f} "
                        f"| {scaled_flag} |"
                    )
                lines.append("")

            # Full ranked table
            lines += [
                "<details><summary>All coefficients (ranked)</summary>",
                "",
                "| Rank | Feature | Coefficient | Scaled? |",
                "|---|---|---|---|",
            ]
            for i, (_, r) in enumerate(exp_df.iterrows(), 1):
                scaled_flag = "std-scaled" if r["scaled"] else "0/1 flag"
                lines.append(
                    f"| {i} | `{r['feature']}` "
                    f"| {_sign(r['coefficient'])} {abs(r['coefficient']):.4f} "
                    f"| {scaled_flag} |"
                )
            lines += ["", "</details>", ""]

    md_path.write_text("\n".join(lines) + "\n")
    print(f"Saved -> {md_path.relative_to(REPO_ROOT)}")


def write_results(
    results: list[dict],
    csv_path: Path,
    md_path: Path,
    label_col: str = TARGET_COL,
):
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"Saved -> {csv_path.relative_to(REPO_ROOT)}")

    summary_cols = [
        "experiment", "model", "n_features",
        "test_roc_auc", "test_pr_auc",
        "test_sensitivity", "test_specificity",
        "test_precision", "test_recall", "test_f1",
        "threshold_youden_val",
    ]

    lines = [
        "# Shareable Tabular Experiment Results",
        "",
        f"Label column: `{label_col}`",
        "",
    ]
    for mode in df["mode"].unique():
        lines += ["", f"## {mode}", "", "| " + " | ".join(summary_cols) + " |",
                  "|" + "---|" * len(summary_cols)]
        sub = df[df["mode"] == mode]
        for _, r in sub.iterrows():
            cells = []
            for c in summary_cols:
                v = r[c]
                cells.append(f"{v:.3f}" if isinstance(v, float) else str(v))
            lines.append("| " + " | ".join(cells) + " |")

    md_path.write_text("\n".join(lines) + "\n")
    print(f"Saved -> {md_path.relative_to(REPO_ROOT)}")


def write_dataset_summary(
    n_total: int,
    df: pd.DataFrame,
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    path: Path,
    target_col: str = TARGET_COL,
):
    lines = [
        "# Shareable Tabular Dataset Summary",
        "",
        f"Source: `data/share/{DATA_PATH.name}`",
        f"Label column: `{target_col}`",
        "",
        f"- Total rows in file: {n_total:,}",
        f"- Rows after common filters (label_join_status == 'coord_match', "
        f"`{target_col}` not null, split in {VALID_SPLITS}): {len(df):,}",
        "",
    ]

    for mode_name, d in [
        ("MODE A — full_geometry_dataset (no extraction_status filter)", df_a),
        ("MODE B — extracted_voxel_subset (extraction_status == 'ok')", df_b),
    ]:
        lines += ["", f"## {mode_name}", "", f"- Total rows: {len(d):,}",
                  f"- Total patients: {d[PATIENT_COL].nunique():,}", ""]
        lines += [f"| Split | Rows | Patients | `{target_col}` prevalence |", "|---|---|---|---|"]
        for split in VALID_SPLITS:
            sub = d[d[SPLIT_COL] == split]
            prev = f"{sub[target_col].mean():.3f}" if len(sub) else "n/a"
            lines.append(f"| {split} | {len(sub):,} | {sub[PATIENT_COL].nunique():,} | {prev} |")

    path.write_text("\n".join(lines) + "\n")
    print(f"Saved -> {path.relative_to(REPO_ROOT)}")


def print_best_models(results: list[dict]):
    df = pd.DataFrame(results)
    if df.empty:
        print("No experiments produced results.")
        return

    print("\n=== Best model per experiment (by test ROC-AUC) ===")
    for mode in df["mode"].unique():
        print(f"\n--- {mode} ---")
        sub = df[df["mode"] == mode]
        for experiment in sub["experiment"].unique():
            exp_rows = sub[sub["experiment"] == experiment]
            best = exp_rows.loc[exp_rows["test_roc_auc"].idxmax()]
            print(f"  {experiment:<40s} best={best['model']:<22s} "
                  f"test ROC-AUC={best['test_roc_auc']:.3f}  "
                  f"PR-AUC={best['test_pr_auc']:.3f}  "
                  f"sens={best['test_sensitivity']:.3f}  spec={best['test_specificity']:.3f}")

    print("\n=== Overall best per mode ===")
    for mode in df["mode"].unique():
        sub = df[df["mode"] == mode]
        best = sub.loc[sub["test_roc_auc"].idxmax()]
        print(f"  {mode}: {best['experiment']} / {best['model']}  "
              f"test ROC-AUC={best['test_roc_auc']:.3f}  PR-AUC={best['test_pr_auc']:.3f}")


# ── 8. Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run tabular experiments on the shareable dataset.")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument(
        "--label-column", default=TARGET_COL,
        help=(
            "Target column to use as y. Must be binary (0/1) and present in the CSV. "
            f"Default: {TARGET_COL}. "
            "Example: --label-column binary_label_gg3plus_int"
        ),
    )
    parser.add_argument("--n-boot", type=int, default=500,
                         help="Number of patient-level bootstrap resamples on the test set.")
    parser.add_argument("--no-bootstrap", action="store_true",
                         help="Skip bootstrap confidence intervals (faster).")
    args = parser.parse_args()

    label_col = args.label_column

    print("=== Shareable tabular experiments ===\n")
    print(f"Label column : {label_col}")
    if not HAS_XGBOOST:
        print("Note: xgboost not installed — XGBoost models will be skipped.")
    print(f"Loading {args.data_path.relative_to(REPO_ROOT)} ...")
    df, n_total = load_dataset(args.data_path, target_col=label_col)
    df = prepare_features(df)
    print(f"  {n_total:,} total rows -> {len(df):,} rows after filters "
          f"({df[PATIENT_COL].nunique():,} patients)\n")

    available_cols = set(df.columns)

    df_a = df  # MODE A: no extraction_status filter
    df_b = df[df["extraction_status"] == "ok"].reset_index(drop=True)
    print(f"MODE A (full_geometry_dataset)   : {len(df_a):,} rows, {df_a[PATIENT_COL].nunique():,} patients")
    print(f"MODE B (extracted_voxel_subset)  : {len(df_b):,} rows, {df_b[PATIENT_COL].nunique():,} patients\n")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Output filenames include the label column so parallel runs don't overwrite each other.
    summary_path = REPORTS_DIR / f"shareable_tabular_dataset_summary_{label_col}.md"
    results_csv  = REPORTS_DIR / f"shareable_tabular_results_{label_col}.csv"
    results_md   = REPORTS_DIR / f"shareable_tabular_results_{label_col}.md"

    write_dataset_summary(n_total, df, df_a, df_b, summary_path, target_col=label_col)

    do_bootstrap = not args.no_bootstrap
    results: list[dict]   = []
    lr_coef_rows: list[dict] = []

    for mode_name, mode_df, experiments in [
        ("full_geometry_dataset", df_a, MODE_A_EXPERIMENTS),
        ("extracted_voxel_subset", df_b, MODE_B_EXPERIMENTS),
    ]:
        print(f"\n--- Running mode: {mode_name} ---")
        model_names = [name for name, _, _ in get_model_specs(mode_df[label_col])]

        for experiment_name, group_names in experiments.items():
            feature_cols = resolve_feature_list(group_names, available_cols)
            if not feature_cols:
                print(f"  [{experiment_name}] no usable features — skipping.")
                continue

            for model_name in model_names:
                print(f"  [{experiment_name}] {model_name} ...")
                row = run_experiment(
                    mode_name, experiment_name, model_name,
                    mode_df, feature_cols, do_bootstrap, args.n_boot,
                    target_col=label_col,
                    coef_accumulator=lr_coef_rows,
                )
                if row is not None:
                    row["label_column"] = label_col
                    results.append(row)

    write_results(results, results_csv, results_md, label_col=label_col)

    lr_coef_csv = REPORTS_DIR / f"logistic_coefficients_{label_col}.csv"
    lr_coef_md  = REPORTS_DIR / f"logistic_coefficients_{label_col}.md"
    write_lr_coefficients(lr_coef_rows, lr_coef_csv, lr_coef_md)

    print_best_models(results)


if __name__ == "__main__":
    main()
