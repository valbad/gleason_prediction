"""
Risk stratification analysis for the all_geometry_no_availability_plus_clinical
experiment on the full_geometry_dataset split.

For each endpoint the model is trained on the train split and predicts
uncalibrated probabilities on the test split.  No hard threshold is applied.

Analyses
--------
1. Risk deciles (decile 10 = highest risk, decile 1 = lowest risk):
   n_cores, n_positives, observed_prevalence, mean/min/max predicted prob.
2. Positive capture: fraction of all positives contained in the top
   5%, 10%, and 20% highest-risk cores.
3. Calibration table: 10 equal-width probability bins with observed
   prevalence — suitable for downstream plotting.

Models run
----------
binary_label_int         : logistic_regression + hist_gradient_boosting
                           + xgboost (if installed)
binary_label_gg3plus_int : logistic_regression

Feature set (mirrors all_geometry_no_availability_plus_clinical in
run_shareable_tabular_experiments.py — target_mesh_available excluded):
  biopsy_geometry    : core_length_mm, n_centerline_voxels, n_tube_voxels,
                       midpoint_inside_prostate,
                       distance_midpoint_to_prostate_surface_mm,
                       approximate_fraction_of_centerline_inside_prostate
  target_geometry    : distance_midpoint_to_target_centroid_mm,
  (no availability)    distance_midpoint_to_target_surface_mm,
                       trajectory_intersects_target,
                       approximate_fraction_of_centerline_inside_target
  clinical           : psa_ng_ml, log_psa_ng_ml, prostate_volume_cc,
                       psa_density

Outputs
-------
  reports/risk_stratification.csv  — tidy rows; "table" column tags each row
                                     as "decile", "capture", or "calibration"
  reports/risk_stratification.md   — human-readable report

Usage
-----
    python src/analyze_risk_stratification.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

REPO_ROOT   = Path(__file__).resolve().parent.parent
DATA_PATH   = REPO_ROOT / "data" / "share" / "needle_features_v1.csv"
REPORTS_DIR = REPO_ROOT / "reports"

RANDOM_STATE = 42
PATIENT_COL  = "patient_number"
SPLIT_COL    = "split"
VALID_SPLITS = ("train", "val", "test")

LABEL_DESC = {
    "binary_label_int":         "GG2+ / csPCa (Gleason ≥ 3+4=7)",
    "binary_label_gg3plus_int": "GG3+ / high-grade (Gleason ≥ 4+3=7)",
}

# Models to run per endpoint.  xgboost entry is filtered out at runtime if
# the package is not installed.
ENDPOINT_MODELS: dict[str, list[str]] = {
    "binary_label_int":         ["logistic_regression", "hist_gradient_boosting", "xgboost"],
    "binary_label_gg3plus_int": ["logistic_regression"],
}

# ── Feature lists (identical to run_shareable_tabular_experiments.py) ─────────

_BIOPSY_GEOMETRY = [
    "core_length_mm",
    "n_centerline_voxels",
    "n_tube_voxels",
    "midpoint_inside_prostate",
    "distance_midpoint_to_prostate_surface_mm",
    "approximate_fraction_of_centerline_inside_prostate",
]

_TARGET_GEOMETRY_NO_AVAIL = [
    "distance_midpoint_to_target_centroid_mm",
    "distance_midpoint_to_target_surface_mm",
    "trajectory_intersects_target",
    "approximate_fraction_of_centerline_inside_target",
]

_CLINICAL = [
    "psa_ng_ml",
    "log_psa_ng_ml",
    "prostate_volume_cc",
    "psa_density",
]

FEATURE_COLS: list[str] = _BIOPSY_GEOMETRY + _TARGET_GEOMETRY_NO_AVAIL + _CLINICAL

# Boolean features: most-frequent imputed, never standard-scaled.
BOOLEAN_FEATURES: set[str] = {"midpoint_inside_prostate", "trajectory_intersects_target"}

CAPTURE_THRESHOLDS = [0.05, 0.10, 0.20]
N_DECILES          = 10
N_CALIB_BINS       = 10


# ── Data loading & preparation (mirrors main experiment script) ──────────────

def _to_binary(series: pd.Series) -> pd.Series:
    truthy = {True, "True", "TRUE", "true", 1, "1", 1.0}
    falsy  = {False, "False", "FALSE", "false", 0, "0", 0.0}
    out = pd.Series(np.nan, index=series.index, dtype=float)
    out[series.isin(truthy)] = 1.0
    out[series.isin(falsy)]  = 0.0
    return out


def load_and_prepare(label_col: str) -> pd.DataFrame:
    raw = pd.read_csv(DATA_PATH)

    mask = (
        (raw["label_join_status"] == "coord_match")
        & raw[label_col].notna()
        & raw[SPLIT_COL].isin(VALID_SPLITS)
    )
    df = raw[mask].copy()
    df[label_col] = df[label_col].astype(int)

    df["log_psa_ng_ml"] = np.log1p(df["psa_ng_ml"])
    valid_vol = df["prostate_volume_cc"].notna() & (df["prostate_volume_cc"] > 0)
    df["psa_density"] = np.where(
        valid_vol & df["psa_ng_ml"].notna(),
        df["psa_ng_ml"] / df["prostate_volume_cc"],
        np.nan,
    )
    for col in BOOLEAN_FEATURES:
        if col in df.columns:
            df[col] = _to_binary(df[col])

    return df.reset_index(drop=True)


def usable_features(feature_cols: list[str], train: pd.DataFrame, available: set[str]) -> list[str]:
    """Drop features missing from the dataset or entirely NaN in train."""
    cols = []
    for c in feature_cols:
        if c not in available:
            warnings.warn(f"Feature '{c}' not in dataset — skipping.")
            continue
        if train[c].isna().all():
            warnings.warn(f"Feature '{c}' all-NaN in train — skipping.")
            continue
        cols.append(c)
    return cols


# ── Pipeline construction (mirrors main experiment script) ───────────────────

def build_pipeline(model_name: str, feature_cols: list[str], y_train: pd.Series) -> Pipeline | None:
    numeric_cols = [c for c in feature_cols if c not in BOOLEAN_FEATURES]
    bool_cols    = [c for c in feature_cols if c in BOOLEAN_FEATURES]

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    if model_name == "logistic_regression":
        estimator    = LogisticRegression(class_weight="balanced", max_iter=5000, C=1.0,
                                          random_state=RANDOM_STATE)
        scale_num    = True
    elif model_name == "hist_gradient_boosting":
        estimator    = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
        scale_num    = False
    elif model_name == "xgboost":
        if not HAS_XGBOOST:
            return None
        estimator    = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
            scale_pos_weight=scale_pos_weight, eval_metric="auc",
            random_state=RANDOM_STATE, verbosity=0,
        )
        scale_num    = False
    else:
        return None

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_num:
        num_steps.append(("scaler", StandardScaler()))

    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline(num_steps), numeric_cols))
    if bool_cols:
        transformers.append(("bool", Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]), bool_cols))

    preprocessor = ColumnTransformer(transformers)
    return Pipeline([("preprocess", preprocessor), ("clf", estimator)])


# ── Risk stratification helpers ───────────────────────────────────────────────

def decile_table(y_true: np.ndarray, probs: np.ndarray) -> list[dict]:
    """
    Assign each test core to a risk decile (10 = highest risk, 1 = lowest).
    Deciles are equal-count bins based on rank.
    """
    n = len(probs)
    order = np.argsort(probs)[::-1]          # descending predicted risk
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)       # rank 1 = highest predicted risk

    # decile: rank 1..ceil(n/10) → decile 10, ..., rank n-ceil(n/10)+1..n → decile 1
    decile_labels = N_DECILES + 1 - np.ceil(ranks / n * N_DECILES).astype(int)
    decile_labels = np.clip(decile_labels, 1, N_DECILES)

    rows = []
    for d in range(N_DECILES, 0, -1):        # report 10 → 1 (highest risk first)
        mask = decile_labels == d
        y_d  = y_true[mask]
        p_d  = probs[mask]
        n_d  = int(mask.sum())
        n_pos = int(y_d.sum())
        rows.append(dict(
            table="decile",
            decile=d,
            n_cores=n_d,
            n_positives=n_pos,
            observed_prevalence=n_pos / n_d if n_d > 0 else np.nan,
            mean_predicted_prob=float(p_d.mean()) if n_d > 0 else np.nan,
            min_predicted_prob=float(p_d.min())  if n_d > 0 else np.nan,
            max_predicted_prob=float(p_d.max())  if n_d > 0 else np.nan,
        ))
    return rows


def capture_table(y_true: np.ndarray, probs: np.ndarray) -> list[dict]:
    """
    For each threshold (5%, 10%, 20%), compute the fraction of all positives
    captured in the top-N% highest-risk cores.
    """
    order = np.argsort(probs)[::-1]
    total_pos = int(y_true.sum())
    rows = []
    for pct in CAPTURE_THRESHOLDS:
        k = max(1, int(np.floor(len(probs) * pct)))
        top_idx = order[:k]
        n_pos_top = int(y_true[top_idx].sum())
        rows.append(dict(
            table="capture",
            top_pct=pct,
            n_cores_in_top=k,
            n_positives_in_top=n_pos_top,
            total_positives=total_pos,
            capture_rate=n_pos_top / total_pos if total_pos > 0 else np.nan,
        ))
    return rows


def calibration_table(y_true: np.ndarray, probs: np.ndarray) -> list[dict]:
    """
    10 equal-width probability bins [0, 0.1), [0.1, 0.2), …, [0.9, 1.0].
    For each bin: n_cores, n_positives, observed_prevalence, mean_predicted_prob.
    """
    edges = np.linspace(0.0, 1.0, N_CALIB_BINS + 1)
    rows = []
    for i in range(N_CALIB_BINS):
        lo, hi = edges[i], edges[i + 1]
        if i < N_CALIB_BINS - 1:
            mask = (probs >= lo) & (probs < hi)
        else:
            mask = (probs >= lo) & (probs <= hi)   # include right edge for last bin
        y_b = y_true[mask]
        p_b = probs[mask]
        n_b = int(mask.sum())
        n_pos = int(y_b.sum())
        rows.append(dict(
            table="calibration",
            prob_bin_low=lo,
            prob_bin_high=hi,
            n_cores=n_b,
            n_positives=n_pos,
            observed_prevalence=n_pos / n_b if n_b > 0 else np.nan,
            mean_predicted_prob=float(p_b.mean()) if n_b > 0 else np.nan,
        ))
    return rows


# ── Markdown formatting ───────────────────────────────────────────────────────

def _f(v, dec: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{dec}f}"
    return str(int(v)) if isinstance(v, (int, np.integer)) else str(v)


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%" if (v is not None and not np.isnan(v)) else "n/a"


def _md_deciles(rows: list[dict]) -> list[str]:
    lines = [
        "| Decile | Risk tier | Cores | Positives | Observed prevalence "
        "| Mean predicted prob | Min | Max |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        d = r["decile"]
        tier = "highest" if d == 10 else ("lowest" if d == 1 else "")
        lines.append(
            f"| {d} | {tier} "
            f"| {_f(r['n_cores'], 0)} | {_f(r['n_positives'], 0)} "
            f"| {_pct(r['observed_prevalence'])} "
            f"| {_f(r['mean_predicted_prob'])} "
            f"| {_f(r['min_predicted_prob'])} "
            f"| {_f(r['max_predicted_prob'])} |"
        )
    return lines


def _md_capture(rows: list[dict]) -> list[str]:
    lines = [
        "| Top % of cores | Cores in top | Positives captured "
        "| Total positives | Capture rate |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {_pct(r['top_pct'])} "
            f"| {_f(r['n_cores_in_top'], 0)} "
            f"| {_f(r['n_positives_in_top'], 0)} "
            f"| {_f(r['total_positives'], 0)} "
            f"| {_pct(r['capture_rate'])} |"
        )
    return lines


def _md_calibration(rows: list[dict]) -> list[str]:
    lines = [
        "| Predicted prob bin | Cores | Positives | Observed prevalence "
        "| Mean predicted |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        bin_label = f"[{r['prob_bin_low']:.1f}, {r['prob_bin_high']:.1f})"
        lines.append(
            f"| {bin_label} "
            f"| {_f(r['n_cores'], 0)} "
            f"| {_f(r['n_positives'], 0)} "
            f"| {_pct(r['observed_prevalence'])} "
            f"| {_f(r['mean_predicted_prob'])} |"
        )
    return lines


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== Risk stratification analysis ===\n")
    print(f"Feature set : all_geometry_no_availability_plus_clinical")
    print(f"  ({len(FEATURE_COLS)} features, target_mesh_available excluded)\n")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    all_csv_rows: list[dict] = []
    md_sections:  list[str]  = []

    for label_col, desc in LABEL_DESC.items():
        model_names = [
            m for m in ENDPOINT_MODELS[label_col]
            if m != "xgboost" or HAS_XGBOOST
        ]

        print(f"Endpoint : {label_col} ({desc})")
        df = load_and_prepare(label_col)
        available = set(df.columns)

        train_df = df[df[SPLIT_COL] == "train"]
        test_df  = df[df[SPLIT_COL] == "test"]

        feat_cols = usable_features(FEATURE_COLS, train_df, available)
        print(f"  {len(feat_cols)}/{len(FEATURE_COLS)} features usable")

        n_test     = len(test_df)
        n_test_pos = int(test_df[label_col].sum())
        prevalence = n_test_pos / n_test if n_test > 0 else 0.0
        print(f"  Test : {n_test:,} cores, {n_test_pos:,} positives ({prevalence*100:.1f}%)")

        X_train = train_df[feat_cols]
        y_train = train_df[label_col]
        X_test  = test_df[feat_cols]
        y_test  = test_df[label_col].values

        md_sections += [
            f"## {label_col} — {desc}",
            "",
            f"- Test cores: {n_test:,}",
            f"- Test positives: {n_test_pos:,} ({_pct(prevalence)})",
            f"- Features: {len(feat_cols)} (all_geometry_no_availability_plus_clinical)",
            "",
        ]

        for model_name in model_names:
            print(f"  [{model_name}] fitting …", end=" ", flush=True)

            pipeline = build_pipeline(model_name, feat_cols, y_train)
            if pipeline is None:
                print("skipped (not available)")
                continue

            pipeline.fit(X_train, y_train)
            probs = pipeline.predict_proba(X_test)[:, 1]

            roc_auc = float(roc_auc_score(y_test, probs)) if len(np.unique(y_test)) > 1 else np.nan
            pr_auc  = float(average_precision_score(y_test, probs)) if len(np.unique(y_test)) > 1 else np.nan
            print(f"ROC-AUC={roc_auc:.3f}  PR-AUC={pr_auc:.3f}")

            d_rows   = decile_table(y_test, probs)
            cap_rows = capture_table(y_test, probs)
            cal_rows = calibration_table(y_test, probs)

            # Tag every row with metadata
            meta = dict(label_col=label_col, model=model_name,
                        test_roc_auc=roc_auc, test_pr_auc=pr_auc,
                        n_test=n_test, n_test_positives=n_test_pos)
            for r in d_rows + cap_rows + cal_rows:
                all_csv_rows.append({**meta, **r})

            # ── Markdown section for this model ──────────────────────────────
            md_sections += [
                f"### {model_name}",
                "",
                f"Test ROC-AUC = **{_f(roc_auc)}**  ·  PR-AUC = **{_f(pr_auc)}**  "
                f"·  baseline PR-AUC = **{_pct(prevalence)}** (no-skill)",
                "",
                "#### Risk deciles (10 = highest risk)",
                "",
            ]
            md_sections += _md_deciles(d_rows)
            md_sections += [
                "",
                "#### Positive capture in top highest-risk cores",
                "",
            ]
            md_sections += _md_capture(cap_rows)
            md_sections += [
                "",
                "#### Calibration (predicted vs observed prevalence)",
                "",
            ]
            md_sections += _md_calibration(cal_rows)
            md_sections.append("")

        print()

    # ── Write outputs ─────────────────────────────────────────────────────────
    csv_path = REPORTS_DIR / "risk_stratification.csv"
    pd.DataFrame(all_csv_rows).to_csv(csv_path, index=False)
    print(f"Saved -> {csv_path.relative_to(REPO_ROOT)}")

    md_header = [
        "# Risk Stratification Analysis",
        "",
        "Feature set: **all_geometry_no_availability_plus_clinical**  ",
        "(`target_mesh_available` excluded by design)",
        "",
        "Probabilities are predicted on the test split.  "
        "No hard threshold is applied.  ",
        "Decile 10 = highest predicted risk; decile 1 = lowest.",
        "",
    ]
    md_path = REPORTS_DIR / "risk_stratification.md"
    md_path.write_text("\n".join(md_header + md_sections) + "\n")
    print(f"Saved -> {md_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
