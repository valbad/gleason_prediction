"""
Probability calibration analysis for the main full_geometry_dataset models.

For each (endpoint, model) the base pipeline is fit on the train split and
probabilities are predicted on val and test.  A logistic recalibration model
(Platt scaling on the logit scale) is then fit on the val predictions and
applied to the test probabilities.  Calibration metrics are reported before
and after recalibration.

Calibration metrics
-------------------
Brier score         : mean squared error between predicted prob and label
                      (lower is better; null model = prevalence × (1-prevalence))
Brier skill score   : 1 - Brier / Brier_null  (higher is better; 0 = no skill)
Calibration intercept (a) :  logistic fit of label ~ a + b*logit(pred);
                              a ≈ 0 → correct calibration-in-the-large
Calibration slope (b) :      b ≈ 1 → correct spread; b < 1 → overconfident;
                              b > 1 → underconfident
ECE                 : expected calibration error over 10 equal-width bins

Recalibration
-------------
Fit LogisticRegression on val-split logit(predicted_probs) → val labels.
Apply to test logit(predicted_probs) to get recalibrated test probabilities.

Feature set
-----------
all_geometry_no_availability_plus_clinical (target_mesh_available excluded).
Identical to analyze_risk_stratification.py.

Outputs
-------
  reports/calibration_analysis.csv   — tidy rows; "table" column tags each
                                        row as "metrics" or "bins"
  reports/calibration_analysis.md

Usage
-----
    python src/analyze_calibration.py
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
from sklearn.metrics import brier_score_loss
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

RANDOM_STATE  = 42
PATIENT_COL   = "patient_number"
SPLIT_COL     = "split"
VALID_SPLITS  = ("train", "val", "test")
N_CALIB_BINS  = 10
LOGIT_EPS     = 1e-6

LABEL_DESC = {
    "binary_label_int":         "GG2+ / csPCa (Gleason ≥ 3+4=7)",
    "binary_label_gg3plus_int": "GG3+ / high-grade (Gleason ≥ 4+3=7)",
}

ENDPOINT_MODELS: dict[str, list[str]] = {
    "binary_label_int":         ["logistic_regression", "xgboost"],
    "binary_label_gg3plus_int": ["logistic_regression"],
}

# ── Feature lists (mirror run_shareable_tabular_experiments.py) ───────────────

_BIOPSY_GEOMETRY = [
    "core_length_mm", "n_centerline_voxels", "n_tube_voxels",
    "midpoint_inside_prostate", "distance_midpoint_to_prostate_surface_mm",
    "approximate_fraction_of_centerline_inside_prostate",
]
_TARGET_GEOMETRY_NO_AVAIL = [
    "distance_midpoint_to_target_centroid_mm",
    "distance_midpoint_to_target_surface_mm",
    "trajectory_intersects_target",
    "approximate_fraction_of_centerline_inside_target",
]
_CLINICAL = ["psa_ng_ml", "log_psa_ng_ml", "prostate_volume_cc", "psa_density"]

FEATURE_COLS: list[str] = _BIOPSY_GEOMETRY + _TARGET_GEOMETRY_NO_AVAIL + _CLINICAL
BOOLEAN_FEATURES: set[str] = {"midpoint_inside_prostate", "trajectory_intersects_target"}


# ── Data loading & preparation ────────────────────────────────────────────────

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


def usable_features(feature_cols: list[str], train: pd.DataFrame) -> list[str]:
    cols = []
    for c in feature_cols:
        if c not in train.columns:
            warnings.warn(f"Feature '{c}' not in dataset — skipping.")
            continue
        if train[c].isna().all():
            warnings.warn(f"Feature '{c}' all-NaN in train — skipping.")
            continue
        cols.append(c)
    return cols


# ── Pipeline (mirrors analyze_risk_stratification.py) ────────────────────────

def build_pipeline(model_name: str, feature_cols: list[str], y_train: pd.Series) -> Pipeline | None:
    numeric_cols = [c for c in feature_cols if c not in BOOLEAN_FEATURES]
    bool_cols    = [c for c in feature_cols if c in BOOLEAN_FEATURES]

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    spw   = n_neg / max(n_pos, 1)

    if model_name == "logistic_regression":
        est, scale_num = (
            LogisticRegression(class_weight="balanced", max_iter=5000,
                               C=1.0, random_state=RANDOM_STATE),
            True,
        )
    elif model_name == "xgboost":
        if not HAS_XGBOOST:
            return None
        est, scale_num = (
            XGBClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                scale_pos_weight=spw, eval_metric="auc",
                random_state=RANDOM_STATE, verbosity=0,
            ),
            False,
        )
    else:
        return None

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_num:
        num_steps.append(("scaler", StandardScaler()))

    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline(num_steps), numeric_cols))
    if bool_cols:
        transformers.append(("bool",
                             Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]),
                             bool_cols))

    return Pipeline([("preprocess", ColumnTransformer(transformers)), ("clf", est)])


# ── Calibration helpers ───────────────────────────────────────────────────────

def logit_safe(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, LOGIT_EPS, 1 - LOGIT_EPS)
    return np.log(p / (1 - p))


def calibration_slope_intercept(
    y_true: np.ndarray,
    probs: np.ndarray,
) -> tuple[float, float]:
    """
    Fit label ~ a + b * logit(prob) and return (intercept a, slope b).

    a ≈ 0 and b ≈ 1 indicate good calibration.
    b < 1 → over-confident (predictions too extreme).
    b > 1 → under-confident (predictions too conservative).
    """
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    X = logit_safe(probs).reshape(-1, 1)
    cal = LogisticRegression(fit_intercept=True, max_iter=1000, C=1e6)
    cal.fit(X, y_true)
    return float(cal.intercept_[0]), float(cal.coef_[0, 0])


def ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = N_CALIB_BINS) -> float:
    """Expected Calibration Error over equal-width probability bins."""
    n     = len(y_true)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    err   = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs >= lo) & (probs <= hi) if i == n_bins - 1 else (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        obs_prev  = y_true[mask].mean()
        mean_pred = probs[mask].mean()
        err += mask.sum() / n * abs(obs_prev - mean_pred)
    return float(err)


def calibration_bins(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = N_CALIB_BINS,
) -> list[dict]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows  = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs >= lo) & (probs <= hi) if i == n_bins - 1 else (probs >= lo) & (probs < hi)
        n_b   = int(mask.sum())
        n_pos = int(y_true[mask].sum()) if n_b > 0 else 0
        rows.append(dict(
            bin_index=i + 1,
            prob_bin=f"[{lo:.1f}, {hi:.1f}{']]'[i == n_bins - 1]}",
            n_cores=n_b,
            n_positives=n_pos,
            observed_prevalence=n_pos / n_b if n_b > 0 else np.nan,
            mean_predicted_prob=float(probs[mask].mean()) if n_b > 0 else np.nan,
        ))
    return rows


def compute_calibration_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    prevalence: float,
) -> dict:
    brier     = float(brier_score_loss(y_true, probs))
    brier_null = prevalence * (1 - prevalence)
    brier_ss  = 1.0 - brier / brier_null if brier_null > 0 else np.nan
    cal_int, cal_slope = calibration_slope_intercept(y_true, probs)
    ece_val   = ece(y_true, probs)
    return dict(
        brier_score=brier,
        brier_null=brier_null,
        brier_skill_score=brier_ss,
        cal_intercept=cal_int,
        cal_slope=cal_slope,
        ece=ece_val,
    )


def fit_recalibrator(
    y_val: np.ndarray,
    val_probs: np.ndarray,
) -> LogisticRegression | None:
    """Fit Platt recalibrator on logit(val_probs) → val labels."""
    if len(np.unique(y_val)) < 2:
        warnings.warn("Val split has only one class — recalibration skipped.")
        return None
    recal = LogisticRegression(fit_intercept=True, max_iter=1000, C=1e6)
    recal.fit(logit_safe(val_probs).reshape(-1, 1), y_val)
    return recal


def apply_recalibrator(
    recal: LogisticRegression,
    probs: np.ndarray,
) -> np.ndarray:
    return recal.predict_proba(logit_safe(probs).reshape(-1, 1))[:, 1]


# ── Formatting ────────────────────────────────────────────────────────────────

def _f(v, dec: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{dec}f}"
    return str(int(v)) if isinstance(v, (int, np.integer)) else str(v)


def _pct(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{float(v) * 100:.1f}%"


def interpretation(
    model_name: str,
    label_col: str,
    brier_before: float,
    brier_after: float,
    slope_before: float,
    slope_after: float,
    ece_before: float,
    ece_after: float,
    prevalence: float,
    low_bin_frac_after: float = 0.0,
) -> list[str]:
    """Generate a short plain-English calibration interpretation."""
    lines = ["**Interpretation:**", ""]

    # Ranking vs calibration note
    if slope_before < 0.8:
        conf_note = (
            f"The base model is **over-confident**: its probability spread is too extreme "
            f"(calibration slope = {_f(slope_before)}).  "
        )
    elif slope_before > 1.2:
        conf_note = (
            f"The base model is **under-confident**: its predicted probabilities cluster "
            f"too close to the prevalence (calibration slope = {_f(slope_before)}).  "
        )
    else:
        conf_note = (
            f"The base model slope is near 1 ({_f(slope_before)}), "
            f"suggesting reasonable spread.  "
        )

    # ECE assessment
    if ece_before < 0.05:
        ece_note = f"ECE ({_f(ece_before)}) is below 0.05, consistent with adequate calibration."
    elif ece_before < 0.10:
        ece_note = f"ECE ({_f(ece_before)}) is moderate; predictions are not strongly miscalibrated."
    else:
        ece_note = f"ECE ({_f(ece_before)}) is high (> 0.10), indicating significant miscalibration."

    # Recalibration gain
    ece_delta = ece_before - ece_after
    if abs(ece_delta) < 0.005:
        recal_note = "Recalibration produces negligible ECE improvement."
    elif ece_delta > 0:
        recal_note = (
            f"Recalibration reduces ECE from {_f(ece_before)} to {_f(ece_after)} "
            f"(−{_f(ece_delta, 3)}), indicating the raw probabilities were shifted "
            f"relative to the true positive rate."
        )
    else:
        recal_note = (
            f"Recalibration slightly increases ECE ({_f(ece_before)} → {_f(ece_after)}), "
            f"which may reflect overfitting on the small val split."
        )

    # Low-bin concentration caution (triggered when >90% of samples fall in [0, 0.1])
    low_bin_caution = (
        "> **Caution:** After recalibration, most predictions are concentrated in the "
        "lowest probability bin, which improves average calibration metrics but suggests "
        "that absolute risk estimates remain conservative and should be interpreted cautiously."
        if low_bin_frac_after > 0.90 else ""
    )

    # Ranking vs calibration conclusion
    conclusion = (
        f"For this endpoint (`{label_col}`, prevalence ≈ {_pct(prevalence)}), "
        f"{'the model is primarily useful as a **ranking tool**' if ece_before > 0.07 else 'the model can be used both for ranking and as a calibrated probability estimator'}."
        f" Recalibration is {'recommended before reporting absolute risk values' if ece_before > 0.05 else 'optional but may help in low-prevalence settings'}."
    )

    lines += [conf_note + ece_note, "", recal_note, ""]
    if low_bin_caution:
        lines += [low_bin_caution, ""]
    lines += [conclusion, ""]
    return lines


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== Calibration analysis ===\n")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    all_csv_rows: list[dict] = []
    md_lines: list[str] = [
        "# Probability Calibration Analysis",
        "",
        "Feature set: `all_geometry_no_availability_plus_clinical`  ",
        "(target_mesh_available excluded).  ",
        "Recalibration: logistic Platt scaling fitted on val-split logit(predicted probabilities).",
        "",
        "**Calibration slope** b ≈ 1 → well-spread; b < 1 → over-confident; b > 1 → under-confident.  ",
        "**ECE** = expected calibration error (10 equal-width bins); lower is better.  ",
        "**Brier skill score** = 1 − Brier / Brier_null; higher is better.",
        "",
    ]

    for label_col, desc in LABEL_DESC.items():
        model_names = [m for m in ENDPOINT_MODELS[label_col]
                       if m != "xgboost" or HAS_XGBOOST]

        print(f"Endpoint : {label_col}")
        df = load_and_prepare(label_col)

        train_df = df[df[SPLIT_COL] == "train"]
        val_df   = df[df[SPLIT_COL] == "val"]
        test_df  = df[df[SPLIT_COL] == "test"]

        feat_cols  = usable_features(FEATURE_COLS, train_df)
        prevalence = float(test_df[label_col].mean())

        md_lines += [
            f"## {label_col} — {desc}",
            "",
            f"- Test cores: {len(test_df):,}  ·  "
            f"Prevalence: **{_pct(prevalence)}**",
            "",
        ]

        # Summary metrics table (built per model, appended to md after loop)
        summary_rows: list[dict] = []

        for model_name in model_names:
            print(f"  [{model_name}]", end=" ", flush=True)

            pipeline = build_pipeline(model_name, feat_cols, train_df[label_col])
            if pipeline is None:
                print("skipped (not available)")
                continue

            pipeline.fit(train_df[feat_cols], train_df[label_col])

            val_probs  = pipeline.predict_proba(val_df[feat_cols])[:, 1]
            test_probs = pipeline.predict_proba(test_df[feat_cols])[:, 1]
            y_val      = val_df[label_col].values
            y_test     = test_df[label_col].values

            # ── Base calibration ──────────────────────────────────────────────
            base_m = compute_calibration_metrics(y_test, test_probs, prevalence)
            print(f"ECE={_f(base_m['ece'])}  slope={_f(base_m['cal_slope'])}", end="  ")

            meta = dict(label_col=label_col, model=model_name, prevalence=prevalence)

            for row in calibration_bins(y_test, test_probs):
                all_csv_rows.append({**meta, "table": "bins", "calibration": "before", **row})
            all_csv_rows.append({**meta, "table": "metrics", "calibration": "before", **base_m})

            # ── Recalibration ─────────────────────────────────────────────────
            recal = fit_recalibrator(y_val, val_probs)
            if recal is not None:
                test_probs_recal = apply_recalibrator(recal, test_probs)
                recal_int  = float(recal.intercept_[0])
                recal_slope = float(recal.coef_[0, 0])
                recal_m = compute_calibration_metrics(y_test, test_probs_recal, prevalence)
                print(f"→ recal ECE={_f(recal_m['ece'])}")

                for row in calibration_bins(y_test, test_probs_recal):
                    all_csv_rows.append({**meta, "table": "bins", "calibration": "after", **row})
                all_csv_rows.append({**meta, "table": "metrics", "calibration": "after",
                                     "recal_intercept": recal_int,
                                     "recal_slope": recal_slope,
                                     **recal_m})
            else:
                recal_m = {k: np.nan for k in base_m}
                recal_int, recal_slope = np.nan, np.nan
                print("(recalibration skipped — val has one class)")

            summary_rows.append(dict(
                model=model_name,
                **{f"before_{k}": v for k, v in base_m.items()},
                **{f"after_{k}":  v for k, v in recal_m.items()},
                recal_fit_intercept=recal_int if recal is not None else np.nan,
                recal_fit_slope=recal_slope if recal is not None else np.nan,
            ))

            # ── Markdown: bin table ───────────────────────────────────────────
            md_lines += [
                f"### {model_name}",
                "",
                "#### Calibration metrics summary",
                "",
                "| Metric | Before recalibration | After recalibration |",
                "|---|---|---|",
                f"| Brier score | {_f(base_m['brier_score'])} | {_f(recal_m['brier_score'])} |",
                f"| Brier skill score | {_f(base_m['brier_skill_score'])} | {_f(recal_m['brier_skill_score'])} |",
                f"| Calibration intercept | {_f(base_m['cal_intercept'])} | {_f(recal_m['cal_intercept'])} |",
                f"| Calibration slope | {_f(base_m['cal_slope'])} | {_f(recal_m['cal_slope'])} |",
                f"| ECE (10 bins) | {_f(base_m['ece'])} | {_f(recal_m['ece'])} |",
                "",
                "Recalibration model fitted on val split:  ",
                f"logit(p_recal) = {_f(recal_int)} + {_f(recal_slope)} × logit(p_base)",
                "",
                "#### Calibration bins — before recalibration",
                "",
                "| Bin | Cores | Positives | Observed prev | Mean predicted |",
                "|---|---|---|---|---|",
            ]
            for r in calibration_bins(y_test, test_probs):
                md_lines.append(
                    f"| {r['prob_bin']} | {r['n_cores']} | {r['n_positives']} "
                    f"| {_pct(r['observed_prevalence'])} "
                    f"| {_f(r['mean_predicted_prob'])} |"
                )
            md_lines += [
                "",
                "#### Calibration bins — after recalibration",
                "",
                "| Bin | Cores | Positives | Observed prev | Mean predicted |",
                "|---|---|---|---|---|",
            ]
            bins_after = (
                calibration_bins(y_test, test_probs_recal)
                if recal is not None
                else calibration_bins(y_test, test_probs)   # fallback: same as before
            )
            for r in bins_after:
                md_lines.append(
                    f"| {r['prob_bin']} | {r['n_cores']} | {r['n_positives']} "
                    f"| {_pct(r['observed_prevalence'])} "
                    f"| {_f(r['mean_predicted_prob'])} |"
                )
            md_lines.append("")

            # ── Interpretation paragraph ──────────────────────────────────────
            n_test = len(y_test)
            low_bin = next((r for r in bins_after if r["bin_index"] == 1), None)
            low_bin_frac_after = low_bin["n_cores"] / n_test if (low_bin and n_test > 0) else 0.0

            md_lines += interpretation(
                model_name, label_col,
                brier_before=base_m["brier_score"],
                brier_after=recal_m["brier_score"],
                slope_before=base_m["cal_slope"] if not np.isnan(base_m["cal_slope"]) else 1.0,
                slope_after=recal_m["cal_slope"]  if not np.isnan(recal_m["cal_slope"]) else 1.0,
                ece_before=base_m["ece"],
                ece_after=recal_m["ece"],
                prevalence=prevalence,
                low_bin_frac_after=low_bin_frac_after,
            )

        # ── Cross-model summary for this endpoint ─────────────────────────────
        if summary_rows:
            md_lines += [
                "### Cross-model calibration summary",
                "",
                "| Model | Brier (before→after) | ECE (before→after) | Slope (before→after) |",
                "|---|---|---|---|",
            ]
            for sr in summary_rows:
                md_lines.append(
                    f"| {sr['model']} "
                    f"| {_f(sr['before_brier_score'])} → {_f(sr['after_brier_score'])} "
                    f"| {_f(sr['before_ece'])} → {_f(sr['after_ece'])} "
                    f"| {_f(sr['before_cal_slope'])} → {_f(sr['after_cal_slope'])} |"
                )
            md_lines.append("")

        print()

    # ── Save outputs ──────────────────────────────────────────────────────────
    csv_path = REPORTS_DIR / "calibration_analysis.csv"
    pd.DataFrame(all_csv_rows).to_csv(csv_path, index=False)
    print(f"Saved -> {csv_path.relative_to(REPO_ROOT)}")

    md_path = REPORTS_DIR / "calibration_analysis.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"Saved -> {md_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
