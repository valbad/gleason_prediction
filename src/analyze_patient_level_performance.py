"""
Patient-level performance analysis.

Biopsy-core predictions are aggregated to the patient level using four
strategies, then evaluated as if each patient had a single risk score.

Aggregation strategies
----------------------
max_prob            : max predicted probability across the patient's cores
top3_mean_prob      : mean of the top-3 core probabilities (or all cores if < 3)
mean_prob           : mean predicted probability across all cores
prop_above_threshold: proportion of cores whose probability exceeds the
                      core-level Youden-J threshold (selected on val)

Patient label
-------------
A patient is positive (label = 1) if at least one of their test-split cores
is positive for the endpoint.

Threshold selection
-------------------
* Core-level Youden-J threshold is selected on the val split. It is used as
  the binarisation cutpoint for prop_above_threshold and reported in results.
* A patient-level Youden-J threshold is re-selected for each aggregation
  method on the patient-level val-split scores. This threshold drives the
  binary sensitivity / specificity / F1 computation.

Feature set (all_geometry_no_availability_plus_clinical)
---------------------------------------------------------
  biopsy_geometry: core_length_mm, n_centerline_voxels, n_tube_voxels,
                   midpoint_inside_prostate,
                   distance_midpoint_to_prostate_surface_mm,
                   approximate_fraction_of_centerline_inside_prostate
  target_geometry: distance_midpoint_to_target_centroid_mm,
  (no avail. flag) distance_midpoint_to_target_surface_mm,
                   trajectory_intersects_target,
                   approximate_fraction_of_centerline_inside_target
  clinical:        psa_ng_ml, log_psa_ng_ml, prostate_volume_cc, psa_density

  target_mesh_available is excluded by design.

Outputs
-------
  reports/patient_level_performance.csv  — tidy rows; "table" column tags
                                           each row as "metrics", "decile",
                                           or "capture"
  reports/patient_level_performance.md   — human-readable report

Usage
-----
    python src/analyze_patient_level_performance.py
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

ENDPOINT_MODELS: dict[str, list[str]] = {
    "binary_label_int":         ["logistic_regression", "xgboost"],
    "binary_label_gg3plus_int": ["logistic_regression"],
}

AGGREGATION_METHODS = [
    "max_prob",
    "top3_mean_prob",
    "mean_prob",
    "prop_above_threshold",
]

CAPTURE_PCTS = [0.10, 0.20]
N_DECILES    = 10

# ── Feature lists (mirror run_shareable_tabular_experiments.py) ──────────────

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

BOOLEAN_FEATURES: set[str] = {"midpoint_inside_prostate", "trajectory_intersects_target"}


# ── Data loading & preparation ───────────────────────────────────────────────

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
    available = set(train.columns)
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


# ── Pipeline (mirrors run_shareable_tabular_experiments.py) ──────────────────

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
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, scale_pos_weight=spw,
                eval_metric="auc", random_state=RANDOM_STATE, verbosity=0,
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


# ── Patient-level aggregation ────────────────────────────────────────────────

def patient_label(df: pd.DataFrame, label_col: str) -> pd.Series:
    """A patient is positive if any of their cores is positive."""
    return df.groupby(PATIENT_COL)[label_col].max().astype(int)


def aggregate_to_patients(
    core_df: pd.DataFrame,
    core_probs: np.ndarray,
    core_threshold: float,
    label_col: str,
) -> pd.DataFrame:
    """
    Combine core-level probabilities into one row per patient.

    Returns a DataFrame indexed by patient_number with columns:
        max_prob, top3_mean_prob, mean_prob, prop_above_threshold, patient_label
    """
    tmp = core_df[[PATIENT_COL, label_col]].copy()
    tmp["prob"] = core_probs

    rows = []
    for pid, grp in tmp.groupby(PATIENT_COL):
        probs_sorted = np.sort(grp["prob"].values)[::-1]
        top3 = probs_sorted[:3]
        rows.append({
            PATIENT_COL:           pid,
            "max_prob":            float(probs_sorted[0]),
            "top3_mean_prob":      float(top3.mean()),
            "mean_prob":           float(probs_sorted.mean()),
            "prop_above_threshold": float((probs_sorted >= core_threshold).mean()),
            "patient_label":       int(grp[label_col].max()),
        })
    return pd.DataFrame(rows).set_index(PATIENT_COL)


# ── Evaluation helpers ───────────────────────────────────────────────────────

def youden_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    j = tpr - fpr
    thr = thresholds[int(np.argmax(j))]
    return float(np.clip(thr, 0.0, 1.0))


def eval_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    if len(np.unique(y_true)) < 2:
        roc_auc = pr_auc = np.nan
    else:
        roc_auc = float(roc_auc_score(y_true, probs))
        pr_auc  = float(average_precision_score(y_true, probs))

    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return dict(
        roc_auc=roc_auc, pr_auc=pr_auc,
        sensitivity=sens, specificity=spec,
        precision=float(precision_score(y_true, preds, zero_division=0)),
        recall=float(recall_score(y_true, preds, zero_division=0)),
        f1=float(f1_score(y_true, preds, zero_division=0)),
    )


def capture_rates(y_true: np.ndarray, probs: np.ndarray, pcts: list[float]) -> dict:
    order     = np.argsort(probs)[::-1]
    total_pos = int(y_true.sum())
    result    = {}
    for pct in pcts:
        k     = max(1, int(np.floor(len(probs) * pct)))
        n_cap = int(y_true[order[:k]].sum())
        key   = f"capture_top{int(pct*100)}pct"
        result[key] = n_cap / total_pos if total_pos > 0 else np.nan
    return result


def decile_rows(
    y_true: np.ndarray,
    probs: np.ndarray,
    meta: dict,
) -> list[dict]:
    """Assign each patient to a risk decile (10 = highest risk, 1 = lowest)."""
    n     = len(probs)
    order = np.argsort(probs)[::-1]
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    deciles = N_DECILES + 1 - np.ceil(ranks / n * N_DECILES).astype(int)
    deciles = np.clip(deciles, 1, N_DECILES)

    rows = []
    for d in range(N_DECILES, 0, -1):
        mask  = deciles == d
        y_d   = y_true[mask]
        p_d   = probs[mask]
        n_d   = int(mask.sum())
        n_pos = int(y_d.sum())
        rows.append({
            **meta,
            "table":               "decile",
            "decile":              d,
            "n_patients":          n_d,
            "n_positive_patients": n_pos,
            "observed_prevalence": n_pos / n_d if n_d > 0 else np.nan,
            "mean_predicted_risk": float(p_d.mean()) if n_d > 0 else np.nan,
        })
    return rows


# ── Markdown formatting ───────────────────────────────────────────────────────

def _f(v, dec: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{dec}f}"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return str(v)


def _pct(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{float(v)*100:.1f}%"


AGG_LABEL = {
    "max_prob":             "Max prob",
    "top3_mean_prob":       "Top-3 mean",
    "mean_prob":            "Mean prob",
    "prop_above_threshold": "Prop > thr",
}


def md_metrics_table(results: list[dict]) -> list[str]:
    cols = [
        ("Aggregation",    lambda r: AGG_LABEL.get(r["aggregation"], r["aggregation"])),
        ("Threshold",      lambda r: _f(r["patient_threshold"])),
        ("ROC-AUC",        lambda r: _f(r["roc_auc"])),
        ("PR-AUC",         lambda r: _f(r["pr_auc"])),
        ("Sensitivity",    lambda r: _f(r["sensitivity"])),
        ("Specificity",    lambda r: _f(r["specificity"])),
        ("F1",             lambda r: _f(r["f1"])),
        ("Capture @10%",   lambda r: _pct(r.get("capture_top10pct"))),
        ("Capture @20%",   lambda r: _pct(r.get("capture_top20pct"))),
    ]
    header = "| " + " | ".join(c[0] for c in cols) + " |"
    sep    = "|" + "---|" * len(cols)
    lines  = [header, sep]
    for r in results:
        lines.append("| " + " | ".join(fn(r) for _, fn in cols) + " |")
    return lines


def md_decile_table(rows: list[dict]) -> list[str]:
    lines = [
        "| Decile | Patients | Positives | Observed prevalence | Mean predicted risk |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        tier = " ← highest risk" if r["decile"] == 10 else (" ← lowest risk" if r["decile"] == 1 else "")
        lines.append(
            f"| {r['decile']}{tier} "
            f"| {_f(r['n_patients'])} "
            f"| {_f(r['n_positive_patients'])} "
            f"| {_pct(r['observed_prevalence'])} "
            f"| {_f(r['mean_predicted_risk'])} |"
        )
    return lines


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== Patient-level performance analysis ===\n")
    print(f"Feature set : all_geometry_no_availability_plus_clinical")
    print(f"  ({len(FEATURE_COLS)} features, target_mesh_available excluded)\n")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    all_csv_rows: list[dict] = []
    md_lines: list[str] = [
        "# Patient-level Performance Analysis",
        "",
        "Core-level predicted probabilities are aggregated to the patient level.",
        "A patient is **positive** if at least one of their cores is positive for the endpoint.",
        "",
        "**Core-level Youden-J threshold** is selected on the val split and used for  ",
        "`prop_above_threshold`.  A **patient-level Youden-J threshold** is also selected  ",
        "for each aggregation method on val-split patient scores — this drives binary metrics.",
        "",
        "Decile 10 = highest predicted risk; decile 1 = lowest.",
        "",
    ]

    for label_col, desc in LABEL_DESC.items():
        model_names = [m for m in ENDPOINT_MODELS[label_col]
                       if m != "xgboost" or HAS_XGBOOST]

        print(f"Endpoint : {label_col} ({desc})")

        df = load_and_prepare(label_col)
        train_df = df[df[SPLIT_COL] == "train"]
        val_df   = df[df[SPLIT_COL] == "val"]
        test_df  = df[df[SPLIT_COL] == "test"]

        feat_cols = usable_features(FEATURE_COLS, train_df)
        print(f"  {len(feat_cols)}/{len(FEATURE_COLS)} features usable")

        # Patient labels (val and test splits)
        pat_label_val  = patient_label(val_df,  label_col)
        pat_label_test = patient_label(test_df, label_col)
        n_test_pat     = len(pat_label_test)
        n_test_pos_pat = int(pat_label_test.sum())
        pat_prev       = n_test_pos_pat / n_test_pat if n_test_pat > 0 else 0.0

        print(f"  Test patients : {n_test_pat}  positives : {n_test_pos_pat} ({pat_prev*100:.1f}%)")

        md_lines += [
            f"## {label_col} — {desc}",
            "",
            f"- Test patients: **{n_test_pat}**",
            f"- Positive test patients: **{n_test_pos_pat}** ({_pct(pat_prev)})",
            f"- Val patients: {val_df[PATIENT_COL].nunique()}",
            "",
        ]

        for model_name in model_names:
            print(f"  [{model_name}]", end=" ")

            pipeline = build_pipeline(model_name, feat_cols, train_df[label_col])
            if pipeline is None:
                print("skipped (not available)")
                continue

            pipeline.fit(train_df[feat_cols], train_df[label_col])

            # Core-level val predictions → core-level Youden threshold
            val_core_probs  = pipeline.predict_proba(val_df[feat_cols])[:, 1]
            core_threshold  = youden_threshold(val_df[label_col].values, val_core_probs)

            # Core-level test predictions
            test_core_probs = pipeline.predict_proba(test_df[feat_cols])[:, 1]

            # Aggregate to patient level (val and test)
            pat_val  = aggregate_to_patients(val_df,  val_core_probs,  core_threshold, label_col)
            pat_test = aggregate_to_patients(test_df, test_core_probs, core_threshold, label_col)

            # Align patient labels with aggregated frames
            pat_val  = pat_val.join(pat_label_val.rename("_true"),  how="inner")
            pat_test = pat_test.join(pat_label_test.rename("_true"), how="inner")

            model_results: list[dict] = []

            md_lines += [
                f"### {model_name}",
                "",
                f"Core-level Youden-J threshold (val): **{core_threshold:.3f}**",
                "",
            ]

            for agg in AGGREGATION_METHODS:
                val_scores  = pat_val[agg].values
                test_scores = pat_test[agg].values
                val_true    = pat_val["_true"].values
                test_true   = pat_test["_true"].values

                # Patient-level threshold selected on val
                pat_thr = youden_threshold(val_true, val_scores)

                metrics = eval_metrics(test_true, test_scores, pat_thr)
                caps    = capture_rates(test_true, test_scores, CAPTURE_PCTS)

                meta = dict(
                    label_col=label_col,
                    model=model_name,
                    aggregation=agg,
                    core_threshold=core_threshold,
                    patient_threshold=pat_thr,
                    n_test_patients=n_test_pat,
                    n_positive_test_patients=n_test_pos_pat,
                    patient_prevalence=pat_prev,
                )
                result_row = {**meta, **metrics, **caps, "table": "metrics"}
                model_results.append(result_row)
                all_csv_rows.append(result_row)

                # Decile rows
                for dr in decile_rows(test_true, test_scores, meta):
                    all_csv_rows.append(dr)

                # Capture CSV rows
                for pct in CAPTURE_PCTS:
                    key = f"capture_top{int(pct*100)}pct"
                    all_csv_rows.append({
                        **meta,
                        "table":             "capture",
                        "top_pct":           pct,
                        "n_patients_in_top": max(1, int(np.floor(n_test_pat * pct))),
                        "capture_rate":      caps[key],
                    })

                print(f"{agg}: ROC-AUC={_f(metrics['roc_auc'])}  ", end="")

            print()

            # ── Markdown: summary table ─────────────────────────────────────
            md_lines += [
                "#### Performance by aggregation method",
                "",
                *md_metrics_table(model_results),
                "",
            ]

            # ── Markdown: decile tables for each aggregation ────────────────
            for agg in AGGREGATION_METHODS:
                val_scores  = pat_val[agg].values
                test_scores = pat_test[agg].values
                test_true   = pat_test["_true"].values

                meta_d = dict(label_col=label_col, model=model_name, aggregation=agg)
                d_rows = decile_rows(test_true, test_scores, meta_d)

                md_lines += [
                    f"<details><summary>Risk deciles — {AGG_LABEL[agg]}</summary>",
                    "",
                    *md_decile_table(d_rows),
                    "",
                    "</details>",
                    "",
                ]

        print()

    # ── Save CSV ─────────────────────────────────────────────────────────────
    csv_path = REPORTS_DIR / "patient_level_performance.csv"
    pd.DataFrame(all_csv_rows).to_csv(csv_path, index=False)
    print(f"Saved -> {csv_path.relative_to(REPO_ROOT)}")

    # ── Save Markdown ─────────────────────────────────────────────────────────
    md_path = REPORTS_DIR / "patient_level_performance.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"Saved -> {md_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
